"""
无人机 RF 检测站 — 全 Python 一体化（N310 采集 / 可视化 / S3R 开放集推理）

启动（工作目录 DroneDetect_V2，解释器 tf210）:
  python s3r_detector_software/drone_rf_station_app.py

模式:
  1. 采集与可视化 — 仅采 .dat + 片段 STFT 预览（等同 MATLAB 采集后看图）
  2. 全流程实时检测 — 采集 → STFT+Z-score → Rim-3σ 推断
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

_pkg = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_pkg)  # .../Desktop/USRP
if __package__ in (None, ''):
    for p in (_root, _pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

from iq_preprocess import DEFAULT_PREPROCESS, iq_to_zscore_spectrogram
from n310_stream import (
    FS_60,
    HAS_UHD,
    HW_RATE_PRESETS,
    N310Config,
    N310LiveStream,
    N310Recorder,
    decim_for_hw_rate_mhz,
    display_spectrogram_db,
    load_dat_matlab,
    signal_quality,
)
# 注意: S3RDetector 依赖 TensorFlow 与上级模型脚本，采集功能并不需要它，
# 因此改为「加载模型」时才延迟导入，保证无模型/无 TF 的机器也能打开做采集。

# 采集数据默认落 E 盘，避免占满 C 盘；可在界面改「数据根目录」
DEFAULT_RECORD_ROOT = r'E:\USRP\recordings'

# 与 MATLAB Receiver_N310_base.m 一致的飞行状态定义
STATE_LABELS = (('ON', '开机不起飞'), ('HO', '起飞悬停'), ('FY', '平移'))


def save_recording_diagnostics(iq_60: np.ndarray, png_path: str, seg_sec: float) -> None:
    """采集后自动存诊断图（|IQ| 时域 + 复信号 FFT + 中段 STFT），对齐 MATLAB 采集即出图。

    使用独立 Figure + Agg 画布，可在工作线程内安全调用（不触碰 pyplot 全局状态）。
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    iq = np.asarray(iq_60)
    iq_dc = iq - np.mean(iq)
    n_total = len(iq_dc)
    if n_total == 0:
        return

    fig = Figure(figsize=(12, 8), dpi=120)
    FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(2, 2, 1)
    n_win = min(n_total, int(FS_60 * 0.002))
    t_ms = np.arange(n_win) / FS_60 * 1e3
    ax1.plot(t_ms, np.abs(iq_dc[:n_win]), lw=0.5, color='#2f7d32')
    ax1.set_xlabel('时间 (ms)')
    ax1.set_ylabel('|I+jQ|')
    ax1.set_title('复信号幅度时域（前 2 ms）')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    nfft = min(n_total, 1 << 19)
    spec = np.fft.fftshift(np.fft.fft(iq_dc[:nfft]))
    f_mhz = (np.arange(-nfft // 2, nfft - nfft // 2) * (FS_60 / nfft)) / 1e6
    ax2.plot(f_mhz, 10 * np.log10(np.abs(spec) ** 2 + 1e-12), lw=0.6, color='#1565c0')
    ax2.set_xlim(-FS_60 / 2 / 1e6, FS_60 / 2 / 1e6)
    ax2.set_xlabel('频率 (MHz)')
    ax2.set_ylabel('功率 (dB)')
    ax2.set_title('复信号 FFT（真实频率）')
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 1, 2)
    seg_n = int(FS_60 * seg_sec)
    if seg_n > 0 and n_total >= seg_n:
        start = max(0, (n_total - seg_n) // 2)
        f, t, db = display_spectrogram_db(iq[start:start + seg_n])
        vmax = float(np.max(db))
        vmin = vmax - 80
        extent = [t[0] * 1e3, t[-1] * 1e3, f[0] / 1e6, f[-1] / 1e6]
        im = ax3.imshow(db, aspect='auto', origin='lower', extent=extent,
                        cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax3, label='幅度 (dB)')
        ax3.set_title(f'中段 STFT（{seg_sec * 1e3:.0f} ms @ {start / FS_60:.2f}s）')
    else:
        ax3.text(0.5, 0.5, '数据不足以绘制 STFT', ha='center', va='center')
    ax3.set_xlabel('时间 (ms)')
    ax3.set_ylabel('频率 (MHz)')

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight')


class DroneRFStationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('无人机 RF 检测站 — N310')
        self.root.minsize(1100, 720)
        self.root.geometry('1280x800')

        self.detector: S3RDetector | None = None
        self.model_path: str | None = None
        self.iq_60: np.ndarray | None = None
        self.iq_source_label = ''
        self.live_stream: N310LiveStream | None = None
        self._live_running = False
        self._record_running = False
        self._infer_stats = {'known': 0, 'unknown': 0, 'total': 0}
        self._infer_lock = threading.Lock()

        self._build_layout()
        self._bind_keys()

    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── 左侧：设备与模型 ──
        left = ttk.Frame(self.root, padding=8, width=300)
        left.grid(row=0, column=0, sticky='ns')
        left.grid_propagate(False)

        ttk.Label(left, text='USRP N310', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        uhd_txt = 'PyUHD 已就绪' if HAS_UHD else '未检测到 PyUHD'
        uhd_color = '#1a7f37' if HAS_UHD else '#a11'
        ttk.Label(left, text=uhd_txt, foreground=uhd_color).pack(anchor='w', pady=(0, 8))

        dev = ttk.LabelFrame(left, text='射频参数', padding=6)
        dev.pack(fill='x', pady=4)
        self._row(dev, 'IP 地址', 'addr_var', N310Config().addr)
        self._row(dev, '中心频率 GHz', 'freq_var', '2.4375')
        self._row(dev, '增益 dB', 'gain_var', '50')
        ttk.Label(dev, text='天线').grid(row=3, column=0, sticky='w', pady=2)
        self.ant_var = tk.StringVar(value='TX/RX')
        ttk.Combobox(dev, textvariable=self.ant_var, values=('TX/RX', 'RX2'), width=12).grid(
            row=3, column=1, sticky='ew', pady=2,
        )
        ttk.Label(dev, text='硬件采样率 MS/s').grid(row=4, column=0, sticky='w', pady=2)
        hw_labels = [p[0] for p in HW_RATE_PRESETS]
        self.hw_rate_var = tk.StringVar(value=hw_labels[0])
        hw_cb = ttk.Combobox(
            dev, textvariable=self.hw_rate_var, values=hw_labels, width=12, state='readonly',
        )
        hw_cb.grid(row=4, column=1, sticky='ew', pady=2)
        ttk.Label(dev, text='片段时长 s').grid(row=5, column=0, sticky='w', pady=2)
        self.seg_var = tk.StringVar(value=str(DEFAULT_PREPROCESS.segment_length))
        ttk.Entry(dev, textvariable=self.seg_var, width=14).grid(row=5, column=1, sticky='ew', pady=2)
        dev.columnconfigure(1, weight=1)

        mdl = ttk.LabelFrame(left, text='检测模型（全流程用）', padding=6)
        mdl.pack(fill='x', pady=4)
        ttk.Button(mdl, text='选择 run 目录…', command=self._load_model).pack(fill='x')
        self.model_lbl = ttk.Label(mdl, text='未加载', wraplength=260, foreground='#444')
        self.model_lbl.pack(anchor='w', pady=(6, 0))

        out = ttk.LabelFrame(left, text='采集保存', padding=6)
        out.pack(fill='x', pady=4)
        out.columnconfigure(1, weight=1)

        ttk.Button(out, text='选择数据根目录…', command=self._pick_save_dir).grid(
            row=0, column=0, columnspan=2, sticky='ew',
        )
        self.save_dir_var = tk.StringVar(value=DEFAULT_RECORD_ROOT)
        ttk.Label(out, textvariable=self.save_dir_var, wraplength=260, foreground='#555').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(4, 6),
        )

        self.test_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            out, text='测试采集（存入 test/，无需命名）',
            variable=self.test_mode_var, command=self._on_test_mode_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky='w', pady=(0, 4))

        ttk.Label(out, text='型号').grid(row=3, column=0, sticky='w', pady=2)
        self.drone_model_var = tk.StringVar(value='')
        self.ent_model = ttk.Entry(out, textvariable=self.drone_model_var, width=14)
        self.ent_model.grid(row=3, column=1, sticky='ew', pady=2)

        ttk.Label(out, text='个体 ID').grid(row=4, column=0, sticky='w', pady=2)
        self.drone_id_var = tk.StringVar(value='')
        self.ent_id = ttk.Entry(out, textvariable=self.drone_id_var, width=14)
        self.ent_id.grid(row=4, column=1, sticky='ew', pady=2)

        ttk.Label(out, text='状态').grid(row=5, column=0, sticky='w', pady=2)
        self.state_var = tk.StringVar(value=STATE_LABELS[0][0])
        self.cb_state = ttk.Combobox(
            out, textvariable=self.state_var, width=12, state='readonly',
            values=[f'{code} ({desc})' for code, desc in STATE_LABELS],
        )
        self.cb_state.current(0)
        self.cb_state.grid(row=5, column=1, sticky='ew', pady=2)

        self.next_path_lbl = ttk.Label(out, text='', wraplength=260, foreground='#1a7f37')
        self.next_path_lbl.grid(row=6, column=0, columnspan=2, sticky='w', pady=(6, 0))

        for var in (self.save_dir_var, self.drone_model_var, self.drone_id_var, self.state_var):
            var.trace_add('write', lambda *_: self._refresh_next_path())

        self.hw_rate_hint = ttk.Label(
            left,
            text=self._hw_rate_hint_text(),
            foreground='#666', justify='left', wraplength=280,
        )
        self.hw_rate_hint.pack(anchor='w', pady=(0, 4))
        self.hw_rate_var.trace_add('write', lambda *_: self._on_hw_rate_change())
        ttk.Label(
            left,
            text='N310 万兆: PC 设 192.168.20.1/24 MTU9000\n千兆口则 192.168.10.1；保存前重采样至 60 MHz',
            foreground='#666', justify='left',
        ).pack(anchor='w', pady=8)

        # ── 右侧：选项卡 ──
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky='nsew')

        self._build_tab_acquire()
        self._build_tab_pipeline()

        self.status = ttk.Label(self.root, text='就绪', relief=tk.SUNKEN, anchor='w', padding=4)
        self.status.grid(row=1, column=0, columnspan=2, sticky='ew')

        self._refresh_next_path()

    def _row(self, parent, label, attr, default):
        r = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky='w', pady=2)
        var = tk.StringVar(value=str(default))
        setattr(self, attr, var)
        ttk.Entry(parent, textvariable=var, width=14).grid(row=r, column=1, sticky='ew', pady=2)

    def _hw_rate_hint_text(self) -> str:
        try:
            hw = float(self.hw_rate_var.get())
        except (ValueError, AttributeError):
            hw = 61.44
        gbps = hw * 4 * 8 / 1000  # sc16 IQ, Gbps
        if hw >= 60:
            net = '网口约 {:.1f} Gbps，建议万兆'.format(gbps)
        elif hw >= 28:
            net = '网口约 {:.1f} Gbps，适合千兆'.format(gbps)
        else:
            net = '网口约 {:.2f} Gbps，千兆宽裕'.format(gbps)
        return f'硬件 {hw} MS/s → 输出 60 MS/s\n{net}'

    def _on_hw_rate_change(self):
        if hasattr(self, 'hw_rate_hint'):
            self.hw_rate_hint.config(text=self._hw_rate_hint_text())

    def _n310_cfg(self) -> N310Config:
        hw_mhz = float(self.hw_rate_var.get())
        return N310Config(
            addr=self.addr_var.get().strip(),
            centre_freq=float(self.freq_var.get()) * 1e9,
            gain=float(self.gain_var.get()),
            antenna=self.ant_var.get().strip(),
            decim_factor=decim_for_hw_rate_mhz(hw_mhz),
            segment_sec=self._seg_sec(),
        )

    def _seg_sec(self) -> float:
        return float(self.seg_var.get())

    def _seg_samples_60(self) -> int:
        return int(FS_60 * self._seg_sec())

    # ────────────────── 保存路径（对齐 MATLAB 目录/自增编号） ──────────────────

    def _state_code(self) -> str:
        return self.state_var.get().split(' ', 1)[0].strip().upper()

    def _on_test_mode_toggle(self):
        is_test = self.test_mode_var.get()
        field_state = tk.DISABLED if is_test else tk.NORMAL
        self.ent_model.config(state=field_state)
        self.ent_id.config(state=field_state)
        self.cb_state.config(state=tk.DISABLED if is_test else 'readonly')
        self._refresh_next_path()

    def _target_dir(self) -> str:
        """正式: {根}/{型号}_{ID}/{状态}；测试: {根}/test。"""
        root = self.save_dir_var.get().strip()
        if self.test_mode_var.get():
            return os.path.join(root, 'test')
        model = self.drone_model_var.get().strip().upper()
        ident = self.drone_id_var.get().strip()
        return os.path.join(root, f'{model}_{ident}', self._state_code())

    @staticmethod
    def _next_index(target_dir: str) -> int:
        """扫描目录内 {整数}.dat，返回下一个可用编号（与 MATLAB 一致）。"""
        idx = 0
        if os.path.isdir(target_dir):
            for name in os.listdir(target_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() == '.dat' and stem.isdigit():
                    idx = max(idx, int(stem) + 1)
        return idx

    def _resolve_save_path(self) -> tuple[str | None, str | None]:
        """返回 (保存路径, 错误信息)；参数不全时路径为 None。"""
        if not self.save_dir_var.get().strip():
            return None, '请先选择数据根目录'
        if not self.test_mode_var.get():
            if not self.drone_model_var.get().strip():
                return None, '请填写无人机型号'
            if not self.drone_id_var.get().strip():
                return None, '请填写个体 ID'
        target = self._target_dir()
        return os.path.join(target, f'{self._next_index(target)}.dat'), None

    def _refresh_next_path(self):
        if not hasattr(self, 'next_path_lbl'):
            return
        try:
            path, err = self._resolve_save_path()
        except Exception as exc:  # 目录/参数异常不应影响界面
            path, err = None, str(exc)
        if path:
            root = self.save_dir_var.get().strip()
            try:
                shown = os.path.relpath(path, root)
            except ValueError:
                shown = path
            self.next_path_lbl.config(text=f'将保存为: {shown}', foreground='#1a7f37')
        else:
            self.next_path_lbl.config(text=err or '', foreground='#a11')

    # ────────────────── Tab1: 采集与可视化 ──────────────────

    def _build_tab_acquire(self):
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text='  采集与可视化  ')
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        bar = ttk.Frame(tab)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        self.btn_record = ttk.Button(bar, text='▶ 开始采集', command=self._toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text='打开 .dat…', command=self._open_dat).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text='录制时长 s').pack(side=tk.LEFT, padx=(16, 4))
        self.dur_var = tk.StringVar(value='2.0')
        ttk.Entry(bar, textvariable=self.dur_var, width=6).pack(side=tk.LEFT)
        self.record_prog = ttk.Progressbar(bar, length=180, mode='determinate')
        self.record_prog.pack(side=tk.LEFT, padx=12)

        info = ttk.Frame(tab)
        info.grid(row=1, column=0, sticky='ew')
        self.qual_lbl = ttk.Label(info, text='信号: —', font=('Consolas', 10))
        self.qual_lbl.pack(side=tk.LEFT)
        ttk.Label(info, text='预览片段').pack(side=tk.LEFT, padx=(24, 4))
        self.seg_slider = ttk.Scale(info, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_seg_slide)
        self.seg_slider.pack(side=tk.LEFT, fill='x', expand=True, padx=4)
        self.seg_idx_lbl = ttk.Label(info, text='#0', width=8)
        self.seg_idx_lbl.pack(side=tk.LEFT)

        fig_frame = ttk.LabelFrame(tab, text='复信号 STFT 幅度 (dB)', padding=4)
        fig_frame.grid(row=2, column=0, sticky='nsew')
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self.fig_acq = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax_acq = self.fig_acq.add_subplot(111)
        self.canvas_acq = FigureCanvasTkAgg(self.fig_acq, master=fig_frame)
        self.canvas_acq.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbar_box = tk.Frame(fig_frame)
        toolbar_box.grid(row=1, column=0, sticky='ew')
        NavigationToolbar2Tk(self.canvas_acq, toolbar_box)

    # ────────────────── Tab2: 全流程 ──────────────────

    def _build_tab_pipeline(self):
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text='  全流程实时检测  ')
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        bar = ttk.Frame(tab)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.btn_live = ttk.Button(bar, text='▶ 开始实时检测', command=self._toggle_live)
        self.btn_live.pack(side=tk.LEFT)
        self.live_stat = ttk.Label(bar, text='已知 0 | 未知 0 | 共 0', font=('Consolas', 10))
        self.live_stat.pack(side=tk.LEFT, padx=16)
        ttk.Label(
            bar,
            text='需先加载模型；每片段自动 STFT+Z-score+Rim-3σ',
            foreground='#555',
        ).pack(side=tk.RIGHT)

        paned = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky='nsew')

        log_frame = ttk.LabelFrame(paned, text='推断结果', padding=4)
        self.log_text = tk.Text(log_frame, height=20, font=('Consolas', 10), wrap=tk.WORD)
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        fig_frame = ttk.LabelFrame(paned, text='当前片段 STFT', padding=4)
        self.fig_live = plt.Figure(figsize=(5, 3.5), dpi=100)
        self.ax_live = self.fig_live.add_subplot(111)
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=fig_frame)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        paned.add(log_frame, weight=2)
        paned.add(fig_frame, weight=3)

    def _bind_keys(self):
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _set_status(self, msg: str):
        self.status.config(text=msg)

    def _load_model(self):
        path = filedialog.askdirectory(title='选择训练 run 目录')
        if not path:
            return
        self._set_status('加载模型…')

        def work():
            err = None
            det = None
            try:
                from s3r_inference_engine import S3RDetector  # 延迟导入，避免无 TF 时启动失败
                det = S3RDetector(path)
            except Exception as exc:
                err = exc

            def done():
                if err:
                    messagebox.showerror('模型加载失败', str(err))
                    return
                self.detector = det
                self.model_path = path
                classes = ', '.join(det.known_classes)
                self.model_lbl.config(text=f'{os.path.basename(path)}\n[{classes}]')
                self._set_status('模型已加载')

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _pick_save_dir(self):
        d = filedialog.askdirectory(initialdir=self.save_dir_var.get())
        if d:
            self.save_dir_var.set(d)

    def _set_iq(self, iq: np.ndarray, label: str):
        self.iq_60 = iq
        self.iq_source_label = label
        n_seg = max(0, len(iq) // self._seg_samples_60() - 1)
        self.seg_slider.config(to=max(0, n_seg))
        self._show_segment(0)
        q = signal_quality(iq)
        self.qual_lbl.config(
            text=f"信号: |IQ|max={q['max']:.4f}  p99={q['p99']:.4f}  — {q['message']}",
        )

    def _open_dat(self):
        path = filedialog.askopenfilename(filetypes=[('DAT', '*.dat'), ('All', '*.*')])
        if not path:
            return
        try:
            iq = load_dat_matlab(path)
            self._set_iq(iq, path)
            self._set_status(f'已加载 {os.path.basename(path)}  {len(iq)/FS_60:.2f}s')
        except Exception as exc:
            messagebox.showerror('读取失败', str(exc))

    def _on_seg_slide(self, _val):
        if self.iq_60 is None:
            return
        idx = int(float(self.seg_slider.get()))
        self._show_segment(idx)

    def _show_segment(self, idx: int):
        if self.iq_60 is None:
            return
        n = self._seg_samples_60()
        start = idx * n
        if start + n > len(self.iq_60):
            return
        seg = self.iq_60[start:start + n]
        self.seg_idx_lbl.config(text=f'#{idx} @ {start/FS_60:.2f}s')
        self._plot_spectrogram(seg, self.ax_acq, self.fig_acq, self.canvas_acq,
                               title=f'{self.iq_source_label}  片段 {idx}')

    def _plot_spectrogram(self, seg, ax, fig, canvas, *, title=''):
        f, t, db = display_spectrogram_db(seg)
        ax.clear()
        vmax = float(np.max(db))
        vmin = vmax - 80
        extent = [t[0] * 1000, t[-1] * 1000, f[0] / 1e6, f[-1] / 1e6]
        ax.imshow(db, aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_xlabel('时间 (ms)')
        ax.set_ylabel('频率 (MHz)')
        ax.set_title(title or '复信号 STFT')
        fig.tight_layout()
        canvas.draw_idle()

    def _toggle_record(self):
        if self._record_running:
            return
        if not HAS_UHD:
            messagebox.showwarning('PyUHD', '未安装 PyUHD，无法采集。可先用「打开 .dat」预览。')
            return
        try:
            dur = float(self.dur_var.get())
        except ValueError:
            messagebox.showerror('参数', '录制时长无效')
            return

        path, path_err = self._resolve_save_path()
        if path_err:
            messagebox.showerror('参数', path_err)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._record_running = True
        self.btn_record.config(text='采集中…', state=tk.DISABLED)
        self.record_prog['value'] = 0
        cfg = self._n310_cfg()
        seg_sec = self._seg_sec()
        hw_mhz = cfg.hw_sample_rate / 1e6
        self._set_status(f'N310 采集中 ({hw_mhz:.2f} MS/s)…')

        def work():
            err = None
            result = None
            diag_path = None
            try:
                rec = N310Recorder(cfg)

                def prog(p):
                    self.root.after(0, lambda: self.record_prog.config(value=p * 100))

                result = rec.record(dur, path, on_progress=prog, segment_sec=seg_sec)
                diag_path = os.path.splitext(path)[0] + '_diag.png'
                try:
                    save_recording_diagnostics(result.iq_60, diag_path, seg_sec)
                except Exception as exc:  # 诊断图失败不影响采集结果
                    print(f'[诊断图] 保存失败: {exc}', file=sys.stderr)
                    diag_path = None
            except Exception as exc:
                err = exc

            def done():
                self._record_running = False
                self.btn_record.config(text='▶ 开始采集', state=tk.NORMAL)
                self.record_prog['value'] = 0
                if err:
                    messagebox.showerror('采集失败', str(err))
                    self._set_status('采集失败')
                    return
                self._set_iq(result.iq_60, path)
                self._refresh_next_path()

                msg = f'已保存:\n{path}\nIQ时长 {result.duration_sec:.2f}s（进度条按收满样点，不是墙上计时）'
                if result.actual_hw_rate:
                    msg += f'\n硬件采样率 {result.actual_hw_rate / 1e6:.4f} MS/s'
                if diag_path:
                    msg += f'\n诊断图: {os.path.basename(diag_path)}'
                seg_ms = result.segment_sec * 1e3
                msg += (
                    f'\n已存 {result.complete_segments} 段×{seg_ms:.0f}ms'
                    f'（每段内部时间连续）'
                )
                if result.overflow_count:
                    warn = (
                        f'overflow×{result.overflow_count}：未凑满的半段已丢弃，'
                        f'只保留完整连续的 {seg_ms:.0f}ms 片段再拼接。\n'
                        f'丢弃硬件样点约 {result.dropped_hw_samples}；'
                        f'段与段之间可能有墙钟缺口，但按 {seg_ms:.0f}ms 从文件头切片时每段连续。\n'
                    )
                    if result.overflow_meta_path:
                        warn += f'标记: {os.path.basename(result.overflow_meta_path)}\n'
                    warn += '建议先降到 30.72 MS/s；USB万兆满载 61.44 仍可能不稳。'
                    self._set_status(f'已保存（{result.complete_segments} 段连续）{path}')
                    messagebox.showwarning('采集完成（overflow 已按段剔除）', f'{msg}\n\n⚠ {warn}')
                else:
                    self._set_status(f'已保存 {path}')
                    messagebox.showinfo('采集完成', msg)

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _toggle_live(self):
        if self._live_running:
            self._stop_live()
            return
        if not HAS_UHD:
            messagebox.showwarning('PyUHD', '未安装 PyUHD')
            return
        if self.detector is None:
            messagebox.showwarning('模型', '请先在左侧加载检测模型')
            return

        self._infer_stats = {'known': 0, 'unknown': 0, 'total': 0}
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, '--- 实时检测开始 ---\n')
        self._live_running = True
        self.btn_live.config(text='■ 停止')
        cfg = self._n310_cfg()
        hw_mhz = cfg.hw_sample_rate / 1e6
        self._set_status(f'检测运行中 ({hw_mhz:.2f} MS/s → 60 MS/s)…')

        det = self.detector
        seg_sec = self._seg_sec()
        cfg = self._n310_cfg()

        def on_seg(seg, idx):
            import time as _time
            t0 = _time.perf_counter()
            try:
                spec = iq_to_zscore_spectrogram(seg)
                with self._infer_lock:
                    pred = det.predict(spec)
                ms = (_time.perf_counter() - t0) * 1000
                self.root.after(0, lambda s=seg, p=pred, i=idx, m=ms: self._on_live_pred(s, p, i, m))
            except Exception as exc:
                self.root.after(0, lambda e=exc, i=idx: self._log(f'[错误] seg{i}: {e}\n'))

        def on_err(exc):
            self.root.after(0, lambda: messagebox.showerror('USRP', str(exc)))
            self.root.after(0, self._stop_live)

        self.live_stream = N310LiveStream(
            cfg, segment_sec=seg_sec,
            on_segment_60=on_seg, on_error=on_err,
        )
        self.live_stream.start()

    def _on_live_pred(self, seg, pred, idx, ms):
        if not self._live_running:
            return
        self._plot_spectrogram(
            seg, self.ax_live, self.fig_live, self.canvas_live,
            title=f'片段 {idx}',
        )
        tag = '已知' if pred.is_known else '未知'
        self._infer_stats['total'] += 1
        if pred.is_known:
            self._infer_stats['known'] += 1
        else:
            self._infer_stats['unknown'] += 1
        s = self._infer_stats
        self.live_stat.config(text=f"已知 {s['known']} | 未知 {s['unknown']} | 共 {s['total']}")
        line = f"[{tag}] {pred.class_name:8s}  margin={pred.min_margin:.3f}  {ms:.0f}ms\n"
        self._log(line)

    def _log(self, text: str):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _stop_live(self):
        self._live_running = False
        if self.live_stream:
            self.live_stream.stop()
            self.live_stream = None
        self.btn_live.config(text='▶ 开始实时检测')
        self._log('--- 已停止 ---\n')
        self._set_status('已停止实时检测')

    def _on_close(self):
        self._stop_live()
        self.root.destroy()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except Exception:
        pass
    DroneRFStationApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
