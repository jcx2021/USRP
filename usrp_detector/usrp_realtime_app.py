"""
USRP 实时开放集检测 GUI。

启动（工作目录 DroneDetect_V2，解释器 tf210）:
  python usrp_detector/usrp_realtime_app.py

依赖:
  - 已训练 run 目录（权重 + S3R 边界）
  - 实时模式需 PyUHD；无硬件可用「DAT 回放」调试
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    for p in (_root, _pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

# S3RDetector 依赖 TensorFlow，改为「加载模型」时延迟导入，无 TF 也能启动界面。
from usrp_realtime import (
    DEFAULT_PREPROCESS,
    DatReplayer,
    N310_DEFAULT,
    RealtimeS3RPipeline,
    UHDReceiver,
    USRPConfig,
    _HAS_UHD,
)


class USRPRealtimeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('USRP N310 实时无人机 RF 检测')
        self.root.geometry('900x640')

        self.detector: S3RDetector | None = None
        self.pipeline: RealtimeS3RPipeline | None = None
        self.source = None
        self._running = False

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 5}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 模型
        mf = ttk.LabelFrame(main, text='1. 加载模型', padding=8)
        mf.pack(fill=tk.X, **pad)
        ttk.Button(mf, text='选择 run 目录', command=self.load_model).grid(row=0, column=0)
        self.model_lbl = ttk.Label(mf, text='未加载')
        self.model_lbl.grid(row=0, column=1, padx=8, sticky=tk.W)

        # 信源
        sf = ttk.LabelFrame(main, text='2. 信号源', padding=8)
        sf.pack(fill=tk.X, **pad)
        self.src_mode = tk.StringVar(value='dat' if not _HAS_UHD else 'uhd')
        ttk.Radiobutton(sf, text='USRP N310 (PyUHD)', variable=self.src_mode, value='uhd',
                        state=tk.NORMAL if _HAS_UHD else tk.DISABLED).grid(row=0, column=0, sticky=tk.W)
        if not _HAS_UHD:
            ttk.Label(sf, text='(未检测到 PyUHD)', foreground='#a00').grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(sf, text='DAT 回放', variable=self.src_mode, value='dat').grid(row=1, column=0, sticky=tk.W)
        ttk.Button(sf, text='选择 .dat', command=self.pick_dat).grid(row=1, column=1, padx=4)
        self.dat_lbl = ttk.Label(sf, text='未选择')
        self.dat_lbl.grid(row=1, column=2, sticky=tk.W)
        self.dat_path: str | None = None

        uf = ttk.Frame(sf)
        uf.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        ttk.Label(uf, text='N310 IP:').pack(side=tk.LEFT)
        self.addr_var = tk.StringVar(value=N310_DEFAULT.addr)
        ttk.Entry(uf, textvariable=self.addr_var, width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(uf, text='中心频率 GHz:').pack(side=tk.LEFT, padx=(8, 0))
        self.freq_var = tk.StringVar(value='2.4375')
        ttk.Entry(uf, textvariable=self.freq_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(uf, text='增益 dB:').pack(side=tk.LEFT, padx=(8, 0))
        self.gain_var = tk.StringVar(value=str(int(N310_DEFAULT.gain)))
        ttk.Entry(uf, textvariable=self.gain_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(uf, text='天线:').pack(side=tk.LEFT, padx=(8, 0))
        self.ant_var = tk.StringVar(value=N310_DEFAULT.antenna)
        ttk.Combobox(uf, textvariable=self.ant_var, width=8, values=('TX/RX', 'RX2')).pack(side=tk.LEFT, padx=4)

        uf2 = ttk.Frame(sf)
        uf2.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
        ttk.Label(
            uf2,
            text='N310 万兆: PC 设 192.168.20.1/24 MTU9000；千兆口 192.168.10.1；先 uhd_find_devices'
            foreground='#555',
        ).pack(side=tk.LEFT)

        # 控制
        cf = ttk.Frame(main)
        cf.pack(fill=tk.X, **pad)
        self.start_btn = ttk.Button(cf, text='开始实时检测', command=self.toggle_run)
        self.start_btn.pack(side=tk.LEFT)
        self.status = ttk.Label(cf, text='就绪')
        self.status.pack(side=tk.LEFT, padx=12)

        ttk.Label(
            main,
            text=(
                f'每 {DEFAULT_PREPROCESS.segment_length}s ({DEFAULT_PREPROCESS.samples_per_segment / 1e6:.1f}M 样点) '
                f'→ STFT {DEFAULT_PREPROCESS.crop_size}×{DEFAULT_PREPROCESS.time_size} → Z-score → Rim-3σ 推断'
            ),
            foreground='#555',
        ).pack(anchor=tk.W, **pad)

        rf = ttk.LabelFrame(main, text='实时结果', padding=8)
        rf.pack(fill=tk.BOTH, expand=True, **pad)
        self.result = tk.Text(rf, height=18, font=('Consolas', 11), wrap=tk.WORD)
        self.result.pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        path = filedialog.askdirectory(title='选择训练 run 目录')
        if not path:
            return
        self.status.config(text='加载模型…')

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
                    messagebox.showerror('失败', str(err))
                    self.status.config(text='加载失败')
                    return
                self.detector = det
                self.model_lbl.config(
                    text=f"{path}  |  {', '.join(det.known_classes)}"
                )
                self.status.config(text='模型已加载')

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def pick_dat(self):
        p = filedialog.askopenfilename(filetypes=[('DAT', '*.dat'), ('All', '*.*')])
        if p:
            self.dat_path = p
            self.dat_lbl.config(text=os.path.basename(p))
            self.src_mode.set('dat')

    def _on_pred(self, pred, elapsed_ms):
        def ui():
            if isinstance(pred, Exception):
                self.result.insert(tk.END, f'[错误] {pred}\n')
            elif pred is None:
                self.result.insert(tk.END, '[错误] 推断失败\n')
            else:
                tag = '已知' if pred.is_known else '未知'
                line = (
                    f"[{tag}] {pred.class_name}  margin={pred.min_margin:.3f}  "
                    f"latency={elapsed_ms:.0f}ms\n"
                )
                self.result.insert(tk.END, line)
                self.result.see(tk.END)
            self.status.config(text=f'运行中 — 最近 {elapsed_ms:.0f} ms/帧')

        self.root.after(0, ui)

    def toggle_run(self):
        if self._running:
            self._stop_run()
            return
        if self.detector is None:
            messagebox.showwarning('提示', '请先加载模型')
            return

        mode = self.src_mode.get()
        if mode == 'dat' and not self.dat_path:
            messagebox.showwarning('提示', '请选择 .dat 文件')
            return

        self.pipeline = RealtimeS3RPipeline(self.detector, on_result=self._on_pred)
        self.pipeline.start()

        try:
            if mode == 'uhd':
                cfg = USRPConfig(
                    addr=self.addr_var.get().strip(),
                    centre_freq=float(self.freq_var.get()) * 1e9,
                    gain=float(self.gain_var.get()),
                    antenna=self.ant_var.get().strip(),
                )
                self.source = UHDReceiver(cfg, self.pipeline)
            else:
                self.source = DatReplayer(self.dat_path, self.pipeline)
            self.source.start()
        except Exception as exc:
            self.pipeline.stop()
            messagebox.showerror('启动失败', str(exc))
            return

        self._running = True
        self.start_btn.config(text='停止')
        self.result.insert(tk.END, f'--- 开始 ({mode}) ---\n')
        self.status.config(text='运行中…')

    def _stop_run(self):
        if self.source:
            self.source.stop()
            self.source = None
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self._running = False
        self.start_btn.config(text='开始实时检测')
        self.status.config(text='已停止')
        self.result.insert(tk.END, '--- 停止 ---\n')


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try:
                s.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    root = tk.Tk()
    USRPRealtimeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
