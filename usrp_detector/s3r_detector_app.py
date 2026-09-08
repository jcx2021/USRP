"""
USRP 无人机 RF 开放集检测 — 桌面演示。

用法:
  python usrp_detector/s3r_detector_app.py
  python -m usrp_detector.s3r_detector_app
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

# 支持直接运行脚本: python usrp_detector/s3r_detector_app.py
if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)

from s3r_inference_engine import S3RDetector, format_prediction, prepare_batch


class S3RDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('USRP 无人机 RF 开放集检测')
        self.root.geometry('1100x720')

        self.detector: S3RDetector | None = None
        self.npy_data: np.ndarray | None = None
        self.npy_path: str | None = None

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        model_frame = ttk.LabelFrame(main, text='模型（训练 run 目录）', padding=8)
        model_frame.pack(fill=tk.X, **pad)

        ttk.Button(model_frame, text='选择 run 目录', command=self.pick_run_dir).grid(row=0, column=0, sticky=tk.W)
        self.run_label = ttk.Label(model_frame, text='未加载', wraplength=900)
        self.run_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        self.model_info = ttk.Label(model_frame, text='', foreground='#444')
        self.model_info.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        input_frame = ttk.LabelFrame(main, text='输入数据（Z-score STFT 谱图）', padding=8)
        input_frame.pack(fill=tk.X, **pad)

        ttk.Button(input_frame, text='选择 NPY 文件', command=self.pick_npy).grid(row=0, column=0, sticky=tk.W)
        self.npy_label = ttk.Label(input_frame, text='未选择文件')
        self.npy_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(input_frame, text='样本模式:').grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.sample_mode = tk.StringVar(value='single')
        ttk.Radiobutton(
            input_frame, text='单样本', variable=self.sample_mode, value='single',
        ).grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Radiobutton(
            input_frame, text='整个文件', variable=self.sample_mode, value='all',
        ).grid(row=1, column=2, sticky=tk.W, pady=(8, 0))

        ttk.Label(input_frame, text='样本索引:').grid(row=2, column=0, sticky=tk.W)
        self.sample_index = tk.StringVar(value='0')
        ttk.Entry(input_frame, textvariable=self.sample_index, width=10).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(
            input_frame,
            text='数据格式: samples_zscore.npy  —  (N,2,H,W) 或 (N,H,W,2)，与训练 OpenSetDataGenerator 一致',
            foreground='#555',
        ).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, **pad)
        self.detect_btn = ttk.Button(action_frame, text='开始检测', command=self.run_detect)
        self.detect_btn.pack(side=tk.LEFT)
        self.status_label = ttk.Label(action_frame, text='就绪')
        self.status_label.pack(side=tk.LEFT, padx=(12, 0))

        result_frame = ttk.LabelFrame(main, text='检测结果', padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.result_text = tk.Text(result_frame, height=24, wrap=tk.WORD, font=('Consolas', 10))
        scroll = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def pick_run_dir(self):
        path = filedialog.askdirectory(title='选择训练 run 目录（含 evaluation_results.json）')
        if not path:
            return
        self._load_detector_async(path)

    def _load_detector_async(self, run_dir: str):
        self.status_label.config(text='正在加载模型…')
        self.detect_btn.config(state=tk.DISABLED)

        def worker():
            err = None
            detector = None
            try:
                detector = S3RDetector(run_dir)
            except Exception as exc:
                err = exc

            def done():
                self.detect_btn.config(state=tk.NORMAL)
                if err:
                    self.status_label.config(text='加载失败')
                    messagebox.showerror('加载失败', str(err))
                    return
                self.detector = detector
                cfg = detector.summary()
                self.run_label.config(text=run_dir)
                inf = cfg.eval_cfg.get('s3r_inference') or {}
                self.model_info.config(
                    text=(
                        f"损失: {cfg.eval_cfg.get('loss_name')}  |  "
                        f"骨干: {cfg.eval_cfg.get('backbone_preset')}  |  "
                        f"已知类: {', '.join(cfg.known_classes)}  |  "
                        f"推断: Rim-3σ ({cfg.metric})  margin_bias={cfg.margin_bias:.4f}  |  "
                        f"θ: g={inf.get('tuned_theta_global_scale')} mp={inf.get('tuned_theta_mp_scale')}"
                    )
                )
                self.status_label.config(text='模型已加载')

            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def pick_npy(self):
        path = filedialog.askopenfilename(
            title='选择 NPY 数据',
            filetypes=[('NumPy', '*.npy'), ('所有文件', '*.*')],
        )
        if not path:
            return
        try:
            data = np.load(path, mmap_mode='r')
            self.npy_data = data
            self.npy_path = path
            self.npy_label.config(text=f'{os.path.basename(path)}  shape={data.shape}  dtype={data.dtype}')
        except Exception as exc:
            messagebox.showerror('读取失败', str(exc))

    def run_detect(self):
        if self.detector is None:
            messagebox.showwarning('提示', '请先选择并加载 run 目录')
            return
        if self.npy_data is None:
            messagebox.showwarning('提示', '请先选择 NPY 文件')
            return

        try:
            if self.sample_mode.get() == 'single':
                idx = int(self.sample_index.get())
                if idx < 0 or idx >= len(self.npy_data):
                    raise IndexError(f'索引超出范围 0..{len(self.npy_data) - 1}')
                batch = np.asarray(self.npy_data[idx:idx + 1])
                indices = [idx]
            else:
                batch = np.asarray(self.npy_data)
                indices = list(range(len(batch)))
        except Exception as exc:
            messagebox.showerror('输入错误', str(exc))
            return

        self.status_label.config(text='检测中…')
        self.detect_btn.config(state=tk.DISABLED)
        detector = self.detector
        input_shape = detector.input_shape

        def worker():
            err = None
            lines = []
            preds = []
            try:
                x = prepare_batch(batch, input_shape)
                preds = detector.predict_batch(x)
                known_n = sum(1 for p in preds if p.is_known)
                unknown_n = len(preds) - known_n
                lines.append(f'文件: {self.npy_path}')
                lines.append(f'样本数: {len(preds)}  已知: {known_n}  未知: {unknown_n}\n')
                show_max = min(len(preds), 50)
                for i in range(show_max):
                    lines.append(f'--- 样本 {indices[i]} ---')
                    lines.append(format_prediction(preds[i]))
                    lines.append('')
                if len(preds) > show_max:
                    lines.append(f'… 另有 {len(preds) - show_max} 条结果未显示')
            except Exception as exc:
                err = exc

            def done():
                self.detect_btn.config(state=tk.NORMAL)
                self.result_text.delete('1.0', tk.END)
                if err:
                    self.status_label.config(text='检测失败')
                    messagebox.showerror('检测失败', str(err))
                    return
                self.result_text.insert(tk.END, '\n'.join(lines))
                self.status_label.config(text=f'完成 — {len(preds)} 样本')

            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    root = tk.Tk()
    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass
    S3RDetectorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
