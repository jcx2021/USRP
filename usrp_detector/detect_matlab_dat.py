"""
MATLAB Receiver_N310_base.m 与 Python 检测的衔接。

MATLAB 负责:
  N310 @ 61.44 MHz → resample(125/128) → 60 MHz → 存 .dat (float32 I/Q 交错)

Python 负责:
  读 .dat → 0.05s 分段 → STFT+zoom+crop+Z-score（训练一致）→ S3R 开放集推断

不必用纯 Python 替代 MATLAB 采集；本模块用于 MATLAB 采完后的检测。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    for p in (_root, _pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

from iq_preprocess import DEFAULT_PREPROCESS, iq_to_zscore_spectrogram
from s3r_inference_engine import S3RDetector, format_prediction

# 与 Receiver_N310_base.m 一致
MATLAB_CENTER_FREQ = 2.4375e9
MATLAB_GAIN_DB = 50
MATLAB_SAMPLE_RATE = 60e6
MATLAB_MASTER_CLOCK = 122.88e6
MATLAB_DECIM = 2
MATLAB_HW_RATE = MATLAB_MASTER_CLOCK / MATLAB_DECIM  # 61.44 MHz
MATLAB_RESAMPLE_P = 125
MATLAB_RESAMPLE_Q = 128


def load_matlab_dat(path: str) -> np.ndarray:
    """读取 MATLAB fwrite 的 float32 小端 I/Q 交错 .dat → 复数向量 @ 60 MHz。"""
    data = np.fromfile(path, dtype='<f4')
    data = data[: len(data) // 2 * 2]
    iq = data.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


def iter_segments(iq: np.ndarray, seg_samples: int):
    """按固定长度滑动分段（不重叠，与离线切片一致）。"""
    n = len(iq)
    start = 0
    while start + seg_samples <= n:
        yield start, iq[start:start + seg_samples]
        start += seg_samples


def detect_matlab_dat(
    run_dir: str,
    dat_path: str,
    *,
    max_segments: int | None = None,
    segment_length: float | None = None,
) -> list[dict]:
    """
    对 MATLAB 保存的 .dat 逐段检测。

    返回每段: {index, start_sample, prediction, ...}
    """
    cfg = DEFAULT_PREPROCESS
    seg_len = segment_length if segment_length is not None else cfg.segment_length
    seg_samples = int(cfg.sample_rate * seg_len)

    iq = load_matlab_dat(dat_path)
    detector = S3RDetector(run_dir)

    results = []
    for i, (start, seg) in enumerate(iter_segments(iq, seg_samples)):
        if max_segments is not None and i >= max_segments:
            break
        spec = iq_to_zscore_spectrogram(seg)
        pred = detector.predict(spec)
        results.append({
            'segment': i,
            'start_sample': start,
            'time_sec': start / cfg.sample_rate,
            'is_known': pred.is_known,
            'class_name': pred.class_name,
            'min_margin': pred.min_margin,
        })
    return results


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description='对 MATLAB Receiver_N310_base.m 保存的 .dat 做 S3R 开放集检测',
    )
    parser.add_argument('run_dir', help='训练 run 目录')
    parser.add_argument('dat_path', help='MATLAB 保存的 .dat 路径')
    parser.add_argument('--max-seg', type=int, default=None, help='最多检测几段（默认全部）')
    parser.add_argument('--seg-len', type=float, default=0.05, help='片段长度秒（默认 0.05）')
    args = parser.parse_args()

    print(f'加载: {args.dat_path}')
    print(f'模型: {args.run_dir}')
    print(f'片段: {args.seg_len}s @ {MATLAB_SAMPLE_RATE/1e6:.0f} MHz\n')

    rows = detect_matlab_dat(
        args.run_dir, args.dat_path,
        max_segments=args.max_seg,
        segment_length=args.seg_len,
    )
    known = sum(1 for r in rows if r['is_known'])
    print(f'共 {len(rows)} 段  已知={known}  未知={len(rows)-known}\n')
    for r in rows:
        tag = '已知' if r['is_known'] else '未知'
        print(
            f"  seg{r['segment']:3d}  t={r['time_sec']:6.2f}s  "
            f"[{tag}] {r['class_name']:8s}  margin={r['min_margin']:.4f}"
        )


if __name__ == '__main__':
    main()
