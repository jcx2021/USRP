"""
IQ → STFT → Z-score 预处理（与训练数据 data_open_set_*_512,320 对齐）。

训练/采集管线:
  segment_length=0.05s, n_fft=1024, hop=50%
  zoom → 512×512 → fftshift → 中心裁 freq=320 → (2, 320, 512)
  complex_zscore_power_norm → 模型输入
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_ROOT = os.path.dirname(_PKG_DIR)


def complex_zscore_power_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """与 USRP/data_normalize.py 一致（避免 import 时触发其批处理主程序）。"""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    real_part = x[:, 0]
    imag_part = x[:, 1]
    mu_real = real_part.mean(axis=(1, 2), keepdims=True)
    mu_imag = imag_part.mean(axis=(1, 2), keepdims=True)
    std_real = real_part.std(axis=(1, 2), keepdims=True)
    std_imag = imag_part.std(axis=(1, 2), keepdims=True)
    std_real = np.where(std_real < eps, 1.0, std_real)
    std_imag = np.where(std_imag < eps, 1.0, std_imag)
    real_norm = (real_part - mu_real) / std_real
    imag_norm = (imag_part - mu_imag) / std_imag
    c = real_norm + 1j * imag_norm
    mean_amp = np.abs(c).mean(axis=(1, 2), keepdims=True)
    c = c / (mean_amp + eps)
    return np.stack([c.real, c.imag], axis=1).astype(np.float32)


@dataclass(frozen=True)
class IQPreprocessConfig:
    """与 USRP 采集及训练数据片段长度一致。"""

    sample_rate: float = 60e6
    centre_freq: float = 2.4375e9
    segment_length: float = 0.05
    n_fft: int = 1024
    hop_ratio: float = 0.5
    freq_size: int = 512
    time_size: int = 512
    crop_size: int = 320

    @property
    def samples_per_segment(self) -> int:
        return int(self.sample_rate * self.segment_length)


DEFAULT_PREPROCESS = IQPreprocessConfig()


def compute_complex_stft(
    signal: np.ndarray,
    cfg: IQPreprocessConfig = DEFAULT_PREPROCESS,
) -> np.ndarray:
    """复数 STFT + zoom，返回 (2, freq, time)。"""
    hop_length = int(cfg.n_fft * (1 - cfg.hop_ratio))
    window = np.hamming(cfg.n_fft)
    _, _, zxx = stft(
        signal,
        fs=cfg.sample_rate,
        window=window,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - hop_length,
        return_onesided=False,
    )
    stft_real = np.real(zxx)
    stft_imag = np.imag(zxx)
    if cfg.freq_size and stft_real.shape[0] > 1:
        stft_real = zoom(stft_real, (cfg.freq_size / stft_real.shape[0], 1), order=1)
        stft_imag = zoom(stft_imag, (cfg.freq_size / stft_imag.shape[0], 1), order=1)
    if cfg.time_size and stft_real.shape[1] > 1:
        stft_real = zoom(stft_real, (1, cfg.time_size / stft_real.shape[1]), order=1)
        stft_imag = zoom(stft_imag, (1, cfg.time_size / stft_imag.shape[1]), order=1)
    return np.stack([stft_real, stft_imag], axis=0)


def fftshift_crop(stft_stack: np.ndarray, crop_size: int = DEFAULT_PREPROCESS.crop_size) -> np.ndarray:
    """fftshift + 频率轴中心裁剪 → (2, crop_size, time)。"""
    full = stft_stack[0] + 1j * stft_stack[1]
    shifted = np.fft.fftshift(full, axes=0)
    h = shifted.shape[0]
    if h < crop_size:
        crop_size = h
    center = h // 2
    start = max(0, center - crop_size // 2)
    end = min(h, center + crop_size // 2 + (crop_size % 2))
    cropped = shifted[start:end, :]
    return np.stack([np.real(cropped), np.imag(cropped)], axis=0).astype(np.float32)


def zscore_single(stft_2hw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(2, H, W) → Z-score + 功率归一化，与 data_normalize 一致。"""
    batch = stft_2hw[np.newaxis, ...]
    return complex_zscore_power_norm(batch, eps=eps)[0]


def iq_to_zscore_spectrogram(
    iq: np.ndarray,
    cfg: IQPreprocessConfig = DEFAULT_PREPROCESS,
) -> np.ndarray:
    """
    一段 IQ 复数信号 → 模型输入谱图 (2, 320, 512)。

    iq: 一维复数，长度 >= cfg.samples_per_segment（多出的尾部忽略）
    """
    n = cfg.samples_per_segment
    if len(iq) < n:
        raise ValueError(
            f'IQ 长度 {len(iq)} < 所需 {n} ({cfg.segment_length}s @ {cfg.sample_rate / 1e6:.0f} MS/s)'
        )
    segment = np.asarray(iq[:n], dtype=np.complex64)
    stft_stack = compute_complex_stft(segment, cfg)
    cropped = fftshift_crop(stft_stack, cfg.crop_size)
    return zscore_single(cropped)


def float32_iq_to_complex(i: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.asarray(i, dtype=np.float32) + 1j * np.asarray(q, dtype=np.float32)
