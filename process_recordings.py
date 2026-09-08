"""
处理 USRP/recordings 下的 .dat 文件。

STFT 逻辑严格对齐 STFT_hamming.py：
  Hamming 窗 STFT -> zoom 到 512x512 -> fftshift -> 中心裁剪
本脚本使用 BANDWIDTH=60e6，crop_size=512，最终输出 shape (N, 2, 512, 512)。

目录结构:
    recordings/{drone_id}/{state}/{file}.dat
"""
import argparse
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft

# 与 STFT_hamming.py 一致，带宽改为 60 MHz
SAMPLE_RATE = 60e6
BANDWIDTH = 60e6
CENTRE_FREQ = 2.4375e9
RECORDING_TIME = 2.0

FREQ_SIZE = 512
TIME_SIZE = 512
CROP_SIZE = 512

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = Path(r"E:\USRP\recordings")
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "stft_512x512"


def load_iq_dat(file_path):
    """加载 I/Q 交错的 .dat 文件，返回复数信号。"""
    data = np.fromfile(file_path, dtype=np.float32)
    data = data[: len(data) // 2 * 2]
    iq = data.reshape(-1, 2)

    i_channel = iq[:, 0]
    q_channel = iq[:, 1]

    print(f"  I mean={np.mean(i_channel):.6f}, Q mean={np.mean(q_channel):.6f}")
    print(f"  I std={np.std(i_channel):.6f}, Q std={np.std(q_channel):.6f}")
    corr = np.corrcoef(i_channel, q_channel)[0, 1]
    print(f"  I-Q correlation={corr:.3f}")

    if np.abs(corr) > 0.9:
        warnings.warn("I 和 Q 通道高度相关，可能是实信号或数据格式错误！")

    return i_channel + 1j * q_channel


def compute_complex_stft(
    signal,
    n_fft=1024,
    hop_ratio=0.5,
    target_height=None,
    target_width=None,
):
    """计算复数 STFT 并 zoom 到目标尺寸（与 STFT_hamming.py 一致）。"""
    hop_length = int(n_fft * (1 - hop_ratio))
    window = np.hamming(n_fft)

    f, t, zxx = stft(
        signal,
        fs=SAMPLE_RATE,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False,
    )

    stft_real = np.real(zxx)
    stft_imag = np.imag(zxx)

    if target_height is not None and stft_real.shape[0] > 1:
        stft_real = zoom(stft_real, (target_height / stft_real.shape[0], 1), order=1)
        stft_imag = zoom(stft_imag, (target_height / stft_imag.shape[0], 1), order=1)
        f = np.linspace(f[0], f[-1], target_height)

    if target_width is not None and stft_real.shape[1] > 1:
        stft_real = zoom(stft_real, (1, target_width / stft_real.shape[1]), order=1)
        stft_imag = zoom(stft_imag, (1, target_width / stft_imag.shape[1]), order=1)
        t = np.linspace(t[0], t[-1], target_width)

    return np.stack([stft_real, stft_imag], axis=0), f, t


def process_segments(
    iq_signal,
    segment_length,
    n_fft=1024,
    hop_ratio=0.5,
    freq_size=FREQ_SIZE,
    time_size=TIME_SIZE,
):
    """将信号分割为片段并处理（与 STFT_hamming.py 一致）。"""
    samples_per_segment = int(SAMPLE_RATE * segment_length)
    num_segments = len(iq_signal) // samples_per_segment

    stft_results = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = iq_signal[start:end]

        try:
            stft_result, _, _ = compute_complex_stft(
                segment,
                n_fft=n_fft,
                hop_ratio=hop_ratio,
                target_height=freq_size,
                target_width=time_size,
            )
            stft_results.append(stft_result)
        except Exception as e:
            warnings.warn(f"片段 {i} 处理失败: {str(e)}")

    return stft_results


def apply_fftshift_and_crop(stft_result, crop_size=CROP_SIZE):
    """fftshift + 中心裁剪（与 STFT_hamming.py process_single_file 一致）。"""
    full_stft = stft_result[0] + 1j * stft_result[1]
    shifted_stft = np.fft.fftshift(full_stft, axes=0)

    if shifted_stft.shape[0] < crop_size:
        warnings.warn(
            f"频率尺寸不足: {shifted_stft.shape[0]} < {crop_size}，将裁剪为实际尺寸"
        )
        crop_size = shifted_stft.shape[0]

    center = shifted_stft.shape[0] // 2
    start = center - crop_size // 2
    end = center + crop_size // 2
    if crop_size % 2 != 0:
        end += 1
    start = max(0, start)
    end = min(shifted_stft.shape[0], end)

    cropped = shifted_stft[start:end, :]
    return np.stack([np.real(cropped), np.imag(cropped)], axis=0).astype(np.float32)


def process_single_file(
    file_path,
    segment_length,
    freq_size=FREQ_SIZE,
    time_size=TIME_SIZE,
    crop_size=CROP_SIZE,
    n_fft=1024,
    hop_ratio=0.5,
):
    """处理单个 .dat，返回 (num_segments, 2, freq, time)。"""
    iq_signal = load_iq_dat(file_path)
    stft_results = process_segments(
        iq_signal,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_ratio=hop_ratio,
        freq_size=freq_size,
        time_size=time_size,
    )

    if not stft_results:
        warnings.warn(f"{os.path.basename(file_path)} 未生成任何有效片段")
        return None, iq_signal

    cropped = [apply_fftshift_and_crop(r, crop_size) for r in stft_results]
    return np.stack(cropped, axis=0), iq_signal


def stft_array_to_mag_db(stft_stack):
    """(2, H, W) -> dB magnitude."""
    return 20 * np.log10(np.abs(stft_stack[0] + 1j * stft_stack[1]) + 1e-12)


def calibrate_db_range(mag_db_list, noise_percentile=25, signal_percentile=99.5, margin_db=6):
    stacked = np.concatenate([m.ravel() for m in mag_db_list])
    noise = float(np.percentile(stacked, noise_percentile))
    signal = float(np.percentile(stacked, signal_percentile))
    vmin = noise - margin_db
    vmax = signal + margin_db
    if vmax - vmin < 25:
        vmax = vmin + 25
    return vmin, vmax, noise, signal


def save_segment_spectrogram(mag_db, save_path, title, vmin, vmax, segment_length):
    """可视化 512x512 STFT 幅度（dB）。"""
    freq_mhz = np.linspace(-BANDWIDTH / 2, BANDWIDTH / 2, mag_db.shape[0]) / 1e6
    t_ms = np.linspace(0, segment_length * 1e3, mag_db.shape[1])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        mag_db,
        aspect="auto",
        extent=[t_ms[0], t_ms[-1], freq_mhz[0], freq_mhz[-1]],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.axhline(y=0, color="white", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(f"Frequency offset from {CENTRE_FREQ / 1e9:.4f} GHz [MHz]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.colorbar(im, ax=ax, label=f"Complex magnitude [dB]  ({vmin:.0f} ~ {vmax:.0f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_dat_file(
    dat_path,
    output_dir,
    segment_length=0.05,
    freq_size=FREQ_SIZE,
    time_size=TIME_SIZE,
    crop_size=CROP_SIZE,
    n_fft=1024,
    hop_ratio=0.5,
):
    dat_path = Path(dat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {dat_path}")
    segments, iq_signal = process_single_file(
        dat_path,
        segment_length=segment_length,
        freq_size=freq_size,
        time_size=time_size,
        crop_size=crop_size,
        n_fft=n_fft,
        hop_ratio=hop_ratio,
    )

    if segments is None:
        return 0, output_dir

    duration = len(iq_signal) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.abs(iq_signal) ** 2)))
    num_segments = segments.shape[0]
    print(
        f"  samples={len(iq_signal)}, duration={duration:.3f}s, rms={rms:.6f}, "
        f"segments={num_segments}, output_shape={segments.shape}"
    )

    mag_list = [stft_array_to_mag_db(segments[i]) for i in range(num_segments)]
    vmin, vmax, noise_db, signal_db = calibrate_db_range(mag_list)
    print(
        f"  dB calibration: noise~{noise_db:.1f}, signal~{signal_db:.1f}, "
        f"plot range [{vmin:.1f}, {vmax:.1f}]"
    )

    samples_per_segment = int(SAMPLE_RATE * segment_length)
    for seg_idx in range(num_segments):
        t_start = seg_idx * segment_length
        t_end = t_start + segment_length
        fig_name = f"{dat_path.stem}_seg{seg_idx:02d}_{t_start:.2f}s-{t_end:.2f}s_stft.png"
        title = (
            f"{dat_path.parent.parent.name}/{dat_path.parent.name}/{dat_path.name} "
            f"| seg {seg_idx} [{t_start:.2f}s, {t_end:.2f}s] | "
            f"512x512 | rms={rms:.4f}"
        )
        save_segment_spectrogram(
            mag_list[seg_idx],
            output_dir / fig_name,
            title=title,
            vmin=vmin,
            vmax=vmax,
            segment_length=segment_length,
        )
        print(f"    saved {fig_name}")

    npy_path = output_dir / f"{dat_path.stem}_stft.npy"
    np.save(npy_path, segments)
    print(f"  saved array -> {npy_path}  shape={segments.shape}")

    return num_segments, output_dir


def collect_dat_files(recordings_root):
    return sorted(Path(recordings_root).rglob("*.dat"))


def main():
    parser = argparse.ArgumentParser(
        description="USRP recordings STFT（对齐 STFT_hamming.py，512x512 输出）"
    )
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--segment-length", type=float, default=0.05)
    parser.add_argument("--freq-size", type=int, default=FREQ_SIZE)
    parser.add_argument("--time-size", type=int, default=TIME_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-ratio", type=float, default=0.5)
    args = parser.parse_args()

    dat_files = collect_dat_files(args.recordings)
    if not dat_files:
        print(f"未找到 .dat 文件: {args.recordings}")
        return

    print(f"Found {len(dat_files)} .dat files under {args.recordings}")
    print(
        f"Params: segment_length={args.segment_length}s, "
        f"freq_size={args.freq_size}, time_size={args.time_size}, crop_size={args.crop_size}, "
        f"BANDWIDTH={BANDWIDTH/1e6:.0f}MHz, n_fft={args.n_fft}, hop_ratio={args.hop_ratio}"
    )

    total_segments = 0
    for dat_path in dat_files:
        rel = dat_path.relative_to(args.recordings)
        n_seg, _ = process_dat_file(
            dat_path,
            args.output / rel.parent,
            segment_length=args.segment_length,
            freq_size=args.freq_size,
            time_size=args.time_size,
            crop_size=args.crop_size,
            n_fft=args.n_fft,
            hop_ratio=args.hop_ratio,
        )
        total_segments += n_seg

    print(f"\nDone. {len(dat_files)} files, {total_segments} segments.")
    print(f"Output: {args.output}  (shape: segments x 2 x {args.crop_size} x {args.time_size})")


if __name__ == "__main__":
    main()
