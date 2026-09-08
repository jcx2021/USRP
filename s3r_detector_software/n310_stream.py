"""
USRP N310 采集流（对齐 MATLAB Receiver_N310_base.m）。

硬件: 122.88 MHz / 2 = 61.44 MHz
重采样: resample_poly(·, 125, 128) → 精确 60 MHz
存盘: float32 小端 I/Q 交错 .dat
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

import numpy as np
from scipy.signal import resample_poly, spectrogram

if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    for p in (_root, _pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

try:
    import uhd  # type: ignore
    HAS_UHD = True
except ImportError:
    uhd = None
    HAS_UHD = False

# MATLAB Receiver_N310_base.m
FS_60 = 60e6
MASTER_CLOCK = 122.88e6
DECIM = 2
HW_RATE = MASTER_CLOCK / DECIM
RESAMPLE_P = 125
RESAMPLE_Q = 128
DEFAULT_CENTER = 2.4375e9
DEFAULT_GAIN = 50.0
DEFAULT_ADDR = '192.168.20.2'  # SFP+1 万兆口；千兆口为 192.168.10.2
DEFAULT_DECIM = 2
STFT_WINDOW = 1024
STFT_HOP = 512

# N310 @ 122.88 MHz 主时钟：decim → 硬件采样率（输出仍重采样至 60 MHz 供模型使用）
HW_RATE_PRESETS: tuple[tuple[str, int], ...] = (
    ('61.44', 2),   # 默认，对齐 MATLAB；约 2 Gbps，需万兆网
    ('30.72', 4),   # 约 1 Gbps，适合千兆网
    ('15.36', 8),
    ('7.68', 16),
)


@dataclass
class N310Config:
    addr: str = DEFAULT_ADDR
    device_args: str = ''
    centre_freq: float = DEFAULT_CENTER
    gain: float = DEFAULT_GAIN
    antenna: str = 'TX/RX'
    rx_channel: int = 0
    rx_subdev_spec: str = 'A:0'
    master_clock_rate: float = MASTER_CLOCK
    decim_factor: int = DEFAULT_DECIM
    output_sample_rate: float = FS_60
    warmup_frames: int = 50
    # 传输调优：帧长须 ≤ 主机实际 MTU。当前 USB 网卡系统 MTU 常卡在 1500，
    # 用 1472（1500−IP/UDP 头）避免 UHD 请求 8000 再被砍；队列仍加大吸抖动。
    # MTU 若真正到 9000，可改回 8000；设为 0 可关闭对应参数。
    recv_frame_size: int = 1472
    num_recv_frames: int = 2048
    # overflow：丢失样点已不在缓冲；带 O 标志的块紧邻时间断裂 → 默认整块丢弃
    drop_overflow_chunks: bool = True
    # overflow 后再额外丢弃的硬件样点数，冲掉段边界脏数据
    overflow_guard_hw_samples: int = 8192
    # 检测/存盘切片长度；保证每段内部时间连续（overflow 时丢弃未凑满半段）
    segment_sec: float = 0.05

    def __post_init__(self):
        if self.decim_factor < 1:
            raise ValueError('decim_factor 须 >= 1')
        if self.segment_sec <= 0:
            raise ValueError('segment_sec 须 > 0')

    @property
    def hw_sample_rate(self) -> float:
        return self.master_clock_rate / self.decim_factor

    def resample_ratio(self) -> tuple[int, int]:
        """硬件 IQ → output_sample_rate 的有理重采样 (P, Q)。"""
        frac = Fraction(
            int(round(self.output_sample_rate * self.decim_factor)),
            int(round(self.master_clock_rate)),
        )
        return frac.numerator, frac.denominator

    def transport_tuning(self) -> str:
        """拼接万兆/USB 网口传输调优参数（供默认 device_args 使用）。"""
        parts = []
        if self.recv_frame_size and self.recv_frame_size > 0:
            parts.append(f'recv_frame_size={int(self.recv_frame_size)}')
        if self.num_recv_frames and self.num_recv_frames > 0:
            parts.append(f'num_recv_frames={int(self.num_recv_frames)}')
        return (',' + ','.join(parts)) if parts else ''

    def resolve_device_args(self) -> str:
        if self.device_args.strip():
            return self.device_args.strip()
        # UHD 发现字段是 type=n3xx（product=n310）；写 type=n310 会导致 No devices found
        # master_clock_rate 必须在构建设备时传入，否则 N310 常落在 125 MHz → 62.5 MS/s
        mcr = f',master_clock_rate={self.master_clock_rate:.0f}'
        return f'type=n3xx,addr={self.addr}{mcr}{self.transport_tuning()}'


def decim_for_hw_rate_mhz(hw_mhz: float) -> int:
    """根据硬件采样率 (MHz) 反查 decim，须为预设之一。"""
    for label, decim in HW_RATE_PRESETS:
        if abs(float(label) - hw_mhz) < 0.01:
            return decim
    decim = int(round(MASTER_CLOCK / (hw_mhz * 1e6)))
    if decim < 1:
        raise ValueError(f'无效硬件采样率 {hw_mhz} MS/s')
    return decim


def hw_samples_for_output(num_out: int, resample_p: int, resample_q: int) -> int:
    """目标输出样点数 → 所需硬件样点数。"""
    return int(np.ceil(num_out * resample_q / resample_p))


def hw_samples_for_60mhz(num_60: int) -> int:
    """兼容旧接口：默认 decim=2 时 60 MHz 样点数 → 硬件样点数。"""
    return hw_samples_for_output(num_60, RESAMPLE_P, RESAMPLE_Q)


def resample_hw_to_output(
    iq_hw: np.ndarray,
    resample_p: int,
    resample_q: int,
    num_out: int | None = None,
) -> np.ndarray:
    """硬件 IQ → 目标输出采样率。"""
    out = resample_poly(np.asarray(iq_hw, dtype=np.complex128), resample_p, resample_q)
    out = out.astype(np.complex64)
    if num_out is not None:
        out = out[:num_out]
    return out


def resample_hw_to_60mhz(iq_hw: np.ndarray, num_60: int | None = None) -> np.ndarray:
    """61.44 MHz IQ → 60 MHz（有理重采样 125/128）。"""
    return resample_hw_to_output(iq_hw, RESAMPLE_P, RESAMPLE_Q, num_60)


def save_dat_matlab(iq_60: np.ndarray, path: str) -> None:
    """保存为 MATLAB fwrite interleaved float32 小端 .dat。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    i = np.real(iq_60).astype('<f4')
    q = np.imag(iq_60).astype('<f4')
    interleaved = np.empty(i.size * 2, dtype='<f4')
    interleaved[0::2] = i
    interleaved[1::2] = q
    interleaved.tofile(path)


def load_dat_matlab(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype='<f4')
    data = data[: len(data) // 2 * 2]
    iq = data.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


def signal_quality(iq: np.ndarray) -> dict:
    amp = np.abs(iq)
    step = max(1, len(amp) // 100_000)
    sample = np.sort(amp[::step])
    p99 = float(sample[max(0, int(0.99 * len(sample)) - 1)])
    mx = float(np.max(amp))
    mn = float(np.mean(amp))
    msg = 'OK'
    if mx >= 1.0:
        msg = '过载/削顶，请降低增益'
    elif mx < 0.1:
        msg = '信号偏弱，可适当提高增益'
    elif mx >= 0.8:
        msg = '接近满幅，可考虑略降增益'
    return {'max': mx, 'p99': p99, 'mean': mn, 'message': msg}


def display_spectrogram_db(
    iq_seg: np.ndarray,
    fs: float = FS_60,
    *,
    window: int = STFT_WINDOW,
    hop: int = STFT_HOP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复信号幅度谱 dB（对齐 MATLAB spectrogram 可视化）。"""
    iq_seg = np.asarray(iq_seg) - np.mean(iq_seg)
    nperseg = window
    noverlap = window - hop
    f, t, s = spectrogram(
        iq_seg, fs=fs, window=np.hamming(nperseg), nperseg=nperseg,
        noverlap=noverlap, nfft=window, mode='complex', return_onesided=False,
    )
    f = np.fft.fftshift(f)
    s = np.fft.fftshift(s, axes=0)
    mag_db = 20 * np.log10(np.abs(s) + 1e-12)
    return f, t, mag_db


class ResampleSegmenter:
    """硬件 IQ → 凑满一段再重采样。

    不变式：只要在时间断裂（overflow）时调用 clear()，发出的每一段在段内都是连续的。
    段与段之间允许有缺口（被丢掉的半段 / overflow）。
    """

    def __init__(
        self,
        segment_sec: float = 0.05,
        *,
        output_rate: float = FS_60,
        resample_p: int = RESAMPLE_P,
        resample_q: int = RESAMPLE_Q,
    ):
        self.segment_sec = float(segment_sec)
        self.output_rate = output_rate
        self.resample_p = resample_p
        self.resample_q = resample_q
        self.seg_out = int(output_rate * segment_sec)
        self.hw_need = hw_samples_for_output(self.seg_out, resample_p, resample_q)
        self._buf: list[complex] = []
        self.emitted = 0
        self.discarded_partial_hw = 0

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        if chunk is None or len(chunk) == 0:
            return []
        self._buf.extend(np.asarray(chunk, dtype=np.complex64).reshape(-1).tolist())
        out = []
        while len(self._buf) >= self.hw_need:
            hw = np.array(self._buf[:self.hw_need], dtype=np.complex64)
            del self._buf[:self.hw_need]
            seg = resample_hw_to_output(hw, self.resample_p, self.resample_q, self.seg_out)
            out.append(seg)
            self.emitted += 1
        return out

    def clear(self) -> int:
        """时间断裂时调用：丢弃未凑满的半段，保证下一段从新连续流开始。"""
        n = len(self._buf)
        self._buf.clear()
        self.discarded_partial_hw += n
        return n

    @property
    def pending_hw(self) -> int:
        return len(self._buf)


@dataclass
class RecordResult:
    path: str
    iq_60: np.ndarray
    duration_sec: float
    quality: dict
    overflow_count: int = 0
    actual_hw_rate: float = 0.0
    dropped_hw_samples: int = 0
    overflow_meta_path: str = ''
    # overflow 发生时已完成的连续段数（段边界下标）
    overflow_at_segment: list[int] = field(default_factory=list)
    complete_segments: int = 0
    segment_sec: float = 0.05


class N310Recorder:
    """一次性采集固定时长（类似 MATLAB 2s 录制）。"""

    def __init__(self, cfg: N310Config | None = None):
        if not HAS_UHD:
            raise RuntimeError('未安装 PyUHD (import uhd 失败)')
        self.cfg = cfg or N310Config()

    def record(
        self,
        duration_sec: float,
        save_path: str,
        *,
        on_progress: Callable[[float], None] | None = None,
        segment_sec: float | None = None,
    ) -> RecordResult:
        """按连续 50ms（可配）段采集：只保留段内时间连续的完整片段再拼接存盘。"""
        p, q = self.cfg.resample_ratio()
        out_rate = self.cfg.output_sample_rate
        seg_sec = float(segment_sec if segment_sec is not None else self.cfg.segment_sec)
        n_need = max(1, int(round(duration_sec / seg_sec)))

        usrp, streamer, ch, buff, md, actual_hw_rate = _open_rx_stream(self.cfg)

        for _ in range(self.cfg.warmup_frames):
            streamer.recv(buff, md)

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        streamer.issue_stream_cmd(cmd)

        segmenter = ResampleSegmenter(
            seg_sec, output_rate=out_rate, resample_p=p, resample_q=q,
        )
        segs: list[np.ndarray] = []
        overflow_count = 0
        dropped_hw = 0
        overflow_at_seg: list[int] = []
        guard_left = 0
        drop_ovf = bool(self.cfg.drop_overflow_chunks)
        guard_n = max(0, int(self.cfg.overflow_guard_hw_samples))
        t0 = time.perf_counter()
        try:
            while len(segs) < n_need:
                n = int(streamer.recv(buff, md))
                ec = md.error_code
                if ec == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_count += 1
                    overflow_at_seg.append(len(segs))
                    # 关键点：未凑满的半段作废，避免段内出现时间断裂
                    dropped_hw += segmenter.clear()
                    if drop_ovf:
                        dropped_hw += max(0, n)
                        guard_left = max(guard_left, guard_n)
                    continue
                if ec != uhd.types.RXMetadataErrorCode.none:
                    print(f'[N310] 接收错误: {md.strerror()}', file=sys.stderr)
                    continue
                if n <= 0:
                    continue
                if guard_left > 0:
                    skip = min(n, guard_left)
                    dropped_hw += skip
                    guard_left -= skip
                    if skip >= n:
                        continue
                    chunk = buff[skip:n]
                else:
                    chunk = buff[:n]
                for seg in segmenter.push(chunk):
                    segs.append(seg)
                    if on_progress:
                        on_progress(min(1.0, len(segs) / n_need))
                    if len(segs) >= n_need:
                        break
        finally:
            streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

        wall_sec = time.perf_counter() - t0
        # 尾部未满一段的半截不写入（不能保证与下一次采集连续）
        dropped_hw += segmenter.clear()
        if not segs:
            raise RuntimeError('未收到任何完整连续片段（overflow 过多或链路异常）')
        iq_out = np.concatenate(segs)
        save_dat_matlab(iq_out, save_path)
        meta_path = ''
        if overflow_count or dropped_hw:
            meta_path = os.path.splitext(save_path)[0] + '.overflow.json'
            meta = {
                'policy': 'contiguous_segments_only',
                'segment_sec': seg_sec,
                'complete_segments': len(segs),
                'overflow_count': overflow_count,
                'dropped_hw_samples': dropped_hw,
                'discarded_partial_hw_samples': segmenter.discarded_partial_hw,
                'overflow_guard_hw_samples': guard_n,
                'overflow_at_completed_segment': overflow_at_seg,
                'note': (
                    f'每段 {seg_sec * 1e3:.0f}ms 内部时间连续；overflow 时丢弃未凑满半段。'
                    '文件为完整连续段的拼接；段与段之间可能有墙钟缺口，'
                    '但按固定段长从文件头切片时，每一刀都落在连续段内。'
                ),
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        qual = signal_quality(iq_out)
        hw_mhz = self.cfg.hw_sample_rate / 1e6
        out_mhz = out_rate / 1e6
        print(
            f'[N310] {hw_mhz:.2f}→{out_mhz:.0f} MS/s  '
            f'{len(segs)}×{seg_sec * 1e3:.0f}ms 连续段 / 墙钟 {wall_sec:.1f}s → {save_path}  '
            f'|IQ|max={qual["max"]:.4f}',
        )
        if overflow_count:
            print(
                f'[N310][警告] overflow×{overflow_count}，已丢弃半段/污染块 '
                f'{dropped_hw} 硬件样点；已保证每段内部连续'
                + (f'；标记: {meta_path}' if meta_path else ''),
                file=sys.stderr,
            )
        return RecordResult(
            save_path, iq_out, len(iq_out) / out_rate, qual,
            overflow_count=overflow_count,
            actual_hw_rate=actual_hw_rate,
            dropped_hw_samples=dropped_hw,
            overflow_meta_path=meta_path,
            overflow_at_segment=overflow_at_seg,
            complete_segments=len(segs),
            segment_sec=seg_sec,
        )


class N310LiveStream:
    """连续采集 + 按片段回调（实时推理 / 实时预览）。"""

    def __init__(
        self,
        cfg: N310Config | None = None,
        *,
        segment_sec: float = 0.05,
        on_segment_60: Callable[[np.ndarray, int], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        if not HAS_UHD:
            raise RuntimeError('未安装 PyUHD')
        self.cfg = cfg or N310Config()
        self.segment_sec = segment_sec
        self.on_segment_60 = on_segment_60
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seg_idx = 0

    def _run(self):
        try:
            usrp, streamer, ch, buff, md, _actual_hw_rate = _open_rx_stream(self.cfg)
            for _ in range(self.cfg.warmup_frames):
                streamer.recv(buff, md)

            p, q = self.cfg.resample_ratio()
            segmenter = ResampleSegmenter(
                self.segment_sec,
                output_rate=self.cfg.output_sample_rate,
                resample_p=p,
                resample_q=q,
            )
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            streamer.issue_stream_cmd(cmd)

            drop_ovf = bool(self.cfg.drop_overflow_chunks)
            guard_n = max(0, int(self.cfg.overflow_guard_hw_samples))
            guard_left = 0
            overflow_count = 0

            while not self._stop.is_set():
                n = int(streamer.recv(buff, md))
                ec = md.error_code
                if ec == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_count += 1
                    # 丢弃未满的 50ms 半段，已发出的完整段不受影响（段内已连续）
                    flushed = segmenter.clear()
                    if drop_ovf:
                        guard_left = max(guard_left, guard_n)
                    if overflow_count == 1 or overflow_count % 20 == 0:
                        print(
                            f'[N310] live overflow×{overflow_count}，'
                            f'丢弃未满段 {flushed} hw 样点（保证下一段内部连续）',
                            file=sys.stderr,
                        )
                    continue
                if ec != uhd.types.RXMetadataErrorCode.none:
                    print(f'[N310] {md.strerror()}', file=sys.stderr)
                    continue
                if n <= 0:
                    continue
                if guard_left > 0:
                    skip = min(n, guard_left)
                    guard_left -= skip
                    if skip >= n:
                        continue
                    chunk = buff[skip:n].copy()
                else:
                    chunk = buff[:n].copy()
                for seg in segmenter.push(chunk):
                    if self.on_segment_60:
                        self.on_segment_60(seg, self._seg_idx)
                    self._seg_idx += 1

            streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        except Exception as exc:
            if self.on_error:
                self.on_error(exc)
            else:
                raise

    def start(self):
        self._stop.clear()
        self._seg_idx = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None


def _open_rx_stream(cfg: N310Config):
    dev = cfg.resolve_device_args()
    ch = int(cfg.rx_channel)
    usrp = uhd.usrp.MultiUSRP(dev)
    if cfg.rx_subdev_spec.strip():
        try:
            usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(cfg.rx_subdev_spec))
        except Exception:
            pass
    try:
        usrp.set_master_clock_rate(cfg.master_clock_rate)
    except Exception as exc:
        print(f'[N310][警告] set_master_clock_rate 失败: {exc}', file=sys.stderr)
    try:
        actual_mcr = float(usrp.get_master_clock_rate())
        if abs(actual_mcr - cfg.master_clock_rate) > 1.0:
            print(
                f'[N310][警告] 主时钟实际 {actual_mcr/1e6:.6f} MHz '
                f'≠ 目标 {cfg.master_clock_rate/1e6:.6f} MHz',
                file=sys.stderr,
            )
    except Exception:
        pass
    usrp.set_rx_rate(cfg.hw_sample_rate, ch)
    actual_hw_rate = float(usrp.get_rx_rate(ch))
    # 硬件采样率必须等于 主时钟/decim，否则 (P,Q) 重采样比失配 → 频率标度错误
    if abs(actual_hw_rate - cfg.hw_sample_rate) > 1.0:
        print(
            f'[N310][警告] 实际硬件采样率 {actual_hw_rate/1e6:.6f} MS/s '
            f'≠ 目标 {cfg.hw_sample_rate/1e6:.6f} MS/s，重采样比将失配，'
            f'请检查 decim/主时钟设置',
            file=sys.stderr,
        )
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(cfg.centre_freq), ch)
    usrp.set_rx_gain(cfg.gain, ch)
    try:
        usrp.set_rx_antenna(cfg.antenna, ch)
    except Exception:
        pass

    args = uhd.usrp.StreamArgs('fc32', 'sc16')
    args.channels = [ch]
    streamer = usrp.get_rx_stream(args)
    max_samps = streamer.get_max_num_samps()
    buff = np.zeros(max_samps, dtype=np.complex64)
    md = uhd.types.RXMetadata()
    return usrp, streamer, ch, buff, md, actual_hw_rate
