"""
USRP 实时检测管线：IQ 采集 → STFT+Z-score → S3R 开放集推断。

模式:
  - uhd   : PyUHD 直连 USRP（需安装 UHD + Python 绑定）
  - dat   : 循环回放 .dat 文件（无硬件时调试）
  - sim   : 正弦+噪声仿真 IQ（快速冒烟测试）

用法:
  python s3r_detector_software/usrp_realtime_app.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    for p in (_root, _pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

from iq_preprocess import DEFAULT_PREPROCESS, IQPreprocessConfig, iq_to_zscore_spectrogram

# S3RDetector / S3RPrediction 仅用于类型注解（本文件启用了 from __future__ import annotations，
# 注解不会在运行时求值），因此不在顶层 import，避免无 TensorFlow 时整包导入失败。
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from s3r_inference_engine import S3RDetector, S3RPrediction

try:
    import uhd  # type: ignore
    _HAS_UHD = True
except ImportError:
    uhd = None
    _HAS_UHD = False


@dataclass
class USRPConfig:
    """USRP N310 默认参数（千兆/万兆网口，与训练 60 MS/s、2.4375 GHz 对齐）。"""

    device_type: str = 'n3xx'  # UHD 发现类型；product 才是 n310
    addr: str = '192.168.20.2'  # SFP+1 万兆口；千兆口为 192.168.10.2
    device_args: str = ''
    rx_channel: int = 0
    rx_subdev_spec: str = 'A:0'
    sample_rate: float = 60e6          # 输出/训练采样率（重采样后）
    master_clock_rate: float = 122.88e6
    decim_factor: int = 2              # 硬件 DDC 抽取：122.88/2 = 61.44 MS/s
    centre_freq: float = 2.4375e9
    gain: float = 50.0
    antenna: str = 'TX/RX'
    # 适配系统 MTU=1500：1472；队列加大。MTU 到 9000 时可改回 8000。
    recv_frame_size: int = 1472
    num_recv_frames: int = 2048
    preprocess: IQPreprocessConfig = DEFAULT_PREPROCESS

    @property
    def hw_sample_rate(self) -> float:
        """N310 实际硬件采样率（主时钟/decim）；60 MHz 无法由 N310 直接抽取得到。"""
        return self.master_clock_rate / self.decim_factor

    def resample_ratio(self) -> tuple[int, int]:
        """硬件率 → sample_rate 的精确有理重采样比 (P, Q)。"""
        frac = Fraction(
            int(round(self.sample_rate * self.decim_factor)),
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
        mcr = f',master_clock_rate={self.master_clock_rate:.0f}'
        return f'type={self.device_type},addr={self.addr}{mcr}{self.transport_tuning()}'


N310_DEFAULT = USRPConfig()


def load_iq_dat(path: str) -> np.ndarray:
    """读取 MATLAB Receiver_N310_base.m 或同格式 float32 I/Q 交错 .dat。"""
    data = np.fromfile(path, dtype=np.float32)
    data = data[: len(data) // 2 * 2]
    iq = data.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


class IQRingBuffer:
    """滑动 IQ 缓冲，凑够一段就触发回调。"""

    def __init__(
        self,
        segment_samples: int,
        on_segment: Callable[[np.ndarray], None],
        *,
        max_segments: int = 4,
    ):
        self.segment_samples = segment_samples
        self.on_segment = on_segment
        self._buf = deque(maxlen=segment_samples * max_segments)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(chunk.reshape(-1))
            while len(self._buf) >= self.segment_samples:
                seg = np.array([self._buf[i] for i in range(self.segment_samples)], dtype=np.complex64)
                for _ in range(self.segment_samples):
                    self._buf.popleft()
                self.on_segment(seg)


class RealtimeS3RPipeline:
    """实时：IQ 段 → 预处理 → S3RDetector.predict。"""

    def __init__(
        self,
        detector: S3RDetector,
        preprocess: IQPreprocessConfig = DEFAULT_PREPROCESS,
        on_result: Callable[[S3RPrediction, float], None] | None = None,
    ):
        self.detector = detector
        self.preprocess = preprocess
        self.on_result = on_result
        self._infer_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_q: deque[np.ndarray] = deque(maxlen=2)
        self._ring = IQRingBuffer(
            preprocess.samples_per_segment,
            self._enqueue_segment,
        )

    def _enqueue_segment(self, iq_seg: np.ndarray) -> None:
        if len(self._segment_q) >= 2:
            return
        self._segment_q.append(iq_seg)

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            if not self._segment_q:
                time.sleep(0.01)
                continue
            iq_seg = self._segment_q.popleft()
            t0 = time.perf_counter()
            try:
                spec = iq_to_zscore_spectrogram(iq_seg, self.preprocess)
                with self._infer_lock:
                    pred = self.detector.predict(spec)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if self.on_result:
                    self.on_result(pred, elapsed_ms)
            except Exception as exc:
                print(f'[实时推断] 错误: {exc}', file=sys.stderr)
                if self.on_result:
                    self.on_result(exc, -1.0)  # type: ignore[arg-type]

    def push_iq(self, chunk: np.ndarray) -> None:
        self._ring.push(chunk)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


class UHDReceiver:
    """PyUHD 连续接收（默认 USRP N310），推入 RealtimeS3RPipeline。"""

    def __init__(self, usrp_cfg: USRPConfig, pipeline: RealtimeS3RPipeline):
        if not _HAS_UHD:
            raise RuntimeError('未安装 PyUHD。请安装 UHD 并确保 import uhd 可用。')
        self.cfg = usrp_cfg
        self.pipeline = pipeline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        # 复用 n310_stream 的重采样分段器，保证与录制/训练完全一致的 60 MHz 输出
        from n310_stream import ResampleSegmenter

        dev = self.cfg.resolve_device_args()
        ch = int(self.cfg.rx_channel)
        hw_rate = self.cfg.hw_sample_rate
        out_rate = self.cfg.sample_rate
        print(
            f'[UHD] 连接 {dev}  RX ch={ch}  '
            f'硬件 {hw_rate/1e6:.2f} MS/s → 重采样 {out_rate/1e6:.0f} MS/s'
        )

        usrp = uhd.usrp.MultiUSRP(dev)
        if self.cfg.rx_subdev_spec.strip():
            try:
                usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(self.cfg.rx_subdev_spec))
            except Exception as exc:
                print(f'[UHD] set_rx_subdev_spec 跳过: {exc}', file=sys.stderr)

        try:
            usrp.set_master_clock_rate(self.cfg.master_clock_rate)
        except Exception:
            pass

        # 关键: N310 主时钟 122.88 MHz 抽取不出精确 60 MHz，必须按硬件率采集后重采样
        usrp.set_rx_rate(hw_rate, ch)
        actual_rate = float(usrp.get_rx_rate(ch))
        print(f'[UHD] 实际硬件采样率: {actual_rate/1e6:.6f} MS/s')
        if abs(actual_rate - hw_rate) > 1.0:
            print(
                f'[UHD][警告] 实际硬件采样率 ≠ 目标 {hw_rate/1e6:.6f} MS/s，'
                f'重采样比失配，请检查 decim/主时钟',
                file=sys.stderr,
            )

        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.cfg.centre_freq), ch)
        usrp.set_rx_gain(self.cfg.gain, ch)
        try:
            usrp.set_rx_antenna(self.cfg.antenna, ch)
        except Exception as exc:
            print(f'[UHD] 天线 {self.cfg.antenna!r} 设置失败: {exc}', file=sys.stderr)

        stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
        stream_args.channels = [ch]
        streamer = usrp.get_rx_stream(stream_args)
        max_samps = streamer.get_max_num_samps()
        buff = np.zeros((max_samps,), dtype=np.complex64)
        md = uhd.types.RXMetadata()

        p, q = self.cfg.resample_ratio()
        segmenter = ResampleSegmenter(
            self.cfg.preprocess.segment_length,
            output_rate=out_rate,
            resample_p=p,
            resample_q=q,
        )

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        streamer.issue_stream_cmd(cmd)

        overflow_count = 0
        # 与 n310_stream 一致：overflow 块丢弃 + 清半段，避免断裂拼进检测
        drop_ovf = True
        guard_n = 8192
        guard_left = 0
        while not self._stop.is_set():
            n = int(streamer.recv(buff, md))
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_count += 1
                    flushed = segmenter.clear()
                    if drop_ovf:
                        guard_left = max(guard_left, guard_n)
                    if overflow_count == 1 or overflow_count % 50 == 0:
                        print(
                            f'[UHD] RX overflow×{overflow_count}，清半段 {flushed} 并跳过污染块',
                            file=sys.stderr,
                        )
                else:
                    print(f'[UHD] {md.strerror()}', file=sys.stderr)
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
                self.pipeline.push_iq(seg)

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(cmd)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None


class DatReplayer:
    """循环回放 .dat，模拟实时流。"""

    def __init__(self, dat_path: str, pipeline: RealtimeS3RPipeline, *, realtime_pacing: bool = True):
        self.dat_path = dat_path
        self.pipeline = pipeline
        self.realtime_pacing = realtime_pacing
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        iq = load_iq_dat(self.dat_path)
        chunk = DEFAULT_PREPROCESS.samples_per_segment // 10
        pos = 0
        dt = (chunk / DEFAULT_PREPROCESS.sample_rate) if self.realtime_pacing else 0
        while not self._stop.is_set():
            if pos + chunk > len(iq):
                pos = 0
            self.pipeline.push_iq(iq[pos:pos + chunk])
            pos += chunk
            if dt > 0:
                time.sleep(dt)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
