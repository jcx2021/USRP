# USRP N310 无人机 RF 检测站

用 **USRP-LW N310** 采集 2.4 GHz 无人机射频，按 50 ms 片段做频谱预览和开放集检测。硬件 IQ 默认 61.44 MS/s，存盘前重采样到 **60 MHz**（与 MATLAB `Receiver_N310_base.m` 对齐）。

## 硬件与网络

| 项目 | 建议 |
|------|------|
| 设备 | N310，SFP+1（10G），默认 IP `192.168.20.2` |
| 主机 | `192.168.20.1/24`，天线用 **TX/RX** |
| 固件 | 设备 MPM 为 **UHD 3.15**；主机 PyUHD 必须同为 3.15，不要用 4.x |
| 采样率 | `61.44` / `30.72` / `15.36` / `7.68` MS/s；USB 万兆口满载 61.44 仍可能 overflow |

本机 conda 环境示例：`n310-uhd315`（Python 3.10 + `uhd` 3.15）。

## 启动

Windows 可改 `启动检测站_UHD315.bat` 里的 Anaconda 路径后双击，或：

```bat
conda activate n310-uhd315
python usrp_detector\drone_rf_station_app.py
```

界面两种用法：

1. **采集与可视化** — 录 `.dat`，按片段看 STFT  
2. **实时检测** — 加载训练 run 目录后，边采边推断  

默认数据根目录：`E:\USRP\recordings`（可在界面改）。

## 采集规则

- 存盘格式：float32 小端 I/Q 交错 `.dat`
- 按片段（默认 50 ms）凑满再写入；**overflow 时丢掉未满半段**，保证每一段内部时间连续
- 段与段之间可能有墙钟缺口；有 overflow 时会旁路写 `*.overflow.json`

## 目录

```
usrp_detector/          # 检测站：采集、实时流、推理、GUI
Receiver_N310_base.m    # MATLAB 对照采集脚本
STFT_hamming.py         # STFT 处理
process_recordings.py   # 批量处理录制文件
```

录制 IQ、驱动安装包不进仓库。

## 其它入口

```bat
python usrp_detector\usrp_realtime_app.py
python process_recordings.py --recordings E:\USRP\recordings
```
