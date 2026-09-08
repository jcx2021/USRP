import numpy as np
from scipy.signal import stft
from scipy.fft import fftshift
from scipy.ndimage import zoom
import os
import warnings

# 信号参数配置
SAMPLE_RATE = 60e6  # 60 MHz采样率
BANDWIDTH = 28e6  # 28 MHz带宽
CENTRE_FREQ = 2.4375e9  # 2.4375 GHz中心频率
RECORDING_TIME = 2.0  # 2秒记录时间
TOTAL_SAMPLES = int(SAMPLE_RATE * RECORDING_TIME)  # 1.2e8 samples


def load_iq_dat(file_path):
    """加载I/Q交错的.dat文件，返回复数信号"""
    data = np.fromfile(file_path, dtype=np.float32)
    data = data[:len(data) // 2 * 2]  # 确保偶数
    iq = data.reshape(-1, 2)

    # 检查：如果I和Q几乎相同，说明可能是实信号重复存储
    i_channel = iq[:, 0]
    q_channel = iq[:, 1]

    print(f"I mean: {np.mean(i_channel):.6f}, Q mean: {np.mean(q_channel):.6f}")
    print(f"I std: {np.std(i_channel):.6f}, Q std: {np.std(q_channel):.6f}")
    print(f"I-Q correlation: {np.corrcoef(i_channel, q_channel)[0, 1]:.3f}")

    # 如果I和Q相关性接近1，说明数据有问题
    if np.abs(np.corrcoef(i_channel, q_channel)[0, 1]) > 0.9:
        warnings.warn("I和Q通道高度相关，可能是实信号或数据格式错误！")

    return i_channel + 1j * q_channel


def compute_complex_stft(signal, n_fft=1024, hop_ratio=0.5,
                         target_height=None, target_width=None):
    # ========== 关键：去DC ==========
    # signal = signal - np.mean(signal)
    # ================================
    """计算复数STFT并调整尺寸"""
    hop_length = int(n_fft * (1 - hop_ratio))
    window = np.hamming(n_fft)

    f, t, Zxx = stft(
        signal,
        fs=SAMPLE_RATE,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False
    )

    # 分离实部和虚部
    stft_real = np.real(Zxx)
    stft_imag = np.imag(Zxx)

    # 独立调整频率和时间轴尺寸
    if target_height is not None and stft_real.shape[0] > 1:
        stft_real = zoom(stft_real, (target_height / stft_real.shape[0], 1), order=1)
        stft_imag = zoom(stft_imag, (target_height / stft_imag.shape[0], 1), order=1)
        f = np.linspace(f[0], f[-1], target_height)

    if target_width is not None and stft_real.shape[1] > 1:
        stft_real = zoom(stft_real, (1, target_width / stft_real.shape[1]), order=1)
        stft_imag = zoom(stft_imag, (1, target_width / stft_imag.shape[1]), order=1)
        t = np.linspace(t[0], t[-1], target_width)

    return np.stack([stft_real, stft_imag], axis=0), f, t


def process_segments(iq_signal, segment_length, n_fft=1024,
                     hop_ratio=0.5, freq_size=1024, time_size=1024):
    """将信号分割为片段并处理，自动处理不足长的信号"""
    samples_per_segment = int(SAMPLE_RATE * segment_length)
    available_samples = len(iq_signal)
    num_segments = available_samples // samples_per_segment

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
                target_width=time_size
            )
            stft_results.append(stft_result)
        except Exception as e:
            warnings.warn(f"片段{i}处理失败: {str(e)}")
            continue

    return stft_results


def process_single_file(file_path, segment_length, freq_size, time_size, crop_size):
    """处理单个文件并返回所有有效片段的STFT结果"""
    try:
        iq_signal = load_iq_dat(file_path)

        # 处理分段
        stft_results = process_segments(
            iq_signal,
            segment_length=segment_length,
            freq_size=freq_size,
            time_size=time_size
        )

        if not stft_results:
            warnings.warn(f"文件 {os.path.basename(file_path)} 未生成任何有效片段")
            return None

        # 检查频率尺寸是否足够
        if stft_results[0].shape[1] < crop_size:
            warnings.warn(f"文件 {os.path.basename(file_path)} 频率尺寸不足: {stft_results[0].shape[1]} < {crop_size}")
            crop_size = stft_results[0].shape[1]

        cropped_stft_results = []
        for stft_result in stft_results:
            # 获取完整的复数STFT
            full_stft = stft_result[0] + 1j * stft_result[1]

            # 执行fftshift将DC分量（0Hz）移到中心
            shifted_stft = np.fft.fftshift(full_stft, axes=0)
            # shifted_stft = full_stft

            # 计算中心截取范围
            center = shifted_stft.shape[0] // 2
            start = center - crop_size // 2
            end = center + crop_size // 2

            # 处理奇数情况
            if crop_size % 2 != 0:
                end += 1

            # 确保不越界
            start = max(0, start)
            end = min(shifted_stft.shape[0], end)

            # 截取中心区域
            cropped = shifted_stft[start:end, :]

            # 分离实部和虚部
            cropped_stft_results.append(np.stack([np.real(cropped), np.imag(cropped)], axis=0))

        return np.stack(cropped_stft_results, axis=0)
    except Exception as e:
        warnings.warn(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None


def process_category(category_dir, output_dir, category_name, segment_length,
                     freq_size, time_size, crop_size):
    """处理一个类别文件夹中的所有.dat文件并合并为一个.npy文件"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing category: {category_name}")

    # 获取所有.dat文件
    dat_files = [f for f in os.listdir(category_dir) if f.endswith('.dat')]

    all_segments = []
    for i, dat_file in enumerate(dat_files, 1):
        file_path = os.path.join(category_dir, dat_file)
        segments = process_single_file(
            file_path, segment_length,
            freq_size, time_size, crop_size
        )

        if segments is not None:
            all_segments.append(segments)
            print(
                f"  [{i}/{len(dat_files)}] Processed {dat_file} - got {segments.shape[0]} segments (shape: {segments.shape})")
        else:
            print(f"  [{i}/{len(dat_files)}] Skipped {dat_file} - no valid segments")

    if all_segments:
        # 合并所有片段
        merged_data = np.concatenate(all_segments, axis=0)
        save_path = os.path.join(output_dir, f"{category_name}.npy")
        np.save(save_path, merged_data)
        print(f"Merged {len(all_segments)} files into {save_path}")
        print(f"Final shape: {merged_data.shape} (segments×2×freq×time)")
        return len(all_segments), merged_data.shape
    else:
        print(f"No valid data processed for category {category_name}")
        return 0, None


def batch_process_all_categories(input_root, output_root, segment_length,
                                 freq_size, time_size, crop_size):
    """批量处理所有类别文件夹"""
    # 获取所有类别文件夹
    categories = [d for d in os.listdir(input_root)
                  if os.path.isdir(os.path.join(input_root, d))]

    print(f"Found {len(categories)} categories to process:")
    for cat in categories:
        print(f"  - {cat}")

    total_files = 0
    category_shapes = {}

    for category in categories:
        category_dir = os.path.join(input_root, category)
        num_files, shape = process_category(
            category_dir, output_root, category, segment_length,
            freq_size, time_size, crop_size
        )
        total_files += num_files
        category_shapes[category] = shape

    print(f"\nAll categories processed! Total {total_files} files with valid data.")
    print("Category shapes:")
    for cat, shape in category_shapes.items():
        print(f"  {cat}: {shape}")

    return total_files, category_shapes


def multi_folder_processing(processing_configs):
    """
    多次处理多个文件夹

    Args:
        processing_configs: 处理配置列表，每个配置包含:
            input_root: 输入根目录
            output_root: 输出根目录
            segment_length: 片段长度
            freq_size: 频率尺寸
            time_size: 时间尺寸
            crop_size: 裁剪尺寸
    """
    results = {}

    for i, config in enumerate(processing_configs, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing batch {i}/{len(processing_configs)}")
        print(f"Input: {config['input_root']}")
        print(f"Output: {config['output_root']}")
        print(f"Parameters: segment_length={config['segment_length']}s, "
              f"freq_size={config['freq_size']}, time_size={config['time_size']}, "
              f"crop_size={config['crop_size']}")
        print(f"{'=' * 60}")

        try:
            total_files, category_shapes = batch_process_all_categories(
                input_root=config['input_root'],
                output_root=config['output_root'],
                segment_length=config['segment_length'],
                freq_size=config['freq_size'],
                time_size=config['time_size'],
                crop_size=config['crop_size']
            )

            results[config['output_root']] = {
                'total_files': total_files,
                'category_shapes': category_shapes,
                'config': config
            }

        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
            results[config['output_root']] = {
                'error': str(e),
                'config': config
            }

    # 打印汇总结果
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")

    for output_dir, result in results.items():
        print(f"\nOutput: {output_dir}")
        if 'error' in result:
            print(f"  Status: FAILED - {result['error']}")
        else:
            print(f"  Status: SUCCESS")
            print(f"  Total files processed: {result['total_files']}")
            print(f"  Category shapes:")
            for cat, shape in result['category_shapes'].items():
                print(f"    {cat}: {shape}")

    return results


if __name__ == "__main__":
    # 定义多个处理配置
    processing_configs = [
        {
            'input_root': "E:/DroneDetect_V2/CLEAN",
            'output_root': "E:/stft_CLEAN_512,320_0.1",
            'segment_length': 0.1,
            'freq_size': 512,
            'time_size': 512,
            'crop_size': 320
        },
        {
            'input_root': "E:/DroneDetect_V2/WIFI",
            'output_root': "E:/stft_WIFI_512,320_0.1",
            'segment_length': 0.1,
            'freq_size': 512,
            'time_size': 512,
            'crop_size': 320
        },
        {
            'input_root': "E:/DroneDetect_V2/BLUE",
            'output_root': "E:/stft_BLUE_512,320_0.1",
            'segment_length': 0.1,
            'freq_size': 512,
            'time_size': 512,
            'crop_size': 320
        },
        {
            'input_root': "E:/DroneDetect_V2/BOTH",
            'output_root': "E:/stft_BOTH_512,320_0.1",
            'segment_length': 0.1,
            'freq_size': 512,
            'time_size': 512,
            'crop_size': 320
        }
    ]

    # 开始批量处理多个文件夹
    results = multi_folder_processing(processing_configs)

    print(f"\nAll processing completed! Processed {len(processing_configs)} folders.")
