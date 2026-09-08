import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift


def load_and_visualize_npy(file_path, save_path=None):
    """分别显示实部、虚部和幅度谱的时频图，全部使用dB幅度表达"""
    # 加载数据
    data = np.load(file_path)
    print(f"数据形状: {data.shape}")

    # 检查数据范围
    print(f"实部范围: [{np.min(data[:, 0])}, {np.max(data[:, 0])}]")
    print(f"虚部范围: [{np.min(data[:, 1])}, {np.max(data[:, 1])}]")

    # 取样本
    sample_idx = 603
    stft_real = data[sample_idx, 0]  # 实部
    stft_imag = data[sample_idx, 1]  # 虚部

    # 频率轴设置
    CENTER_FREQ = 2.4375e9 / 1e9  # 中心频率2.4GHz
    SAMPLE_RATE = 60e6 / 1e6  # 采样率60MHz
    BANDWIDTH = 28e6  # 带宽28MHz
    freq_bins = stft_real.shape[0]
    max_freq_offset = SAMPLE_RATE / 4
    freq_offset = np.linspace(-max_freq_offset, max_freq_offset, freq_bins)

    # 时间轴设置
    time_bins = stft_real.shape[1]
    time_axis = np.linspace(0, 0.05, time_bins)  # 50ms

    # 将实部和虚部转换为dB幅度
    real_magnitude_db = 20 * np.log10(np.abs(stft_real) + 1e-10)
    imag_magnitude_db = 20 * np.log10(np.abs(stft_imag) + 1e-10)
    complex_magnitude_db = 20 * np.log10(np.abs(stft_real + 1j * stft_imag) + 1e-10)

    # 设置统一的dB动态范围
    db_range = 80  # dB动态范围

    # 计算统一的颜色范围
    all_data = np.concatenate([real_magnitude_db, imag_magnitude_db, complex_magnitude_db])
    vmax = np.max(all_data)
    vmin = vmax - db_range

    print(f"\n实部dB范围: [{np.min(real_magnitude_db):.2f}, {np.max(real_magnitude_db):.2f}] dB")
    print(f"虚部dB范围: [{np.min(imag_magnitude_db):.2f}, {np.max(imag_magnitude_db):.2f}] dB")
    print(f"幅度谱dB范围: [{np.min(complex_magnitude_db):.2f}, {np.max(complex_magnitude_db):.2f}] dB")
    print(f"统一颜色范围: [{vmin:.2f}, {vmax:.2f}] dB")

    # 1. 实部时频图 - 使用dB幅度
    plt.figure(figsize=(12, 6))
    im1 = plt.imshow(
        real_magnitude_db,
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], freq_offset[0], freq_offset[-1]],
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        origin='lower'
    )
    # plt.axhline(y=BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    # plt.axhline(y=-BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    plt.colorbar(label='Real Part Magnitude (dB)')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel(f'Frequency Offset from {CENTER_FREQ} GHz [MHz]', fontsize=12)
    plt.title(f'Real Part - Magnitude in dB', fontsize=14)
    plt.grid(alpha=0.3)

    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base_path}_real_part_dB.png", dpi=300, bbox_inches='tight')
        print(f"实部图片已保存至: {base_path}_real_part_dB.png")

    plt.show()

    # 2. 虚部时频图 - 使用dB幅度
    plt.figure(figsize=(12, 6))
    im2 = plt.imshow(
        imag_magnitude_db,
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], freq_offset[0], freq_offset[-1]],
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        origin='lower'
    )
    # plt.axhline(y=BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    # plt.axhline(y=-BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    plt.colorbar(label='Imaginary Part Magnitude (dB)')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel(f'Frequency Offset from {CENTER_FREQ} GHz [MHz]', fontsize=12)
    plt.title(f'Imaginary Part - Magnitude in dB', fontsize=14)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(f"{base_path}_imaginary_part_dB.png", dpi=300, bbox_inches='tight')
        print(f"虚部图片已保存至: {base_path}_imaginary_part_dB.png")

    plt.show()

    # 3. 幅度谱时频图（复数模值的dB）
    plt.figure(figsize=(12, 6))
    im3 = plt.imshow(
        complex_magnitude_db,
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], freq_offset[0], freq_offset[-1]],
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        origin='lower'
    )
    # plt.axhline(y=BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    # plt.axhline(y=-BANDWIDTH / 2e6, color='white', linestyle='--', linewidth=1.2, alpha=0.9)
    plt.colorbar(label='Complex Magnitude (dB)')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel(f'Frequency Offset from {CENTER_FREQ} GHz [MHz]', fontsize=12)
    plt.title('Complex Magnitude Spectrum (dB)', fontsize=14)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(f"{base_path}_complex_magnitude_spectrum.png", dpi=300, bbox_inches='tight')
        print(f"幅度谱图片已保存至: {base_path}_complex_magnitude_spectrum.png")

    plt.show()

    # 打印统计信息
    print(f"\n实部统计 (dB):")
    print(f"  均值: {np.mean(real_magnitude_db):.2f} dB")
    print(f"  标准差: {np.std(real_magnitude_db):.2f} dB")
    print(f"  最小值: {np.min(real_magnitude_db):.2f} dB")
    print(f"  最大值: {np.max(real_magnitude_db):.2f} dB")

    print(f"\n虚部统计 (dB):")
    print(f"  均值: {np.mean(imag_magnitude_db):.2f} dB")
    print(f"  标准差: {np.std(imag_magnitude_db):.2f} dB")
    print(f"  最小值: {np.min(imag_magnitude_db):.2f} dB")
    print(f"  最大值: {np.max(imag_magnitude_db):.2f} dB")

    print(f"\n复数幅度统计 (dB):")
    print(f"  均值: {np.mean(complex_magnitude_db):.2f} dB")
    print(f"  标准差: {np.std(complex_magnitude_db):.2f} dB")
    print(f"  最小值: {np.min(complex_magnitude_db):.2f} dB")
    print(f"  最大值: {np.max(complex_magnitude_db):.2f} dB")


if __name__ == "__main__":
    # load_and_visualize_npy("E:/stft_BLUE_0.1s/AIR_HO.npy")
    load_and_visualize_npy("data_open_set_CLEAN_WIFI_BLUE_BOTH_512,320/test/samples_zscore.npy", "figure/spectrum")
