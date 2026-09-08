import numpy as np


def complex_zscore_power_norm(x, eps=1e-6):
    """
    复数信号归一化：Z-score去直流 + 功率归一化
    保持I/Q相对关系，同时消除功率差异和直流偏移
    """
    # 先清理输入中的 nan/inf（仅用于防御性编程，正常情况下不应出现）
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    real_part = x[:, 0]
    imag_part = x[:, 1]

    # ===== Step 1: 逐样本去直流（Z-score去均值）=====
    mu_real = real_part.mean(axis=(1, 2), keepdims=True)
    mu_imag = imag_part.mean(axis=(1, 2), keepdims=True)

    std_real = real_part.std(axis=(1, 2), keepdims=True)
    std_imag = imag_part.std(axis=(1, 2), keepdims=True)

    # 防止除零
    std_real = np.where(std_real < eps, 1.0, std_real)
    std_imag = np.where(std_imag < eps, 1.0, std_imag)

    real_norm = (real_part - mu_real) / std_real
    imag_norm = (imag_part - mu_imag) / std_imag

    # ===== Step 2: 功率归一化（保持I/Q相对比例）=====
    c = real_norm + 1j * imag_norm

    # 计算平均幅度（不是功率，是幅度）
    amplitude = np.abs(c)
    mean_amp = amplitude.mean(axis=(1, 2), keepdims=True)

    # 归一化使平均幅度为1
    c = c / (mean_amp + eps)

    return np.stack([c.real, c.imag], axis=1).astype(np.float32)


# 主处理流程
input_path = 'data_close_set_CLEAN_WIFI_BLUE_BOTH_512,320_0dB_old/augmented_train(2)/samples.npy'
output_path = 'data_close_set_CLEAN_WIFI_BLUE_BOTH_512,320_0dB_old/augmented_train(2)/samples_zscore.npy'

print(f"正在加载数据: {input_path}")
raw = np.load(input_path, mmap_mode='r')
print(f"原始数据形状: {raw.shape}, dtype: {raw.dtype}")

# 预分配输出数组
norm = np.empty_like(raw, dtype=np.float32)

# 分块处理，避免内存溢出
bs = 2048
total_samples = len(raw)

print(f"开始处理，批次大小: {bs}, 总样本数: {total_samples}")

for i in range(0, total_samples, bs):
    end_idx = min(i + bs, total_samples)
    batch = raw[i:end_idx]
    norm[i:end_idx] = complex_zscore_power_norm(batch)

    if (i // bs) % 10 == 0 or end_idx == total_samples:
        print(f"  进度: {end_idx}/{total_samples} ({100 * end_idx / total_samples:.1f}%)")

# 保存结果
np.save(output_path, norm)
print(f"\n处理完成，已保存至: {output_path}")

# 验证结果
print("\n=== 验证结果 ===")
sample_power = np.abs(norm[:, 0] + 1j * norm[:, 1]).mean(axis=(1, 2))
print(f"平均幅度: {sample_power.mean():.6f} (应≈1.0)")
print(f"幅度标准差: {sample_power.std():.6f}")
print(f"I路均值: {norm[:, 0].mean():.6f} (应≈0)")
print(f"I路标准差: {norm[:, 0].std():.4f}")
print(f"Q路均值: {norm[:, 1].mean():.6f} (应≈0)")
print(f"Q路标准差: {norm[:, 1].std():.4f}")

# 检查功率分布是否保留
power = (norm[:, 0] ** 2 + norm[:, 1] ** 2).mean(axis=(1, 2))
print(f"\n复数功率均值: {power.mean():.6f}")
print(f"复数功率标准差: {power.std():.6f} (应>0，保留功率分布差异)")