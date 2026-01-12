# comm.py

import torch
import numpy as np

def apply_mock_channel_noise(quantized_params, enable=False, snr_db=20.0, device="cuda"):
    """
    模拟无线信道 y = h * x + z
    - h: 信道增益（假设为1，平坦衰落）
    - z: AWGN 噪声
    - SNR (dB) 控制噪声强度

    Args:
        quantized_params (dict): {'xyz', 'scale', 'rotation', 'f_dc', 'f_rest', ...}
        enable (bool): 是否启用噪声
        snr_db (float): 信噪比（dB）
        device: 设备

    Returns:
        corrupted_params (dict): 加噪后的参数（同结构）
    """
    if not enable:
        return quantized_params  # 无损通过

    corrupted = {}
    for key, tensor in quantized_params.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            corrupted[key] = tensor
            continue

        # 计算信号功率
        signal_power = torch.mean(tensor ** 2)
        if signal_power == 0:
            corrupted[key] = tensor
            continue

        # 转换 SNR 到线性尺度
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # 生成高斯噪声
        noise = torch.randn_like(tensor, device=device) * torch.sqrt(noise_power)

        # 应用 y = x + z （h=1）
        corrupted[key] = tensor + noise

        # 可选：裁剪到合理范围（如 opacity ∈ [0,1]）
        if key == "opacity":
            corrupted[key] = torch.clamp(corrupted[key], 0.0, 1.0)
        elif key in ["scale", "xyz"]:
            # scale 不能为负
            if key == "scale":
                corrupted[key] = torch.clamp(corrupted[key], min=1e-6)

    return corrupted


# JSCC函数：实际 JSCC 编码/解码逻辑需根据具体实现替换
def apply_real_jscc(quantized_params, codebook, channel_model):
    bits = encode_to_bits(quantized_params, codebook)
    transmitted_bits = channel_model(bits)
    recovered_indices = decode_bits(transmitted_bits)
    return decode_from_codebook(recovered_indices, codebook)