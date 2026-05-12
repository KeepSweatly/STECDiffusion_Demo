"""
models/pos_encoding.py
=======================
地理坐标傅里叶位置编码 + 扩散时间步嵌入。

设计说明：
  - 输入是 2D 连续坐标 (lat_norm, lon_norm) ∈ [-1, 1]
  - 传统 RoPE 依赖整数序列位置，不适合不规则坐标
  - 改为：对每个坐标分量分别应用正弦/余弦傅里叶特征（Nerf 风格）
  - 傅里叶特征维度 = 2 * fourier_bands * 2（lat 和 lon 各贡献一份）

时间步嵌入：
  - 正弦/余弦嵌入（参考 JustImageTransformer 和 DDPM）
  - 通过两层 MLP 映射到模型维度
"""

import math
import torch
import torch.nn as nn


class FourierPosEncoding(nn.Module):
    """
    2D 地理坐标傅里叶位置编码。

    对 (lat_norm, lon_norm) 各分量分别计算多频率正弦余弦特征，
    然后通过线性层映射到模型维度 dim。

    Args:
        fourier_bands: 频率数量（每个坐标分量各生成 2*fourier_bands 维特征）
        dim:           输出维度（模型隐层维度）
        max_freq:      最大频率（控制编码分辨率）
    """

    def __init__(self, fourier_bands: int = 64, dim: int = 256, max_freq: float = 10.0):
        super().__init__()
        self.fourier_bands = fourier_bands
        # 频率：对数均匀分布在 [1, max_freq]
        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), fourier_bands)
        )  # [fourier_bands]
        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        self.register_buffer("freqs", freqs)

        # 傅里叶特征维度：lat 和 lon 各贡献 2*fourier_bands 维
        fourier_dim = 4 * fourier_bands  # sin(lat*f), cos(lat*f), sin(lon*f), cos(lon*f)
        # 线性层将傅里叶特征映射到 dim
        self.proj = nn.Linear(fourier_dim, dim, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 2]  归一化坐标 (lat_norm, lon_norm) ∈ [-1, 1]

        Returns:
            pos_emb: [B, N, dim]  位置嵌入
        """
        lat = coords[..., 0:1]  # [B, N, 1]
        lon = coords[..., 1:2]  # [B, N, 1]

        # 计算各频率的正弦余弦特征
        # freqs: [fourier_bands] → 广播到 [B, N, fourier_bands]
        lat_freqs = lat * self.freqs * math.pi  # [B, N, fourier_bands]
        lon_freqs = lon * self.freqs * math.pi  # [B, N, fourier_bands]

        # 拼接 sin 和 cos [B, N, 4*fourier_bands]
        fourier_feat = torch.cat([
            torch.sin(lat_freqs),
            torch.cos(lat_freqs),
            torch.sin(lon_freqs),
            torch.cos(lon_freqs),
        ], dim=-1)

        return self.proj(fourier_feat)  # [B, N, dim]


class AngleFourierEncoding(nn.Module):
    """
    角度傅里叶位置编码（用于 azimuth 和 elevation）。

    与 FourierPosEncoding 逻辑相同，但用于编码角度信息。
    对 (azimuth_norm, elevation_norm) 各分量分别计算多频率正弦余弦特征。

    Args:
        fourier_bands: 频率数量（每个角度分量各生成 2*fourier_bands 维特征）
        dim:           输出维度（模型隐层维度）
        max_freq:      最大频率（控制编码分辨率）
    """

    def __init__(self, fourier_bands: int = 32, dim: int = 256, max_freq: float = 10.0):
        super().__init__()
        self.fourier_bands = fourier_bands
        # 频率：对数均匀分布在 [1, max_freq]
        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), fourier_bands)
        )  # [fourier_bands]
        self.register_buffer("freqs", freqs)

        # 傅里叶特征维度：azimuth 和 elevation 各贡献 2*fourier_bands 维
        fourier_dim = 4 * fourier_bands  # sin(az*f), cos(az*f), sin(el*f), cos(el*f)
        # 线性层将傅里叶特征映射到 dim
        self.proj = nn.Linear(fourier_dim, dim, bias=True)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: [B, N, 2]  归一化角度 (azimuth_norm, elevation_norm) ∈ [-1, 1]

        Returns:
            angle_emb: [B, N, dim]  角度嵌入
        """
        azimuth = angles[..., 0:1]    # [B, N, 1]
        elevation = angles[..., 1:2]  # [B, N, 1]

        # 计算各频率的正弦余弦特征
        az_freqs = azimuth * self.freqs * math.pi      # [B, N, fourier_bands]
        el_freqs = elevation * self.freqs * math.pi    # [B, N, fourier_bands]

        # 拼接 sin 和 cos [B, N, 4*fourier_bands]
        fourier_feat = torch.cat([
            torch.sin(az_freqs),
            torch.cos(az_freqs),
            torch.sin(el_freqs),
            torch.cos(el_freqs),
        ], dim=-1)

        return self.proj(fourier_feat)  # [B, N, dim]


class TimestepEmbedding(nn.Module):
    """
    扩散时间步正弦嵌入 + MLP 映射。

    来源：参考 JustImageTransformer get_timestep_embedding 和 DDPM。
    时间步先通过正弦/余弦编码，再通过两层 MLP 映射到条件维度。

    Args:
        dim:     输出维度（与模型 dim 一致）
        sinusoidal_dim: 正弦嵌入的中间维度（默认与 dim 相同）
    """

    def __init__(self, dim: int, sinusoidal_dim: int = 256):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        # 两层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        正弦/余弦时间步编码（参考 DDPM / JustImageTransformer）。

        Args:
            t: [B]  时间步（整数或浮点数）

        Returns:
            emb: [B, sinusoidal_dim]
        """
        half_dim = self.sinusoidal_dim // 2
        # 频率：exp(-log(10000) * k / (half_dim - 1))
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / (half_dim - 1)
        )  # [half_dim]
        # t 归一化（可选，这里直接用整数步）
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, sinusoidal_dim]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B]  扩散时间步

        Returns:
            emb: [B, dim]
        """
        sinusoidal = self._sinusoidal_embedding(t)  # [B, sinusoidal_dim]
        return self.mlp(sinusoidal)                 # [B, dim]


def precompute_rope_freqs(head_dim: int, max_seq_len: int = 4096,
                          theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 旋转频率（用于 STECAttention 中的 apply_rope）。

    注意：本项目中 RoPE 用于序列位置（点的序号），
    地理空间位置由 FourierPosEncoding 独立编码。

    Args:
        head_dim:    每个注意力头的维度
        max_seq_len: 预计算的最大序列长度
        theta:       频率基数

    Returns:
        freqs_cis: [max_seq_len, head_dim//2]  复数频率（极坐标形式）
    """
    freqs = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    ))  # [head_dim//2]
    t = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    freqs = torch.outer(t, freqs)  # [max_seq_len, head_dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # 复数极坐标
