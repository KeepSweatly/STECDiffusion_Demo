"""
utils/normalizer.py
====================
STEC 和地理坐标的归一化工具。

- CoordNormalizer: 对经纬度做 minmax 归一化，归一化到 [-1, 1]
- STECNormalizer: 对 STEC 做标准化（均值0方差1）或 minmax 归一化

设计说明：
- 归一化参数从训练集统计，测试集复用同一参数
- 支持 fit / transform / inverse_transform 接口
"""

import numpy as np
import torch


class CoordNormalizer:
    """
    经纬度归一化器（MinMax → [-1, 1]）
    """

    def __init__(self):
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None

    def fit(self, lats: np.ndarray, lons: np.ndarray):
        """从训练集统计归一化参数"""
        self.lat_min = float(lats.min())
        self.lat_max = float(lats.max())
        self.lon_min = float(lons.min())
        self.lon_max = float(lons.max())

    def transform(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        归一化经纬度到 [-1, 1]
        返回 shape: [N, 2]，列顺序为 [lat_norm, lon_norm]
        """
        lat_norm = 2.0 * (lats - self.lat_min) / (self.lat_max - self.lat_min + 1e-8) - 1.0
        lon_norm = 2.0 * (lons - self.lon_min) / (self.lon_max - self.lon_min + 1e-8) - 1.0
        return np.stack([lat_norm, lon_norm], axis=-1).astype(np.float32)

    def inverse_transform_lat(self, lat_norm: np.ndarray) -> np.ndarray:
        return (lat_norm + 1.0) / 2.0 * (self.lat_max - self.lat_min) + self.lat_min

    def inverse_transform_lon(self, lon_norm: np.ndarray) -> np.ndarray:
        return (lon_norm + 1.0) / 2.0 * (self.lon_max - self.lon_min) + self.lon_min

    def state_dict(self) -> dict:
        return {
            "lat_min": self.lat_min, "lat_max": self.lat_max,
            "lon_min": self.lon_min, "lon_max": self.lon_max,
        }

    def load_state_dict(self, d: dict):
        self.lat_min = d["lat_min"]
        self.lat_max = d["lat_max"]
        self.lon_min = d["lon_min"]
        self.lon_max = d["lon_max"]


class STECNormalizer:
    """
    STEC 值归一化器（标准化：均值0方差1）
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, stec: np.ndarray):
        """从训练集统计均值和标准差"""
        self.mean = float(stec.mean())
        self.std = float(stec.std() + 1e-8)

    def transform(self, stec: np.ndarray) -> np.ndarray:
        """标准化 STEC"""
        return ((stec - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, stec_norm: np.ndarray) -> np.ndarray:
        """反标准化"""
        return stec_norm * self.std + self.mean

    def inverse_transform_tensor(self, stec_norm: torch.Tensor) -> torch.Tensor:
        """反标准化（Tensor版本）"""
        return stec_norm * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.std = d["std"]
