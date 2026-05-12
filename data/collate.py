"""
data/collate.py
================
变长样本批处理与 context/target mask 生成（多星联合版本）

概述：
    本模块提供两个核心功能：
    1. collate_fn：将变长 IPP 点集样本 padding 到统一长度，构建 batch
    2. generate_context_target_mask：在有效点中划分 context 和 target 点

核心功能：
    1. 变长样本 padding（collate_fn）
       - 计算 batch 内最大点数 N_max
       - 将所有样本 padding 到 N_max（填充 0）
       - 生成 valid_mask 标识真实观测点
       - 支持多种字段：coords, angles, stec, system_ids, satellite_ids

    2. Context/Target 划分（generate_context_target_mask）
       训练模式（mode="train"）：
         - 从训练点池中随机抽取 target 点
         - mask_ratio 在 [mask_ratio_min, mask_ratio_max] 范围内随机采样
         - 剩余点作为 context
         - 每个 batch 动态生成，增强鲁棒性

       验证模式（mode="val"）：
         - 使用数据集预计算的 context_indices 和 target_indices
         - 固定划分，确保评估一致性
         - 前 80% 作 context，后 20% 作 target

    3. Mask 约束
       - context_mask 和 target_mask 互不重叠
       - 两者都是 valid_mask 的子集
       - padding 点既不是 context 也不是 target

数据流：
    [变长样本] → collate_fn → [padding 后的 batch + valid_mask]
                                        ↓
                          generate_context_target_mask
                                        ↓
                          [context_mask + target_mask]

Batch 数据格式：
    {
        "coords":           [B, N_max, 2]   归一化坐标（padding 处为 0）
        "angles":           [B, N_max, 2]   归一化角度（padding 处为 0）
        "stec":             [B, N_max, 1]   归一化 STEC（padding 处为 0）
        "system_ids":       [B, N_max]      卫星系统 ID（padding 处为 0）
        "satellite_ids":    [B, N_max]      卫星编号（padding 处为 0）
        "valid_mask":       [B, N_max]      bool，True 表示真实观测点
        "n_points":         [B]             每个样本有效点数
        "epoch_times":      list[str]       历元时间戳
        "station_names":    list[list[str]] 测站名称列表
        "context_indices":  list[np.ndarray] 可选，context 点索引（仅 val 模式）
        "target_indices":   list[np.ndarray] 可选，target 点索引（仅 val 模式）
    }

设计原则：
    - 灵活 padding：支持任意长度的样本
    - 模式分离：训练/验证使用不同的 mask 生成策略
    - 可复现性：验证模式使用固定 indices
    - 高效批处理：向量化操作，避免循环

使用示例：
    # 构建 DataLoader
    train_loader = build_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    # 训练时生成 mask
    for batch in train_loader:
        context_mask, target_mask = generate_context_target_mask(
            valid_mask=batch["valid_mask"],
            mode="train",
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
        )

    # 验证时使用预计算 indices
    for batch in val_loader:
        context_mask, target_mask = generate_context_target_mask(
            valid_mask=batch["valid_mask"],
            mode="val",
            context_indices_batch=batch["context_indices"],
            target_indices_batch=batch["target_indices"],
        )
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Set, Optional


def collate_fn(batch: list) -> dict:
    """
    将一个 batch 的变长样本 padding 到同一长度。

    Args:
        batch: list of dict，每个 dict 来自 STECEpochDataset.__getitem__

    Returns:
        dict:
            coords           [B, N_max, 2]   float32，归一化坐标（padding 处为 0）
            angles           [B, N_max, 2]   float32，归一化角度（padding 处为 0）
            stec             [B, N_max, 1]   float32，归一化 STEC（padding 处为 0）
            system_ids       [B, N_max]      int64，系统ID（padding 处为 0）
            satellite_ids    [B, N_max]      int64，卫星ID（padding 处为 0）
            valid_mask       [B, N_max]      bool，True 表示真实观测点
            n_points         [B]             int，每个样本有效点数
            epoch_times      list[str]       时间戳
            station_names    list[list[str]] 站点名列表（每个样本一个列表）
            context_indices  list[np.ndarray] 可选，context 点索引（仅 val_eval 模式）
            target_indices   list[np.ndarray] 可选，target 点索引（仅 val_eval/test_target 模式）
    """
    B = len(batch)
    # 计算 batch 内最大点数，用于统一 padding 长度
    n_max = max(item["n_points"] for item in batch)

    # 初始化 padding 后的张量（全零）
    coords_batch = np.zeros((B, n_max, 2), dtype=np.float32)
    angles_batch = np.zeros((B, n_max, 2), dtype=np.float32)
    stec_batch = np.zeros((B, n_max, 1), dtype=np.float32)
    system_ids_batch = np.zeros((B, n_max), dtype=np.int64)
    satellite_ids_batch = np.zeros((B, n_max), dtype=np.int64)
    valid_mask_batch = np.zeros((B, n_max), dtype=bool)
    n_points_list = []
    epoch_times_list = []
    station_names_list = []
    context_indices_list = []
    target_indices_list = []

    for i, item in enumerate(batch):
        n = item["n_points"]
        coords_batch[i, :n, :] = item["coords"]          # [n, 2]
        angles_batch[i, :n, :] = item["angles"]          # [n, 2]
        stec_batch[i, :n, :] = item["stec"]              # [n, 1]
        system_ids_batch[i, :n] = item["system_ids"]     # [n]
        satellite_ids_batch[i, :n] = item["satellite_ids"]  # [n]
        valid_mask_batch[i, :n] = True                   # 前 n 个是有效点
        n_points_list.append(n)
        epoch_times_list.append(item["epoch_time"])
        station_names_list.append(item["station_names"])

        # 可选字段：context_indices 和 target_indices
        if "context_indices" in item:
            context_indices_list.append(item["context_indices"])
        if "target_indices" in item:
            target_indices_list.append(item["target_indices"])

    result = {
        "coords":         torch.from_numpy(coords_batch),         # [B, N_max, 2]
        "angles":         torch.from_numpy(angles_batch),         # [B, N_max, 2]
        "stec":           torch.from_numpy(stec_batch),           # [B, N_max, 1]
        "system_ids":     torch.from_numpy(system_ids_batch),     # [B, N_max]
        "satellite_ids":  torch.from_numpy(satellite_ids_batch),  # [B, N_max]
        "valid_mask":     torch.from_numpy(valid_mask_batch),     # [B, N_max]
        "n_points":       torch.tensor(n_points_list, dtype=torch.long),  # [B]
        "epoch_times":    epoch_times_list,
        "station_names":  station_names_list,  # list[list[str]]
    }

    # 添加可选字段
    if context_indices_list:
        result["context_indices"] = context_indices_list  # list[np.ndarray]
    if target_indices_list:
        result["target_indices"] = target_indices_list    # list[np.ndarray]

    return result


def generate_context_target_mask(
    valid_mask: torch.Tensor,
    mask_ratio_min: float = 0.10,
    mask_ratio_max: float = 0.50,
    mode: str = "train",
    context_indices_batch: list = None,
    target_indices_batch: list = None,
) -> tuple:
    """
    在每个样本的有效点中划分 context 和 target（IPP 点级划分版本）。

    模式说明：
    - mode="train": 训练模式
      - 从训练点池（已经是前 80% IPP 点）中随机抽取 target
      - target 数量 = 训练点数 × mask_ratio（mask_ratio ∈ [mask_ratio_min, mask_ratio_max]）
      - 其余训练点作为 context
    - mode="val": 验证模式
      - 使用数据集提供的 context_indices 和 target_indices
      - 无随机性，确保验证结果可复现

    Args:
        valid_mask:              [B, N_max]  bool，True 表示有效观测点
        mask_ratio_min:          target 点最小比例（仅 train 模式）
        mask_ratio_max:          target 点最大比例（仅 train 模式）
        mode:                    "train" 或 "val"
        context_indices_batch:   list[np.ndarray]，每个样本的 context 点索引（仅 val 模式）
        target_indices_batch:    list[np.ndarray]，每个样本的 target 点索引（仅 val 模式）

    Returns:
        context_mask: [B, N_max]  bool
        target_mask:  [B, N_max]  bool
    """
    B, N_max = valid_mask.shape

    context_mask = torch.zeros(B, N_max, dtype=torch.bool)
    target_mask = torch.zeros(B, N_max, dtype=torch.bool)

    if mode == "train":
        # 训练模式：随机抽取 target
        # 本 step 全局采样一个遮挡比例
        mask_ratio = np.random.uniform(mask_ratio_min, mask_ratio_max)

        for i in range(B):
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(-1).cpu().numpy()  # [N_valid]
            n_valid = len(valid_indices)

            if n_valid == 0:
                continue  # 该样本没有有效点，跳过

            # 从所有有效点中随机抽取 target
            n_target = max(1, min(int(n_valid * mask_ratio), n_valid - 1))

            perm = np.random.permutation(n_valid)
            target_idx = valid_indices[perm[:n_target]]
            context_idx = valid_indices[perm[n_target:]]

            target_mask[i, target_idx] = True
            context_mask[i, context_idx] = True

    elif mode == "val":
        # 验证模式：使用预计算的 indices
        if context_indices_batch is None or target_indices_batch is None:
            raise ValueError("验证模式下必须提供 context_indices_batch 和 target_indices_batch")

        for i in range(B):
            if i < len(context_indices_batch):
                context_idx = context_indices_batch[i]
                context_mask[i, context_idx] = True

            if i < len(target_indices_batch):
                target_idx = target_indices_batch[i]
                target_mask[i, target_idx] = True

    else:
        raise ValueError(f"未知的 mode：{mode}，必须为 'train' 或 'val'")

    return context_mask, target_mask


def build_dataloader(dataset, batch_size: int, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """
    构建 DataLoader，使用自定义 collate_fn 处理变长样本。

    Args:
        dataset:      STECEpochDataset 实例
        batch_size:   批大小
        shuffle:      是否打乱
        num_workers:  数据加载线程数（Windows 下建议设为 0）

    Returns:
        DataLoader 实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
