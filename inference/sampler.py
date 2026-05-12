"""
inference/sampler.py
=====================
STEC 条件扩散模型推理采样器

概述：
    本模块实现完整的反向 SDE 推理采样流程，给定 context 点（已知 STEC），
    通过迭代去噪恢复 target 点的 STEC 值。设计参考 EDiffSR 的推理流程，
    适配为离散点任务。

核心功能：
    STECSampler 类：
        - 封装预训练模型和 SDE 实例
        - 提供单批次推理接口（sample）
        - 提供数据集评估接口（evaluate_dataset）
        - 自动处理归一化/反归一化

推理流程：
    1. 输入准备
       - 指定 context 点（已知 STEC）和 target 点（待预测）
       - 归一化坐标、角度、STEC 值

    2. 条件均值构建
       - context 点：μ = context STEC 全局均值
       - target 点：μ = IDW 距离加权插值（基于最近 k 个 context 点）

    3. 初始化噪声状态
       - context 点：保持真实 STEC 值（条件固定）
       - target 点：添加最大噪声 x_T = μ + max_sigma * ε

    4. 反向 SDE 去噪
       - 从 t=T 到 t=1 迭代去噪
       - 每步：模型预测噪声 → 更新状态 → context 点保持不变
       - 最终得到去噪结果 x0

    5. 反归一化输出
       - 将归一化空间的预测值转换回原始 TECU 单位
       - 计算评估指标（MAE, RMSE）

使用示例：
    # 初始化采样器
    sampler = STECSampler(
        model=trained_model,
        sde=sde_instance,
        stec_normalizer=normalizer,
        device=torch.device("cuda"),
    )

    # 单批次推理
    result = sampler.sample(
        coords=coords,
        angles=angles,
        system_ids=system_ids,
        stec_full=stec_full,
        context_mask=context_mask,
        target_mask=target_mask,
        valid_mask=valid_mask,
    )

    # 数据集评估
    metrics = sampler.evaluate_dataset(test_loader, verbose=True)
    print(f"MAE: {metrics['mae']:.4f} TECU")

设计特点：
    - 完整 T 步去噪：不使用加速采样，确保最佳质量
    - 条件固定：context 点在每步保持真实值，提供稳定条件信息
    - 批量处理：支持批量推理，提升效率
    - 自动归一化：内部处理归一化/反归一化，简化使用

参考来源：
    EDiffSR/codes/config/sisr/test.py（适配为离散点任务）
"""

import os
import torch
import numpy as np
from typing import Optional

from models.transformer import STECDiffTransformer
from diffusion.sde import STEC_IRSDE
from data.collate import generate_context_target_mask
from utils.normalizer import STECNormalizer, CoordNormalizer
from utils.logger import get_logger


class STECSampler:
    """
    STEC 条件扩散模型推理采样器（多星联合版本，第三~五阶段：带 Guidance）。

    Args:
        model:            预训练的 STECDiffTransformer
        sde:              STEC_IRSDE 实例
        stec_normalizer:  STECNormalizer（用于反归一化）
        device:           推理设备
        num_steps:        反向 SDE 步数（默认与训练 T 相同）
        use_guidance:     是否使用 guidance（默认 True，第三~五阶段）
    """

    def __init__(
        self,
        model: STECDiffTransformer,
        sde: STEC_IRSDE,
        stec_normalizer: STECNormalizer,
        device: torch.device,
        num_steps: Optional[int] = None,
        use_guidance: bool = True,
    ):
        self.model           = model.to(device).eval()
        self.sde             = sde
        self.stec_normalizer = stec_normalizer
        self.device          = device
        self.num_steps       = num_steps if num_steps is not None else sde.T
        self.use_guidance    = use_guidance  # 第三~五阶段：是否使用 guidance

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        angles: torch.Tensor,
        system_ids: torch.Tensor,
        stec_full: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        verbose: bool = False,
    ) -> dict:
        """
        对单个批次进行条件扩散采样推理。

        Args:
            coords:       [B, N, 2]  归一化坐标
            angles:       [B, N, 2]  归一化角度
            system_ids:   [B, N]     int64 系统ID
            stec_full:    [B, N, 1]  完整 STEC（推理时只用 context 处的值）
            context_mask: [B, N]     bool，context 点
            target_mask:  [B, N]     bool，target 点（待预测）
            valid_mask:   [B, N]     bool，有效点
            verbose:      是否打印反向扩散进度

        Returns:
            dict:
                pred_stec_norm:  [B, N, 1]  归一化空间下的预测 STEC（target 处）
                pred_stec_orig:  [B, N, 1]  原始单位（TECU）的预测 STEC（target 处）
                true_stec_orig:  [B, N, 1]  原始单位的真实 STEC（target 处，用于评估）
        """
        coords       = coords.to(self.device)
        angles       = angles.to(self.device)
        system_ids   = system_ids.to(self.device)
        stec_full    = stec_full.to(self.device)
        context_mask = context_mask.to(self.device)
        target_mask  = target_mask.to(self.device)
        valid_mask   = valid_mask.to(self.device)
        B, N, _      = stec_full.shape

        # ---- 1. 构建角色类型编码 ----
        role_type = torch.zeros(B, N, dtype=torch.long, device=self.device)
        role_type[context_mask] = 1
        role_type[target_mask]  = 2

        # ---- 2. context_stec：target 和 padding 处为 0 ----
        context_stec = stec_full * context_mask.unsqueeze(-1).float()  # [B, N, 1]

        # ---- 3. 构建条件均值 μ 和先验特征（context 均值 + target IDW）----
        mu, prior_features = self.sde.build_mu_batch(
            coords, stec_full, context_mask, target_mask,
            return_prior_features=True,
        )  # mu: [B, N, 1], prior_features: [B, N, 3]

        # ---- 4. 初始化推理起始噪声状态 ----
        # context 点：使用真实 STEC；target 点：添加最大噪声
        x_T = stec_full.clone()
        # 对 target 点加最大噪声
        target_noise = mu + self.sde.max_sigma * torch.randn_like(stec_full)
        x_T[target_mask] = target_noise[target_mask]

        # ---- 5. 完整反向 SDE 去噪（T 步，第三~五阶段：带 Guidance）----
        x0_pred = self.sde.reverse_sde(
            x_T          = x_T,
            mu           = mu,
            model        = self.model,
            coords       = coords,
            angles       = angles,
            system_ids   = system_ids,
            context_stec = context_stec,
            role_type    = role_type,
            valid_mask   = valid_mask,
            target_mask  = target_mask,
            device       = self.device,
            prior_features = prior_features,
            use_guidance = self.use_guidance,  # 第三~五阶段：控制是否使用 guidance
            verbose      = verbose,
        )  # [B, N, 1]

        # ---- 6. 反归一化到原始 TECU ----
        pred_np = x0_pred.cpu().numpy()    # [B, N, 1]
        true_np = stec_full.cpu().numpy()  # [B, N, 1]
        pred_orig = self.stec_normalizer.inverse_transform(pred_np.reshape(-1)).reshape(B, N, 1)
        true_orig = self.stec_normalizer.inverse_transform(true_np.reshape(-1)).reshape(B, N, 1)

        return {
            "pred_stec_norm": x0_pred.cpu(),
            "pred_stec_orig": torch.from_numpy(pred_orig.astype(np.float32)),
            "true_stec_orig": torch.from_numpy(true_orig.astype(np.float32)),
            "target_mask":    target_mask.cpu(),
            "context_mask":   context_mask.cpu(),
        }

    def evaluate_dataset(
        self,
        test_loader,
        verbose: bool = False,
    ) -> dict:
        """
        在完整测试集上评估推理性能。
        使用数据集预计算的 context_indices / target_indices（val_eval 或 test_target 模式）。

        Args:
            test_loader:  测试集 DataLoader（来自 val_eval 或 test_target 模式数据集）
            verbose:      是否打印进度

        Returns:
            metrics: {"mae": float, "rmse": float, "n_target_points": int}
        """
        all_pred   = []
        all_target = []
        n_target   = 0

        for i, batch in enumerate(test_loader):
            if verbose:
                print(f"  推理 batch [{i+1}/{len(test_loader)}] ...")

            coords       = batch["coords"]         # [B, N, 2]
            angles       = batch["angles"]         # [B, N, 2]
            system_ids   = batch["system_ids"]     # [B, N]
            stec         = batch["stec"]           # [B, N, 1]
            valid_mask   = batch["valid_mask"]     # [B, N]
            context_indices_batch = batch.get("context_indices", None)
            target_indices_batch  = batch.get("target_indices", None)

            # 从预计算 indices 生成 mask
            context_mask, target_mask = generate_context_target_mask(
                valid_mask,
                mode="val",
                context_indices_batch=context_indices_batch,
                target_indices_batch=target_indices_batch,
            )

            if target_mask.sum() == 0:
                continue

            result = self.sample(
                coords=coords,
                angles=angles,
                system_ids=system_ids,
                stec_full=stec,
                context_mask=context_mask,
                target_mask=target_mask,
                valid_mask=valid_mask,
                verbose=False,
            )

            t_mask    = result["target_mask"]                          # [B, N]
            pred_orig = result["pred_stec_orig"][t_mask].numpy()      # [M]
            true_orig = result["true_stec_orig"][t_mask].numpy()      # [M]

            all_pred.append(pred_orig)
            all_target.append(true_orig)
            n_target += t_mask.sum().item()

        if len(all_pred) == 0:
            return {"mae": float("inf"), "rmse": float("inf"), "n_target_points": 0}

        all_pred   = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)

        mae  = float(np.mean(np.abs(all_pred - all_target)))
        rmse = float(np.sqrt(np.mean((all_pred - all_target) ** 2)))

        return {"mae": mae, "rmse": rmse, "n_target_points": n_target}
