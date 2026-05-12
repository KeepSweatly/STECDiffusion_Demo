"""
scripts/test.py
================
STEC 条件扩散模型推理/测试入口脚本（多星联合版本）

概述：
    本脚本实现多卫星联合推理，通过历元对齐方式，利用 model_stations 的观测数据
    作为 context，预测 val_stations 的 STEC 值，并按卫星分组导出结果。

使用方式：
    cd D:/Phd/IonoModeling/stec_diffusionv2
    python scripts/test.py
    python scripts/test.py --config configs/default.yaml
    python scripts/test.py --checkpoint experiments/exp_joint_epoch/checkpoints/ckpt_best.pth
    python scripts/test.py --verbose

核心功能：
    1. 历元文件扫描与对齐
       - 扫描 val_stations/ 目录中的历元文件（测试目标）
       - 匹配 model_stations/ 中同名历元文件（context 来源）
       - 仅处理两个目录中共同存在的历元（文件名匹配）

    2. 多星联合推理
       - model_stations 所有 IPP 点 → context（已知 STEC）
       - val_stations 所有 IPP 点 → target（待预测 STEC）
       - 所有卫星的 IPP 点在同一样本中联合处理

    3. 完整反向 SDE 推理
       - 构建 IDW 条件均值 μ（context 均值 + target IDW 插值）
       - 初始化 target 点为最大噪声状态
       - 迭代 T 步反向 SDE 去噪，恢复 STEC 值

    4. 结果导出与分析
       - 导出完整预测结果（all_predictions.csv）
       - 按 satellite_id + system_id 分组统计（summary_by_satellite.csv）
       - 整体指标 JSON（metrics.json）

设计原则：
    - 多星联合：不同卫星系统的 IPP 点在同一样本中联合建模
    - 历元对齐：确保 context 和 target 来自同一时刻的观测
    - 结果分组：支持按卫星分析不同系统的预测性能
    - 可扩展性：支持任意数量的卫星系统和测站

输出文件：
    - results/final_test/all_predictions.csv: 完整预测结果（包含所有元信息）
    - results/final_test/summary_by_satellite.csv: 按卫星分组的统计摘要
    - results/final_test/metrics.json: 整体评估指标（MAE, RMSE）
"""

import sys
import os
import argparse
import json
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.sde import STEC_IRSDE
from models.transformer import build_model
from utils.normalizer import CoordNormalizer, STECNormalizer
from data.dataset import map_system_id_to_index, get_system_ascii_code


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_normalizers(norm_path: str):
    """从 JSON 文件加载归一化参数"""
    with open(norm_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    coord_norm = CoordNormalizer()
    coord_norm.load_state_dict(d["coord"])
    stec_norm = STECNormalizer()
    stec_norm.load_state_dict(d["stec"])
    angle_norm = CoordNormalizer()
    if "angle" in d:
        angle_norm.load_state_dict(d["angle"])
    return coord_norm, stec_norm, angle_norm


def predict_epoch(
    model,
    sde: STEC_IRSDE,
    model_df: pd.DataFrame,
    val_df: pd.DataFrame,
    coord_norm: CoordNormalizer,
    stec_norm: STECNormalizer,
    angle_norm: CoordNormalizer,
    device: torch.device,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    对单个历元进行推理预测。

    context 点：model_stations 的所有 IPP 点（全部）
    target 点：val_stations 的所有 IPP 点（全部）

    Args:
        model:      训练好的模型
        sde:        SDE 实例
        model_df:   该历元的 model_stations 数据（context）
        val_df:     该历元的 val_stations 数据（target）
        coord_norm: 坐标归一化器
        stec_norm:  STEC 归一化器
        angle_norm: 角度归一化器
        device:     设备
        verbose:    是否打印进度

    Returns:
        result_df: 包含预测结果和元信息的 DataFrame
    """
    # 提取 context（model_stations）数据
    ctx_lats     = model_df["ipp_latitude"].values.astype(np.float32)
    ctx_lons     = model_df["ipp_longitude"].values.astype(np.float32)
    ctx_az       = model_df["azimuth_deg"].values.astype(np.float32)
    ctx_el       = model_df["elevation_deg"].values.astype(np.float32)
    ctx_stec     = model_df["stec"].values.astype(np.float32)
    ctx_sys_ids  = model_df["system_id"].values.astype(np.int64)
    ctx_sys_ids  = map_system_id_to_index(ctx_sys_ids)
    ctx_sat_ids  = model_df["satellite_id"].values.astype(np.int64)
    ctx_stations = model_df["station_name"].tolist()

    # 提取 target（val_stations）数据
    tgt_lats     = val_df["ipp_latitude"].values.astype(np.float32)
    tgt_lons     = val_df["ipp_longitude"].values.astype(np.float32)
    tgt_az       = val_df["azimuth_deg"].values.astype(np.float32)
    tgt_el       = val_df["elevation_deg"].values.astype(np.float32)
    tgt_stec     = val_df["stec"].values.astype(np.float32)
    tgt_sys_ids  = val_df["system_id"].values.astype(np.int64)
    tgt_sys_ids  = map_system_id_to_index(tgt_sys_ids)
    tgt_sat_ids  = val_df["satellite_id"].values.astype(np.int64)
    tgt_stations = val_df["station_name"].tolist()

    n_ctx = len(ctx_lats)
    n_tgt = len(tgt_lats)
    n_total = n_ctx + n_tgt

    # 合并 context + target 为完整样本
    all_lats    = np.concatenate([ctx_lats, tgt_lats])
    all_lons    = np.concatenate([ctx_lons, tgt_lons])
    all_az      = np.concatenate([ctx_az, tgt_az])
    all_el      = np.concatenate([ctx_el, tgt_el])
    all_stec    = np.concatenate([ctx_stec, tgt_stec])
    all_sys_ids = np.concatenate([ctx_sys_ids, tgt_sys_ids])

    # 归一化
    coords_norm = coord_norm.transform(all_lats, all_lons)            # [N, 2]
    angles_norm = angle_norm.transform(all_az, all_el)                # [N, 2]
    stec_norm_v = stec_norm.transform(all_stec)[:, np.newaxis]       # [N, 1]

    # 转为 tensor（batch_size=1）
    coords    = torch.from_numpy(coords_norm).unsqueeze(0).to(device)   # [1, N, 2]
    angles    = torch.from_numpy(angles_norm).unsqueeze(0).to(device)   # [1, N, 2]
    sys_ids_t = torch.from_numpy(all_sys_ids).unsqueeze(0).to(device)   # [1, N]
    stec_full = torch.from_numpy(stec_norm_v).unsqueeze(0).to(device)   # [1, N, 1]

    # 构建 mask
    N = n_total
    valid_mask   = torch.ones(1, N, dtype=torch.bool, device=device)
    context_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
    target_mask  = torch.zeros(1, N, dtype=torch.bool, device=device)

    context_mask[0, :n_ctx] = True
    target_mask[0, n_ctx:]  = True

    # 构建 role_type
    role_type = torch.zeros(1, N, dtype=torch.long, device=device)
    role_type[context_mask] = 1
    role_type[target_mask]  = 2

    # context_stec
    context_stec = stec_full * context_mask.unsqueeze(-1).float()

    # 构建条件均值 μ 和先验特征
    mu, prior_features = sde.build_mu_batch(
        coords, stec_full, context_mask, target_mask,
        return_prior_features=True,
    )

    # 初始化推理起始噪声状态
    x_T = stec_full.clone()
    target_noise = mu + sde.max_sigma * torch.randn_like(stec_full)
    x_T[target_mask] = target_noise[target_mask]

    # 完整反向 SDE 去噪
    x0_pred = sde.reverse_sde(
        x_T=x_T,
        mu=mu,
        model=model,
        coords=coords,
        angles=angles,
        system_ids=sys_ids_t,
        context_stec=context_stec,
        role_type=role_type,
        valid_mask=valid_mask,
        target_mask=target_mask,
        device=device,
        prior_features=prior_features,
        verbose=verbose,
    )

    # 提取 target 点的预测结果
    pred_norm_vals = x0_pred[0, n_ctx:, 0].cpu().numpy()     # [n_tgt]
    true_norm_vals = stec_full[0, n_ctx:, 0].cpu().numpy()   # [n_tgt]

    # 反归一化
    pred_orig = stec_norm.inverse_transform(pred_norm_vals)
    true_orig = stec_norm.inverse_transform(true_norm_vals)

    # 构建结果 DataFrame（保留所有元信息，便于后续按 satellite_id 分析）
    result_df = pd.DataFrame({
        "station_name":  tgt_stations,
        "ipp_latitude":  tgt_lats,
        "ipp_longitude": tgt_lons,
        "azimuth_deg":   tgt_az,
        "elevation_deg": tgt_el,
        "system_id":     tgt_sys_ids,
        "satellite_id":  tgt_sat_ids,
        "true_stec":     true_orig,
        "pred_stec":     pred_orig,
        "abs_error":     np.abs(pred_orig - true_orig),
    })

    return result_df


def main():
    parser = argparse.ArgumentParser(description="STEC 条件扩散模型测试脚本（多星联合版）")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "default.yaml"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="指定 checkpoint 路径（不指定则自动查找 best checkpoint）"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="是否打印反向扩散进度"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 加载配置
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    print(f"[Config] 已加载：{args.config}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 系统过滤配置
    system_filter = cfg["data"].get("system_filter", None)
    system_ascii_code = get_system_ascii_code(system_filter) if system_filter else None
    if system_filter:
        print(f"[System] 卫星系统过滤：{system_filter}")
    else:
        print(f"[System] 全系统联合测试")

    # 设备
    device_str = cfg["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    # 1. 扫描历元文件 & 历元对齐
    # ------------------------------------------------------------------
    print("\n[1/5] 扫描并对齐历元文件...")
    model_stations_dir = os.path.join(project_root, cfg["data"]["model_stations_dir"])
    val_stations_dir   = os.path.join(project_root, cfg["data"]["val_stations_dir"])

    if not os.path.exists(model_stations_dir):
        print(f"[Error] model_stations 目录不存在：{model_stations_dir}")
        return
    if not os.path.exists(val_stations_dir):
        print(f"[Error] val_stations 目录不存在：{val_stations_dir}")
        return

    model_files = {f.stem: f for f in sorted(Path(model_stations_dir).glob("*.csv"))}
    val_files   = {f.stem: f for f in sorted(Path(val_stations_dir).glob("*.csv"))}

    # 找共同历元（文件名匹配）
    common_stems = sorted(set(model_files.keys()) & set(val_files.keys()))
    print(f"  model_stations: {len(model_files)} 个历元文件")
    print(f"  val_stations:   {len(val_files)} 个历元文件")
    print(f"  共同历元:       {len(common_stems)} 个")

    if not common_stems:
        print("[Error] 没有共同的历元文件，无法测试")
        return

    # 过滤：context IPP 数 < min_test_context_ipps 或 target IPP 数 < min_test_target_ipps
    min_ctx_ipps = cfg["data"].get("min_test_context_ipps", 20)
    min_tgt_ipps = cfg["data"].get("min_test_target_ipps", 10)

    valid_epochs = []
    for stem in common_stems:
        model_df_tmp = pd.read_csv(model_files[stem])
        val_df_tmp   = pd.read_csv(val_files[stem])
        if system_ascii_code is not None:
            model_df_tmp = model_df_tmp[model_df_tmp["system_id"] == system_ascii_code]
            val_df_tmp   = val_df_tmp[val_df_tmp["system_id"] == system_ascii_code]
        if len(model_df_tmp) >= min_ctx_ipps and len(val_df_tmp) >= min_tgt_ipps:
            valid_epochs.append(stem)

    print(f"  过滤后可用历元：{len(valid_epochs)} 个（context >= {min_ctx_ipps}, target >= {min_tgt_ipps}）")

    if not valid_epochs:
        print("[Error] 没有满足条件的历元，无法测试")
        return

    # ------------------------------------------------------------------
    # 2. 加载模型和归一化参数
    # ------------------------------------------------------------------
    print("\n[2/5] 加载模型和归一化参数...")
    exp_name = cfg["experiment"]["name"]
    if system_filter:
        exp_name = f"{exp_name}_{system_filter}"
    out_dir  = cfg["experiment"]["output_dir"]
    exp_dir  = os.path.join(project_root, out_dir, exp_name)

    norm_path = os.path.join(exp_dir, "normalizer.json")
    if not os.path.exists(norm_path):
        print(f"[Error] 归一化参数文件不存在：{norm_path}")
        return

    coord_norm, stec_norm, angle_norm = load_normalizers(norm_path)
    print(f"  已加载归一化参数：{norm_path}")

    # 确定 checkpoint 路径
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(exp_dir, "checkpoints", "ckpt_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Error] checkpoint 不存在：{ckpt_path}")
        return

    model_cfg = dict(cfg["model"])
    model = build_model(model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"  已加载 checkpoint：{ckpt_path}")
    print(f"  训练轮次：{ckpt.get('epoch', '?')}  最佳验证 MAE：{ckpt.get('best_val_mae', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 3. 构建 SDE
    # ------------------------------------------------------------------
    print("\n[3/5] 构建 SDE...")
    sde_cfg = cfg["sde"]
    mu_reg_cfg = cfg.get("mu_reg", {})
    sde = STEC_IRSDE(
        max_sigma=sde_cfg.get("max_sigma", 50.0),
        T=sde_cfg.get("T", 100),
        schedule=sde_cfg.get("schedule", "cosine"),
        eps=sde_cfg.get("eps", 1e-8),
        idw_power=cfg["inference"].get("idw_power", 2.0),
        idw_k=cfg["inference"].get("idw_k", 5),
        theta=sde_cfg.get("theta", 1.0),
        guidance_scale_max=mu_reg_cfg.get("guidance_scale_max", 2.0),
        guidance_beta=mu_reg_cfg.get("guidance_beta", 1.0),
        guidance_schedule=mu_reg_cfg.get("guidance_schedule", "sin2"),
        weak_context_dropout=mu_reg_cfg.get("weak_context_dropout", 0.3),
        use_reg=mu_reg_cfg.get("use_reg", True),
    )

    # ------------------------------------------------------------------
    # 4. 推理预测
    # ------------------------------------------------------------------
    print(f"\n[4/5] 推理预测（共 {len(valid_epochs)} 个历元）...")
    all_results = []

    for i, stem in enumerate(valid_epochs):
        if args.verbose or (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(valid_epochs)}] 处理历元：{stem}")

        model_df_i = pd.read_csv(model_files[stem])
        val_df_i   = pd.read_csv(val_files[stem])

        # 按系统过滤
        if system_ascii_code is not None:
            model_df_i = model_df_i[model_df_i["system_id"] == system_ascii_code].reset_index(drop=True)
            val_df_i   = val_df_i[val_df_i["system_id"] == system_ascii_code].reset_index(drop=True)

        try:
            result_df = predict_epoch(
                model=model,
                sde=sde,
                model_df=model_df_i,
                val_df=val_df_i,
                coord_norm=coord_norm,
                stec_norm=stec_norm,
                angle_norm=angle_norm,
                device=device,
                verbose=args.verbose,
            )
            result_df["epoch_time"] = stem
            all_results.append(result_df)
        except Exception as e:
            print(f"    [Warning] 历元 {stem} 推理失败：{e}")
            continue

    if not all_results:
        print("[Error] 所有历元推理均失败")
        return

    final_results = pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. 导出结果
    # ------------------------------------------------------------------
    print(f"\n[5/5] 导出结果...")
    result_dir = os.path.join(project_root, cfg["inference"].get("result_output_dir", "results/final_test"))
    if system_filter:
        result_dir = f"{result_dir}_{system_filter}"
    os.makedirs(result_dir, exist_ok=True)

    # 导出完整结果（所有卫星，所有历元）
    if cfg["inference"].get("export_all_predictions", True):
        all_pred_path = os.path.join(result_dir, "all_predictions.csv")
        final_results.to_csv(all_pred_path, index=False)
        print(f"  完整预测结果：{all_pred_path}（{len(final_results)} 条记录）")

    # 整体指标（全局模式：汇总所有 target 点）
    mae_all  = float(final_results["abs_error"].mean())
    rmse_all = float(np.sqrt((final_results["abs_error"] ** 2).mean()))
    print(f"\n  整体指标（全局模式，汇总全部 target 点）：")
    print(f"    总 target 点数：{len(final_results)}")
    print(f"    MAE  = {mae_all:.4f} TECU")
    print(f"    RMSE = {rmse_all:.4f} TECU")

    # 按历元（样本）计算 per-sample RMSE/MAE，并导出 CSV
    print(f"\n  按历元分析指标：")
    epoch_metric_rows = []
    for epoch_stem, grp in final_results.groupby("epoch_time"):
        ep_mae  = float(grp["abs_error"].mean())
        ep_rmse = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts   = len(grp)
        epoch_metric_rows.append({
            "epoch_time": epoch_stem,
            "n_target_points": n_pts,
            "mae_tecu": ep_mae,
            "rmse_tecu": ep_rmse,
        })

    epoch_metric_df = pd.DataFrame(epoch_metric_rows)
    epoch_metric_path = os.path.join(result_dir, "rmse_per_epoch.csv")
    epoch_metric_df.to_csv(epoch_metric_path, index=False)

    avg_sample_mae  = float(epoch_metric_df["mae_tecu"].mean())
    avg_sample_rmse = float(epoch_metric_df["rmse_tecu"].mean())
    print(f"    Per-sample 平均 MAE  = {avg_sample_mae:.4f} TECU")
    print(f"    Per-sample 平均 RMSE = {avg_sample_rmse:.4f} TECU")
    print(f"    Per-sample 指标 CSV：{epoch_metric_path}（{len(epoch_metric_df)} 个历元）")

    # 按 satellite_id + system_id 分组统计
    print(f"\n  按卫星分组指标：")
    summary_rows = []
    for (sys_id, sat_id), grp in final_results.groupby(["system_id", "satellite_id"]):
        mae_i  = float(grp["abs_error"].mean())
        rmse_i = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts  = len(grp)
        print(f"    system_id={sys_id} satellite_id={sat_id:3d}: "
              f"MAE={mae_i:.4f}  RMSE={rmse_i:.4f}  N={n_pts}")
        summary_rows.append({
            "system_id":    sys_id,
            "satellite_id": sat_id,
            "n_points":     n_pts,
            "mae_tecu":     mae_i,
            "rmse_tecu":    rmse_i,
        })

    # 导出分组统计摘要
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(result_dir, "summary_by_satellite.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  分组统计摘要：{summary_path}")

    # 导出整体指标 JSON
    metrics = {
        "n_epochs":            len(valid_epochs),
        "n_target_points":     len(final_results),
        "global_mae_tecu":     mae_all,
        "global_rmse_tecu":    rmse_all,
        "per_sample_mae_tecu": avg_sample_mae,
        "per_sample_rmse_tecu": avg_sample_rmse,
    }
    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  整体指标 JSON：{metrics_path}")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
