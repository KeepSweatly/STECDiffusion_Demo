"""
测试验证集 target_indices 是否正确生成
"""
import sys
import yaml
from pathlib import Path

# 加载配置
with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 构建数据集
from data.dataset import build_train_val_datasets

print("构建数据集...")
train_ds, val_ds, coord_norm, stec_norm, angle_norm = build_train_val_datasets(
    model_stations_dir="model_stations",
    cfg=cfg["data"],
)

print(f"\n训练集大小: {len(train_ds)}")
print(f"验证集大小: {len(val_ds)}")

# 检查前几个验证样本
print("\n检查前5个验证样本的 target_indices:")
for i in range(min(5, len(val_ds))):
    sample = val_ds[i]
    n_points = sample["n_points"]
    context_indices = sample.get("context_indices", None)
    target_indices = sample.get("target_indices", None)

    print(f"\n样本 {i}:")
    print(f"  n_points: {n_points}")
    if context_indices is not None:
        print(f"  context_indices: {len(context_indices)} 个, 范围 [{context_indices.min()}, {context_indices.max()}]")
    if target_indices is not None:
        print(f"  target_indices: {len(target_indices)} 个, 范围 [{target_indices.min()}, {target_indices.max()}]")
        if len(target_indices) == 0:
            print(f"  ⚠️ 警告: target_indices 为空!")
        elif target_indices.max() >= n_points:
            print(f"  ⚠️ 警告: target_indices 超出范围 (max={target_indices.max()} >= n_points={n_points})")
        else:
            print(f"  ✓ target_indices 正常")

print("\n测试完成!")
