"""
data/station_splitter.py
=========================
站点级划分模块。

功能：
1. 收集某颗卫星所有历元涉及的全部站点
2. 按 80/20 划分为 train_stations / val_stations
3. 支持随机种子、可复现
4. 保存划分结果到文件（JSON格式）
5. 支持加载已有划分结果

设计原则：
- 可复现：固定随机种子确保多次运行结果一致
- 可持久化：划分结果保存到文件，避免重复计算
- 可扩展：支持任意卫星、任意站点数量
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Set


def split_stations(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    将数据中的所有站点按比例划分为训练集和验证集。

    Args:
        df: 包含 "Station" 列的 DataFrame
        train_ratio: 训练集比例（默认 0.8）
        seed: 随机种子

    Returns:
        train_stations: 训练站点集合
        val_stations: 验证站点集合
    """
    # 收集所有唯一站点
    all_stations = sorted(df["Station"].unique())
    n_total = len(all_stations)

    if n_total == 0:
        raise ValueError("数据中没有站点信息")

    # 计算训练集站点数量
    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_train, n_total - 1))  # 至少1个训练站点，至少1个验证站点

    # 随机打乱并划分
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_stations)

    train_stations = set(shuffled[:n_train])
    val_stations = set(shuffled[n_train:])

    return train_stations, val_stations


def save_station_split(
    train_stations: Set[str],
    val_stations: Set[str],
    save_path: str,
):
    """
    保存站点划分结果到 JSON 文件。

    Args:
        train_stations: 训练站点集合
        val_stations: 验证站点集合
        save_path: 保存路径（JSON 文件）
    """
    split_dict = {
        "train_stations": sorted(list(train_stations)),
        "val_stations": sorted(list(val_stations)),
        "n_train": len(train_stations),
        "n_val": len(val_stations),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)

    print(f"[StationSplit] 站点划分已保存：{save_path}")
    print(f"  训练站点：{len(train_stations)} 个")
    print(f"  验证站点：{len(val_stations)} 个")


def load_station_split(load_path: str) -> Tuple[Set[str], Set[str]]:
    """
    从 JSON 文件加载站点划分结果。

    Args:
        load_path: JSON 文件路径

    Returns:
        train_stations: 训练站点集合
        val_stations: 验证站点集合
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"站点划分文件不存在：{load_path}")

    with open(load_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    train_stations = set(split_dict["train_stations"])
    val_stations = set(split_dict["val_stations"])

    print(f"[StationSplit] 站点划分已加载：{load_path}")
    print(f"  训练站点：{len(train_stations)} 个")
    print(f"  验证站点：{len(val_stations)} 个")

    return train_stations, val_stations


def get_or_create_station_split(
    df: pd.DataFrame,
    split_file_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    force_recreate: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """
    获取或创建站点划分。

    若划分文件已存在，则加载；否则创建新划分并保存。

    Args:
        df: 包含 "Station" 列的 DataFrame
        split_file_path: 划分文件保存路径
        train_ratio: 训练集比例
        seed: 随机种子
        force_recreate: 是否强制重新创建（忽略已有文件）

    Returns:
        train_stations: 训练站点集合
        val_stations: 验证站点集合
    """
    if os.path.exists(split_file_path) and not force_recreate:
        print(f"[StationSplit] 发现已有划分文件，直接加载")
        return load_station_split(split_file_path)
    else:
        print(f"[StationSplit] 创建新的站点划分")
        train_stations, val_stations = split_stations(df, train_ratio, seed)
        save_station_split(train_stations, val_stations, split_file_path)
        return train_stations, val_stations


if __name__ == "__main__":
    # 测试代码
    print("=== 测试站点划分模块 ===")

    # 创建测试数据
    test_data = {
        "Time": ["2024-02-18 00:00:00"] * 10,
        "Station": ["sta1", "sta2", "sta3", "sta4", "sta5",
                    "sta6", "sta7", "sta8", "sta9", "sta10"],
        "IPP_Lat": [60.0] * 10,
        "IPP_Lon": [10.0] * 10,
        "STEC": [20.0] * 10,
    }
    df = pd.DataFrame(test_data)

    print(f"\n测试数据：{len(df['Station'].unique())} 个站点")

    # 测试划分
    train_set, val_set = split_stations(df, train_ratio=0.8, seed=42)
    print(f"\n划分结果：")
    print(f"  训练站点：{sorted(train_set)}")
    print(f"  验证站点：{sorted(val_set)}")

    # 测试保存和加载
    test_save_path = "test_station_split.json"
    save_station_split(train_set, val_set, test_save_path)

    train_loaded, val_loaded = load_station_split(test_save_path)
    print(f"\n加载验证：")
    print(f"  训练站点一致：{train_set == train_loaded}")
    print(f"  验证站点一致：{val_set == val_loaded}")

    # 清理测试文件
    if os.path.exists(test_save_path):
        os.remove(test_save_path)
        print(f"\n测试文件已清理：{test_save_path}")
