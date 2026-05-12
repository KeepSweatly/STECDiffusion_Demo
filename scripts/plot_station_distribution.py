"""
scripts/plot_station_distribution.py
=====================================
绘制站点空间分布图，区分 model_stations 和 val_stations。

功能：
1. 读取 model_stations 和 val_stations 数据
2. 提取每个站点第一次采样时的坐标
3. 绘制站点空间分布图，用不同颜色和标记区分
4. 支持指定卫星编号

使用方式：
    python scripts/plot_station_distribution.py --satellite BDS-6
    python scripts/plot_station_distribution.py --satellite BDS-6 --output figures/station_dist.png
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入模块，避免触发 torch 导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "satellite_grouper",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "satellite_grouper.py")
)
satellite_grouper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(satellite_grouper)
load_satellite_pair = satellite_grouper.load_satellite_pair

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib 未安装，将使用文本输出模式")


def get_first_occurrence_coordinates(df: pd.DataFrame):
    """
    获取每个站点第一次出现时的坐标。

    Args:
        df: 包含 Time, Station, IPP_Lat, IPP_Lon 列的 DataFrame

    Returns:
        DataFrame with columns: Station, IPP_Lat, IPP_Lon
    """
    # 按时间排序，确保获取最早的记录
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    # 对每个站点，取第一次出现的坐标
    station_coords = df_sorted.groupby("Station", as_index=False).first()[["Station", "IPP_Lat", "IPP_Lon"]]

    return station_coords


def plot_station_distribution(
    model_coords: pd.DataFrame,
    val_coords: pd.DataFrame,
    satellite_prn: str,
    output_path: str = None,
):
    """
    绘制站点空间分布图

    Args:
        model_coords: model_stations 站点坐标 (Station, IPP_Lat, IPP_Lon)
        val_coords: val_stations 站点坐标 (Station, IPP_Lat, IPP_Lon)
        satellite_prn: 卫星编号
        output_path: 输出图片路径
    """
    if not HAS_MATPLOTLIB:
        print("\n[Text Mode] 站点分布统计：")
        print(f"\n卫星：{satellite_prn}")
        print(f"model_stations 站点数：{len(model_coords)}")
        print(f"val_stations 站点数：{len(val_coords)}")
        print(f"\n纬度范围：{model_coords['IPP_Lat'].min():.2f}° ~ {model_coords['IPP_Lat'].max():.2f}°")
        print(f"经度范围：{model_coords['IPP_Lon'].min():.2f}° ~ {model_coords['IPP_Lon'].max():.2f}°")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制 model_stations 站点（蓝色圆点）
    ax.scatter(
        model_coords["IPP_Lon"],
        model_coords["IPP_Lat"],
        c="blue",
        marker="o",
        s=120,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
        label=f"Model Stations (n={len(model_coords)})",
        zorder=3,
    )

    # 绘制 val_stations 站点（红色三角）
    ax.scatter(
        val_coords["IPP_Lon"],
        val_coords["IPP_Lat"],
        c="red",
        marker="^",
        s=150,
        alpha=0.8,
        edgecolors="black",
        linewidths=2,
        label=f"Val Stations (n={len(val_coords)})",
        zorder=4,
    )

    # 添加 model_stations 站点标签
    for _, row in model_coords.iterrows():
        ax.annotate(
            row["Station"],
            (row["IPP_Lon"], row["IPP_Lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            color="darkblue",
        )

    # 添加 val_stations 站点标签
    for _, row in val_coords.iterrows():
        ax.annotate(
            row["Station"],
            (row["IPP_Lon"], row["IPP_Lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            alpha=0.9,
            color="darkred",
        )

    # 设置坐标轴
    ax.set_xlabel("Longitude (°)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude (°)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Station Distribution for Satellite {satellite_prn}\n"
        f"Model: {len(model_coords)} | Val: {len(val_coords)}",
        fontsize=14,
        fontweight="bold",
    )

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle="--")

    # 添加图例
    ax.legend(loc="best", fontsize=11, framealpha=0.9)

    # 设置坐标轴范围（留出边距）
    all_lats = pd.concat([model_coords["IPP_Lat"], val_coords["IPP_Lat"]])
    all_lons = pd.concat([model_coords["IPP_Lon"], val_coords["IPP_Lon"]])

    lat_margin = (all_lats.max() - all_lats.min()) * 0.1
    lon_margin = (all_lons.max() - all_lons.min()) * 0.1

    ax.set_xlim(all_lons.min() - lon_margin, all_lons.max() + lon_margin)
    ax.set_ylim(all_lats.min() - lat_margin, all_lats.max() + lat_margin)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n[Success] 图片已保存：{output_path}")
    else:
        # 默认保存路径
        default_output = f"figures/station_distribution_{satellite_prn}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(default_output, dpi=300, bbox_inches="tight")
        print(f"\n[Success] 图片已保存：{default_output}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="绘制站点空间分布图")
    parser.add_argument(
        "--satellite",
        type=str,
        required=True,
        help="卫星编号，如 BDS-6",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（默认：figures/station_distribution_{satellite}.png）",
    )
    args = parser.parse_args()

    satellite_prn = args.satellite

    print(f"{'='*80}")
    print(f"绘制卫星 {satellite_prn} 的站点分布图")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("[1/3] 加载数据...")
    model_dir = "model_stations"
    val_dir = "val_stations"

    model_df, val_df = load_satellite_pair(satellite_prn, model_dir, val_dir)

    if model_df.empty:
        print(f"[Error] 卫星 {satellite_prn} 在 {model_dir} 中无数据")
        return

    if val_df.empty:
        print(f"[Error] 卫星 {satellite_prn} 在 {val_dir} 中无数据")
        return

    print(f"  model_stations: {len(model_df)} 行数据")
    print(f"  val_stations: {len(val_df)} 行数据")

    # ------------------------------------------------------------------
    # 2. 获取站点第一次出现时的坐标
    # ------------------------------------------------------------------
    print("\n[2/3] 提取站点第一次采样坐标...")
    model_coords = get_first_occurrence_coordinates(model_df)
    val_coords = get_first_occurrence_coordinates(val_df)

    print(f"  model_stations 站点数: {len(model_coords)}")
    print(f"  val_stations 站点数: {len(val_coords)}")

    # ------------------------------------------------------------------
    # 3. 绘制分布图
    # ------------------------------------------------------------------
    print("\n[3/3] 绘制分布图...")
    plot_station_distribution(
        model_coords=model_coords,
        val_coords=val_coords,
        satellite_prn=satellite_prn,
        output_path=args.output,
    )

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
