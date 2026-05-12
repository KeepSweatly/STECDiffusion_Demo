"""
data/satellite_grouper.py
==========================
多卫星数据文件扫描与分组模块。

功能：
1. 扫描 model_stations 和 val_stations 目录
2. 从文件名解析卫星编号（支持 BDS-X-YYYYMMDD_IPP.csv 格式）
3. 按卫星编号分组，返回 {prn: [file_paths]}
4. 支持同一卫星多天数据合并加载

设计原则：
- 可扩展：支持任意数量卫星
- 自适应：自动识别文件名中的卫星编号
- 容错：文件名格式不匹配时给出警告
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_satellite_prn_from_filename(filename: str) -> str:
    """
    从文件名中解析卫星编号。

    支持格式：
    - BDS-6-20240219_IPP.csv  → "BDS-6"
    - GPS-12-20240301_IPP.csv → "GPS-12"

    Args:
        filename: 文件名（不含路径）

    Returns:
        satellite_prn: 卫星编号字符串，如 "BDS-6"
        若解析失败返回 None
    """
    # 正则匹配：系统名-卫星编号-日期_IPP.csv
    # 例如：BDS-6-20240219_IPP.csv
    pattern = r'^([A-Z]+)-(\d+)-\d{8}_IPP\.csv$'
    match = re.match(pattern, filename)

    if match:
        system = match.group(1)  # BDS, GPS, etc.
        prn = match.group(2)     # 6, 12, etc.
        return f"{system}-{prn}"

    return None


def scan_satellite_files(data_dir: str) -> Dict[str, List[str]]:
    """
    扫描目录，按卫星编号分组文件。

    Args:
        data_dir: 数据目录路径（如 "model_stations" 或 "val_stations"）

    Returns:
        satellite_files: {satellite_prn: [file_path1, file_path2, ...]}

    示例：
        {
            "BDS-6": ["model_stations/BDS-6-20240219_IPP.csv",
                      "model_stations/BDS-6-20240220_IPP.csv"],
            "BDS-12": ["model_stations/BDS-12-20240219_IPP.csv"]
        }
    """
    if not os.path.exists(data_dir):
        print(f"[Warning] 数据目录不存在：{data_dir}")
        return {}

    satellite_files = defaultdict(list)

    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        prn = parse_satellite_prn_from_filename(filename)
        if prn is None:
            print(f"[Warning] 无法解析卫星编号，跳过文件：{filename}")
            continue

        file_path = os.path.join(data_dir, filename)
        satellite_files[prn].append(file_path)

    # 对每个卫星的文件列表按文件名排序（确保多天数据按时间顺序）
    for prn in satellite_files:
        satellite_files[prn] = sorted(satellite_files[prn])

    return dict(satellite_files)


def load_satellite_data(file_list: List[str]) -> pd.DataFrame:
    """
    合并同一卫星的多天数据。

    Args:
        file_list: 该卫星的所有数据文件路径列表

    Returns:
        merged_df: 合并后的 DataFrame

    注意：
    - 会检查必要字段是否存在
    - 自动解析 Time 字段为 datetime 类型
    - 按 Time 升序排序
    """
    if not file_list:
        raise ValueError("文件列表为空，无法加载数据")

    df_list = []
    required_cols = {"Time", "Station", "IPP_Lat", "IPP_Lon", "STEC"}

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"[Warning] 文件不存在，跳过：{file_path}")
            continue

        df = pd.read_csv(file_path)

        # 检查必要字段
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"文件 {file_path} 缺少字段：{missing}")

        df_list.append(df)

    if not df_list:
        raise ValueError(f"所有文件都无法加载：{file_list}")

    # 合并所有数据
    merged_df = pd.concat(df_list, ignore_index=True)

    # 解析时间字段
    merged_df["Time"] = pd.to_datetime(merged_df["Time"])

    # 按时间排序
    merged_df = merged_df.sort_values("Time").reset_index(drop=True)

    return merged_df


def get_all_satellites(model_dir: str, val_dir: str) -> List[str]:
    """
    获取所有卫星编号列表（model_stations 和 val_stations 的并集）。

    Args:
        model_dir: model_stations 目录
        val_dir: val_stations 目录

    Returns:
        satellite_list: 卫星编号列表，按字母顺序排序
    """
    model_sats = set(scan_satellite_files(model_dir).keys())
    val_sats = set(scan_satellite_files(val_dir).keys())

    all_sats = model_sats | val_sats
    return sorted(list(all_sats))


def load_satellite_pair(
    satellite_prn: str,
    model_dir: str,
    val_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载某颗卫星的 model_stations 和 val_stations 数据。

    Args:
        satellite_prn: 卫星编号，如 "BDS-6"
        model_dir: model_stations 目录
        val_dir: val_stations 目录

    Returns:
        model_df: model_stations 数据（可能为空 DataFrame）
        val_df: val_stations 数据（可能为空 DataFrame）
    """
    model_files = scan_satellite_files(model_dir).get(satellite_prn, [])
    val_files = scan_satellite_files(val_dir).get(satellite_prn, [])

    # 加载 model_stations 数据
    if model_files:
        model_df = load_satellite_data(model_files)
    else:
        print(f"[Warning] 卫星 {satellite_prn} 在 {model_dir} 中无数据文件")
        model_df = pd.DataFrame()

    # 加载 val_stations 数据
    if val_files:
        val_df = load_satellite_data(val_files)
    else:
        print(f"[Warning] 卫星 {satellite_prn} 在 {val_dir} 中无数据文件")
        val_df = pd.DataFrame()

    return model_df, val_df


if __name__ == "__main__":
    # 测试代码
    print("=== 测试卫星文件扫描模块 ===")

    model_dir = "model_stations"
    val_dir = "val_stations"

    print(f"\n1. 扫描 {model_dir}:")
    model_sats = scan_satellite_files(model_dir)
    for prn, files in model_sats.items():
        print(f"  {prn}: {len(files)} 个文件")
        for f in files:
            print(f"    - {f}")

    print(f"\n2. 扫描 {val_dir}:")
    val_sats = scan_satellite_files(val_dir)
    for prn, files in val_sats.items():
        print(f"  {prn}: {len(files)} 个文件")
        for f in files:
            print(f"    - {f}")

    print("\n3. 获取所有卫星列表:")
    all_sats = get_all_satellites(model_dir, val_dir)
    print(f"  共 {len(all_sats)} 颗卫星：{all_sats}")

    if all_sats:
        test_prn = all_sats[0]
        print(f"\n4. 加载卫星 {test_prn} 的数据:")
        model_df, val_df = load_satellite_pair(test_prn, model_dir, val_dir)
        print(f"  model_stations: {len(model_df)} 行")
        print(f"  val_stations: {len(val_df)} 行")
