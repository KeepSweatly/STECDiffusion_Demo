"""
data/__init__.py
"""
from .dataset import STECEpochDataset, build_train_val_datasets
from .collate import collate_fn, generate_context_target_mask, build_dataloader
from .satellite_grouper import (
    scan_satellite_files,
    load_satellite_data,
    get_all_satellites,
    load_satellite_pair,
)
from .station_splitter import (
    split_stations,
    save_station_split,
    load_station_split,
    get_or_create_station_split,
)

__all__ = [
    "STECEpochDataset",
    "build_train_val_datasets",
    "collate_fn",
    "generate_context_target_mask",
    "build_dataloader",
    "scan_satellite_files",
    "load_satellite_data",
    "get_all_satellites",
    "load_satellite_pair",
    "split_stations",
    "save_station_split",
    "load_station_split",
    "get_or_create_station_split",
]
