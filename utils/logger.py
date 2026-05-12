"""
utils/logger.py
================
简单日志工具，同时输出到终端和文件。
支持可选的 TensorBoard 写入。
"""

import logging
import os
from datetime import datetime


def get_logger(log_dir: str, name: str = "stec_diffusion") -> logging.Logger:
    """
    创建并返回 Logger。

    Args:
        log_dir: 日志文件保存目录
        name: logger名称

    Returns:
        logging.Logger 实例
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加handler
    if logger.handlers:
        return logger

    # 终端输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
