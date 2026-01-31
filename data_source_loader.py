"""
数据源加载：若本地无 数据源/ 目录，则从 数据源.pack.zip 解压后使用。
打包后为 zip 二进制，GitHub 上不直接显示明文；clone 仓库后运行即可使用。
"""
import os
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录（本文件所在目录的上一级为项目根）
PROJECT_ROOT = Path(__file__).resolve().parent
PACK_FILE = PROJECT_ROOT / "数据源.pack.zip"
DATA_SOURCE_DIR_NAME = "数据源"


def ensure_data_source_available(root_dir=None):
    """
    确保数据源目录可用：若已有 数据源/ 则直接返回路径；否则从 数据源.pack.zip 解压到项目根下 数据源/。
    Returns:
        数据源目录的绝对路径（str）
    """
    root = Path(root_dir).resolve() if root_dir else PROJECT_ROOT
    data_dir = root / DATA_SOURCE_DIR_NAME
    pack_path = root / "数据源.pack.zip"

    if data_dir.is_dir():
        try:
            next(data_dir.iterdir())
        except StopIteration:
            pass
        else:
            return str(data_dir)

    if pack_path.is_file():
        logger.info("未找到 数据源 目录，正在从 数据源.pack.zip 解压...")
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(pack_path, "r") as zf:
            zf.extractall(root)
        logger.info("数据源解压完成: %s", data_dir)
        return str(data_dir)

    return str(data_dir)


def get_data_source_dir(root_dir=None):
    """
    返回当前应使用的数据源目录路径（先确保已解压）。
    供 web_chat、api、rebuild_index 等入口在启动时调用并设置 DATA_SOURCE_DIR。
    """
    return ensure_data_source_available(root_dir=root_dir)
