"""从数据源加载人物/别名，供观测结果校验。"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Set

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_known_entity_names() -> FrozenSet[str]:
    """加载人物主名与别名；失败时返回空集合。"""
    names: Set[str] = set()
    try:
        from data_source_loader import get_data_source_dir

        data_dir = Path(os.environ.get("DATA_SOURCE_DIR") or get_data_source_dir())
        char_file = data_dir / "人物.json"
        if not char_file.is_file():
            return frozenset()
        raw = char_file.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data, _ = json.JSONDecoder().raw_decode(raw)
        profiles = data.get("人物画像系统", {}).get("数据", {})
        for main_name, profile in profiles.items():
            if main_name:
                names.add(main_name.strip())
            if isinstance(profile, dict):
                for alias in profile.get("aliases", []) or []:
                    if isinstance(alias, str) and alias.strip():
                        names.add(alias.strip())
        # 书名
        names.update({"雪落成诗", "影化成殇"})
    except Exception as e:
        logger.debug("加载人物索引失败: %s", e)
    return frozenset(names)
