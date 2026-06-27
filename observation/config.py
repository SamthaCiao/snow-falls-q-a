"""观测模块配置加载。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "observer_config.yaml"


@dataclass
class ObserverConfig:
    """用户观测模块运行参数。"""

    trigger_every_n_turns: int = 10
    trigger_char_threshold: int = 8000
    poll_interval_seconds: int = 15
    max_turns_per_extract: int = 20
    max_llm_calls_per_run: int = 5
    extractor_model: str = ""
    use_llm_extractor: bool = True
    enable_rule_fallback: bool = True
    visible_summary_max_chars: int = 600
    dev_mode: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObserverConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ObserverConfig":
        cfg_path = path or Path(os.environ.get("OBSERVER_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
        data: Dict[str, Any] = {}
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                data = raw.get("observer", raw)
        cfg = cls.from_dict(data)
        if os.environ.get("OBSERVER_DEV_MODE", "").lower() in ("1", "true", "yes"):
            cfg.dev_mode = True
        if os.environ.get("OBSERVER_TRIGGER_N"):
            cfg.trigger_every_n_turns = int(os.environ["OBSERVER_TRIGGER_N"])
        if os.environ.get("OBSERVER_POLL_SECONDS"):
            cfg.poll_interval_seconds = int(os.environ["OBSERVER_POLL_SECONDS"])
        return cfg


def get_runtime_data_dir() -> Path:
    """运行时数据目录（gitignore，部署时挂载持久卷）。"""
    root = Path(os.environ.get("RUNTIME_DATA_DIR", str(PROJECT_ROOT / "runtime_data")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_db_path() -> Path:
    return get_runtime_data_dir() / "observation.db"
