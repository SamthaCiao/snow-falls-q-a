"""叙事星表：人物/事件/意象节点布局。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_MAX_CHAR = 20
_MAX_EVENT = 28
_MAX_IMAGERY = 18
_GOLDEN = 2.399963229728653
_CATALOG_PATH = Path(__file__).resolve().parents[1] / "static" / "data" / "exploration_star_catalog.json"


def _data_dir() -> Path:
    from data_source_loader import get_data_source_dir
    return Path(os.environ.get("DATA_SOURCE_DIR") or get_data_source_dir())


def _load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from json import JSONDecoder
        obj, _ = JSONDecoder().raw_decode(raw)
        return obj


def _character_weight(name: str, profile: Dict[str, Any]) -> float:
    w = 1.0
    role = str(profile.get("role") or "")
    if "主角" in role or "叙述" in role:
        w += 14
    elif "重要" in role or "核心" in role:
        w += 8
    else:
        w += 3
    for ev in profile.get("key_events") or []:
        if isinstance(ev, dict):
            w += float(ev.get("importance") or 2)
    w += len(profile.get("relationships") or []) * 0.4
    return w


def _parse_event_character(raw: str, char_names: Set[str]) -> Optional[str]:
    c = str(raw).split("（")[0].split("(")[0].strip()
    if c in char_names:
        return c
    for name in char_names:
        if name in c or c in name:
            return name
    return None


def _orbit_position(cx: float, cy: float, seed: str, base_r: float = 0.038) -> Tuple[float, float]:
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    angle = int(h[0:4], 16) / 65535.0 * 2 * math.pi
    r = base_r + (int(h[4:6], 16) / 65535.0) * 0.025
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def build_narrative_catalog() -> Dict[str, Any]:
    """从数据源构建叙事节点星表（uv 0-1）。"""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    char_names: List[str] = []
    char_pos: Dict[str, Tuple[float, float]] = {}
    char_symbols: Dict[str, List[str]] = {}
    rel_edges: List[Tuple[str, str]] = []

    char_file = _data_dir() / "人物.json"
    profiles: Dict[str, Any] = {}
    if char_file.is_file():
        data = _load_json_file(char_file) or {}
        profiles = data.get("人物画像系统", {}).get("数据", {}) or {}

    weighted = [(n, _character_weight(n, p)) for n, p in profiles.items() if n]
    weighted.sort(key=lambda x: x[1], reverse=True)
    max_w = weighted[0][1] if weighted else 1.0

    for i, (name, weight) in enumerate(weighted[:_MAX_CHAR]):
        char_names.append(name)
        importance = weight / max_w if max_w else 0.5
        r = 0.14 + (1.0 - importance) * 0.28
        angle = i * _GOLDEN
        x = 0.48 + r * math.cos(angle) * 0.82
        y = 0.46 + r * math.sin(angle) * 0.55
        char_pos[name] = (x, y)
        prof = profiles.get(name, {})
        char_symbols[name] = [str(s) for s in (prof.get("symbols") or []) if s]
        for rel in prof.get("relationships") or []:
            if isinstance(rel, dict):
                to = rel.get("to")
                if to and to in profiles:
                    rel_edges.append((name, str(to).split("（")[0].split("(")[0].strip()))

        mag = 2.0 + (1.0 - importance) * 4.5
        nodes.append({
            "id": f"char:{name}",
            "label": name,
            "type": "character",
            "importance": round(importance, 3),
            "magnitude": round(mag, 2),
            "x": round(x, 4),
            "y": round(y, 4),
        })

    char_set = set(char_names)
    for a, b in rel_edges:
        if a in char_set and b in char_set:
            edges.append({
                "from": f"char:{a}",
                "to": f"char:{b}",
                "kind": "bond",
                "style": "constellation",
            })

    event_file = _data_dir() / "关键事件链条.json"
    if event_file.is_file():
        edata = _load_json_file(event_file) or {}
        for ev in (edata.get("events") or [])[:_MAX_EVENT]:
            eid = ev.get("event_id") or ev.get("title", "")
            title = (ev.get("title") or eid)[:24]
            nid = f"event:{eid}"
            parents: List[str] = []
            for raw in ev.get("characters") or []:
                p = _parse_event_character(str(raw), char_set)
                if p and p not in parents:
                    parents.append(p)
            parent = parents[0] if parents else None
            if parent and parent in char_pos:
                cx, cy = char_pos[parent]
                x, y = _orbit_position(cx, cy, nid, base_r=0.032)
                if parent in char_set:
                    edges.append({"from": f"char:{parent}", "to": nid, "kind": "satellite", "style": "orbit"})
            else:
                h = hashlib.md5(nid.encode()).hexdigest()
                x = 0.18 + (int(h[0:4], 16) / 65535.0) * 0.64
                y = 0.22 + (int(h[4:8], 16) / 65535.0) * 0.56
            nodes.append({
                "id": nid,
                "label": title,
                "type": "event",
                "source": ev.get("source", ""),
                "parent": parent,
                "parents": parents,
                "importance": 0.55,
                "magnitude": 5.5,
                "x": round(x, 4),
                "y": round(y, 4),
            })

    imagery_file = _data_dir() / "意象系统.txt"
    if imagery_file.is_file():
        idata = _load_json_file(imagery_file)
        if isinstance(idata, list):
            for item in idata[:_MAX_IMAGERY]:
                name = (item.get("元素") or item.get("name") or "").strip()
                if not name:
                    continue
                anchor = None
                for cname, syms in char_symbols.items():
                    if any(name in s or s in name for s in syms):
                        anchor = cname
                        break
                if anchor and anchor in char_pos:
                    cx, cy = char_pos[anchor]
                    x, y = _orbit_position(cx, cy, f"imagery:{name}", base_r=0.048)
                else:
                    x, y = _orbit_position(0.48, 0.46, f"imagery:{name}", base_r=0.22)
                nodes.append({
                    "id": f"imagery:{name}",
                    "label": name,
                    "type": "imagery",
                    "parent": anchor,
                    "importance": 0.35,
                    "magnitude": 6.2,
                    "x": round(x, 4),
                    "y": round(y, 4),
                })

    return {
        "version": "1.0",
        "nodes": nodes,
        "edges": edges,
        "mobius": {"center": [0.52, 0.48], "strength": "weak"},
    }


@lru_cache(maxsize=2)
def _load_static_catalog_cached(mtime_ns: int) -> Dict[str, Any]:
    if _CATALOG_PATH.is_file():
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_narrative_catalog()


def load_static_catalog(*, force_reload: bool = False) -> Dict[str, Any]:
    if force_reload:
        _load_static_catalog_cached.cache_clear()
    mtime_ns = int(_CATALOG_PATH.stat().st_mtime_ns) if _CATALOG_PATH.is_file() else 0
    return _load_static_catalog_cached(mtime_ns)
