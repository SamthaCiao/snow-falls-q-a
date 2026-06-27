#!/usr/bin/env python3
"""从事件链补全 parents 字段，并重算事件/意象沿 ∞ 轨迹的 path_index。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.nebula_starmap import CATALOG_PATH, resnap_catalog_paths
from observation.star_map_catalog import _parse_event_character, load_static_catalog

EVENT_FILE = ROOT / "数据源" / "关键事件链条.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def enrich_parents(catalog: dict) -> dict:
    if not EVENT_FILE.is_file():
        return catalog
    edata = json.loads(EVENT_FILE.read_text(encoding="utf-8"))
    char_set = {n["label"] for n in catalog.get("nodes") or [] if n.get("type") == "character"}
    by_id: dict[str, list[str]] = {}
    for ev in edata.get("events") or []:
        eid = ev.get("event_id") or ev.get("title", "")
        parents: list[str] = []
        for raw in ev.get("characters") or []:
            p = _parse_event_character(str(raw), char_set)
            if p and p not in parents:
                parents.append(p)
        if parents:
            by_id[f"event:{eid}"] = parents

    nodes = []
    for node in catalog.get("nodes") or []:
        n = dict(node)
        if n.get("type") == "event":
            ps = by_id.get(n.get("id", ""))
            if ps:
                n["parents"] = ps
                n["parent"] = ps[0]
        elif n.get("parent") and not n.get("parents"):
            n["parents"] = [n["parent"]]
        nodes.append(n)
    return {**catalog, "nodes": nodes}


def main() -> int:
    if not CATALOG_PATH.is_file():
        print(f"星表不存在: {CATALOG_PATH}")
        return 1
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = enrich_parents(catalog)
    catalog = resnap_catalog_paths(catalog)
    load_static_catalog(force_reload=True)
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    events = [n for n in catalog["nodes"] if n.get("type") == "event"]
    multi = sum(1 for n in events if len(n.get("parents") or []) > 1)
    idx_span = {n["label"]: n.get("path_index") for n in events[:5]}
    print(f"已更新 {CATALOG_PATH.name}: events={len(events)}, 多人物事件={multi}")
    print(f"  样例 path_index: {idx_span}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
