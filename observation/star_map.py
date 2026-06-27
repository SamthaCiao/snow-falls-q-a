"""探索星图 API 数据：静态 catalog + 预计算 render。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from observation.schema import SessionObservationState
from observation.star_map_catalog import load_static_catalog


def build_starmap_response(state: Optional[SessionObservationState]) -> Dict[str, Any]:
    """返回前端 WebGL 所需数据；render 每次按最新星表 + 观测 state 重算（避免缓存旧坐标）。"""
    catalog = load_static_catalog()
    bg = catalog.get("background") or {}
    resp: Dict[str, Any] = {
        "catalog_version": catalog.get("version", "1.0"),
        "background_url": bg.get("url", "/static/images/exploration_nebula_bg.webp"),
        "background_mode": bg.get("mode", "nebula_subject"),
        "background_size": [bg.get("width", 2048), bg.get("height", 1152)],
        "nodes": catalog.get("nodes") or [],
        "edges": catalog.get("edges") or [],
        "mobius": catalog.get("mobius") or {},
        "lit_node_ids": [],
        "lit_count": 0,
        "constellation_lines": [],
        "progress": 0.0,
        "mobius_arc_points": [],
        "mobius_full_points": [],
        "lit_nodes": [],
        "ready": False,
    }
    if not state:
        return resp

    from observation.star_map_render import compute_star_map_render

    r = compute_star_map_render(state)
    resp.update({
        "ready": True,
        "lit_node_ids": r.lit_node_ids,
        "lit_count": r.lit_count,
        "constellation_lines": r.constellation_lines,
        "progress": r.progress,
        "mobius_arc_points": r.mobius_arc_points,
        "mobius_full_points": r.mobius_full_points,
        "lit_nodes": r.lit_nodes,
        "computed_at": r.computed_at,
    })
    return resp
