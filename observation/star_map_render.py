"""P2+P3：观测 state 更新时预计算星图 render payload（observer 进程，非主进程、非打开 UI 时）。"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from observation.character_index import get_known_entity_names
from observation.schema import SessionObservationState, StarMapRenderPayload
from observation.star_map_catalog import load_static_catalog
from observation.nebula_starmap import path_reveal_index

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _match_lit(label: str, needles: Set[str]) -> bool:
    if not label or not needles:
        return False
    nl = _norm(label)
    for n in needles:
        nn = _norm(n)
        if nn and len(nn) >= 2 and (nn in nl or nl in nn):
            return True
    return False


def _collect_needles(state: SessionObservationState) -> Set[str]:
    """从用户已讨论内容提取点亮关键词（人物/意象/主题），避免 confirmed_facts 全文误匹配。"""
    ui = state.user_interest
    exp = ui.exploration
    needles: Set[str] = set()
    needles.update(c for c in ui.characters if c and len(c.strip()) >= 2)
    needles.update(i for i in ui.imagery_discussed if i and len(i.strip()) >= 2)
    needles.update(t for t in ui.themes if t and 2 <= len(t.strip()) <= 16)
    needles.update(t for t in exp.main_themes if t and 2 <= len(t.strip()) <= 16)
    if exp.topic_focus and len(exp.topic_focus.strip()) >= 2:
        needles.add(exp.topic_focus.strip())
    if exp.analysis_angle and len(exp.analysis_angle.strip()) >= 2:
        needles.add(exp.analysis_angle.strip())
    for q in getattr(ui, "unresolved", None) or []:
        q = (q or "").strip()
        if len(q) >= 2:
            needles.add(q[:24])
    if not needles:
        known = get_known_entity_names()
        blob = " ".join([
            state.visible_summary or "",
            exp.topic_focus or "",
            exp.analysis_angle or "",
            " ".join(exp.main_themes or []),
        ])
        for name in sorted(known, key=len, reverse=True):
            if len(name) >= 2 and name in blob:
                needles.add(name)
                if len(needles) >= 8:
                    break
    return needles


def _bezier_points(
    x0: float, y0: float, x1: float, y1: float, bend: float = 0.15, n: int = 12,
) -> List[List[float]]:
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    pts: List[List[float]] = []
    for i in range(n + 1):
        t = i / n
        omt = 1 - t
        px = omt * omt * x0 + 2 * omt * t * cx + t * t * x1
        py = omt * omt * y0 + 2 * omt * t * cy + t * t * y1
        pts.append([round(px, 4), round(py, 4)])
    return pts


def compute_star_map_render(state: SessionObservationState) -> StarMapRenderPayload:
    catalog = load_static_catalog()
    nodes = catalog.get("nodes") or []
    edges = catalog.get("edges") or []
    needles = _collect_needles(state)

    node_by_id = {n["id"]: n for n in nodes}
    lit_ids: List[str] = []
    lit_detail: List[Dict[str, Any]] = []

    for n in nodes:
        label = n.get("label", "")
        source = n.get("source", "")
        if _match_lit(label, needles) or _match_lit(source, needles):
            lit_ids.append(n["id"])
            lit_detail.append({
                "id": n["id"],
                "label": n.get("label", ""),
                "type": n.get("type", ""),
                "importance": n.get("importance", 0.5),
                "magnitude": n.get("magnitude", 5.0),
                "x": n.get("x"),
                "y": n.get("y"),
                "path_index": n.get("path_index", 0),
                "parent": n.get("parent"),
            })

    char_lit_detail = [d for d in lit_detail if d.get("type") == "character"]
    char_total = max(1, len([n for n in nodes if n.get("type") == "character"]))

    mobius_path = _mobius_guide_path(catalog.get("mobius") or {})
    path_len = len(mobius_path) if mobius_path else 1
    nebula_mode = (catalog.get("background") or {}).get("mode") == "nebula_subject"

    if not char_lit_detail:
        reveal_cap = 0
        progress = 0.0
    else:
        explore_ratio = len(char_lit_detail) / char_total
        reveal_cap = path_reveal_index(explore_ratio, path_len) if nebula_mode else path_len - 1
        progress = min(1.0, (reveal_cap + 1) / max(path_len, 1)) if nebula_mode else len(lit_ids) / len(nodes)

    char_lit_detail = [
        d for d in char_lit_detail
        if int(d.get("path_index", 0)) <= reveal_cap
    ]
    for d in char_lit_detail:
        pi = int(d.get("path_index", 0))
        cat_node = node_by_id.get(d["id"])
        if cat_node and cat_node.get("x") is not None:
            d["x"] = float(cat_node["x"])
            d["y"] = float(cat_node["y"])
        elif 0 <= pi < path_len:
            d["x"] = float(mobius_path[pi][0])
            d["y"] = float(mobius_path[pi][1])
    char_lit_ids = [d["id"] for d in char_lit_detail]

    lit_set = set(lit_ids)
    constellation: List[Dict[str, Any]] = []
    max_edge_dist = 0.28
    for e in edges:
        fid, tid = e.get("from"), e.get("to")
        if fid in lit_set and tid in lit_set:
            a, b = node_by_id.get(fid), node_by_id.get(tid)
            if not a or not b:
                continue
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > max_edge_dist:
                continue
            style = e.get("style", "constellation")
            bend = 0.12 if style == "orbit" else 0.18
            constellation.append({
                "from": fid,
                "to": tid,
                "kind": e.get("kind", "bond"),
                "style": style,
                "points": e.get("points") or _bezier_points(a["x"], a["y"], b["x"], b["y"], bend=bend),
            })

    arc_len = reveal_cap + 1 if nebula_mode else max(1, int(len(mobius_path) * progress))

    return StarMapRenderPayload(
        lit_node_ids=char_lit_ids,
        lit_count=len(char_lit_ids),
        constellation_lines=constellation,
        progress=round(progress, 3),
        mobius_arc_points=mobius_path[:arc_len],
        mobius_full_points=mobius_path,
        lit_nodes=char_lit_detail,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _mobius_guide_path(mobius: Dict[str, Any], samples: int = 64) -> List[List[float]]:
    intrinsic = mobius.get("path")
    if intrinsic and len(intrinsic) >= 4:
        return [[float(p[0]), float(p[1])] for p in intrinsic]
    center = mobius.get("center") or [0.52, 0.48]
    cx, cy = float(center[0]), float(center[1])
    R = 0.32
    pts: List[List[float]] = []
    for i in range(samples):
        u = 2 * math.pi * i / samples
        x3 = R * math.cos(u)
        y3 = R * math.sin(u)
        px = cx + x3 * 0.95
        py = cy - y3 * 0.72
        pts.append([round(px, 4), round(py, 4)])
    return pts


def attach_star_map_render(state: SessionObservationState) -> SessionObservationState:
    try:
        state.star_map_render = compute_star_map_render(state)
        logger.info(
            "[用户观测][星图] 预计算 lit=%d progress=%.0f%%",
            state.star_map_render.lit_count,
            state.star_map_render.progress * 100,
        )
    except Exception as e:
        logger.warning("[用户观测][星图] 预计算失败: %s", e)
    return state
