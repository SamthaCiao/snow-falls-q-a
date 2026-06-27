#!/usr/bin/env python3
"""生成星图演示数据：模拟用户已探索的人物/事件，并校验「揭示弧段 ↔ 星点」对齐。

用法:
  python scripts/build_starmap_demo.py
  python scripts/build_starmap_demo.py --session-id sess_xxx   # 从 observation.db 读取真实 session

产物:
  static/data/starmap_demo_payload.json
  static/data/starmap_demo_report.md
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.nebula_starmap import path_reveal_index
from observation.schema import SessionMeta, SessionObservationState, UserInterestState
from observation.star_map_catalog import load_static_catalog
from observation.star_map_render import compute_star_map_render, _mobius_guide_path

OUT_JSON = ROOT / "static" / "data" / "starmap_demo_payload.json"
OUT_MD = ROOT / "static" / "data" / "starmap_demo_report.md"

# 演示：用户聊过这些人物（含弧段外人物，用于验证「只显示弧段内星点」）
DEMO_CHARACTERS = ["雪遗诗", "夏空山", "肖潇", "印子", "Steven", "Samantha"]


def _mock_state(characters: list[str]) -> SessionObservationState:
    meta = SessionMeta(session_id="demo_starmap", user_id="demo", created_at="2026-06-26T00:00:00Z")
    ui = UserInterestState(characters=list(characters))
    return SessionObservationState(session_meta=meta, user_interest=ui)


def _load_session_characters(session_id: str) -> list[str]:
    db = ROOT / "runtime_data" / "observation.db"
    if not db.exists():
        raise FileNotFoundError(f"未找到 {db}")
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT state_json FROM observation_states WHERE session_id = ? ORDER BY version DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError(f"session {session_id} 无观测数据")
    data = json.loads(row[0])
    ui = data.get("user_interest") or {}
    return list(ui.get("characters") or [])


def _related_events(catalog: dict, char_label: str) -> list[dict]:
    out = []
    for n in (catalog.get("nodes") or []):
        if n.get("type") not in ("event", "imagery"):
            continue
        parents = n.get("parents") or ([n["parent"]] if n.get("parent") else [])
        if char_label in parents:
            out.append(n)
    return out


def _alignment_check(render, path: list) -> list[str]:
    issues: list[str] = []
    cap = path_reveal_index(render.progress, len(path))
    for n in render.lit_nodes or []:
        pi = int(n.get("path_index", 0))
        if pi > cap:
            issues.append(f"弧段外星点: {n.get('label')} path_index={pi} > cap={cap}")
        if 0 <= pi < len(path):
            px, py = path[pi]
            if abs(n.get("x", 0) - px) > 1e-4 or abs(n.get("y", 0) - py) > 1e-4:
                issues.append(f"坐标未吸附路径: {n.get('label')}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", help="从 observation.db 读取真实 session 人物列表")
    args = parser.parse_args()

    chars = _load_session_characters(args.session_id) if args.session_id else DEMO_CHARACTERS
    catalog = load_static_catalog()
    state = _mock_state(chars)
    render = compute_star_map_render(state)
    path = _mobius_guide_path(catalog.get("mobius") or {})

    payload = {
        "ready": True,
        "progress": render.progress,
        "lit_count": render.lit_count,
        "lit_node_ids": render.lit_node_ids,
        "lit_nodes": render.lit_nodes,
        "constellation_lines": render.constellation_lines,
        "mobius_arc_points": render.mobius_arc_points,
        "mobius_full_points": render.mobius_full_points,
        "background_mode": "nebula_subject",
        "background_url": "/static/images/exploration_nebula_bg.webp",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 星图演示报告",
        "",
        f"- 模拟已讨论人物（{len(chars)}）: {', '.join(chars)}",
        f"- 弧段内可见人物（{render.lit_count}）: "
        + ", ".join(n.get("label", "") for n in (render.lit_nodes or [])),
        f"- 探索进度（星云揭示）: {render.progress * 100:.1f}%",
        f"- 路径 reveal_cap: {path_reveal_index(render.progress, len(path))} / {len(path) - 1}",
        "",
        "## 人物 → 关联事件/意象（hover 展开）",
        "",
    ]
    for n in render.lit_nodes or []:
        label = n.get("label", "")
        rel = _related_events(catalog, label)
        lines.append(f"### {label}")
        if rel:
            for r in rel[:8]:
                lines.append(f"- [{r.get('type')}] {r.get('label')}")
            if len(rel) > 8:
                lines.append(f"- … 另有 {len(rel) - 8} 条")
        else:
            lines.append("- （catalog 无 parent 关联）")
        lines.append("")

    issues = _alignment_check(render, path)
    lines.append("## 对齐校验")
    lines.append("")
    if issues:
        lines.extend(f"- ❌ {i}" for i in issues)
    else:
        lines.append("- ✅ 所有可见星点均在揭示弧段内，且坐标已吸附到 ∞ 路径")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT_JSON.relative_to(ROOT)}")
    print(f"已写入 {OUT_MD.relative_to(ROOT)}")
    print(f"可见人物 {render.lit_count}，进度 {render.progress * 100:.1f}%")
    if issues:
        for i in issues:
            print(f"  警告: {i}")


if __name__ == "__main__":
    main()
