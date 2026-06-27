#!/usr/bin/env python3
"""星点分布图 + 随机探索路径动画（离线验收用）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.nebula_starmap import (  # noqa: E402
    CATALOG_PATH,
    TYPE_COLOR,
    nodes_in_revealed_arc,
    render_nebula_progress,
)

OUT_DIR = ROOT / "static" / "images" / "nebula_previews"
BG_PREVIEW = ROOT / "static" / "images" / "exploration_nebula_bg_preview.jpg"

TYPE_LABEL = {
    "character": "人物",
    "event": "事件",
    "imagery": "意象",
}


def _load_font(size: int):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _random_progress_curve(
    path_len: int,
    rng: np.random.Generator,
    n_stops: int = 10,
    frames: int = 120,
) -> List[float]:
    """随机探索：在 ∞ 路径上随机停驻点，帧间 smoothstep 插值。"""
    inner = sorted(rng.choice(range(1, path_len - 1), size=n_stops - 2, replace=False))
    stops = [0] + list(inner) + [path_len - 1]
    key_progress = [s / max(path_len - 1, 1) for s in stops]
    dwell = rng.integers(8, 18, size=len(key_progress))
    timeline: List[float] = []
    for i in range(len(key_progress) - 1):
        p0, p1 = key_progress[i], key_progress[i + 1]
        seg_frames = max(6, int(dwell[i] * (0.6 + 0.8 * abs(p1 - p0))))
        for f in range(seg_frames):
            t = _smoothstep(f / max(seg_frames - 1, 1))
            timeline.append(p0 + (p1 - p0) * t)
    while len(timeline) < frames:
        timeline.append(1.0)
    if len(timeline) > frames:
        idx = np.linspace(0, len(timeline) - 1, frames).astype(int)
        timeline = [timeline[i] for i in idx]
    timeline[-1] = 1.0
    return timeline


def render_star_points_map(
    rgb: np.ndarray,
    catalog: Dict[str, Any],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    path = catalog["mobius"]["path"]
    nodes = catalog["nodes"]

    base = render_nebula_progress(rgb, path, nodes, 1.0)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil, "RGBA")
    font_sm = _load_font(14)
    font_md = _load_font(16)
    font_lg = _load_font(18)

    path_xy = [(p[0] * w, (1.0 - p[1]) * h) for p in path]
    for i in range(len(path_xy) - 1):
        draw.line([path_xy[i], path_xy[i + 1]], fill=(80, 200, 255, 90), width=2)
    draw.line([path_xy[-1], path_xy[0]], fill=(80, 200, 255, 90), width=2)

    for n in sorted(nodes, key=lambda x: x.get("path_index", 0)):
        px = int(n["x"] * w)
        py = int((1.0 - n["y"]) * h)
        tp = n.get("type", "character")
        tc = TYPE_COLOR.get(tp, TYPE_COLOR["character"])
        rgba = (int(tc[0] * 255), int(tc[1] * 255), int(tc[2] * 255), 230)
        r = 7 if tp == "character" else 5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=rgba, outline=(255, 255, 255, 180), width=1)
        idx = n.get("path_index", 0)
        if tp == "character":
            label = n.get("label", "")
            draw.text((px + 9, py - 10), f"{idx}:{label}", fill=(255, 252, 240, 240), font=font_sm)
        else:
            draw.text((px + 7, py - 6), str(idx), fill=(200, 220, 255, 200), font=font_sm)

    legend_x, legend_y = 24, h - 118
    draw.rectangle([legend_x - 8, legend_y - 8, legend_x + 200, h - 16], fill=(8, 10, 20, 200))
    draw.text((legend_x, legend_y), "星点分布（沿 ∞ 路径 path_index）", fill=(235, 238, 248), font=font_md)
    y = legend_y + 28
    for tp, name in TYPE_LABEL.items():
        tc = TYPE_COLOR[tp]
        c = (int(tc[0] * 255), int(tc[1] * 255), int(tc[2] * 255))
        draw.ellipse([legend_x, y, legend_x + 12, y + 12], fill=c + (230,))
        draw.text((legend_x + 18, y - 2), name, fill=(220, 225, 235), font=font_sm)
        y += 22
    draw.text((legend_x, y + 4), f"共 {len(nodes)} 节点 · 路径 {len(path)} 采样点", fill=(160, 170, 190), font=font_sm)

    title_y = 20
    draw.rectangle([16, title_y - 4, 320, title_y + 28], fill=(8, 10, 20, 210))
    draw.text((24, title_y), "叙事星点落位图（100% 底图）", fill=(235, 238, 248), font=font_lg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path, "JPEG", quality=93)
    print(f"  → 星点图 {out_path}")


def _progress_curve_from_render(
    render: Dict[str, Any],
    path_len: int,
    frames: int = 96,
) -> List[float]:
    """按已点亮节点的 path_index 生成探索轨迹（0 → 当前 progress）。"""
    target = float(render.get("progress") or 0)
    if target <= 0:
        return [0.0] * max(24, frames // 4)

    lit = render.get("lit_nodes") or []
    stops = {0.0, target}
    for n in lit:
        idx = int(n.get("path_index", 0))
        stops.add(min(1.0, (idx + 1) / max(path_len, 1)))
    keys = sorted(stops)
    if len(keys) < 2:
        keys = [0.0, target]

    per_seg = max(8, frames // max(len(keys) - 1, 1))
    curve: List[float] = []
    for i in range(len(keys) - 1):
        p0, p1 = keys[i], keys[i + 1]
        seg_n = per_seg if i < len(keys) - 2 else max(8, frames - len(curve))
        for f in range(seg_n):
            t = _smoothstep(f / max(seg_n - 1, 1))
            curve.append(p0 + (p1 - p0) * t)
    while len(curve) < frames:
        curve.append(target)
    curve[-1] = target
    return curve[:frames]


def render_exploration_gif(
    rgb: np.ndarray,
    catalog: Dict[str, Any],
    out_path: Path,
    progress_curve: Sequence[float],
    width: int = 1280,
    fps: int = 10,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if rgb.shape[1] != width:
        h_tgt = int(rgb.shape[0] * width / rgb.shape[1])
        rgb = np.array(
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize(
                (width, h_tgt), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0

    path = catalog["mobius"]["path"]
    path_len = len(path)
    all_nodes = sorted(catalog["nodes"], key=lambda n: n.get("path_index", 0))
    font = _load_font(18)
    frames_pil: List[Any] = []
    total = len(progress_curve)

    for i, progress in enumerate(progress_curve):
        lit = nodes_in_revealed_arc(all_nodes, progress, path_len)
        frame = render_nebula_progress(rgb, path, lit, progress, bloom_radius=10)
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        pct = int(round(progress * 100))
        draw.rectangle([12, 12, 200, 46], fill=(12, 14, 24, 220))
        draw.text((20, 18), f"探索进度 {pct}%", fill=(235, 238, 248), font=font)
        frames_pil.append(pil.convert("P", palette=Image.ADAPTIVE, colors=256))
        if i % max(1, total // 6) == 0 or i == total - 1:
            print(f"    frame {i + 1}/{total}  progress={pct}%")

    duration_ms = int(1000 / max(fps, 1))
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"  → 探索动画 {out_path} ({total} 帧, {total / fps:.1f}s, GIF)")


def load_session_render(session_id: str) -> Dict[str, Any] | None:
    from observation.state_store import ObservationStateStore

    store = ObservationStateStore()
    state = store.load(session_id)
    if not state or not state.star_map_render:
        return None
    r = state.star_map_render
    return {
        "progress": r.progress,
        "lit_nodes": r.lit_nodes,
        "lit_node_ids": r.lit_node_ids,
    }


def main() -> int:
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="星点分布图 + 探索轨迹 GIF")
    parser.add_argument("--session-id", help="用观测 session 的真实探索进度生成动画")
    parser.add_argument("--seed", type=int, default=20260626, help="随机路径种子（无 session 时）")
    args = parser.parse_args()

    if not BG_PREVIEW.is_file():
        print(f"缺少底图: {BG_PREVIEW}，请先运行 setup_nebula_starmap.py")
        return 1
    if not CATALOG_PATH.is_file():
        print(f"缺少星表: {CATALOG_PATH}")
        return 1

    rgb = np.array(Image.open(BG_PREVIEW).convert("RGB"), dtype=np.float32) / 255.0
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    path_len = len(catalog["mobius"]["path"])

    print("[demo] 渲染星点分布图 …")
    render_star_points_map(rgb, catalog, OUT_DIR / "star_points_map.jpg")

    mp4_path = OUT_DIR / "exploration_journey.mp4"
    if mp4_path.is_file():
        mp4_path.unlink()

    print("[demo] 渲染探索轨迹动画 …")
    if args.session_id:
        render_data = load_session_render(args.session_id)
        if not render_data:
            print(f"  session {args.session_id} 无 star_map_render，请先触发观测抽取")
            return 1
        curve = _progress_curve_from_render(render_data, path_len, frames=48)
        print(f"  使用 session 探索进度 target={render_data['progress']:.0%}")
    else:
        rng = np.random.default_rng(args.seed)
        curve = _random_progress_curve(path_len, rng, n_stops=12, frames=48)
        print(f"  使用随机探索路径 seed={args.seed}")

    render_exploration_gif(rgb, catalog, OUT_DIR / "exploration_journey.gif", curve)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
