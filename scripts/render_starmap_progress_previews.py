#!/usr/bin/env python3
"""生成 30/50/75/100% 探索进度星图预览（离线 PIL，供验收）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.star_map_render import _bezier_points, _mobius_guide_path  # noqa: E402

W_OUT, H_OUT = 1280, 720


def _moffat(r, beta=3.5, rc=1.2):
    return (1 + (r / rc) ** 2) ** (-beta)


def _draw_psf(acc: np.ndarray, px: int, py: int, flux: float, color, radius=18):
    h, w, _ = acc.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = px + dx, py + dy
            if 0 <= x < w and 0 <= y < h:
                r = math.hypot(dx, dy)
                i = flux * _moffat(r)
                if i > 0.001:
                    acc[y, x, 0] += color[0] * i
                    acc[y, x, 1] += color[1] * i
                    acc[y, x, 2] += color[2] * i


def _draw_line(acc: np.ndarray, pts, color, width=2, alpha=0.85):
    from PIL import Image, ImageDraw

    img = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")
    xy = [(p[0] * W_OUT, (1 - p[1]) * H_OUT) for p in pts]
    c = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(alpha * 255))
    if len(xy) >= 2:
        draw.line(xy, fill=c, width=width, joint="curve")
    return np.array(img, dtype=np.float32) / 255.0


def render_progress(bg_rgb: np.ndarray, catalog: dict, progress: float) -> np.ndarray:
    nodes = sorted(catalog["nodes"], key=lambda n: -n.get("importance", 0))
    n_lit = max(1, int(len(nodes) * progress))
    lit = nodes[:n_lit]
    lit_ids = {n["id"] for n in lit}
    lit_map = {n["id"]: n for n in lit}

    acc = bg_rgb.astype(np.float32).copy()
    mobius_full = _mobius_guide_path(catalog.get("mobius") or {}, samples=72)
    arc_n = max(2, int(len(mobius_full) * progress))
    mobius_arc = mobius_full[:arc_n]

    TYPE_COLOR = {
        "character": (1.0, 0.9, 0.65),
        "event": (0.5, 0.78, 1.0),
        "imagery": (0.72, 0.52, 1.0),
    }

    for e in catalog.get("edges") or []:
        if e["from"] in lit_ids and e["to"] in lit_ids:
            a, b = lit_map[e["from"]], lit_map[e["to"]]
            pts = _bezier_points(a["x"], a["y"], b["x"], b["y"], bend=0.15)
            acc = _draw_line(acc, pts, (0.7, 0.85, 1.0), width=2, alpha=0.55)

    acc = _draw_line(acc, mobius_full, (0.42, 0.32, 0.72), width=1, alpha=0.35)
    acc = _draw_line(acc, mobius_arc, (0.62, 0.48, 0.95), width=3, alpha=0.9)

    for n in lit:
        px = int(n["x"] * W_OUT)
        py = int((1 - n["y"]) * H_OUT)
        imp = n.get("importance", 0.5)
        mag = n.get("magnitude", 5)
        flux = (10 ** (-0.4 * (mag - 2))) * (0.8 + imp * 1.2) * 2.2
        _draw_psf(acc, px, py, flux, TYPE_COLOR.get(n["type"], TYPE_COLOR["character"]), radius=22)

    lum = acc.max(axis=2, keepdims=True)
    acc = acc / (1.0 + lum * 0.8)
    return (np.clip(acc, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    from PIL import Image, ImageDraw, ImageFont

    bg_path = ROOT / "static" / "images" / "exploration_starfield_bg.webp"
    catalog_path = ROOT / "static" / "data" / "exploration_star_catalog.json"
    out_dir = ROOT / "static" / "images" / "progress_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = np.array(Image.open(bg_path).convert("RGB").resize((W_OUT, H_OUT))) / 255.0
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    for pct in (30, 50, 75, 100):
        img = render_progress(bg, catalog, pct / 100.0)
        out = out_dir / f"starmap_progress_{pct}.jpg"
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        label = f"探索进度 {pct}%"
        draw.rectangle([12, 12, 180, 44], fill=(20, 20, 30, 200))
        draw.text((20, 18), label, fill=(230, 230, 240))
        pil.save(out, "JPEG", quality=92)
        print(f"  → {out}")

    combo = Image.new("RGB", (W_OUT * 2, H_OUT * 2))
    for i, pct in enumerate((30, 50, 75, 100)):
        tile = Image.open(out_dir / f"starmap_progress_{pct}.jpg")
        combo.paste(tile, ((i % 2) * W_OUT, (i // 2) * H_OUT))
    combo.save(out_dir / "starmap_progress_grid.jpg", "JPEG", quality=90)
    print(f"  → grid {out_dir / 'starmap_progress_grid.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
