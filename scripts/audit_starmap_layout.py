#!/usr/bin/env python3
"""诊断星表：越界坐标、人物/事件重合、跨图连线。输出 overlay JPG。"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATALOG = ROOT / "static" / "data" / "exploration_star_catalog.json"
BG = ROOT / "static" / "images" / "exploration_nebula_bg.webp"
OUT = ROOT / "static" / "images" / "nebula_previews" / "starmap_layout_audit.jpg"


def main() -> int:
    from PIL import Image, ImageDraw, ImageFont

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    nodes = cat["nodes"]
    path = cat["mobius"]["path"]
    w, h = 1280, int(1280 * cat["background"]["height"] / cat["background"]["width"])

    if BG.is_file():
        img = Image.open(BG).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    else:
        img = Image.new("RGB", (w, h), (5, 5, 8))
    draw = ImageDraw.Draw(img)

    def xy(n):
        return n["x"] * w, (1 - n["y"]) * h

    # path
    pts = [xy({"x": p[0], "y": p[1]}) for p in path]
    draw.line(pts + [pts[0]], fill=(80, 60, 120), width=2)

    issues = []
    by_idx: dict[int, list] = defaultdict(list)
    for n in nodes:
        by_idx[n.get("path_index", -1)].append(n)
        x, y = n["x"], n["y"]
        if x < 0.02 or x > 0.98 or y < 0.02 or y > 0.98:
            issues.append(f"越界 {n['label']}: ({x:.3f},{y:.3f})")
        px, py = xy(n)
        if px < 4 or px > w - 4 or py < 4 or py > h - 4:
            issues.append(f"像素越界 {n['label']}: ({px:.0f},{py:.0f})")

    for idx, group in by_idx.items():
        if len(group) > 1:
            labels = ", ".join(f"{g['type']}:{g['label']}" for g in group)
            issues.append(f"path_index={idx} 重合: {labels}")

    chars = {n["label"]: n for n in nodes if n["type"] == "character"}
    for n in nodes:
        if n["type"] not in ("event", "imagery"):
            continue
        parents = n.get("parents") or ([n["parent"]] if n.get("parent") else [])
        for p in parents:
            c = chars.get(p)
            if not c:
                continue
            dist = math.hypot(n["x"] - c["x"], n["y"] - c["y"])
            if dist < 0.018:
                issues.append(f"距人物过近 {n['label']} @ {p} d={dist:.4f}")

    # draw nodes
    for n in nodes:
        px, py = xy(n)
        col = {"character": (255, 235, 180), "event": (120, 200, 255), "imagery": (200, 150, 255)}.get(
            n["type"], (200, 200, 200)
        )
        r = 6 if n["type"] == "character" else 4
        draw.ellipse((px - r, py - r, px + r, py + r), outline=col, width=2)
        draw.line([(px, py), (px + 8, py - 10)], fill=col, width=1)

    # draw edges (long lines highlighted)
    for e in cat.get("edges") or []:
        a = next((n for n in nodes if n["id"] == e["from"]), None)
        b = next((n for n in nodes if n["id"] == e["to"]), None)
        if not a or not b:
            continue
        p0, p1 = xy(a), xy(b)
        dist = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
        col = (255, 80, 80) if dist > w * 0.35 else (60, 120, 180)
        draw.line([p0, p1], fill=col, width=1 if dist <= w * 0.35 else 2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=92)
    print(f"审计图 → {OUT.relative_to(ROOT)}")
    print(f"问题 {len(issues)} 条:")
    for i in issues[:30]:
        print(f"  - {i}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
