#!/usr/bin/env python3
"""
P0+P1：离线生成探索星图底场（Gaia 风银盘密度 + 弱莫比乌斯星云）与叙事星表。

输出：
  static/images/exploration_starfield_bg.webp
  static/images/exploration_starfield_preview.jpg
  static/data/exploration_star_catalog.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.star_map_catalog import build_narrative_catalog  # noqa: E402

W, H = 2048, 1152
N_FIELD = 80_000
SEED = 42


def _moffat_psf(r: np.ndarray, beta: float = 3.5, r_core: float = 1.2) -> np.ndarray:
    return (1 + (r / r_core) ** 2) ** (-beta)


def _galactic_density(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    rho = np.exp(-0.5 * (b_deg / 14.0) ** 2)
    rho *= 1.0 + 0.12 * np.cos(2.0 * np.radians(l_deg))
    rho *= 1.0 + 0.08 * np.cos(4.0 * np.radians(l_deg + 35.0))
    bulge = np.exp(-0.5 * ((l_deg - 0) / 25.0) ** 2) * np.exp(-0.5 * (b_deg / 8.0) ** 2)
    return rho + 0.45 * bulge


def _magnitude_flux(m: np.ndarray) -> np.ndarray:
    return 10.0 ** (-0.4 * (m - 8.0))


def render_galaxy_field(rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros((H, W, 3), dtype=np.float32)
    l_deg = rng.uniform(-60, 60, N_FIELD)
    b_deg = rng.normal(0, 9, N_FIELD)
    weights = _galactic_density(l_deg, b_deg)
    weights /= weights.sum() + 1e-12
    idx = rng.choice(N_FIELD, size=N_FIELD, p=weights)
    l_deg, b_deg = l_deg[idx], b_deg[idx]
    px = ((l_deg + 60) / 120.0 * (W - 1)).astype(np.int32)
    py = ((0.5 - b_deg / 90.0) * (H - 1)).astype(np.int32)
    mag = rng.uniform(8.0, 15.5, N_FIELD)
    flux = _magnitude_flux(mag) * rng.uniform(0.8, 1.2, N_FIELD) * 2.8
    bp_rp = rng.normal(0.8, 0.6, N_FIELD)
    colors = np.stack([
        np.clip(1.0 + 0.15 * bp_rp, 0.75, 1.15),
        np.clip(1.0 - 0.05 * np.abs(bp_rp), 0.85, 1.05),
        np.clip(1.0 - 0.25 * bp_rp, 0.65, 1.1),
    ], axis=1)
    yy, xx = np.mgrid[-8:9, -8:9]
    kernel = _moffat_psf(np.sqrt(xx ** 2 + yy ** 2)).astype(np.float32)
    kernel /= kernel.sum() + 1e-12
    for i in range(N_FIELD):
        x, y = int(px[i]), int(py[i])
        f = float(flux[i])
        if f < 0.002:
            continue
        c = colors[i] * f
        x0, x1 = max(0, x - 8), min(W, x + 9)
        y0, y1 = max(0, y - 8), min(H, y + 9)
        kx0, ky0 = 8 - (x - x0), 8 - (y - y0)
        acc[y0:y1, x0:x1, :] += kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0), None] * c
    return acc


def render_mobius_nebula(rng: np.random.Generator, strength: float = 0.24) -> np.ndarray:
    from PIL import Image, ImageFilter

    layer = np.zeros((H, W, 4), dtype=np.float32)
    R, w = 0.32, 0.055
    cx, cy = 0.52, 0.48
    # 环带采样：形成可辨认的莫比乌斯轮廓
    for u_steps in range(360):
        u = 2 * math.pi * u_steps / 360
        for v in np.linspace(-0.85, 0.85, 8):
            x3 = (R + v * w * math.cos(u / 2)) * math.cos(u)
            y3 = (R + v * w * math.cos(u / 2)) * math.sin(u)
            z3 = v * w * math.sin(u / 2)
            px = int((cx + x3 * 0.95) * W)
            py = int((cy - y3 * 0.72 + z3 * 0.15) * H)
            if 0 <= px < W and 0 <= py < H:
                a = strength * (1.0 - abs(v) * 0.6)
                layer[py, px, 0] += 0.5 * a
                layer[py, px, 1] += 0.32 * a
                layer[py, px, 2] += 0.82 * a
                layer[py, px, 3] += a
    rgba = np.clip(layer * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    return np.array(img, dtype=np.float32) / 255.0


def composite_and_tone_map(field: np.ndarray, nebula: np.ndarray) -> np.ndarray:
    rgb = field.copy()
    neb_a = nebula[:, :, 3:4]
    rgb = rgb * (1.0 - neb_a * 0.55) + nebula[:, :, :3] * neb_a
    rgb += 0.012
    lum = rgb.max(axis=2, keepdims=True)
    rgb = rgb / (1.0 + lum * 1.2)
    rgb = np.power(np.clip(rgb, 0, 1), 0.82)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow: pip install Pillow")
        return 1

    out_img = ROOT / "static" / "images" / "exploration_starfield_bg.webp"
    out_preview = ROOT / "static" / "images" / "exploration_starfield_preview.jpg"
    out_catalog = ROOT / "static" / "data" / "exploration_star_catalog.json"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print(f"[P0] 渲染银盘星场 {N_FIELD} 颗星 @ {W}x{H} …")
    field = render_galaxy_field(rng)
    print("[P0] 叠加弱莫比乌斯星云 …")
    nebula = render_mobius_nebula(rng)
    rgb = composite_and_tone_map(field, nebula)
    Image.fromarray(rgb, "RGB").save(out_img, "WEBP", quality=88, method=6)
    Image.fromarray(rgb, "RGB").save(out_preview, "JPEG", quality=92)
    print(f"  → {out_img} ({out_img.stat().st_size // 1024} KB)")

    print("[P1] 叙事星表 …")
    catalog = build_narrative_catalog()
    catalog["background"] = {
        "url": "/static/images/exploration_starfield_bg.webp",
        "width": W,
        "height": H,
        "mobius_strength": "weak",
    }
    out_catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → nodes={len(catalog.get('nodes', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
