"""以星云主体图反向布点：沿 ∞ 形亮带采样路径、映射叙事节点、计算点亮光晕与渐进揭示。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "数据源" / "星图.png"
DEFAULT_BG = ROOT / "static" / "images" / "exploration_nebula_bg.webp"
CATALOG_PATH = ROOT / "static" / "data" / "exploration_star_catalog.json"

TYPE_COLOR = {
    "character": (1.0, 0.92, 0.72),
    "event": (0.55, 0.82, 1.0),
    "imagery": (0.78, 0.58, 1.0),
}


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def figure8_norm(t: float, cx: float, cy: float, ax: float, ay: float) -> Tuple[float, float]:
    """水平 ∞ 参数曲线，返回 catalog 坐标 (x∈[0,1] 左→右, y∈[0,1] 下→上)。"""
    x = cx + ax * math.cos(t)
    y = cy + ay * math.sin(2.0 * t) / 2.0
    return x, y


def extract_nebula_path(
    lum: np.ndarray,
    samples: int = 96,
    cx: float = 0.50,
    cy: float = 0.52,
    ax: float = 0.36,
    ay: float = 0.26,
    search_px: int = 28,
) -> List[List[float]]:
    """沿理论 ∞ 曲线在亮带内找局部峰值，得到贴合星云主体的路径点。"""
    h, w = lum.shape
    pts: List[List[float]] = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        nx, ny = figure8_norm(t, cx, cy, ax, ay)
        px, py = int(nx * w), int((1.0 - ny) * h)
        y0, y1 = max(0, py - search_px), min(h, py + search_px + 1)
        x0, x1 = max(0, px - search_px), min(w, px + search_px + 1)
        patch = lum[y0:y1, x0:x1]
        if patch.size == 0:
            pts.append([round(nx, 4), round(ny, 4)])
            continue
        dy, dx = np.unravel_index(int(np.argmax(patch)), patch.shape)
        bx, by = x0 + dx, y0 + dy
        pts.append([round(bx / w, 4), round(1.0 - by / h, 4)])
    return pts


def path_reveal_index(progress: float, path_len: int) -> int:
    """progress∈[0,1] → 已揭示的路径点下标（含），与视觉 mask 严格对齐。"""
    if path_len <= 1:
        return 0
    p = max(0.0, min(1.0, progress))
    return int(round(p * (path_len - 1)))


def _watermark_ui_mask(patch: np.ndarray) -> np.ndarray:
    """检测角部 UI 像素：白字/图标、灰边框、半透明 badge。"""
    lum = _luminance(patch)
    std = np.std(patch, axis=2)
    white = (lum > 0.72) & (np.min(patch, axis=2) > 0.65)
    gray = (lum > 0.18) & (lum < 0.58) & (std < 0.045)
    border = (lum > 0.28) & (lum < 0.52) & (std < 0.035)
    return white | gray | border


def _build_watermark_mask(src_f: np.ndarray) -> np.ndarray:
    """在源图上构建水印 mask（仅角部小块，不误伤星云主体）。"""
    import cv2

    h, w = src_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    zones = [
        (0, 145, 0, 365, 5, 2),
        (895, h, 1080, w, 7, 3),
        (1002, h, 790, 1135, 5, 2),
    ]
    for y0, y1, x0, x1, k, dil in zones:
        y1 = min(y1, h)
        x1 = min(x1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        m = (_watermark_ui_mask(src_f[y0:y1, x0:x1]).astype(np.uint8) * 255)
        m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=dil)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], m)
    return mask


def _fill_ui_pixels_from_neighbors(
    arr: np.ndarray,
    zones: Sequence[Tuple[int, int, int, int]],
    lum_threshold: float = 0.52,
) -> np.ndarray:
    """角部 zone 内残留高亮 UI 像素，从邻近暗星野复制。"""
    out = arr.copy()
    h, w = out.shape[:2]
    offsets = [
        (15, 0), (-15, 0), (0, 15), (0, -15),
        (25, 0), (0, -25), (-25, 0), (18, -12), (-18, 12),
    ]
    for _ in range(3):
        lum = _luminance(out)
        coords: List[Tuple[int, int]] = []
        for y0, y1, x0, x1 in zones:
            y1 = min(y1, h)
            x1 = min(x1, w)
            ys, xs = np.where(lum[y0:y1, x0:x1] > lum_threshold)
            coords.extend((y0 + y, x0 + x) for y, x in zip(ys, xs))
        if not coords:
            break
        for y, x in coords:
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and _luminance(out[ny : ny + 1, nx : nx + 1])[0, 0] < 0.14:
                    out[y, x] = out[ny, nx]
                    break
    return out


def blend_watermark_regions(rgb: np.ndarray) -> np.ndarray:
    """不裁切 16:9：角部 UI 精准 inpaint + 邻域暗星野补点（无大块镜像/复制）。"""
    import cv2

    h, w = rgb.shape[:2]
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()
    mask = _build_watermark_mask(rgb)
    if mask.any():
        out = cv2.inpaint(out, mask, 10, cv2.INPAINT_NS)

    sf = out.astype(np.float32) / 255.0
    touch_zones = [
        (0, 145, 0, 365),
        (895, h, 1080, w),
        (930, h, 1580, 1920),
    ]
    sf = _fill_ui_pixels_from_neighbors(sf, touch_zones)

    lum = _luminance(sf)
    residual = np.zeros((h, w), dtype=np.uint8)
    for y0, y1, x0, x1, thr in [
        (0, 145, 0, 365, 0.44),
        (930, h, 1580, 1920, 0.36),
    ]:
        y1 = min(y1, h)
        x1 = min(x1, w)
        r = (lum[y0:y1, x0:x1] > thr).astype(np.uint8) * 255
        r = cv2.dilate(r, np.ones((3, 3), np.uint8), iterations=1)
        residual[y0:y1, x0:x1] = r
    if residual.any():
        out = cv2.inpaint((sf * 255).astype(np.uint8), residual, 5, cv2.INPAINT_TELEA)
        sf = out.astype(np.float32) / 255.0

    return sf


def _path_arc_lengths(path: Sequence[Sequence[float]]) -> Tuple[np.ndarray, float]:
    arr = np.array(path, dtype=np.float64)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, float(cum[-1])


def _node_parent_labels(node: Dict[str, Any]) -> List[str]:
    parents = node.get("parents")
    if isinstance(parents, list):
        return [str(p) for p in parents if p]
    p = node.get("parent")
    return [str(p)] if p else []


def _inward_xy(
    x: float,
    y: float,
    center: Tuple[float, float] = (0.52, 0.48),
) -> Tuple[float, float]:
    """距画面边缘越近，向星云中心收拢越多。"""
    inset = min(x, 1.0 - x, y, 1.0 - y)
    if inset >= 0.22:
        pull = 0.10
    elif inset >= 0.16:
        pull = 0.20
    else:
        pull = 0.30
    ox = x * (1.0 - pull) + center[0] * pull
    oy = y * (1.0 - pull) + center[1] * pull
    return _clamp_xy(ox, oy, margin=0.04)


def _usable_path_indices(path: List[List[float]], min_inset: float = 0.20) -> List[int]:
    """排除 ∞ 路径外侧尖角，只在星云主体环带布点。"""
    usable = [
        i for i, (x, y) in enumerate(path)
        if min(x, 1.0 - x, y, 1.0 - y) >= min_inset
        and 0.24 <= x <= 0.80
        and 0.20 <= y <= 0.82
    ]
    if len(usable) < max(12, len(path) // 5):
        relaxed = min_inset * 0.55
        usable = [
            i for i, (x, y) in enumerate(path)
            if min(x, 1.0 - x, y, 1.0 - y) >= relaxed
        ]
    return usable if usable else list(range(len(path)))


def _nearest_usable(desired: int, usable: List[int]) -> int:
    return min(usable, key=lambda u: abs(u - desired))


def _clamp_xy(x: float, y: float, margin: float = 0.035) -> Tuple[float, float]:
    return (
        max(margin, min(1.0 - margin, x)),
        max(margin, min(1.0 - margin, y)),
    )


def _path_tangent(path: List[List[float]], idx: int) -> Tuple[float, float]:
    i0 = max(0, idx - 1)
    i1 = min(len(path) - 1, idx + 1)
    tx = path[i1][0] - path[i0][0]
    ty = path[i1][1] - path[i0][1]
    norm = math.hypot(tx, ty) or 1.0
    return tx / norm, ty / norm


def _display_xy_on_path(
    path: List[List[float]],
    idx: int,
    side: int,
    magnitude: float = 0.016,
    center: Tuple[float, float] = (0.52, 0.48),
) -> Tuple[float, float]:
    """沿路径法向微偏移，避免与人物点重合；略向画面中心收拢。"""
    px, py = path[idx]
    tx, ty = _path_tangent(path, idx)
    nx, ny = -ty, tx
    sign = 1.0 if side % 2 == 0 else -1.0
    ox = px + sign * magnitude * nx
    oy = py + sign * magnitude * ny
    return _inward_xy(ox, oy, center=center)


def _min_dist_to_chars(
    x: float, y: float, char_xy: Dict[str, Tuple[float, float]],
) -> float:
    if not char_xy:
        return 1.0
    return min(math.hypot(x - cx, y - cy) for cx, cy in char_xy.values())


def _snap_nodes_to_path(
    nodes: List[Dict[str, Any]],
    path: List[List[float]],
) -> List[Dict[str, Any]]:
    """人物沿 ∞ 路径分散；事件/意象沿轨迹段采样（非环形）。"""
    path_len = len(path)
    if path_len < 2:
        return list(nodes)

    chars = sorted(
        [n for n in nodes if n.get("type") == "character"],
        key=lambda n: (-n.get("importance", 0), n.get("id", "")),
    )
    others = [n for n in nodes if n.get("type") != "character"]
    usable = _usable_path_indices(path)

    char_pos: Dict[str, int] = {}
    for i, node in enumerate(chars):
        t = i / max(len(chars) - 1, 1)
        idx = usable[int(round(t * max(len(usable) - 1, 0)))]
        char_pos[node["label"]] = idx

    placed: List[Dict[str, Any]] = []
    for node in chars:
        idx = char_pos[node["label"]]
        px, py = path[idx]
        x, y = _inward_xy(px, py)
        placed.append({
            **node,
            "x": round(x, 4),
            "y": round(y, 4),
            "path_index": idx,
            "path_t": round(idx / max(path_len - 1, 1), 4),
        })

    used_indices: set[int] = set(char_pos.values())
    char_reserved = set(char_pos.values())
    char_xy: Dict[str, Tuple[float, float]] = {
        node["label"]: (float(placed[i]["x"]), float(placed[i]["y"]))
        for i, node in enumerate(chars)
    }

    groups: Dict[str, List[Dict[str, Any]]] = {}
    orphans: List[Dict[str, Any]] = []
    for node in others:
        labels = _node_parent_labels(node)
        if labels:
            groups.setdefault(labels[0], []).append(node)
        else:
            orphans.append(node)

    def _allocate_index(desired: int) -> int:
        for idx in sorted(usable, key=lambda u: (abs(u - desired), u)):
            if idx in char_reserved or idx in used_indices:
                continue
            used_indices.add(idx)
            return idx
        for idx in usable:
            if idx not in used_indices and idx not in char_reserved:
                used_indices.add(idx)
                return idx
        idx = _nearest_usable(desired, usable)
        used_indices.add(idx)
        return idx

    def _pick_path_index(parent_indices: List[int], ordinal: int, total: int) -> int:
        if len(parent_indices) >= 2:
            lo, hi = min(parent_indices), max(parent_indices)
            inner_lo = lo + 2
            inner_hi = hi - 1
            if inner_hi > inner_lo:
                span = inner_hi - inner_lo + 1
                t = (ordinal + 1) / (total + 1)
                desired = inner_lo + int(round(t * max(span - 1, 0)))
            else:
                desired = min(path_len - 1, lo + 2 + ordinal)
        elif parent_indices:
            base = parent_indices[0]
            step = max(2, (path_len - base - 3) // max(total, 1))
            desired = min(path_len - 1, base + step * (ordinal + 1))
        else:
            desired = int(round((ordinal + 1) / (total + 1) * (path_len - 1)))
        return _allocate_index(desired)

    placed_xy: List[Tuple[float, float]] = list(char_xy.values())
    side_counter = 0
    for _parent_key, group in groups.items():
        group_sorted = sorted(group, key=lambda n: n.get("id", ""))
        total = len(group_sorted)
        for i, node in enumerate(group_sorted):
            labels = _node_parent_labels(node)
            parent_indices = [char_pos[l] for l in labels if l in char_pos]
            if not parent_indices and labels:
                parent_indices = [char_pos.get(labels[0], 0)]
            idx = _pick_path_index(parent_indices, i, total)
            x, y = path[idx][0], path[idx][1]
            for attempt in range(8):
                mag = 0.014 + attempt * 0.007
                sc = side_counter + attempt
                x, y = _display_xy_on_path(path, idx, sc, magnitude=mag)
                ok_char = _min_dist_to_chars(x, y, char_xy) >= 0.024
                ok_peer = all(math.hypot(x - px, y - py) >= 0.02 for px, py in placed_xy)
                if ok_char and ok_peer:
                    break
            side_counter += 1
            for pl in placed:
                if pl.get("type") != "character":
                    continue
                plabel = pl.get("label", "")
                if plabel not in labels and plabel not in (labels or []):
                    continue
                if math.hypot(x - pl["x"], y - pl["y"]) >= 0.02:
                    continue
                tx, ty = _path_tangent(path, idx)
                x, y = _inward_xy(x + tx * 0.028, y + ty * 0.028)
                break
            placed_xy.append((x, y))
            out: Dict[str, Any] = {
                **node,
                "x": round(x, 4),
                "y": round(y, 4),
                "path_index": idx,
                "path_t": round(idx / max(path_len - 1, 1), 4),
            }
            if labels:
                out["parents"] = labels
                out["parent"] = labels[0]
            placed.append(out)

    for i, node in enumerate(orphans):
        idx = _pick_path_index([], i, len(orphans))
        x, y = _display_xy_on_path(path, idx, side_counter)
        side_counter += 1
        placed_xy.append((x, y))
        placed.append({
            **node,
            "x": round(x, 4),
            "y": round(y, 4),
            "path_index": idx,
            "path_t": round(idx / max(path_len - 1, 1), 4),
        })

    return placed


def resnap_catalog_paths(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """仅重算节点 path_index / 坐标，不重新提取星云路径。"""
    path = (catalog.get("mobius") or {}).get("path") or []
    if len(path) < 2:
        return catalog
    nodes = _snap_nodes_to_path(catalog.get("nodes") or [], path)
    node_by_id = {n["id"]: n for n in nodes}
    edges_out: List[Dict[str, Any]] = []
    max_edge_dist = 0.28
    for e in catalog.get("edges") or []:
        fid, tid = e.get("from"), e.get("to")
        a, b = node_by_id.get(fid), node_by_id.get(tid)
        if not a or not b:
            continue
        dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        if dist > max_edge_dist:
            continue
        bend = 0.10 if e.get("style") == "orbit" else 0.12
        edges_out.append({
            **e,
            "points": _bezier_points(a["x"], a["y"], b["x"], b["y"], bend=bend),
        })
    return {**catalog, "nodes": nodes, "edges": edges_out}


def _moffat(r: float, beta: float = 3.2, rc: float = 1.1) -> float:
    return (1.0 + (r / rc) ** 2) ** (-beta)


def _draw_bloom(
    acc: np.ndarray,
    px: int,
    py: int,
    color: Tuple[float, float, float],
    flux: float,
    radius: int = 24,
) -> None:
    h, w, _ = acc.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = px + dx, py + dy
            if 0 <= x < w and 0 <= y < h:
                r = math.hypot(dx, dy)
                i = flux * _moffat(r)
                if i > 0.002:
                    acc[y, x, 0] += color[0] * i
                    acc[y, x, 1] += color[1] * i
                    acc[y, x, 2] += color[2] * i


def _bezier_points(x0, y0, x1, y1, bend=0.14, n=10) -> List[List[float]]:
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    pts: List[List[float]] = []
    for i in range(n + 1):
        t = i / n
        omt = 1 - t
        pts.append([
            omt * omt * x0 + 2 * omt * t * cx + t * t * x1,
            omt * omt * y0 + 2 * omt * t * cy + t * t * y1,
        ])
    return pts


def nodes_in_revealed_arc(
    nodes: Sequence[Dict[str, Any]],
    progress: float,
    path_len: int,
) -> List[Dict[str, Any]]:
    """仅返回落在已揭示弧段上的节点（按 path_index 排序）。"""
    cap = path_reveal_index(progress, path_len)
    return sorted(
        [n for n in nodes if int(n.get("path_index", 0)) <= cap],
        key=lambda n: n.get("path_index", 0),
    )


def build_contour_mask(
    path: Sequence[Sequence[float]],
    w: int,
    h: int,
    corridor: float = 0.032,
    strength: float = 1.0,
) -> np.ndarray:
    """整条 ∞ 路径的淡轮廓（未揭示区也可见，不是全图涂色）。"""
    path_arr = np.array(path)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx + 0.5) / w
    ny = 1.0 - (yy + 0.5) / h
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(0, len(path), 2):
        px, py = path_arr[i]
        d = np.hypot(nx - px, ny - py)
        local = np.clip(1.0 - d / corridor, 0.0, 1.0) ** 1.4
        mask = np.maximum(mask, local)
    from PIL import Image, ImageFilter

    mimg = Image.fromarray((np.clip(mask * strength, 0, 1) * 255).astype(np.uint8))
    mimg = mimg.filter(ImageFilter.GaussianBlur(radius=3))
    return np.array(mimg, dtype=np.float32) / 255.0


def build_reveal_mask(
    path: Sequence[Sequence[float]],
    w: int,
    h: int,
    progress: float,
    corridor: float = 0.058,
) -> np.ndarray:
    """沿路径弧长渐进揭示（第一版渐变逻辑），progress 严格对应弧长。"""
    cum, total = _path_arc_lengths(path)
    if total <= 0:
        return np.zeros((h, w), dtype=np.float32)
    p = max(0.0, min(1.0, progress))
    reveal_len = max(0.001, p) * total
    path_arr = np.array(path)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx + 0.5) / w
    ny = 1.0 - (yy + 0.5) / h
    mask = np.zeros((h, w), dtype=np.float32)
    step = max(1, len(path) // 72)
    for i in range(0, len(path), step):
        if cum[i] > reveal_len:
            break
        px, py = path_arr[i]
        d = np.hypot(nx - px, ny - py)
        t = np.clip(1.0 - d / corridor, 0.0, 1.0)
        local = t * t * (3.0 - 2.0 * t)
        mask = np.maximum(mask, local)
    from PIL import Image, ImageFilter

    mimg = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=10))
    return np.array(mimg, dtype=np.float32) / 255.0


def render_nebula_progress(
    full_rgb: np.ndarray,
    path: Sequence[Sequence[float]],
    lit_nodes: Sequence[Dict[str, Any]],
    progress: float,
    dim_factor: float = 0.08,
    draw_lines: bool = False,
    lit_edges: Sequence[Dict[str, Any]] | None = None,
    bloom_radius: int = 22,
) -> np.ndarray:
    """
    100% = 原图；低进度 = 暗场 + 全路径淡轮廓 + 沿 ∞ 渐变揭示 + 弧段内星点光晕。
    不做全图涂色/剪影。
    """
    h, w, _ = full_rgb.shape
    full = full_rgb.astype(np.float32)
    lum = _luminance(full)
    dim = full * dim_factor + np.array([0.02, 0.025, 0.05], dtype=np.float32)

    reveal_mask = build_reveal_mask(path, w, h, progress)
    contour = build_contour_mask(path, w, h, corridor=0.034, strength=1.0)
    if progress >= 0.999:
        reveal_mask = np.ones((h, w), dtype=np.float32)

    unrevealed = (1.0 - reveal_mask)[..., None]
    contour_hint = full * (contour[..., None] * 0.18) * unrevealed
    base = dim + contour_hint
    acc = base * unrevealed + full * reveal_mask[..., None]

    if draw_lines and lit_edges:
        from PIL import Image, ImageDraw

        pil = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil, "RGBA")
        for e in lit_edges:
            pts = e.get("points") or []
            if len(pts) < 2:
                continue
            xy = [(p[0] * w, (1 - p[1]) * h) for p in pts]
            draw.line(xy, fill=(180, 210, 255, 100), width=2, joint="curve")
        acc = np.array(pil, dtype=np.float32) / 255.0

    path_len = len(path)
    reveal_cap = path_reveal_index(progress, path_len)
    for n in lit_nodes:
        if int(n.get("path_index", 0)) > reveal_cap:
            continue
        px = int(n["x"] * w)
        py = int((1.0 - n["y"]) * h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        local = full[py, px]
        imp = n.get("importance", 0.5)
        mag = n.get("magnitude", 5.0)
        flux = (10 ** (-0.4 * (mag - 2.0))) * (0.9 + imp * 1.4) * 3.2
        flux *= 0.55 + 0.45 * lum[py, px]
        tc = TYPE_COLOR.get(n.get("type", "character"), TYPE_COLOR["character"])
        color = (
            local[0] * 0.65 + tc[0] * 0.35,
            local[1] * 0.65 + tc[1] * 0.35,
            local[2] * 0.65 + tc[2] * 0.35,
        )
        _draw_bloom(acc, px, py, color, flux, radius=bloom_radius)
        _draw_bloom(acc, px, py, (1.0, 0.98, 0.92), flux * 0.3, radius=max(4, bloom_radius // 4))

    lum_out = acc.max(axis=2, keepdims=True)
    acc = acc / (1.0 + lum_out * 0.5)
    return (np.clip(acc, 0, 1) * 255).astype(np.uint8)


def ingest_nebula_image(
    src_path: Path | None = None,
    max_width: int = 1920,
) -> Tuple[np.ndarray, Path]:
    from PIL import Image

    src = src_path or DEFAULT_SRC
    if not src.is_file():
        raise FileNotFoundError(f"星云源图不存在: {src}")
    img = Image.open(src).convert("RGB")
    if img.width > max_width:
        nh = int(img.height * max_width / img.width)
        img = img.resize((max_width, nh), Image.Resampling.LANCZOS)
    rgb = np.array(img, dtype=np.float32) / 255.0
    rgb = blend_watermark_regions(rgb)
    out = DEFAULT_BG
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_out = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    pil_out.save(out, "WEBP", quality=88, method=6)
    pil_out.save(out.with_name("exploration_nebula_bg_preview.jpg"), "JPEG", quality=93)
    return rgb, out


def rebuild_catalog_from_nebula(
    rgb: np.ndarray,
    catalog: Dict[str, Any] | None = None,
    path_samples: int = 96,
) -> Dict[str, Any]:
    cat = dict(catalog or {})
    if not cat.get("nodes"):
        if CATALOG_PATH.is_file():
            cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        else:
            raise ValueError("无可用星表")

    lum = _luminance(rgb)
    path = extract_nebula_path(lum, samples=path_samples)
    nodes = _snap_nodes_to_path(cat.get("nodes") or [], path)
    node_by_id = {n["id"]: n for n in nodes}

    edges_out: List[Dict[str, Any]] = []
    for e in cat.get("edges") or []:
        fid, tid = e.get("from"), e.get("to")
        a, b = node_by_id.get(fid), node_by_id.get(tid)
        if not a or not b:
            continue
        bend = 0.10 if e.get("style") == "orbit" else 0.12
        edges_out.append({
            **e,
            "points": _bezier_points(a["x"], a["y"], b["x"], b["y"], bend=bend),
        })

    h, w = rgb.shape[:2]
    return {
        **cat,
        "version": "2.1-nebula",
        "nodes": nodes,
        "edges": edges_out,
        "mobius": {
            "center": [0.50, 0.52],
            "strength": "intrinsic",
            "path": path,
        },
        "background": {
            "url": "/static/images/exploration_nebula_bg.webp",
            "width": w,
            "height": h,
            "mode": "nebula_subject",
            "source": "数据源/星图.png",
        },
    }


def preview_progress_set(
    rgb: np.ndarray,
    catalog: Dict[str, Any],
    out_dir: Path,
    pcts: Sequence[int] = (30, 50, 75, 100),
) -> None:
    from PIL import Image, ImageDraw

    path = catalog["mobius"]["path"]
    path_len = len(path)
    nodes = sorted(catalog["nodes"], key=lambda n: n.get("path_index", 0))
    out_dir.mkdir(parents=True, exist_ok=True)

    for pct in pcts:
        progress = pct / 100.0
        lit = nodes_in_revealed_arc(nodes, progress, path_len)
        img = render_nebula_progress(rgb, path, lit, progress)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([12, 12, 200, 46], fill=(12, 14, 24, 210))
        draw.text((20, 18), f"探索进度 {pct}%", fill=(235, 238, 248))
        out = out_dir / f"nebula_progress_{pct}.jpg"
        pil.save(out, "JPEG", quality=93)
        print(f"  → {out}")

    w, h = Image.open(out_dir / f"nebula_progress_{pcts[0]}.jpg").size
    grid = Image.new("RGB", (w * 2, h * 2))
    for i, pct in enumerate(pcts):
        grid.paste(Image.open(out_dir / f"nebula_progress_{pct}.jpg"), ((i % 2) * w, (i // 2) * h))
    grid_path = out_dir / "nebula_progress_grid.jpg"
    grid.save(grid_path, "JPEG", quality=90)
    print(f"  → grid {grid_path}")
