#!/usr/bin/env python3
"""导入用户星云主体图 → 反向布点 → 更新星表 → 渲染 30/50/75/100% 预览。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observation.nebula_starmap import (  # noqa: E402
    CATALOG_PATH,
    DEFAULT_SRC,
    ingest_nebula_image,
    preview_progress_set,
    rebuild_catalog_from_nebula,
)


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    print(f"[星云] 读取源图: {src}")
    rgb, bg_path = ingest_nebula_image(src)
    print(f"  → 底图 {bg_path} ({rgb.shape[1]}x{rgb.shape[0]})")

    catalog = rebuild_catalog_from_nebula(rgb)
    CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  → 星表 {CATALOG_PATH} (nodes={len(catalog['nodes'])}, path={len(catalog['mobius']['path'])})")

    out_dir = ROOT / "static" / "images" / "nebula_previews"
    print("[星云] 渲染进度预览 …")
    preview_progress_set(rgb, catalog, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
