"""Fail when the deploy lock's distributions reach the 256 MiB builder budget.

The Superlinear nano builder has about 256 MiB of memory. This checks the
Python 3.11 / manylinux2014 files that image actually installs: every direct
and transitive distribution in requirements-deploy.lock, compressed size
summed together. jieba publishes only a source tarball, so that tarball is
counted too.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-deploy.lock"
BUDGET_BYTES = 256 * 1024 * 1024
# No manylinux wheel; the sdist is what pip installs.
SDIST_ONLY = {"jieba"}
PIP_BASE = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--dry-run",
    "--ignore-installed",
    "--python-version",
    "3.11",
    "--implementation",
    "cp",
    "--abi",
    "cp311",
    "--platform",
    "manylinux2014_x86_64",
    "--only-binary=:all:",
]


def _parse_lock() -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise SystemExit(f"锁文件必须是 name==version：{raw}")
        name, version = line.split("==", 1)
        pins.append((name.strip(), version.strip()))
    return pins


def _content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    if not value:
        raise SystemExit(f"镜像没有返回体积：{url}")
    return int(value)


def _sdist_size(name: str, version: str) -> int:
    url = f"https://pypi.tuna.tsinghua.edu.cn/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    files = payload["releases"].get(version) or []
    sdists = [item for item in files if item.get("packagetype") == "sdist"]
    if not sdists:
        raise SystemExit(f"{name}=={version} 没有源码包可计入预算")
    return int(sdists[0]["size"])


def _wheel_rows(pins: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    wheel_pins = [(name, version) for name, version in pins if name.lower() not in SDIST_ONLY]
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "wheels.txt"
        report = Path(tmp) / "report.json"
        req.write_text("".join(f"{name}=={version}\n" for name, version in wheel_pins), encoding="utf-8")
        completed = subprocess.run(
            [*PIP_BASE, "--report", str(report), "-r", str(req)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout[-2000:])
            sys.stderr.write(completed.stderr[-2000:])
            raise SystemExit("无法按 Python 3.11 / manylinux2014 解析部署锁")
        payload = json.loads(report.read_text(encoding="utf-8"))
    items = payload["install"]

    def one(item):
        meta = item["metadata"]
        return _content_length(item["download_info"]["url"]), meta["name"], meta["version"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(one, items))


def main() -> int:
    pins = _parse_lock()
    rows = _wheel_rows(pins)
    for name, version in pins:
        if name.lower() in SDIST_ONLY:
            rows.append((_sdist_size(name, version), name, version))
    total = sum(size for size, _, _ in rows)
    rows.sort(reverse=True)
    print(f"distributions={len(rows)} total={total / 1024 / 1024:.1f} MiB budget=256 MiB")
    for size, name, version in rows[:12]:
        print(f"{size / 1024 / 1024:7.2f} MiB  {name}=={version}")
    locked = {name.lower().replace("_", "-") for name, _ in pins}
    resolved = {name.lower().replace("_", "-") for _, name, _ in rows}
    if locked != resolved:
        missing = sorted(locked - resolved)
        extra = sorted(resolved - locked)
        raise SystemExit(f"锁和解析结果不一致 missing={missing[:8]} extra={extra[:8]}")
    if total >= BUDGET_BYTES:
        print("超过 256 MiB 构建预算", file=sys.stderr)
        return 1
    print("预算内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
