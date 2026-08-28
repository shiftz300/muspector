#!/usr/bin/env python3
"""Prune rebuildable Inspector training data while preserving audit reports."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from layout import ACTIVE_CACHES, ACTIVE_RUNS, ROOT


CACHE_ROOT = ROOT / "train/cache"
RUN_ROOT = ROOT / "train/runs"
ARCHIVE_ROOT = RUN_ROOT / "archive"
LARGE_ARTIFACT_SUFFIXES = {".ckpt", ".npz", ".onnx", ".pt", ".safetensors"}
LARGE_ARTIFACT_BYTES = 1024 * 1024


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def inactive_caches() -> list[Path]:
    active = {path.resolve() for path in ACTIVE_CACHES}
    return sorted(
        path
        for path in CACHE_ROOT.iterdir()
        if path.is_dir() and path.resolve() not in active
    ) if CACHE_ROOT.exists() else []


def historical_payloads() -> list[Path]:
    active = {path.resolve() for path in ACTIVE_RUNS}
    result = []
    if not RUN_ROOT.exists():
        return result
    for run in RUN_ROOT.iterdir():
        if not run.is_dir() or run.resolve() in active:
            continue
        environment = run / "venv"
        if environment.is_dir():
            result.append(environment)
        result.extend(
            path
            for path in run.rglob("*")
            if path.is_file()
            and path.suffix in LARGE_ARTIFACT_SUFFIXES
            and path.stat().st_size >= LARGE_ARTIFACT_BYTES
        )
    return sorted(set(result))


def historical_runs() -> list[Path]:
    active = {path.resolve() for path in ACTIVE_RUNS}
    return sorted(
        path
        for path in RUN_ROOT.iterdir()
        if path.is_dir()
        and path != ARCHIVE_ROOT
        and path.resolve() not in active
    ) if RUN_ROOT.exists() else []


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the listed removals")
    parser.add_argument(
        "--historical-runs",
        action="store_true",
        help="also prune large payloads from inactive runs while retaining reports",
    )
    parser.add_argument(
        "--archive-runs",
        action="store_true",
        help="move inactive run reports under train/runs/archive",
    )
    args = parser.parse_args()

    targets = inactive_caches()
    if args.historical_runs:
        targets.extend(historical_payloads())
    archives = historical_runs() if args.archive_runs else []
    total = sum(size(path) for path in targets)
    for path in targets:
        print(f"{'remove' if args.apply else 'would remove'} {path.relative_to(ROOT)}")
    print(f"targets={len(targets)} bytes={total}")
    for path in archives:
        print(f"{'archive' if args.apply else 'would archive'} {path.relative_to(ROOT)}")
    print(f"archives={len(archives)}")
    if args.apply:
        for path in targets:
            remove(path)
        if archives:
            ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
            for path in archives:
                shutil.move(path, ARCHIVE_ROOT / path.name)


if __name__ == "__main__":
    main()
