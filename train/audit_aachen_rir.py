#!/usr/bin/env python3
"""Audit the release-compatible Aachen chapel measured RIR corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import soundfile

from data import discover_aachen_rirs


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 271_472_748
EXPECTED_MD5 = "6d20fffe4ab0c3e0ce85d7774232487f"
EXPECTED_FILES = 46


def digest(path: Path) -> str:
    value = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/downloads/aachen-chapel-rir/impulse_responses.zip",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data/corpus/aachen-chapel-rir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "train/runs/aachen-chapel-rir-manifest.json",
    )
    args = parser.parse_args()

    size = args.archive.stat().st_size
    archive_md5 = digest(args.archive)
    if size != EXPECTED_BYTES or archive_md5 != EXPECTED_MD5:
        raise RuntimeError(f"unexpected archive: {size} bytes, md5 {archive_md5}")

    rirs = discover_aachen_rirs(args.root)
    if len(rirs) != EXPECTED_FILES:
        raise RuntimeError(f"expected {EXPECTED_FILES} measured RIRs, found {len(rirs)}")

    contracts: Counter[str] = Counter()
    durations: list[float] = []
    inventory = hashlib.sha256()
    for path in rirs:
        relative = path.relative_to(args.root).as_posix()
        inventory.update(relative.encode())
        inventory.update(str(path.stat().st_size).encode())
        info = soundfile.info(path)
        contracts[f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"] += 1
        durations.append(info.frames / info.samplerate)
        if info.samplerate != 48_000 or info.channels != 4:
            raise RuntimeError(f"unexpected RIR audio contract: {path}: {info}")
        if not 6.0 <= durations[-1] <= 10.0:
            raise RuntimeError(f"unexpected RIR duration: {path}: {durations[-1]:.3f}s")

    first = int(len(rirs) * 0.60)
    second = int(len(rirs) * 0.76)
    third = int(len(rirs) * 0.88)
    report = {
        "schema": 1,
        "title": "Room acoustic measurement and simulation data of the St. Nicholas Chapel, Aachen Cathedral",
        "record": "https://zenodo.org/records/20428705",
        "doi": "10.5281/zenodo.20428705",
        "license": "CC-BY-4.0",
        "creators": ["Martin Zerwas", "Selin Kayku", "FH Aachen"],
        "archive": {
            "file": str(args.archive.resolve()),
            "bytes": size,
            "md5": archive_md5,
        },
        "policy": {
            "input": "measured B-format BRIRs only; simulated responses are excluded",
            "rendering": "ephemeral FFT convolution of channel 1 with 25/40/55/70 percent wet mix",
            "derived_audio_written": False,
            "split": "sorted RIR inventory split 60/16/12/12 percent; no RIR crosses train, validation, calibration, and test",
        },
        "counts": {
            "files": len(rirs),
            "splits": {
                "train": first,
                "valid": second - first,
                "calibrate": third - second,
                "test": len(rirs) - third,
            },
            "audio_contracts": dict(sorted(contracts.items())),
            "duration_seconds": {
                "minimum": min(durations),
                "maximum": max(durations),
            },
        },
        "inventory_sha256": inventory.hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
