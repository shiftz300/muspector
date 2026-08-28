#!/usr/bin/env python3
"""Audit the release-compatible DAFx25 guitar effect-chain corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile

from data import discover_guitar_effect_chains, guitar_effect_chain_is_clean, label


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 32_969_803_616
EXPECTED_MD5 = "56773ca55a4eb4c3be81f5c3418053a5"
EXPECTED_FILES = 38_800
EXPECTED_GROUPS = 400


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
        default=ROOT / "data/downloads/guitar-effects-chains/DATASET_guitar_effects.zip",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data/corpus/guitar-effects-chains",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "train/runs/guitar-effects-chains-manifest.json",
    )
    args = parser.parse_args()

    size = args.archive.stat().st_size
    archive_md5 = digest(args.archive)
    if size != EXPECTED_BYTES or archive_md5 != EXPECTED_MD5:
        raise RuntimeError(f"unexpected archive: {size} bytes, md5 {archive_md5}")

    parts = discover_guitar_effect_chains(args.root)
    items = [item for values in parts.values() for item in values]
    if len(items) != EXPECTED_FILES:
        raise RuntimeError(f"expected {EXPECTED_FILES} audio files, found {len(items)}")

    groups: dict[str, list] = defaultdict(list)
    inventory = hashlib.sha256()
    audio_contracts: Counter[str] = Counter()
    for item in sorted(items, key=lambda value: value.source.path.as_posix()):
        groups[item.source.group].append(item)
        relative = item.source.path.relative_to(args.root).as_posix()
        inventory.update(relative.encode())
        inventory.update(str(item.source.path.stat().st_size).encode())
        info = soundfile.info(item.source.path)
        audio_contracts[f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"] += 1
    if len(groups) != EXPECTED_GROUPS:
        raise RuntimeError(f"expected {EXPECTED_GROUPS} performances, found {len(groups)}")
    group_sizes = Counter(len(values) for values in groups.values())
    if group_sizes != Counter({97: EXPECTED_GROUPS}):
        raise RuntimeError(f"unexpected files per performance: {dict(group_sizes)}")

    split_counts = {name: len(values) for name, values in parts.items()}
    split_roles = {
        name: dict(sorted(Counter(label(item.target) for item in values).items()))
        for name, values in parts.items()
    }
    split_clean_references = {
        name: sum(guitar_effect_chain_is_clean(item.source) for item in values)
        for name, values in parts.items()
    }
    split_domains = {
        name: dict(sorted(Counter(item.source.domain for item in values).items()))
        for name, values in parts.items()
    }
    report = {
        "schema": 1,
        "title": "Guitar improvisations with chains of five effects",
        "record": "https://zenodo.org/records/7871720",
        "doi": "10.5281/zenodo.7871720",
        "license": "CC-BY-4.0",
        "creators": ["Michele Rossi"],
        "archive": {
            "file": str(args.archive.resolve()),
            "bytes": size,
            "md5": archive_md5,
        },
        "policy": {
            "upstream_bits": ["overdrive", "chorus", "tremolo", "delay", "reverb"],
            "targets": ["drive", "delay", "reverb"],
            "hard_negatives": ["chorus", "tremolo"],
            "split": "PRS+Les Paul train; Strat tracks 1-12 per playing condition validation and 13-25 calibration; Telecaster test",
            "group": "all dry and 96 processed variants of one improvisation stay together",
            "generated_by_muspector": False,
        },
        "counts": {
            "files": len(items),
            "performances": len(groups),
            "files_per_performance": dict(sorted(group_sizes.items())),
            "splits": split_counts,
            "split_roles": split_roles,
            "split_clean_references": split_clean_references,
            "split_domains": split_domains,
            "audio_contracts": dict(sorted(audio_contracts.items())),
        },
        "inventory_sha256": inventory.hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
