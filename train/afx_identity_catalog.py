#!/usr/bin/env python3
"""Fit and evaluate a global pedal catalog on frozen AFx-Rep embeddings.

This is the quality ceiling for compact identity distillation.  The command
never updates the AFx-Rep encoder and never trains on end-user audio.  The
developer hardware folder is used only to select and report a development
threshold, so a separate device-disjoint final test is still required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from afx_identity_teacher import UNKNOWN as SOURCE_UNKNOWN
from compact_identity_distill import SEED, examples, load_teacher_targets, request_key
from compact_identity_verifier import (
    CATALOG,
    CATEGORY_CATALOG,
    UNKNOWN,
    CatalogHead,
    Verifier,
    aggregate_recordings,
    hardware_report,
    infer_catalog,
    infer_verifier,
    metrics,
    scores,
    select_threshold,
    train_catalog,
    train_verifier,
)
from layout import PEDAL_IDENTITY_CACHE, PEDAL_IDENTITY_RUN


def hardware_scores(
    embeddings: np.ndarray,
    request_index: dict[tuple[Path, int, int], int],
    catalog: CatalogHead,
    verifier: Verifier,
    directory: Path,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths = [
        path
        for path in sorted(directory.glob("*.wav"))
        if any(token in path.stem.lower() for token in ("drive", "fuzz", "rat"))
    ]
    combined = []
    truth = []
    for path in paths:
        indices = np.asarray(
            [request_index[(path, segment, 3)] for segment in range(3)]
        )
        identity = infer_catalog(catalog, embeddings[indices])
        known = infer_verifier(verifier, embeddings[indices])
        combined.append(scores(identity, known).mean(axis=0))
        truth.append(CATALOG.index("RAT") if "rat" in path.stem.lower() else UNKNOWN)
    return paths, np.stack(combined), np.asarray(truth)


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=PEDAL_IDENTITY_CACHE / "afx-rep-teacher",
    )
    parser.add_argument(
        "--output", type=Path, default=PEDAL_IDENTITY_RUN / "afx-rep-catalog"
    )
    parser.add_argument(
        "--development", type=Path, default=Path.home() / "Downloads/test"
    )
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows, inventory = examples(include_remfx=True)
    requests = {(row.path, row.segment, row.segments) for row in rows}
    hardware_paths = [
        path
        for path in args.development.glob("*.wav")
        if any(token in path.stem.lower() for token in ("drive", "fuzz", "rat"))
    ]
    requests.update(
        (path, segment, 3) for path in hardware_paths for segment in range(3)
    )
    requests = sorted(requests, key=request_key)
    request_index = {request: index for index, request in enumerate(requests)}
    embeddings = load_teacher_targets(requests, args.teacher_cache)
    row_indices = np.asarray(
        [request_index[(row.path, row.segment, row.segments)] for row in rows]
    )
    row_embeddings = embeddings[row_indices]
    labels = np.asarray(
        [
            CATALOG.index(CATEGORY_CATALOG[row.category])
            if row.category in CATEGORY_CATALOG
            else UNKNOWN
            if row.label == SOURCE_UNKNOWN
            else row.label
            for row in rows
        ],
        dtype=np.int64,
    )
    splits = np.asarray([row.split for row in rows])
    categories = np.asarray([row.category for row in rows])

    args.output.mkdir(parents=True, exist_ok=True)
    catalog, catalog_training = train_catalog(
        row_embeddings, labels, splits, args.epochs, args.output
    )
    verifier, verifier_training = train_verifier(
        row_embeddings, labels, splits, categories, args.epochs, args.output
    )
    row_scores = scores(
        infer_catalog(catalog, row_embeddings),
        infer_verifier(verifier, row_embeddings),
    )

    calibrate = splits == "calibrate"
    calibration_scores, calibration_truth, _ = aggregate_recordings(
        [row for row, selected in zip(rows, calibrate) if selected],
        row_scores[calibrate],
        labels[calibrate],
        categories[calibrate],
    )
    public_threshold = select_threshold(calibration_scores, calibration_truth, 0.05)
    paths, development_scores, development_truth = hardware_scores(
        embeddings, request_index, catalog, verifier, args.development
    )
    development_threshold = select_threshold(
        development_scores, development_truth, 0.20
    )
    test = splits == "test"
    test_scores, test_truth, test_categories = aggregate_recordings(
        [row for row, selected in zip(rows, test) if selected],
        row_scores[test],
        labels[test],
        categories[test],
    )
    reports = {
        "public_calibration": {
            "threshold": public_threshold,
            "public_test": metrics(
                test_scores, test_truth, test_categories, public_threshold
            ),
            "hardware_development": hardware_report(
                paths, development_scores, development_truth, public_threshold
            ),
        },
        "hardware_development_calibration": {
            "threshold": development_threshold,
            "uses_hardware_labels": True,
            "public_test": metrics(
                test_scores, test_truth, test_categories, development_threshold
            ),
            "hardware_development": hardware_report(
                paths, development_scores, development_truth, development_threshold
            ),
        },
    }
    selected = reports["hardware_development_calibration"]
    public = selected["public_test"]
    hardware = selected["hardware_development"]
    failures = []
    if hardware["rat_recall"] < 0.5:
        failures.append("hardware-development RAT recall is below 50%")
    if hardware["noncatalog_false_accept"] > 0.2:
        failures.append(
            "hardware-development non-catalog false accept exceeds 20%"
        )
    if public["negative_false_accept_rate"] > 0.05:
        failures.append("public negative false accept exceeds 5%")
    for category, false_accept in public[
        "negative_category_false_accept"
    ].items():
        if false_accept > 0.20:
            failures.append(category + " false accept exceeds 20%")
    for name, value in public["per_class"].items():
        if value["recall"] < 0.65:
            failures.append(name + " public recall is below 65%")

    payload = {
        "experiment": "afx-rep-global-pedal-catalog",
        "architecture": {
            "encoder": "frozen ST-ITO AFx-Rep Cnn14",
            "embedding": 512,
            "identity_head": "512-128 pedal-catalog MLP",
            "catalog": list(CATALOG),
            "verifier": "independent 512-64 knownness MLP",
            "user_gradient_updates": 0,
        },
        "data": {
            "examples": len(rows),
            "tone_twist": inventory,
            "license_scope": (
                "non-commercial: ToneTwist CC BY-NC 4.0 and RemFX Zenodo cc-nc"
            ),
            "split_counts": {
                name: int((splits == name).sum())
                for name in ("train", "valid", "calibrate", "test")
            },
            "external_test_folder_role": (
                "labeled hardware development threshold; not model training and not a final test"
            ),
        },
        "training": {
            "catalog": catalog_training,
            "verifier": verifier_training,
        },
        "reports": reports,
        "development_gate": {"passed": not failures, "failures": failures},
        "integration_eligible": not failures,
        "noncommercial_release_eligible": not failures,
        "commercial_release_eligible": False,
        "release_reason": (
            "the fitted heads use CC BY-NC 4.0 ToneTwist and Zenodo cc-nc "
            "RemFX recordings; distribution must remain non-commercial with "
            "attribution, and the limited test support is an accuracy caveat "
            "rather than a license gate"
        ),
        "artifacts": {
            "catalog": {
                "path": str((args.output / "compact-drive-catalog.pt").resolve()),
                "sha256": checkpoint_hash(args.output / "compact-drive-catalog.pt"),
            },
            "verifier": {
                "path": str((args.output / "compact-drive-verifier.pt").resolve()),
                "sha256": checkpoint_hash(args.output / "compact-drive-verifier.pt"),
            },
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "calibration.json").write_text(
        json.dumps(
            {
                "labels": list(CATALOG),
                "threshold": development_threshold,
                "development_calibrated": True,
            },
            indent=2,
        )
        + "\n"
    )
    summary = {
        "threshold": development_threshold,
        "public": public,
        "hardware": hardware,
        "gate": payload["development_gate"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
