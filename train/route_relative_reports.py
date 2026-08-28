"""Build an auditable label-routed Inspector report from completed runs.

This performs no training and does not select thresholds.  It routes Drive and
Delay from one frozen Clean-relative run and Reverb from another, then combines
their already-recorded external-development predictions by filename.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ("drive", "delay", "reverb")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def duration_report(run: dict, seconds: int) -> dict:
    return run["external_reference_duration_ablation"][str(seconds)]


def routed_external(primary: dict, reverb: dict) -> dict:
    primary_rows = {row["file"]: row for row in primary["results"]}
    reverb_rows = {row["file"]: row for row in reverb["results"]}
    if set(primary_rows) != set(reverb_rows):
        raise RuntimeError("external result filenames differ between routed runs")

    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    exact = 0
    rows = []
    for name in sorted(primary_rows):
        expected = set(primary_rows[name]["expected"])
        predicted = set(primary_rows[name]["predicted"]) & {"drive", "delay"}
        predicted |= set(reverb_rows[name]["predicted"]) & {"reverb"}
        exact += predicted == expected
        for label in LABELS:
            counts[label]["tp"] += int(label in expected and label in predicted)
            counts[label]["fp"] += int(label not in expected and label in predicted)
            counts[label]["fn"] += int(label in expected and label not in predicted)
        rows.append(
            {
                "file": name,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
            }
        )

    classes = {}
    for label, count in counts.items():
        precision_denominator = count["tp"] + count["fp"]
        recall_denominator = count["tp"] + count["fn"]
        precision = count["tp"] / precision_denominator if precision_denominator else 0.0
        recall = count["tp"] / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **count,
        }

    clean_fp_upper_bound = min(
        1.0,
        float(primary["clean_false_positive"])
        + float(reverb["clean_false_positive"]),
    )
    return {
        "samples": len(rows),
        "exact_match": exact / len(rows),
        "macro_f1": sum(classes[label]["f1"] for label in LABELS) / len(LABELS),
        "clean_false_positive_upper_bound": clean_fp_upper_bound,
        "classes": classes,
        "results": rows,
    }


def clean_loo(report: dict, windows_per_file: int) -> float:
    for config in report["configurations"]:
        if config["windows_per_file"] == windows_per_file:
            return float(config["strict_clean_leave_one_out"]["false_positive_rate"])
    raise RuntimeError(f"missing {windows_per_file}-window Clean LOO configuration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--reverb-run", type=Path, required=True)
    parser.add_argument("--primary-seconds", type=int, default=5)
    parser.add_argument("--reverb-seconds", type=int, default=10)
    parser.add_argument("--primary-multiclean", type=Path, required=True)
    parser.add_argument("--reverb-multiclean", type=Path, required=True)
    parser.add_argument("--multiclean-windows-per-file", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_run = load(args.primary_run / "metrics.json")
    reverb_run = load(args.reverb_run / "metrics.json")
    primary_external = duration_report(primary_run, args.primary_seconds)
    reverb_external = duration_report(reverb_run, args.reverb_seconds)
    external = routed_external(primary_external, reverb_external)

    primary_clean_fp = clean_loo(
        load(args.primary_multiclean / "metrics.json"),
        args.multiclean_windows_per_file,
    )
    reverb_clean_fp = clean_loo(
        load(args.reverb_multiclean / "metrics.json"),
        args.multiclean_windows_per_file,
    )
    multiclean_fp_upper_bound = min(1.0, primary_clean_fp + reverb_clean_fp)

    internal = {
        "clean_false_positive_upper_bound": min(
            1.0,
            float(primary_run["test"]["clean_false_positive"])
            + float(reverb_run["test"]["clean_false_positive"]),
        ),
        "drive": primary_run["test"]["drive"],
        "delay": primary_run["test"]["delay"],
        "reverb": reverb_run["test"]["reverb"],
    }
    failures = []
    if internal["clean_false_positive_upper_bound"] > 0.05:
        failures.append("internal routed Clean FP upper bound exceeds 5%")
    if external["clean_false_positive_upper_bound"] > 0.05:
        failures.append("external held-out Clean FP upper bound exceeds 5%")
    if multiclean_fp_upper_bound > 0.05:
        failures.append("multi-recording Clean LOO FP upper bound exceeds 5%")
    for label in LABELS:
        if internal[label]["recall"] < 0.80:
            failures.append(f"internal {label} recall is below 80%")
        if external["classes"][label]["recall"] < 0.80:
            failures.append(f"external {label} recall is below 80%")
        if external["classes"][label]["precision"] < 0.50:
            failures.append(f"external {label} precision is below 50%")

    report = {
        "experiment": args.output.name,
        "architecture": {
            "routing": {
                "drive": f"{args.primary_run.name} Clean-relative pair",
                "delay": f"{args.primary_run.name} Clean-relative pair",
                "reverb": f"{args.reverb_run.name} Clean-relative pair",
            },
            "encoders": 2,
            "encoder_parameters_total": 1_401_318,
            "relative_head_parameters_total": 697_862,
            "user_gradient_updates": 0,
            "minimum_clean_reference_seconds": max(
                args.primary_seconds, args.reverb_seconds
            ),
        },
        "training": {"new_gradient_updates": 0},
        "internal_routed_test": internal,
        "external_development": external,
        "multi_recording_clean_loo": {
            "recordings": 6,
            "windows_per_file": args.multiclean_windows_per_file,
            "support_seconds": 6 * args.multiclean_windows_per_file * 5,
            "primary_false_positive": primary_clean_fp,
            "reverb_false_positive": reverb_clean_fp,
            "routed_upper_bound": multiclean_fp_upper_bound,
        },
        "quality_gate": {
            "passed": not failures,
            "failures": failures,
            "requirements": {
                "clean_false_positive_max": 0.05,
                "per_class_recall_min": 0.80,
                "external_per_class_precision_min": 0.50,
            },
        },
        "artifacts": {
            "drive_delay_encoder": str(
                (args.primary_run / "relative-encoder.onnx").resolve()
            ),
            "drive_delay_head": str(
                (args.primary_run / "relative-head.onnx").resolve()
            ),
            "reverb_encoder": str(
                (args.reverb_run / "relative-encoder.onnx").resolve()
            ),
            "reverb_head": str((args.reverb_run / "relative-head.onnx").resolve()),
        },
        "data_policy": {
            "query_labels_used_for_routing_selection": True,
            "external_role": "development only; not an unbiased final hardware test",
            "restricted_dataset_weights": False,
            "primary_backbone_renderer": (
                "Spotify Pedalboard 0.9.24 plus local DSP; research training-time only"
            ),
        },
        "limitations": [
            "The external effect recordings are a development set used to select routing.",
            "The external Clean check uses held-out windows from the reference recording.",
            "The six-recording Clean check is independent leave-one-recording-out, but has no matching wet recordings.",
            "A fully untouched device-disjoint hardware wet set remains required for a final product claim.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    routing = {
        "schema": 1,
        "reference_seconds": max(args.primary_seconds, args.reverb_seconds),
        "branches": {
            "drive_delay": {
                "labels": ["drive", "delay"],
                "encoder": report["artifacts"]["drive_delay_encoder"],
                "head": report["artifacts"]["drive_delay_head"],
                "calibration": str((args.primary_run / "calibration.json").resolve()),
                "profile_seconds": args.primary_seconds,
            },
            "reverb": {
                "labels": ["reverb"],
                "encoder": report["artifacts"]["reverb_encoder"],
                "head": report["artifacts"]["reverb_head"],
                "calibration": str((args.reverb_run / "calibration.json").resolve()),
                "profile_seconds": args.reverb_seconds,
            },
        },
    }
    (args.output / "routing.json").write_text(
        json.dumps(routing, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
