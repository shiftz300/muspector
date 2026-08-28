"""Evaluate a frozen Inspector with a multi-recording Clean device profile.

This experiment performs no gradient updates and never uses query labels for
profile construction or threshold selection.  Several random Clean recordings
cover pickup/tone states; each strict Clean fold holds out one whole recording.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from detect import calibrated, metrics
from evaluate import expected as external_expected
from model import LABELS, Detector
from layout import (
    REVERB_BLIND_RUN,
    REVERB_CLEAN_AUDIT_RUN,
    REVERB_ENCODER_RUN,
    REVERB_PAIR_RUN,
)
from relative import (
    RelativeHead,
    encode_relative_windows,
    fused_with_profile,
    infer,
    target_device,
)


ROOT = Path(__file__).resolve().parents[1]


def take_windows(values: np.ndarray, count: int) -> np.ndarray:
    """Spread a small window budget across the whole recording."""

    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[indices]


def profile(values: list[np.ndarray], windows_per_file: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.concatenate(
        [take_windows(value, windows_per_file) for value in values], axis=0
    )
    return selected, selected.mean(axis=0), selected.std(axis=0) + 1.0e-4


def clean_threshold(
    head: RelativeHead,
    support: np.ndarray,
    mean: np.ndarray,
    deviation: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    floors: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = calibrated(
        infer(head, fused_with_profile(support, mean, deviation), device),
        scales,
        biases,
    )
    threshold = np.minimum(
        np.maximum(probabilities.max(axis=0) + 0.02, floors), 0.95
    )
    return threshold, probabilities


def file_score(
    model: Detector,
    head: RelativeHead,
    path: Path,
    mean: np.ndarray,
    deviation: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    threshold: np.ndarray,
    blind_scales: np.ndarray,
    blind_biases: np.ndarray,
    blind_threshold: np.ndarray,
    delay_route: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embedding, blind_logits = encode_relative_windows(
        model, path, "embedding-logits", device
    )
    probabilities = calibrated(
        infer(head, fused_with_profile(embedding, mean, deviation), device),
        scales,
        biases,
    )
    strongest = min(2, len(probabilities))
    score = np.partition(probabilities, len(probabilities) - strongest, axis=0)[
        -strongest:
    ].mean(axis=0)
    blind_probabilities = calibrated(blind_logits, blind_scales, blind_biases)
    blind_score = np.partition(
        blind_probabilities, len(blind_probabilities) - strongest, axis=0
    )[-strongest:].mean(axis=0)
    predicted = score >= threshold
    if delay_route == "blind":
        predicted[1] = blind_score[1] >= blind_threshold[1]
    return score, predicted, blind_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-directory",
        type=Path,
        default=Path.home() / "Downloads/clean test",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path.home() / "Downloads/test",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=REVERB_PAIR_RUN,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REVERB_ENCODER_RUN / "backbone.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REVERB_CLEAN_AUDIT_RUN,
    )
    parser.add_argument(
        "--blind-calibration",
        type=Path,
        default=REVERB_BLIND_RUN / "calibration.json",
    )
    parser.add_argument("--window-counts", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument(
        "--delay-route",
        choices=("blind", "pair"),
        default="blind",
        help="route Delay through the frozen blind branch or the Clean-relative pair head",
    )
    args = parser.parse_args()

    clean_paths = sorted(args.clean_directory.glob("*.wav"))
    external_paths = sorted(args.external.glob("*.wav"))
    if len(clean_paths) < 2:
        parser.error("multi-Clean evaluation needs at least two Clean recordings")
    if not external_paths:
        parser.error("external directory contains no WAV files")

    calibration_payload = json.loads((args.run / "calibration.json").read_text())
    scales = np.asarray(
        [calibration_payload["scale"][label] for label in LABELS], dtype=np.float32
    )
    biases = np.asarray(
        [calibration_payload["bias"][label] for label in LABELS], dtype=np.float32
    )
    floors = np.asarray([0.05, 0.55, 0.59], dtype=np.float32)
    blind_payload = json.loads(args.blind_calibration.read_text())
    blind_scales = np.asarray(
        [blind_payload["scale"][label] for label in LABELS], dtype=np.float32
    )
    blind_biases = np.asarray(
        [blind_payload["bias"][label] for label in LABELS], dtype=np.float32
    )
    blind_threshold = np.asarray(
        [blind_payload["threshold"][label] for label in LABELS], dtype=np.float32
    )

    device = target_device()
    model = Detector(stem_stride=1).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    head = RelativeHead(1295).to(device)
    head.load_state_dict(
        torch.load(args.run / "relative-head.pt", map_location=device, weights_only=True)
    )
    head.eval()

    started = time.perf_counter()
    clean_encoded = [
        encode_relative_windows(model, path, "embedding-logits", device)
        for path in clean_paths
    ]
    clean_embeddings = [value[0] for value in clean_encoded]
    encode_seconds = time.perf_counter() - started

    configurations = []
    for windows_per_file in args.window_counts:
        support, mean, deviation = profile(clean_embeddings, windows_per_file)
        threshold, support_probabilities = clean_threshold(
            head,
            support,
            mean,
            deviation,
            scales,
            biases,
            floors,
            device,
        )
        blind_support = np.concatenate(
            [
                take_windows(value[1], windows_per_file)
                for value in clean_encoded
            ],
            axis=0,
        )
        blind_support_probabilities = calibrated(
            blind_support, blind_scales, blind_biases
        )
        user_blind_threshold = np.minimum(
            np.maximum(
                blind_support_probabilities.max(axis=0) + 0.02,
                blind_threshold,
            ),
            0.95,
        )

        clean_rows = []
        clean_predictions = []
        for index, path in enumerate(clean_paths):
            fold_values = [
                value for other, value in enumerate(clean_embeddings) if other != index
            ]
            fold_support, fold_mean, fold_deviation = profile(
                fold_values, windows_per_file
            )
            fold_threshold, _ = clean_threshold(
                head,
                fold_support,
                fold_mean,
                fold_deviation,
                scales,
                biases,
                floors,
                device,
            )
            fold_blind_support = np.concatenate(
                [
                    take_windows(value[1], windows_per_file)
                    for other, value in enumerate(clean_encoded)
                    if other != index
                ],
                axis=0,
            )
            fold_blind_threshold = np.minimum(
                np.maximum(
                    calibrated(
                        fold_blind_support, blind_scales, blind_biases
                    ).max(axis=0)
                    + 0.02,
                    blind_threshold,
                ),
                0.95,
            )
            held_out = take_windows(clean_embeddings[index], windows_per_file)
            held_out_probabilities = calibrated(
                infer(
                    head,
                    fused_with_profile(
                        held_out, fold_mean, fold_deviation
                    ),
                    device,
                ),
                scales,
                biases,
            )
            strongest = min(2, len(held_out_probabilities))
            score = np.partition(
                held_out_probabilities,
                len(held_out_probabilities) - strongest,
                axis=0,
            )[-strongest:].mean(axis=0)
            predicted = score >= fold_threshold
            held_out_blind = calibrated(
                take_windows(clean_encoded[index][1], windows_per_file),
                blind_scales,
                blind_biases,
            )
            blind_score = np.partition(
                held_out_blind,
                len(held_out_blind) - strongest,
                axis=0,
            )[-strongest:].mean(axis=0)
            if args.delay_route == "blind":
                predicted[1] = blind_score[1] >= fold_blind_threshold[1]
            clean_predictions.append(predicted)
            clean_rows.append(
                {
                    "file": path.name,
                    "predicted": [
                        LABELS[i] for i in np.flatnonzero(predicted)
                    ],
                    "score": dict(zip(LABELS, map(float, score))),
                    "blind_score": dict(zip(LABELS, map(float, blind_score))),
                    "threshold": dict(zip(LABELS, map(float, fold_threshold))),
                    "blind_threshold": dict(
                        zip(LABELS, map(float, fold_blind_threshold))
                    ),
                }
            )

        rows = []
        truths = []
        predictions = []
        query_started = time.perf_counter()
        for path in external_paths:
            score, predicted, blind_score = file_score(
                model,
                head,
                path,
                mean,
                deviation,
                scales,
                biases,
                threshold,
                blind_scales,
                blind_biases,
                user_blind_threshold,
                args.delay_route,
                device,
            )
            truth = external_expected(path)
            truths.append(truth)
            predictions.append(predicted)
            rows.append(
                {
                    "file": path.name,
                    "expected": [LABELS[i] for i in np.flatnonzero(truth)],
                    "predicted": [LABELS[i] for i in np.flatnonzero(predicted)],
                    "score": dict(zip(LABELS, map(float, score))),
                    "blind_score": dict(zip(LABELS, map(float, blind_score))),
                }
            )
        truth_values = np.stack(truths)
        prediction_values = np.stack(predictions)
        external_metrics = metrics(
            truth_values,
            prediction_values.astype(np.float32),
            np.full(len(LABELS), 0.5, dtype=np.float32),
        )
        clean_prediction_values = np.stack(clean_predictions)
        clean_fp = float(clean_prediction_values.any(axis=1).mean())
        external_metrics["clean_false_positive"] = float(
            prediction_values[~truth_values.any(axis=1)].any(axis=1).mean()
        )
        failures = []
        if clean_fp > 0.05:
            failures.append("strict leave-one-recording-out Clean FP exceeds 5%")
        if external_metrics["clean_false_positive"] > 0.05:
            failures.append("independent external Clean FP exceeds 5%")
        for label in LABELS:
            if external_metrics[label]["recall"] < 0.80:
                failures.append(f"external {label} recall is below 80%")
            if external_metrics[label]["precision"] < 0.50:
                failures.append(f"external {label} precision is below 50%")

        configurations.append(
            {
                "windows_per_file": windows_per_file,
                "support_windows": int(len(support)),
                "support_seconds": int(len(support) * 5),
                "threshold": dict(zip(LABELS, map(float, threshold))),
                "blind_threshold": dict(
                    zip(LABELS, map(float, user_blind_threshold))
                ),
                "blind_support_probability_max": dict(
                    zip(
                        LABELS,
                        map(float, blind_support_probabilities.max(axis=0)),
                    )
                ),
                "support_probability_max": dict(
                    zip(LABELS, map(float, support_probabilities.max(axis=0)))
                ),
                "strict_clean_leave_one_out": {
                    "files": len(clean_paths),
                    "false_positive_rate": clean_fp,
                    "per_class_false_positive_rate": dict(
                        zip(
                            LABELS,
                            map(float, clean_prediction_values.mean(axis=0)),
                        )
                    ),
                    "results": clean_rows,
                },
                "external": {
                    "metrics": external_metrics,
                    "query_seconds": time.perf_counter() - query_started,
                    "results": rows,
                },
                "quality_gate": {"passed": not failures, "failures": failures},
                "profile": {
                    "mean": mean.tolist(),
                    "standard_deviation": deviation.tolist(),
                },
            }
        )

    # Configuration selection is based only on strict Clean LOO FP and burden;
    # external labels are retained as development evidence, never as selectors.
    best = min(
        configurations,
        key=lambda item: (
            item["strict_clean_leave_one_out"]["false_positive_rate"],
            item["support_seconds"],
        ),
    )
    report = {
        "experiment": args.output.name,
        "hypothesis": (
            "A multi-recording Clean profile covering pickup and tone states reduces "
            "capture-domain false positives without gradient updates or aligned playing."
        ),
        "architecture": {
            "frozen_encoder": str(args.checkpoint.resolve()),
            "frozen_head": str((args.run / "relative-head.pt").resolve()),
            "feature_mode": "embedding-logits",
            "gradient_updates": 0,
            "routing": (
                "pair Drive/Reverb; frozen blind Delay"
                if args.delay_route == "blind"
                else "pair Drive/Delay/Reverb"
            ),
            "blind_calibration": str(args.blind_calibration.resolve()),
        },
        "data_policy": {
            "clean_support": [str(path.resolve()) for path in clean_paths],
            "external_development": str(args.external.resolve()),
            "query_labels_used_for_selection": False,
            "derived_audio": False,
        },
        "timing": {"clean_encode_seconds": encode_seconds, "device": str(device)},
        "selection_policy": "minimum strict Clean LOO FP, then minimum support seconds",
        "selected_windows_per_file": best["windows_per_file"],
        "selected_quality_gate": best["quality_gate"],
        "configurations": configurations,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output / "clean-profile.npz",
        mean=np.asarray(best["profile"]["mean"], dtype=np.float32),
        standard_deviation=np.asarray(
            best["profile"]["standard_deviation"], dtype=np.float32
        ),
        threshold=np.asarray(
            [best["threshold"][label] for label in LABELS], dtype=np.float32
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
