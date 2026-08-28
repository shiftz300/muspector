#!/usr/bin/env python3
"""Evaluate frozen blind-detector artifacts on strictly external audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile
import torch
from scipy.signal import resample_poly

from model import LABELS, RATE, SAMPLES, frontend


ROOT = Path(__file__).resolve().parents[1]
STEP = SAMPLES // 2
ALIASES = {
    "drive": ("drive", "distortion", "fuzz", "rat", "muff"),
    "delay": ("delay", "echo"),
    "reverb": ("ambience", "dream", "reverb", "room", "hall", "plate"),
}


def expected(path: Path) -> np.ndarray:
    name = path.stem.lower()
    return np.asarray(
        [float(any(alias in name for alias in ALIASES[label])) for label in LABELS],
        dtype=np.float32,
    )


def waveform(path: Path) -> np.ndarray:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        divisor = np.gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor).astype(np.float32)
    return np.nan_to_num(audio, copy=False).clip(-4.0, 4.0)


def windows(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) <= SAMPLES:
        return [np.pad(audio, (0, max(0, SAMPLES - len(audio))))[:SAMPLES].astype(np.float32)]
    starts = list(range(0, len(audio) - SAMPLES + 1, STEP))
    final = len(audio) - SAMPLES
    if starts[-1] != final:
        starts.append(final)
    return [np.asarray(audio[start : start + SAMPLES], dtype=np.float32) for start in starts]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def probabilities(
    session: onnxruntime.InferenceSession,
    audio: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for value in windows(audio):
        mel = frontend(torch.from_numpy(value[None, :])).numpy()
        logits = session.run(None, {"mel": mel})[0][0]
        values.append(sigmoid(logits * scales + biases))
    matrix = np.stack(values)
    count = min(2, len(matrix))
    top = np.partition(matrix, len(matrix) - count, axis=0)[-count:]
    return top.mean(axis=0), matrix.max(axis=0)


def rate(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def score(expected_values: np.ndarray, predicted: np.ndarray) -> dict:
    classes = {}
    f1s = []
    for index, name in enumerate(LABELS):
        truth = expected_values[:, index].astype(bool)
        actual = predicted[:, index].astype(bool)
        true_positive = np.logical_and(truth, actual).sum()
        false_positive = np.logical_and(~truth, actual).sum()
        false_negative = np.logical_and(truth, ~actual).sum()
        precision = (
            float(true_positive / (true_positive + false_positive))
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            float(true_positive / (true_positive + false_negative))
            if true_positive + false_negative
            else 0.0
        )
        value = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(value)
        classes[name] = {"precision": precision, "recall": recall, "f1": value}
    clean = expected_values.sum(axis=1) == 0
    return {
        "classes": classes,
        "macro_f1": float(np.mean(f1s)),
        "exact_match": rate((expected_values.astype(bool) == predicted).all(axis=1)),
        "clean_false_positive": rate(predicted[clean].any(axis=1)),
        "files": int(len(expected_values)),
    }


def quality_gate(metrics: dict) -> dict:
    failures = []
    if metrics["clean_false_positive"] > 0.05:
        failures.append("external clean false-positive rate exceeds 5%")
    for name in LABELS:
        if metrics["classes"][name]["recall"] < 0.80:
            failures.append(f"external {name} recall is below 80%")
    return {
        "passed": not failures,
        "requirements": {
            "clean_false_positive_max": 0.05,
            "per_class_recall_min": 0.80,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "train" / "runs" / "blind" / "blind.onnx"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "train" / "runs" / "blind" / "calibration.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "train" / "runs" / "hardware-external-dev.json",
    )
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.data.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".aif", ".aiff"}
    )
    if not files:
        parser.error(f"no supported audio files found in {args.data}")
    calibration = json.loads(args.calibration.read_text())
    scales = np.asarray([calibration["scale"][label] for label in LABELS], dtype=np.float32)
    biases = np.asarray([calibration["bias"][label] for label in LABELS], dtype=np.float32)
    threshold = np.asarray(
        [calibration["threshold"][label] for label in LABELS], dtype=np.float32
    )
    session = onnxruntime.InferenceSession(str(args.model))
    rows = []
    truths, predictions = [], []
    for path in files:
        truth = expected(path)
        top2, peak = probabilities(session, waveform(path), scales, biases)
        predicted = top2 >= threshold
        truths.append(truth)
        predictions.append(predicted)
        rows.append(
            {
                "file": path.name,
                "expected": [LABELS[index] for index in np.flatnonzero(truth)],
                "predicted": [LABELS[index] for index in np.flatnonzero(predicted)],
                "top2_mean": dict(zip(LABELS, map(float, top2))),
                "peak": dict(zip(LABELS, map(float, peak))),
            }
        )
    scored = score(np.stack(truths), np.stack(predictions))
    report = {
        "role": "external-development; never used for training, early stopping, or calibration",
        "data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "aggregation": "mean of top two calibrated five-second windows; 2.5-second hop",
        "thresholds": dict(zip(LABELS, map(float, threshold))),
        "metrics": scored,
        "quality_gate": quality_gate(scored),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
