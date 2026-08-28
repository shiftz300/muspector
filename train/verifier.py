#!/usr/bin/env python3
"""Train license-clean temporal verifiers for pair-model Delay/Reverb candidates.

The frozen public-RIR pair model remains the candidate generator. This script trains
one tiny shared MLP from deterministic 65-value log-Mel temporal statistics and
selects verifier thresholds only on the grouped calibration split.  The
external development directory is evaluated after selection and never updates
weights or thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from data import (
    add_rir_reverb,
    audit,
    discover,
    discover_aachen_rirs,
    discover_guitar_effect_chains,
    eligible_clean,
    guitar_effect_chain_is_clean,
)
from detect import calibrated, metrics
from evaluate import expected as external_expected
from model import LABELS, Detector
from layout import CORPUS, REVERB_CACHE, REVERB_ENCODER_RUN, REVERB_PAIR_RUN
from reference import real_only_split
from relative import (
    PHYSICS,
    RelativeHead,
    actual_clean_mask,
    encode_relative_windows,
    episode_features,
    fused_with_profile,
    infer,
    reference_window_count,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260828
DELAY = LABELS.index("delay")
REVERB = LABELS.index("reverb")
VERIFIED = (DELAY, REVERB)
VERIFIER_INPUT = PHYSICS * 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoded(cache: Path, split: str) -> dict[str, np.ndarray]:
    paths = sorted(cache.glob(f"{split}-relative-*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {split} encoded cache below {cache}, found {paths}")
    payload = np.load(paths[0])
    return {key: payload[key] for key in payload.files}


class TemporalVerifier(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("deviation", torch.from_numpy(deviation.astype(np.float32)))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(VERIFIER_INPUT, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(VERIFIED)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((value - self.mean) / self.deviation, -6.0, 6.0)
        return self.network(normalized)


def verifier_logits(model: TemporalVerifier, values: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            result.append(model(torch.from_numpy(values[start : start + 1024])).numpy())
    return np.concatenate(result)


def fit_calibration(expected: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales, biases = [], []
    for index in range(len(VERIFIED)):
        values = torch.tensor(logits[:, index], dtype=torch.float64)
        labels = torch.tensor(expected[:, index], dtype=torch.float64)
        log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (log_scale, bias), lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                values * log_scale.exp() + bias, labels
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        scales.append(float(log_scale.detach().exp().clamp(0.05, 20.0)))
        biases.append(float(bias.detach().clamp(-20.0, 20.0)))
    return np.asarray(scales, dtype=np.float32), np.asarray(biases, dtype=np.float32)


def verifier_probabilities(
    logits: np.ndarray, scales: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    value = logits * scales + biases
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def train(
    model: TemporalVerifier,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    valid_values: np.ndarray,
    valid_labels: np.ndarray,
    output: Path,
    epochs: int,
    patience: int,
) -> tuple[int, float, float]:
    x = torch.from_numpy(train_values)
    y = torch.from_numpy(train_labels[:, VERIFIED])
    vx = torch.from_numpy(valid_values)
    vy = torch.from_numpy(valid_labels[:, VERIFIED])
    positives = y.sum(dim=0)
    positive_weight = torch.clamp((len(y) - positives) / positives.clamp_min(1.0), 1.0, 8.0)
    # Drive-only negatives are the known failure mode.  Delay-only and Reverb-
    # only recordings are also strong cross-family negatives for the other
    # verifier, while ordinary clean/modulation negatives keep unit weight.
    weights = torch.ones_like(y)
    drive = torch.from_numpy(train_labels[:, 0] > 0.5)
    delay = torch.from_numpy(train_labels[:, DELAY] > 0.5)
    reverb = torch.from_numpy(train_labels[:, REVERB] > 0.5)
    weights[torch.logical_and(drive, ~delay), 0] = 2.5
    weights[torch.logical_and(drive, ~reverb), 1] = 3.0
    weights[torch.logical_and(reverb, ~delay), 0] = torch.maximum(
        weights[torch.logical_and(reverb, ~delay), 0], torch.tensor(1.5)
    )
    weights[torch.logical_and(delay, ~reverb), 1] = torch.maximum(
        weights[torch.logical_and(delay, ~reverb), 1], torch.tensor(1.5)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    best, stale, completed = float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(order), 512):
            selected = order[start : start + 512]
            logits = model(x[selected])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y[selected],
                pos_weight=positive_weight,
                weight=weights[selected],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    model(vx), vy, pos_weight=positive_weight
                )
            )
        completed = epoch + 1
        print("verifier_epoch", completed, "valid_loss", valid_loss)
        if valid_loss < best - 1.0e-5:
            best, stale = valid_loss, 0
            torch.save(model.state_dict(), output / "temporal-verifier.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(
        torch.load(output / "temporal-verifier.pt", map_location="cpu", weights_only=True)
    )
    return completed, best, time.perf_counter() - started


def select_thresholds(
    expected: np.ndarray,
    pair_predicted: np.ndarray,
    verifier: np.ndarray,
    recall_floor: float = 0.80,
) -> np.ndarray:
    selected = []
    for column, label_index in enumerate(VERIFIED):
        truth = expected[:, label_index] > 0.5
        candidates = []
        for threshold in np.linspace(0.0, 1.0, 1001):
            predicted = np.logical_and(pair_predicted[:, label_index], verifier[:, column] >= threshold)
            recall = float(predicted[truth].mean()) if truth.any() else 0.0
            if recall >= recall_floor:
                candidates.append(float(threshold))
        selected.append(max(candidates) if candidates else 0.0)
    return np.asarray(selected, dtype=np.float32)


def gated_predictions(
    pair_probabilities: np.ndarray,
    pair_thresholds: np.ndarray,
    verifier: np.ndarray,
    verifier_thresholds: np.ndarray,
) -> np.ndarray:
    predicted = pair_probabilities >= pair_thresholds
    for column, label_index in enumerate(VERIFIED):
        predicted[:, label_index] = np.logical_and(
            predicted[:, label_index], verifier[:, column] >= verifier_thresholds[column]
        )
    return predicted


def binary_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict:
    return metrics(
        expected,
        predicted.astype(np.float32),
        np.full(len(LABELS), 0.5, dtype=np.float32),
    )


def top_two(values: np.ndarray) -> np.ndarray:
    count = min(2, len(values))
    return np.partition(values, len(values) - count, axis=0)[-count:].mean(axis=0)


def external_report(
    encoder: Detector,
    head: RelativeHead,
    verifier: TemporalVerifier,
    reference: Path,
    directory: Path,
    pair_scales: np.ndarray,
    pair_biases: np.ndarray,
    verifier_scales: np.ndarray,
    verifier_biases: np.ndarray,
    verifier_thresholds: np.ndarray,
) -> dict:
    reference_query, _ = encode_relative_windows(
        encoder, reference, "embedding-logits", torch.device("cpu")
    )
    reference_physics, _ = encode_relative_windows(
        encoder, reference, "physics", torch.device("cpu")
    )
    count = reference_window_count(10, len(reference_query))
    selected = reference_query[:count]
    mean = selected.mean(axis=0)
    deviation = selected.std(axis=0) + 1.0e-4
    physics_selected = reference_physics[:count]
    physics_mean = physics_selected.mean(axis=0)
    physics_deviation = physics_selected.std(axis=0) + 1.0e-4
    clean_pair = calibrated(
        infer(head, fused_with_profile(selected, mean, deviation), torch.device("cpu")),
        pair_scales,
        pair_biases,
    )
    pair_thresholds = np.minimum(
        np.maximum(clean_pair.max(axis=0) + 0.02, np.asarray([0.05, 0.55, 0.59])),
        0.95,
    )
    held_query = reference_query[count + 1 :]
    held_physics = reference_physics[count + 1 :]
    if len(held_query):
        held_pair = calibrated(
            infer(head, fused_with_profile(held_query, mean, deviation), torch.device("cpu")),
            pair_scales,
            pair_biases,
        )
        held_verifier = verifier_probabilities(
            verifier_logits(
                verifier,
                fused_with_profile(held_physics, physics_mean, physics_deviation),
            ),
            verifier_scales,
            verifier_biases,
        )
        held_prediction = gated_predictions(
            held_pair, pair_thresholds, held_verifier, verifier_thresholds
        )
        clean_fp = float(held_prediction.any(axis=1).mean())
    else:
        clean_fp = None
    rows, truths, pair_predictions, predictions = [], [], [], []
    for path in sorted(directory.glob("*.wav")):
        if path.resolve() == reference.resolve():
            continue
        query, _ = encode_relative_windows(
            encoder, path, "embedding-logits", torch.device("cpu")
        )
        physics, _ = encode_relative_windows(
            encoder, path, "physics", torch.device("cpu")
        )
        pair = top_two(
            calibrated(
                infer(head, fused_with_profile(query, mean, deviation), torch.device("cpu")),
                pair_scales,
                pair_biases,
            )
        )
        verified = top_two(
            verifier_probabilities(
                verifier_logits(
                    verifier,
                    fused_with_profile(physics, physics_mean, physics_deviation),
                ),
                verifier_scales,
                verifier_biases,
            )
        )
        pair_prediction = pair >= pair_thresholds
        prediction = gated_predictions(
            pair[None], pair_thresholds, verified[None], verifier_thresholds
        )[0]
        truth = external_expected(path)
        truths.append(truth)
        pair_predictions.append(pair_prediction)
        predictions.append(prediction)
        rows.append(
            {
                "file": path.name,
                "expected": [LABELS[index] for index in np.flatnonzero(truth)],
                "pair_predicted": [LABELS[index] for index in np.flatnonzero(pair_prediction)],
                "predicted": [LABELS[index] for index in np.flatnonzero(prediction)],
                "pair_probability": dict(zip(LABELS, map(float, pair))),
                "verifier_probability": dict(
                    zip((LABELS[index] for index in VERIFIED), map(float, verified))
                ),
            }
        )
    truth = np.stack(truths)
    pair_prediction = np.stack(pair_predictions)
    prediction = np.stack(predictions)
    return {
        "role": "external development; no fitting or threshold selection",
        "reference": str(reference.resolve()),
        "queries": len(rows),
        "held_out_clean_windows": int(len(held_query)),
        "held_out_clean_false_positive": clean_fp,
        "pair_threshold": dict(zip(LABELS, map(float, pair_thresholds))),
        "pair_metrics": binary_metrics(truth, pair_prediction),
        "gated_metrics": binary_metrics(truth, prediction),
        "results": rows,
        "profile": {
            "query_mean": mean,
            "query_standard_deviation": deviation,
            "physics_mean": physics_mean,
            "physics_standard_deviation": physics_deviation,
        },
    }


def export(model: TemporalVerifier, output: Path) -> float:
    model = model.cpu().eval()
    example = torch.randn(1, VERIFIER_INPUT)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["temporal_features"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))
    expected = model(example).detach().numpy()
    actual = onnxruntime.InferenceSession(str(output)).run(
        None, {"temporal_features": example.numpy()}
    )[0]
    return float(np.max(np.abs(expected - actual)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=REVERB_CACHE
    )
    parser.add_argument(
        "--base-run", type=Path, default=REVERB_PAIR_RUN
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REVERB_ENCODER_RUN / "backbone.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "train/runs/rejected-temporal-statistics-verifier"
    )
    parser.add_argument("--data", type=Path, default=CORPUS)
    parser.add_argument("--external", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument(
        "--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)

    encoded = {name: load_encoded(args.cache, name) for name in ("train", "valid", "calibrate", "test")}
    clean, anchors = discover(args.data)
    parts = real_only_split(clean, anchors)
    chain = discover_guitar_effect_chains(args.data / "guitar-effects-chains")
    for name in parts:
        parts[name].extend(chain[name])
    clean_paths = {source.path.resolve() for source in clean if eligible_clean(source)}
    clean_paths.update(
        item.source.path.resolve()
        for items in chain.values()
        for item in items
        if guitar_effect_chain_is_clean(item.source)
    )
    parts = add_rir_reverb(
        parts, discover_aachen_rirs(args.data / "aachen-chapel-rir"), clean_paths
    )
    audit(parts)
    for name in parts:
        if len(parts[name]) != len(encoded[name]["labels"]):
            raise RuntimeError(f"{name} cache/item mismatch")
    masks = {name: actual_clean_mask(parts[name], clean_paths) for name in parts}
    pair_episode = {
        name: episode_features(
            encoded[name], masks[name], "embedding-logits", SEED + index * 10_007,
            2 if name == "train" else 1,
        )
        for index, name in enumerate(("train", "valid", "calibrate", "test"))
    }
    physics_episode = {
        name: episode_features(
            encoded[name], masks[name], "physics", SEED + index * 10_007,
            2 if name == "train" else 1,
        )
        for index, name in enumerate(("train", "valid", "calibrate", "test"))
    }
    mean = physics_episode["train"][0].mean(axis=0).astype(np.float32)
    deviation = (physics_episode["train"][0].std(axis=0) + 1.0e-4).astype(np.float32)
    verifier = TemporalVerifier(mean, deviation)
    epochs, best, seconds = train(
        verifier,
        physics_episode["train"][0],
        physics_episode["train"][1],
        physics_episode["valid"][0],
        physics_episode["valid"][1],
        args.output,
        args.epochs,
        args.patience,
    )
    head = RelativeHead(259 * 5)
    head.load_state_dict(
        torch.load(args.base_run / "relative-head.pt", map_location="cpu", weights_only=True)
    )
    head.eval()
    base_metrics = json.loads((args.base_run / "metrics.json").read_text())
    pair_scales = np.asarray(
        [base_metrics["calibration"]["scale"][label] for label in LABELS], dtype=np.float32
    )
    pair_biases = np.asarray(
        [base_metrics["calibration"]["bias"][label] for label in LABELS], dtype=np.float32
    )
    pair_thresholds = np.asarray(
        [base_metrics["calibration"]["threshold"][label] for label in LABELS], dtype=np.float32
    )
    pair_probabilities = {
        name: calibrated(
            infer(head, pair_episode[name][0], torch.device("cpu")),
            pair_scales,
            pair_biases,
        )
        for name in ("calibrate", "test")
    }
    verifier_logits_by_split = {
        name: verifier_logits(verifier, physics_episode[name][0])
        for name in ("calibrate", "test")
    }
    verifier_scales, verifier_biases = fit_calibration(
        physics_episode["calibrate"][1][:, VERIFIED],
        verifier_logits_by_split["calibrate"],
    )
    verifier_probabilities_by_split = {
        name: verifier_probabilities(values, verifier_scales, verifier_biases)
        for name, values in verifier_logits_by_split.items()
    }
    verifier_thresholds = select_thresholds(
        pair_episode["calibrate"][1],
        pair_probabilities["calibrate"] >= pair_thresholds,
        verifier_probabilities_by_split["calibrate"],
    )
    final_predictions = {
        name: gated_predictions(
            pair_probabilities[name], pair_thresholds,
            verifier_probabilities_by_split[name], verifier_thresholds,
        )
        for name in ("calibrate", "test")
    }
    test_domains = encoded["test"]["domain"][pair_episode["test"][2]].astype(str)
    domain_reports = {
        domain: binary_metrics(
            pair_episode["test"][1][test_domains == domain],
            final_predictions["test"][test_domains == domain],
        )
        for domain in sorted(set(test_domains))
    }

    encoder = Detector(stem_stride=1)
    encoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    encoder.eval()
    external = external_report(
        encoder, head, verifier, args.reference, args.external,
        pair_scales, pair_biases, verifier_scales, verifier_biases, verifier_thresholds,
    )
    onnx_path = args.output / "temporal-verifier.onnx"
    parity = export(verifier, onnx_path)
    profile = external.pop("profile")
    profile_values = np.concatenate(
        (
            profile["query_mean"],
            profile["query_standard_deviation"],
            profile["physics_mean"],
            profile["physics_standard_deviation"],
        )
    ).astype("<f4")
    profile_path = args.output / "device-profile.bin"
    profile_path.write_bytes(profile_values.tobytes())
    calibration = {
        "scale": dict(zip((LABELS[index] for index in VERIFIED), map(float, verifier_scales))),
        "bias": dict(zip((LABELS[index] for index in VERIFIED), map(float, verifier_biases))),
        "threshold": dict(
            zip((LABELS[index] for index in VERIFIED), map(float, verifier_thresholds))
        ),
    }
    calibration_report = binary_metrics(
        pair_episode["calibrate"][1], final_predictions["calibrate"]
    )
    test_report = binary_metrics(pair_episode["test"][1], final_predictions["test"])
    failures = []
    for name, report in (("calibrate", calibration_report), ("test", test_report)):
        if report["clean_false_positive"] > 0.05:
            failures.append(f"{name} clean false-positive rate exceeds 5%")
        for label in ("delay", "reverb"):
            if report[label]["recall"] < 0.80:
                failures.append(f"{name} {label} recall is below 80%")
    for label in ("delay", "reverb"):
        if external["gated_metrics"][label]["precision"] < 0.80:
            failures.append(f"external {label} precision is below 80%")
        if external["gated_metrics"][label]["recall"] < 0.80:
            failures.append(f"external {label} recall is below 80%")
    report = {
        "experiment": args.output.name,
        "hypothesis": (
            "A tiny temporal verifier can reject pair-model Delay/Reverb confounders "
            "without changing the frozen public-RIR encoder or requiring user wet audio."
        ),
        "architecture": {
            "candidate": "public-RIR non-aligned clean-reference pair model",
            "verifier": "325-64-2 MLP",
            "parameters": sum(value.numel() for value in verifier.parameters()),
            "features": (
                "query/clean/difference/absolute-difference/deviation over deterministic "
                "65-value temporal log-Mel statistics"
            ),
            "fusion": "pair candidate AND temporal verifier",
            "user_gradient_updates": 0,
        },
        "data_policy": {
            "weights": "CC-BY/CC0 public-RIR grouped train split only",
            "threshold_selection": "grouped Stratocaster calibration split only",
            "external": "evaluation only; no fitting or threshold selection",
            "excluded": ["IDMT", "RemFX", "Apple AU", "generated plugin captures"],
        },
        "training": {
            "epochs": epochs,
            "best_validation_loss": best,
            "seconds": seconds,
            "seed": SEED,
        },
        "calibration": calibration,
        "pair_baseline": {
            "calibrate": base_metrics["pair_calibrate"],
            "test": base_metrics["pair_test"],
            "external": external["pair_metrics"],
        },
        "calibrate": calibration_report,
        "test": test_report,
        "test_domains": domain_reports,
        "external": external,
        "export": {
            "onnx": str(onnx_path.resolve()),
            "sha256": sha256(onnx_path),
            "max_absolute_difference": parity,
            "device_profile": str(profile_path.resolve()),
            "device_profile_sha256": sha256(profile_path),
            "device_profile_values": int(len(profile_values)),
        },
        "quality_gate": {
            "passed": not failures,
            "requirements": {
                "internal_clean_false_positive_max": 0.05,
                "internal_delay_reverb_recall_min": 0.80,
                "external_delay_reverb_precision_min": 0.80,
                "external_delay_reverb_recall_min": 0.80,
            },
            "failures": failures,
        },
    }
    (args.output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
