#!/usr/bin/env python3
"""Train a compact temporal-convolution verifier for public-RIR pair candidates."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader, Dataset

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
from evaluate import expected as external_expected, waveform, windows
from model import FRAMES, LABELS, MELS, Detector, frontend
from layout import CORPUS, REVERB_CACHE, REVERB_ENCODER_RUN, REVERB_PAIR_RUN
from reference import real_only_split
from relative import (
    RelativeHead,
    actual_clean_mask,
    encode_relative_windows,
    episode_features,
    fused_with_profile,
    infer,
    reference_window_count,
)
from verifier import (
    binary_metrics,
    fit_calibration,
    gated_predictions,
    load_encoded,
    select_thresholds,
    sha256,
    top_two,
    verifier_probabilities,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260828
DELAY = LABELS.index("delay")
REVERB = LABELS.index("reverb")
VERIFIED = (DELAY, REVERB)
BANDS = 8


def mel_path(cache: Path, split: str) -> Path:
    paths = sorted(cache.glob(f"{split}-*-mel-f16.npy"))
    paths = [path for path in paths if "physics" not in path.name]
    if len(paths) != 1:
        raise RuntimeError(f"expected one {split} Mel cache, found {paths}")
    return paths[0]


def sequence_cache(cache: Path, split: str) -> Path:
    source_path = mel_path(cache, split)
    output = source_path.with_name(source_path.stem + "-temporal-bands.npy")
    source = np.load(source_path, mmap_mode="r")
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        if existing.shape == (len(source), BANDS, FRAMES):
            return output
    result = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(len(source), BANDS, FRAMES)
    )
    for start in range(0, len(source), 256):
        value = np.asarray(source[start : start + 256], dtype=np.float32)[:, 0]
        result[start : start + len(value)] = value.reshape(
            len(value), BANDS, MELS // BANDS, FRAMES
        ).mean(axis=2)
    result.flush()
    return output


class SequenceVerifier(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None, :, None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None, :, None]
        )
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(BANDS, 24, 9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(24, 32, 9, padding=16, dilation=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, 9, padding=48, dilation=12),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, len(VERIFIED)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp((value - self.mean) / self.deviation, -6.0, 6.0)
        value = self.network(value)
        pooled = torch.cat((value.mean(dim=2), value.amax(dim=2)), dim=1)
        return self.output(pooled)


class SequenceDataset(Dataset):
    def __init__(self, path: Path, labels: np.ndarray, domains: np.ndarray) -> None:
        self.values = np.load(path, mmap_mode="r")
        self.labels = labels
        self.domains = domains.astype(str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = torch.from_numpy(np.array(self.values[index], dtype=np.float32, copy=True))
        labels = self.labels[index]
        target = torch.from_numpy(labels[list(VERIFIED)].astype(np.float32, copy=True))
        weight = np.ones(len(VERIFIED), dtype=np.float32)
        drive, delay, reverb = labels > 0.5
        if drive and not delay:
            weight[0] = 2.5
        if drive and not reverb:
            weight[1] = 4.0
        if reverb and not delay:
            weight[0] = max(weight[0], 1.5)
        if delay and not reverb:
            weight[1] = max(weight[1], 1.5)
        domain = self.domains[index]
        if domain.startswith("egfx"):
            weight *= 8.0
        elif domain.startswith(("guitarset", "techs")):
            weight *= 2.0
        return value, target, torch.from_numpy(weight)


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sequence_logits(
    model: SequenceVerifier, values: np.ndarray, target: torch.device
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), 256):
            batch = torch.from_numpy(
                np.asarray(values[start : start + 256], dtype=np.float32).copy()
            ).to(target)
            result.append(model(batch).cpu().numpy())
    return np.concatenate(result)


def train(
    model: SequenceVerifier,
    train_path: Path,
    train_labels: np.ndarray,
    train_domains: np.ndarray,
    valid_path: Path,
    valid_labels: np.ndarray,
    output: Path,
    epochs: int,
    patience: int,
    target: torch.device,
) -> tuple[int, float, float]:
    train_loader = DataLoader(
        SequenceDataset(train_path, train_labels, train_domains),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    valid_values = np.load(valid_path, mmap_mode="r")
    positives = train_labels[:, VERIFIED].sum(axis=0)
    positive_weight = torch.tensor(
        np.clip((len(train_labels) - positives) / np.maximum(positives, 1.0), 1.0, 8.0),
        dtype=torch.float32,
        device=target,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2.0e-4)
    best, stale, completed = float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for values, labels, weights in train_loader:
            logits = model(values.to(target))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.to(target),
                pos_weight=positive_weight,
                weight=weights.to(target),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        valid_logits = sequence_logits(model, valid_values, target)
        valid_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(valid_logits),
                torch.from_numpy(valid_labels[:, VERIFIED]),
                pos_weight=positive_weight.cpu(),
            )
        )
        completed = epoch + 1
        print("sequence_epoch", completed, "valid_loss", valid_loss)
        if valid_loss < best - 1.0e-5:
            best, stale = valid_loss, 0
            torch.save(model.state_dict(), output / "temporal-sequence-verifier.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(
        torch.load(
            output / "temporal-sequence-verifier.pt", map_location=target, weights_only=True
        )
    )
    return completed, best, time.perf_counter() - started


def path_sequences(path: Path) -> np.ndarray:
    audio = waveform(path)
    values = np.stack(windows(audio))
    mel = frontend(torch.from_numpy(values)).numpy()[:, 0]
    return mel.reshape(len(mel), BANDS, MELS // BANDS, FRAMES).mean(axis=2).astype(np.float32)


def external_report(
    encoder: Detector,
    head: RelativeHead,
    verifier: SequenceVerifier,
    target: torch.device,
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
    reference_sequences = path_sequences(reference)
    count = reference_window_count(10, len(reference_query))
    selected = reference_query[:count]
    mean = selected.mean(axis=0)
    deviation = selected.std(axis=0) + 1.0e-4
    clean_pair = calibrated(
        infer(head, fused_with_profile(selected, mean, deviation), torch.device("cpu")),
        pair_scales,
        pair_biases,
    )
    pair_thresholds = np.minimum(
        np.maximum(clean_pair.max(axis=0) + 0.02, np.asarray([0.05, 0.55, 0.59])), 0.95
    )
    held_query = reference_query[count + 1 :]
    held_sequences = reference_sequences[count + 1 :]
    if len(held_query):
        held_pair = calibrated(
            infer(head, fused_with_profile(held_query, mean, deviation), torch.device("cpu")),
            pair_scales,
            pair_biases,
        )
        held_verifier = verifier_probabilities(
            sequence_logits(verifier, held_sequences, target), verifier_scales, verifier_biases
        )
        clean_fp = float(
            gated_predictions(
                held_pair, pair_thresholds, held_verifier, verifier_thresholds
            ).any(axis=1).mean()
        )
    else:
        clean_fp = None
    rows, truths, pair_predictions, predictions = [], [], [], []
    for path in sorted(directory.glob("*.wav")):
        if path.resolve() == reference.resolve():
            continue
        query, _ = encode_relative_windows(
            encoder, path, "embedding-logits", torch.device("cpu")
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
                sequence_logits(verifier, path_sequences(path), target),
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
    }


def export(model: SequenceVerifier, output: Path) -> float:
    cpu = model.cpu().eval()
    example = torch.randn(1, BANDS, FRAMES)
    torch.onnx.export(
        cpu,
        example,
        output,
        input_names=["temporal_bands"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))
    expected = cpu(example).detach().numpy()
    actual = onnxruntime.InferenceSession(str(output)).run(
        None, {"temporal_bands": example.numpy()}
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
        "--output",
        type=Path,
        default=ROOT / "train/runs/rejected-temporal-sequence-verifier",
    )
    parser.add_argument("--data", type=Path, default=CORPUS)
    parser.add_argument("--external", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument(
        "--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav"
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    target = device()

    encoded = {name: load_encoded(args.cache, name) for name in ("train", "valid", "calibrate", "test")}
    sequences = {name: sequence_cache(args.cache, name) for name in encoded}
    train_values = np.load(sequences["train"], mmap_mode="r")
    mean = np.asarray(train_values, dtype=np.float32).mean(axis=(0, 2))
    deviation = np.asarray(train_values, dtype=np.float32).std(axis=(0, 2)) + 1.0e-4
    verifier = SequenceVerifier(mean, deviation).to(target)
    epochs, best, seconds = train(
        verifier,
        sequences["train"],
        encoded["train"]["labels"],
        encoded["train"]["domain"],
        sequences["valid"],
        encoded["valid"]["labels"],
        args.output,
        args.epochs,
        args.patience,
        target,
    )

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
    masks = {name: actual_clean_mask(parts[name], clean_paths) for name in parts}
    episode = {
        name: episode_features(
            encoded[name], masks[name], "embedding-logits", SEED + index * 10_007,
            2 if name == "train" else 1,
        )
        for index, name in enumerate(("train", "valid", "calibrate", "test"))
    }
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
            infer(head, episode[name][0], torch.device("cpu")), pair_scales, pair_biases
        )
        for name in ("calibrate", "test")
    }
    verifier_logits_by_split = {
        name: sequence_logits(
            verifier,
            np.load(sequences[name], mmap_mode="r")[episode[name][2]],
            target,
        )
        for name in ("calibrate", "test")
    }
    verifier_scales, verifier_biases = fit_calibration(
        episode["calibrate"][1][:, VERIFIED], verifier_logits_by_split["calibrate"]
    )
    verifier_probability = {
        name: verifier_probabilities(values, verifier_scales, verifier_biases)
        for name, values in verifier_logits_by_split.items()
    }
    verifier_thresholds = select_thresholds(
        episode["calibrate"][1],
        pair_probabilities["calibrate"] >= pair_thresholds,
        verifier_probability["calibrate"],
    )
    prediction = {
        name: gated_predictions(
            pair_probabilities[name], pair_thresholds,
            verifier_probability[name], verifier_thresholds,
        )
        for name in ("calibrate", "test")
    }
    test_domains = encoded["test"]["domain"][episode["test"][2]].astype(str)
    domain_reports = {
        domain: binary_metrics(
            episode["test"][1][test_domains == domain],
            prediction["test"][test_domains == domain],
        )
        for domain in sorted(set(test_domains))
    }
    encoder = Detector(stem_stride=1)
    encoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    encoder.eval()
    external = external_report(
        encoder, head, verifier, target, args.reference, args.external,
        pair_scales, pair_biases, verifier_scales, verifier_biases, verifier_thresholds,
    )
    onnx_path = args.output / "temporal-sequence-verifier.onnx"
    parity = export(verifier, onnx_path)
    calibration_report = binary_metrics(episode["calibrate"][1], prediction["calibrate"])
    test_report = binary_metrics(episode["test"][1], prediction["test"])
    failures = []
    for split, report in (("calibrate", calibration_report), ("test", test_report)):
        if report["clean_false_positive"] > 0.05:
            failures.append(f"{split} clean false-positive rate exceeds 5%")
        for label in ("delay", "reverb"):
            if report[label]["recall"] < 0.80:
                failures.append(f"{split} {label} recall is below 80%")
    for label in ("delay", "reverb"):
        if external["gated_metrics"][label]["precision"] < 0.80:
            failures.append(f"external {label} precision is below 80%")
        if external["gated_metrics"][label]["recall"] < 0.80:
            failures.append(f"external {label} recall is below 80%")
    report = {
        "experiment": args.output.name,
        "hypothesis": (
            "A compact dilated temporal CNN over ordered Mel-band sequences can reject "
            "Drive sustain and modulation confounders that summary statistics cannot."
        ),
        "architecture": {
            "candidate": "public-RIR clean-reference pair model",
            "verifier": "8x216 band sequence -> dilated Conv1d -> mean/max pool -> 2",
            "parameters": sum(value.numel() for value in verifier.parameters()),
            "fusion": "pair candidate AND temporal verifier",
            "user_gradient_updates": 0,
        },
        "data_policy": {
            "weights": "CC-BY/CC0 public-RIR grouped train split only",
            "domain_weighting": "EGFx real hardware 8x; GuitarSet/Guitar-TECHS clean 2x",
            "threshold_selection": "grouped Stratocaster calibration split only",
            "external": "evaluation only; no fitting or threshold selection",
            "excluded": ["IDMT", "RemFX", "Apple AU", "generated plugin captures"],
        },
        "training": {
            "device": str(target),
            "epochs": epochs,
            "best_validation_loss": best,
            "seconds": seconds,
            "seed": SEED,
        },
        "calibration": {
            "scale": dict(
                zip((LABELS[index] for index in VERIFIED), map(float, verifier_scales))
            ),
            "bias": dict(
                zip((LABELS[index] for index in VERIFIED), map(float, verifier_biases))
            ),
            "threshold": dict(
                zip((LABELS[index] for index in VERIFIED), map(float, verifier_thresholds))
            ),
        },
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
        json.dumps(report["calibration"], indent=2, sort_keys=True) + "\n"
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
