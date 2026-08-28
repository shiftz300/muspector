#!/usr/bin/env python3
"""Train, calibrate, evaluate, and export Muspector's blind detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    Audio,
    Item,
    add_rir_reverb,
    audit,
    discover,
    discover_aachen_rirs,
    discover_apple_au,
    discover_guitar_effect_chains,
    discover_idmt_guitar,
    discover_remfx_1_1,
    egfx_hardware_test,
    eligible_clean,
    guitar_effect_chain_is_clean,
    partition,
    split,
    vector,
)
from model import Detector, FRAMES, LABELS, MELS, augment_features, frontend, parameters


ROOT = Path(__file__).resolve().parents[1]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loader(dataset: Audio, batch: int, workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def features(waveform: torch.Tensor, target: torch.device, augment: bool = False) -> torch.Tensor:
    # Keep the frontend on CPU so its exact behavior is portable to the Rust
    # implementation. Only the fixed Mel tensor crosses to the accelerator.
    value = frontend(waveform).to(target)
    return augment_features(value) if augment else value


def evaluate(model: Detector, source: DataLoader, target: torch.device):
    model.eval()
    expected, logits = [], []
    with torch.no_grad():
        for waveform, labels in source:
            logits.append(model(features(waveform, target)).cpu().numpy())
            expected.append(labels.numpy())
    return np.concatenate(expected), np.concatenate(logits)


def f1(expected: np.ndarray, actual: np.ndarray) -> float:
    true_positive = np.logical_and(expected, actual).sum()
    false_positive = np.logical_and(~expected, actual).sum()
    false_negative = np.logical_and(expected, ~actual).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 1.0


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_platt(expected: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales, biases = [], []
    for index in range(len(LABELS)):
        values = torch.tensor(logits[:, index], dtype=torch.float64)
        labels = torch.tensor(expected[:, index], dtype=torch.float64)
        log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (log_scale, bias), lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            predicted = values * log_scale.exp() + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        scales.append(float(log_scale.detach().exp().clamp(0.05, 20.0)))
        biases.append(float(bias.detach().clamp(-20.0, 20.0)))
    return np.asarray(scales, dtype=np.float32), np.asarray(biases, dtype=np.float32)


def calibrated(logits: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return sigmoid(logits * scales + biases)


def thresholds(
    expected: np.ndarray, probabilities: np.ndarray, clean_fp_max: float = 0.05
) -> np.ndarray:
    result = np.full(len(LABELS), 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, 91)
    for index in range(len(LABELS)):
        truth = expected[:, index] > 0.5
        result[index] = max(
            ((f1(truth, probabilities[:, index] >= value), value) for value in candidates),
            key=lambda pair: (pair[0], pair[1]),
        )[1]
    clean = expected.sum(axis=1) == 0
    while clean.any() and (probabilities[clean] >= result).any(axis=1).mean() > clean_fp_max:
        contributions = [
            (probabilities[clean, index] >= result[index]).mean()
            if result[index] < 0.99
            else -1.0
            for index in range(len(LABELS))
        ]
        index = int(np.argmax(contributions))
        if contributions[index] < 0.0:
            break
        result[index] = min(0.99, result[index] + 0.01)
    return result


def gate_feasibility(expected: np.ndarray, probabilities: np.ndarray) -> dict:
    """Test whether independent thresholds can satisfy calibration gate bounds.

    For each class, choose the highest threshold that still reaches 80% recall.
    This is the most conservative independent threshold vector under the recall
    constraint and therefore gives the lowest attainable combined Clean FP.
    """

    required_recall = 0.80
    candidates = np.linspace(0.0, 1.0, 1001)
    selected = np.zeros(len(LABELS), dtype=np.float32)
    recalls = {}
    for index, name in enumerate(LABELS):
        truth = expected[:, index] > 0.5
        feasible = [
            value
            for value in candidates
            if truth.any()
            and float((probabilities[truth, index] >= value).mean()) >= required_recall
        ]
        selected[index] = max(feasible) if feasible else 0.0
        recalls[name] = float(
            (probabilities[truth, index] >= selected[index]).mean()
        ) if truth.any() else 0.0
    clean = expected.sum(axis=1) == 0
    clean_false_positive = (
        float((probabilities[clean] >= selected).any(axis=1).mean())
        if clean.any()
        else 0.0
    )
    return {
        "feasible": clean_false_positive <= 0.05,
        "method": (
            "highest independent 0.001-grid thresholds retaining at least 80% "
            "per-class recall; this minimizes combined calibration Clean FP"
        ),
        "threshold": dict(zip(LABELS, map(float, selected))),
        "recall": recalls,
        "minimum_clean_false_positive": clean_false_positive,
    }


def metrics(expected: np.ndarray, probabilities: np.ndarray, threshold: np.ndarray) -> dict:
    predicted = probabilities >= threshold
    clean = expected.sum(axis=1) == 0
    values = {}
    f1s = []
    for index, name in enumerate(LABELS):
        truth = expected[:, index] > 0.5
        actual = predicted[:, index]
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
        value = f1(truth, actual)
        f1s.append(value)
        values[name] = {"precision": precision, "recall": recall, "f1": value}
    values["macro_f1"] = float(np.mean(f1s))
    values["exact_match"] = float((predicted == (expected > 0.5)).all(axis=1).mean())
    values["clean_false_positive"] = (
        float(predicted[clean].any(axis=1).mean()) if clean.any() else 0.0
    )
    values["samples"] = int(len(expected))
    return values


def domain_metrics(
    model: Detector,
    items: list[Item],
    threshold: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    batch: int,
    workers: int,
    seed: int,
    target: torch.device,
    pedalboard_renderer: bool = False,
) -> dict:
    report = {}
    for domain in sorted({item.source.domain for item in items}):
        selected = [item for item in items if item.source.domain == domain]
        expected, logits = evaluate(
            model,
            loader(
                Audio(selected, False, seed, pedalboard_renderer), batch, workers
            ),
            target,
        )
        report[domain] = metrics(expected, calibrated(logits, scales, biases), threshold)
        report[domain]["positive_counts"] = {
            label: int((expected[:, index] > 0.5).sum())
            for index, label in enumerate(LABELS)
        }
    return report


def quality_gate(
    calibration: dict, test: dict, domains: dict, benchmarks: dict
) -> dict:
    """Apply the acceptance rules before a model can be considered shippable."""

    failures = []
    if calibration["clean_false_positive"] > 0.05:
        failures.append("calibration clean false-positive rate exceeds 5%")
    for name in LABELS:
        if calibration[name]["recall"] < 0.80:
            failures.append(f"calibration {name} recall is below 80%")
        if test[name]["recall"] < 0.80:
            failures.append(f"test {name} recall is below 80%")
    if test["clean_false_positive"] > 0.05:
        failures.append("test clean false-positive rate exceeds 5%")
    for name, values in domains.items():
        if values["clean_false_positive"] > 0.05:
            failures.append(f"{name} clean false-positive rate exceeds 5%")
        for label in LABELS:
            if values.get("positive_counts", {}).get(label, 0) and values[label]["recall"] < 0.80:
                failures.append(f"{name} {label} recall is below 80%")
    for benchmark, values in benchmarks.items():
        if values["clean_false_positive"] > 0.05:
            failures.append(f"{benchmark} clean false-positive rate exceeds 5%")
        for name in LABELS:
            if values[name]["recall"] < 0.80:
                failures.append(f"{benchmark} {name} recall is below 80%")
    return {
        "passed": not failures,
        "requirements": {
            "clean_false_positive_max": 0.05,
            "per_class_recall_min": 0.80,
        },
        "failures": failures,
    }


def export(model: Detector, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.randn(1, 1, MELS, FRAMES)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))
    expected = model(example).detach().numpy()
    actual = onnxruntime.InferenceSession(str(output)).run(None, {"mel": example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "train" / "runs" / "blind")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--realizations", type=int, default=4)
    parser.add_argument(
        "--all-train-anchors",
        action="store_true",
        help="retain every partition-safe EGFxSet wet anchor in training",
    )
    parser.add_argument(
        "--fixed-clean-train",
        action="store_true",
        help="reserve the first training realization as untreated clean audio",
    )
    parser.add_argument(
        "--apple-au-captures",
        action="store_true",
        help="include fixed captures declared by data/corpus/apple-au/manifest.json",
    )
    parser.add_argument(
        "--remfx-1-1",
        action="store_true",
        help="include official CC-NC RemFX 1-1 wet/dry pairs in training only",
    )
    parser.add_argument(
        "--idmt-guitar",
        action="store_true",
        help="include the audited CC-BY-NC-ND IDMT guitar subset with device-disjoint splits",
    )
    parser.add_argument(
        "--guitar-effects-chains",
        action="store_true",
        help="include the CC-BY-4.0 DAFx25 archived five-effect-chain dataset",
    )
    parser.add_argument(
        "--archived-only",
        action="store_true",
        help="fit only fixed archived audio; do not render effects dynamically",
    )
    parser.add_argument(
        "--pedalboard-renderer",
        action="store_true",
        help="use the audited Spotify Pedalboard/local multi-implementation renderer for dynamic items",
    )
    parser.add_argument(
        "--aachen-rir",
        action="store_true",
        help="add fixed ephemeral reverb examples from CC-BY-4.0 measured chapel RIRs",
    )
    parser.add_argument(
        "--anchors-evaluation-only",
        action="store_true",
        help="exclude EGFx wet anchors from fitting and use test-partition anchors only as a hardware benchmark",
    )
    parser.add_argument(
        "--experiment",
        help="stable experiment identifier written to metrics.json",
    )
    parser.add_argument(
        "--hypothesis",
        help="single experimental hypothesis written to metrics.json",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="skip fitting and finalize calibration/test/export from this checkpoint",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="initialize fitting from this checkpoint instead of random weights",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="freeze the encoder and BatchNorm state; fit only the 771-parameter head",
    )
    parser.add_argument("--epochs-completed", type=int, default=0)
    parser.add_argument("--best-validation-loss", type=float, default=float("nan"))
    args = parser.parse_args()
    if args.fixed_clean_train and args.realizations < 2:
        parser.error("--fixed-clean-train requires --realizations >= 2")
    if args.archived_only and args.pedalboard_renderer:
        parser.error("--pedalboard-renderer requires dynamic rendering")
    if args.archived_only and args.anchors_evaluation_only:
        parser.error("--archived-only cannot use --anchors-evaluation-only")
    if args.checkpoint and args.initial_checkpoint:
        parser.error("--checkpoint and --initial-checkpoint are mutually exclusive")
    if args.initial_checkpoint and not math.isfinite(args.best_validation_loss):
        parser.error("--initial-checkpoint requires its --best-validation-loss")
    if args.head_only and not args.initial_checkpoint:
        parser.error("--head-only requires --initial-checkpoint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    clean, anchors = discover(args.data)
    chain_items = {name: [] for name in ("train", "valid", "calibrate", "test")}
    if args.guitar_effects_chains:
        chain_items = discover_guitar_effect_chains(args.data / "guitar-effects-chains")
        if not any(chain_items.values()):
            parser.error("--guitar-effects-chains found no archived audio")
    captures = discover_apple_au(args.data / "apple-au") if args.apple_au_captures else None
    apple_capture_count = sum(map(len, captures.values())) if captures else 0
    remfx_items = (
        discover_remfx_1_1(args.data / "remfx-1-1") if args.remfx_1_1 else []
    )
    remfx_pair_manifest = None
    if args.remfx_1_1:
        if not remfx_items:
            parser.error("--remfx-1-1 found no complete pairs below data/corpus/remfx-1-1")
        remfx_manifest_path = ROOT / "train" / "runs" / "remfx-1-1-pairs.json"
        if not remfx_manifest_path.exists():
            parser.error("RemFX is excluded from the active Inspector pipeline")
        remfx_manifest_bytes = remfx_manifest_path.read_bytes()
        remfx_manifest_payload = json.loads(remfx_manifest_bytes)
        if len(remfx_manifest_payload.get("pairs", [])) * 2 != len(remfx_items):
            parser.error("RemFX audit manifest does not match discovered training items")
        remfx_pair_manifest = {
            "path": str(remfx_manifest_path.resolve()),
            "sha256": hashlib.sha256(remfx_manifest_bytes).hexdigest(),
            "counts": remfx_manifest_payload["counts"],
            "detector_policy": remfx_manifest_payload["detector_policy"],
        }
        if captures is None:
            captures = {name: [] for name in ("train", "valid", "calibrate", "test")}
        captures["train"].extend(remfx_items)
    idmt_manifest = None
    idmt_items = {name: [] for name in ("train", "valid", "calibrate", "test")}
    if args.idmt_guitar:
        idmt_manifest_path = ROOT / "train" / "runs" / "idmt-guitar-manifest.json"
        if not idmt_manifest_path.exists():
            parser.error("IDMT is excluded from the active Inspector pipeline")
        idmt_manifest_bytes = idmt_manifest_path.read_bytes()
        idmt_manifest_payload = json.loads(idmt_manifest_bytes)
        idmt_items = discover_idmt_guitar(
            args.data / "idmt-smt-audio-effects", idmt_manifest_path
        )
        if sum(map(len, idmt_items.values())) != len(idmt_manifest_payload.get("records", [])):
            parser.error("IDMT manifest does not match discovered guitar items")
        if captures is None:
            captures = {name: [] for name in ("train", "valid", "calibrate", "test")}
        for name, items in idmt_items.items():
            captures[name].extend(items)
        idmt_manifest = {
            "path": str(idmt_manifest_path.resolve()),
            "sha256": hashlib.sha256(idmt_manifest_bytes).hexdigest(),
            "doi": idmt_manifest_payload["doi"],
            "license": idmt_manifest_payload["license"],
            "policy": idmt_manifest_payload["policy"],
            "counts": idmt_manifest_payload["counts"],
        }
    apple_au_manifest = None
    if args.apple_au_captures:
        manifest_path = args.data / "apple-au" / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest_payload = json.loads(manifest_bytes)
        apple_au_manifest = {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "renderer": manifest_payload["renderer"],
            "selection": manifest_payload["selection"],
        }
    if args.archived_only:
        if captures:
            parser.error(
                "--archived-only cannot be combined with Apple AU, RemFX, or IDMT inputs"
            )
        selected = {name: [] for name in ("train", "valid", "calibrate", "test")}
        for source in clean:
            selected[partition(source)].append(
                Item(source, vector(()), augment=False)
            )
        for item in anchors:
            selected[partition(item.source)].append(
                Item(item.source, item.target, augment=False)
            )
        for name in selected:
            selected[name].extend(chain_items[name])
        if args.aachen_rir:
            rirs = discover_aachen_rirs(args.data / "aachen-chapel-rir")
            clean_paths = {
                source.path.resolve() for source in clean if eligible_clean(source)
            }
            clean_paths.update(
                item.source.path.resolve()
                for items in chain_items.values()
                for item in items
                if guitar_effect_chain_is_clean(item.source)
            )
            selected = add_rir_reverb(selected, rirs, clean_paths)
        audit(selected)
    else:
        selected = split(
            clean,
            anchors,
            args.realizations,
            captures=captures,
            all_train_anchors=args.all_train_anchors,
            fixed_clean_train=args.fixed_clean_train,
            anchors_evaluation_only=args.anchors_evaluation_only,
        )
        for name in selected:
            selected[name].extend(chain_items[name])
        if args.aachen_rir:
            rirs = discover_aachen_rirs(args.data / "aachen-chapel-rir")
            clean_paths = {
                source.path.resolve() for source in clean if eligible_clean(source)
            }
            clean_paths.update(
                item.source.path.resolve()
                for items in chain_items.values()
                for item in items
                if guitar_effect_chain_is_clean(item.source)
            )
            selected = add_rir_reverb(selected, rirs, clean_paths)
        audit(selected)
    print(
        "sources",
        len(clean),
        "hardware anchors",
        len(anchors),
        "Apple AU captures",
        apple_capture_count,
        "RemFX 1-1 items",
        len(remfx_items),
        "IDMT guitar items",
        sum(map(len, idmt_items.values())),
        "DAFx25 chain items",
        sum(map(len, chain_items.values())),
    )
    for name, items in selected.items():
        domains = {}
        for item in items:
            domains[item.source.domain] = domains.get(item.source.domain, 0) + 1
        print(name, len(items), json.dumps(domains, sort_keys=True))

    datasets = {
        name: Audio(
            items, name == "train", args.seed, args.pedalboard_renderer
        )
        for name, items in selected.items()
    }
    loaders = {
        name: loader(dataset, args.batch, args.workers, name == "train")
        for name, dataset in datasets.items()
    }
    target = device()
    model = Detector().to(target)
    if args.initial_checkpoint:
        model.load_state_dict(
            torch.load(args.initial_checkpoint, map_location=target, weights_only=True)
        )
    if args.head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("head.")
    print("device", target, "parameters", parameters(model))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    steps = max(1, args.epochs * len(loaders["train"]))
    warmup = max(1, len(loaders["train"]))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        position = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(position, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = (
        args.best_validation_loss
        if args.checkpoint or args.initial_checkpoint
        else float("inf")
    )
    if args.initial_checkpoint:
        # Fine-tuning must never discard a better initialization merely because
        # the first new epoch is the first value observed in this process.
        torch.save(model.state_dict(), args.output / "best.pt")
    stale = 0
    completed = args.epochs_completed if args.checkpoint else 0
    if args.checkpoint is None:
        for epoch in range(args.epochs):
            if args.head_only:
                # Frozen BatchNorm running statistics are part of the learned
                # domain representation and must not drift on RemFX.
                model.eval()
                model.head.train()
            else:
                model.train()
            total = 0.0
            progress = tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{args.epochs}")
            for waveform, expected in progress:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features(waveform, target, augment=True))
                loss = loss_fn(logits, expected.to(target))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach().cpu())
                progress.set_postfix(loss=f"{total / (progress.n + 1):.4f}")
            valid_expected, valid_logits = evaluate(model, loaders["valid"], target)
            valid_loss = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.from_numpy(valid_logits), torch.from_numpy(valid_expected)
                )
            )
            valid_report = metrics(
                valid_expected, sigmoid(valid_logits), np.full(len(LABELS), 0.5)
            )
            print("valid_loss", valid_loss, json.dumps(valid_report, sort_keys=True))
            completed = epoch + 1
            if valid_loss < best_loss - 1.0e-4:
                best_loss = valid_loss
                stale = 0
                torch.save(model.state_dict(), args.output / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop after {stale} epochs without validation improvement")
                    break

    checkpoint = args.checkpoint or args.output / "best.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location=target, weights_only=True)
    )
    final_valid_expected, final_valid_logits = evaluate(model, loaders["valid"], target)
    final_valid_loss = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(final_valid_logits), torch.from_numpy(final_valid_expected)
        )
    )
    final_valid_report = metrics(
        final_valid_expected,
        sigmoid(final_valid_logits),
        np.full(len(LABELS), 0.5),
    )
    if args.checkpoint and not math.isfinite(best_loss):
        best_loss = final_valid_loss
    calibration_expected, calibration_logits = evaluate(model, loaders["calibrate"], target)
    scales, biases = fit_platt(calibration_expected, calibration_logits)
    calibration_probabilities = calibrated(calibration_logits, scales, biases)
    threshold = thresholds(calibration_expected, calibration_probabilities)
    test_expected, test_logits = evaluate(model, loaders["test"], target)
    test_probabilities = calibrated(test_logits, scales, biases)
    calibration_report = metrics(calibration_expected, calibration_probabilities, threshold)
    test_report = metrics(test_expected, test_probabilities, threshold)
    domains = domain_metrics(
        model,
        selected["test"],
        threshold,
        scales,
        biases,
        args.batch,
        args.workers,
        args.seed,
        target,
        args.pedalboard_renderer,
    )
    benchmark_reports = {}
    if args.anchors_evaluation_only:
        hardware_items = egfx_hardware_test(clean, anchors)
        hardware_expected, hardware_logits = evaluate(
            model,
            loader(Audio(hardware_items, False, args.seed), args.batch, args.workers),
            target,
        )
        benchmark_reports["egfx-hardware-device-disjoint"] = metrics(
            hardware_expected,
            calibrated(hardware_logits, scales, biases),
            threshold,
        )
    split_report = {}
    for name, items in selected.items():
        split_domains = {}
        labels = {}
        for item in items:
            split_domains[item.source.domain] = (
                split_domains.get(item.source.domain, 0) + 1
            )
            if item.target is None:
                item_label = "dynamic"
            else:
                item_label = "+".join(
                    LABELS[index]
                    for index, value in enumerate(item.target)
                    if value > 0.5
                ) or (
                    "target-negative-anchor"
                    if item.augment
                    else "source-clean-hard-negative"
                )
            labels[item_label] = labels.get(item_label, 0) + 1
        split_report[name] = {
            "items": len(items),
            "domains": dict(sorted(split_domains.items())),
            "item_roles": dict(sorted(labels.items())),
        }
    report = {
        "experiment": args.experiment or args.output.name,
        "labels": list(LABELS),
        "hypothesis": args.hypothesis
        or (
            "Retain the three-family label contract and compact ResNet18 while "
            "increasing partition-safe real EGFxSet wet-anchor coverage."
        ),
        "data": {
            "root": str(args.data.resolve()),
            "license_manifest": str((ROOT / "train" / "LICENSES.md").resolve()),
            "apple_au_manifest": apple_au_manifest,
            "remfx_pair_manifest": remfx_pair_manifest,
            "idmt_guitar_manifest": idmt_manifest,
            "sources": {
                "egfxset": "CC-BY-4.0",
                "guitar-techs": "CC-BY-4.0",
                "guitarset": "CC-BY-4.0",
                "guitarjam": "CC0-1.0",
                **(
                    {"apple-au": "macOS system components; generated audio is not redistributed"}
                    if args.apple_au_captures
                    else {}
                ),
                **(
                    {"remfx-1-1": "CC-NC; non-commercial research training only"}
                    if args.remfx_1_1
                    else {}
                ),
                **(
                    {"idmt-smt-audio-effects": "CC-BY-NC-ND-4.0; non-commercial research only"}
                    if args.idmt_guitar
                    else {}
                ),
                **(
                    {"guitar-effects-chains": "CC-BY-4.0"}
                    if args.guitar_effects_chains
                    else {}
                ),
                **(
                    {"aachen-chapel-rir": "CC-BY-4.0"}
                    if args.aachen_rir
                    else {}
                ),
                **(
                    {"spotify-pedalboard": "GPL-3.0; local research renderer only"}
                    if args.pedalboard_renderer
                    else {}
                ),
            },
        },
        "training_config": {
            "batch_size": args.batch,
            "workers": args.workers,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 1.0e-4,
            "patience": args.patience,
            "minimum_validation_improvement": 1.0e-4,
            "loss": "BCEWithLogitsLoss",
            "target_complexity_weights": {"0": 30, "1": 50, "2": 17, "3": 3},
            "modulation_nuisance_probability": 0.30,
            "train_anchor_policy": (
                "all-partition-safe" if args.all_train_anchors else "stable-one-third"
            ),
            "fixed_clean_train_realization": args.fixed_clean_train,
            "archived_only": args.archived_only,
            "dynamic_renderer": (
                "spotify-pedalboard-0.9.24-plus-local-dsp"
                if args.pedalboard_renderer
                else "local-scipy-dsp"
            ),
            "aachen_rir": {
                "enabled": args.aachen_rir,
                "record": "https://zenodo.org/records/20428705"
                if args.aachen_rir
                else None,
                "license": "CC-BY-4.0" if args.aachen_rir else None,
                "render_policy": "ephemeral measured-RIR convolution; no derived audio corpus stored"
                if args.aachen_rir
                else None,
            },
            "guitar_effects_chains": {
                "enabled": args.guitar_effects_chains,
                "record": "https://zenodo.org/records/7871720"
                if args.guitar_effects_chains
                else None,
                "license": "CC-BY-4.0" if args.guitar_effects_chains else None,
                "items": {name: len(items) for name, items in chain_items.items()},
                "split_policy": "PRS+Les Paul train; Strat validation/calibration; Telecaster test",
            },
            "apple_au_captures": args.apple_au_captures,
            "remfx_1_1_training_items": len(remfx_items),
            "idmt_guitar_items": {
                name: len(items) for name, items in idmt_items.items()
            },
            "initial_checkpoint": (
                str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None
            ),
            "head_only": args.head_only,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "egfx_anchor_policy": (
                "test-partition-device-disjoint-benchmark-only"
                if args.anchors_evaluation_only
                else "included-in-partitioned-training-and-evaluation"
            ),
        },
        "architecture": "compact-resnet18-16-32-64-128",
        "parameters": parameters(model),
        "frontend": {
            "rate": 44_100,
            "seconds": 5,
            "fft": 2_048,
            "hop": 1_024,
            "mels": 128,
            "frames": 216,
            "fmin": 30,
            "fmax": 16_000,
        },
        "calibration": {
            "scale": dict(zip(LABELS, map(float, scales))),
            "bias": dict(zip(LABELS, map(float, biases))),
            "threshold": dict(zip(LABELS, map(float, threshold))),
        },
        "calibrate": calibration_report,
        "calibration_gate_feasibility": gate_feasibility(
            calibration_expected, calibration_probabilities
        ),
        "test": test_report,
        "validation_loss": final_valid_loss,
        "validation": final_valid_report,
        "test_domains": domains,
        "hardware_benchmarks": benchmark_reports,
        "quality_gate": quality_gate(
            calibration_report, test_report, domains, benchmark_reports
        ),
        "device": str(target),
        "epochs_completed": completed,
        "epochs_requested": args.epochs,
        "evaluation_checkpoint": str(checkpoint.resolve()),
        "best_validation_loss": best_loss,
        "train_realizations": args.realizations,
        "split": split_report,
        "seed": args.seed,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "calibration.json").write_text(
        json.dumps(report["calibration"], indent=2, sort_keys=True) + "\n"
    )
    export(model, args.output / "blind.onnx")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
