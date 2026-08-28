#!/usr/bin/env python3
"""Train and personalize a real-audio clean-reference detector.

The offline backbone sees only archived recordings: EGFxSet hardware wet audio
and explicitly untreated recordings. User personalization freezes that backbone
and updates only a small embedding adapter using a clean reference plus a
compact replay bank. No DSP or plugin-rendered positive audio is created here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import (
    Item,
    Source,
    add_rir_reverb,
    audit,
    discover,
    discover_aachen_rirs,
    discover_guitar_effect_chains,
    discover_tonetwist_big_muff_nc,
    eligible_clean,
    guitar_effect_chain_is_clean,
    partition,
    render_item,
    vector,
)
from detect import calibrated, fit_platt, gate_feasibility, metrics, thresholds
from evaluate import expected as external_expected
from model import EMBEDDING, FRAMES, LABELS, MELS, RATE, SAMPLES, Detector, frontend, parameters


ROOT = Path(__file__).resolve().parents[1]
STEP = SAMPLES // 2


def target_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_domain(source: Source) -> str:
    if source.domain == "egfx":
        parts = source.group.split(":", 2)
        return ":".join(parts[:2]) if len(parts) >= 2 else "egfx"
    return source.domain


def real_only_split(clean: list[Source], anchors: list[Item]) -> dict[str, list[Item]]:
    parts: dict[str, list[Item]] = {name: [] for name in ("train", "valid", "calibrate", "test")}
    for source in clean:
        if eligible_clean(source):
            parts[partition(source)].append(Item(source, vector(()), augment=False))
    for item in anchors:
        parts[partition(item.source)].append(Item(item.source, item.target, augment=False))
    audit(parts)
    return parts


class FixedAudio(Dataset):
    def __init__(self, items: list[Item], seed: int) -> None:
        self.items = items
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        rng = random.Random(self.seed + index * 104_729)
        audio, target = render_item(item, rng, False)
        if item.target is None:
            raise RuntimeError("real-only dataset cannot contain dynamic targets")
        return torch.from_numpy(audio), torch.from_numpy(target)


def make_loader(items: list[Item], batch: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(FixedAudio(items, 20260828), batch_size=batch, shuffle=shuffle, num_workers=0)


class CachedFeatures(Dataset):
    def __init__(self, feature_path: Path, label_path: Path) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Copy one record out of the read-only mmap before PyTorch wraps it.
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.float32, copy=True))
        return feature, label


def cache_key(items: list[Item]) -> str:
    digest = hashlib.sha256()
    digest.update(b"reference-real-logmel-v1")
    for item in items:
        digest.update(str(item.source.path.resolve()).encode())
        digest.update(
            f":{item.source.offset}:{item.target}:{item.effect_ir}:{item.effect_mix}".encode()
        )
    return digest.hexdigest()[:16]


def cache_features(items: list[Item], name: str, root: Path, batch: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(items)
    feature_path = root / f"{name}-{key}-mel-f16.npy"
    label_path = root / f"{name}-{key}-labels.npy"
    if feature_path.exists() and label_path.exists():
        features = np.load(feature_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        if features.shape == (len(items), 1, MELS, FRAMES) and labels.shape == (len(items), len(LABELS)):
            return feature_path, label_path
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16, shape=(len(items), 1, MELS, FRAMES)
    )
    labels = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.float32, shape=(len(items), len(LABELS))
    )
    position = 0
    for waveform, expected in tqdm(make_loader(items, batch), desc=f"cache {name}"):
        count = len(waveform)
        features[position : position + count] = frontend(waveform).numpy().astype(np.float16)
        labels[position : position + count] = expected.numpy()
        position += count
    features.flush()
    labels.flush()
    return feature_path, label_path


def feature_loader(paths: tuple[Path, Path], batch: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(CachedFeatures(*paths), batch_size=batch, shuffle=shuffle, num_workers=0)


def mel(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    return frontend(waveform).to(device)


def class_weight(items: list[Item], device: torch.device) -> torch.Tensor:
    labels = np.asarray([item.target for item in items], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    return torch.tensor(np.clip(negatives / np.maximum(positives, 1.0), 1.0, 8.0), device=device)


def evaluate_detector(model: Detector, cached: tuple[Path, Path], batch: int, device: torch.device):
    expected, logits = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in feature_loader(cached, batch):
            logits.append(model(features.to(device)).cpu().numpy())
            expected.append(labels.numpy())
    return np.concatenate(expected), np.concatenate(logits)


def encode_items(
    model: Detector,
    items: list[Item],
    cached: tuple[Path, Path],
    batch: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    embeddings, logits, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for features, expected in feature_loader(cached, batch):
            value = model.encode(features.to(device))
            embeddings.append(value.cpu().numpy())
            logits.append(model.head(value).cpu().numpy())
            labels.append(expected.numpy())
    return {
        "embedding": np.concatenate(embeddings).astype(np.float32),
        "base_logits": np.concatenate(logits).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.float32),
        "domain": np.asarray([reference_domain(item.source) for item in items]),
    }


def prototypes(encoded: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    clean = encoded["labels"].sum(axis=1) == 0
    for domain in sorted(set(encoded["domain"])):
        selected = encoded["embedding"][np.logical_and(clean, encoded["domain"] == domain)]
        if len(selected):
            result[domain] = (selected.mean(axis=0), selected.std(axis=0) + 1.0e-4)
    global_clean = encoded["embedding"][clean]
    result["__global__"] = (global_clean.mean(axis=0), global_clean.std(axis=0) + 1.0e-4)
    return result


def fuse(encoded: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = prototypes(encoded)
    means, deviations = [], []
    for domain in encoded["domain"]:
        mean, deviation = refs.get(str(domain), refs["__global__"])
        means.append(mean)
        deviations.append(deviation)
    z = encoded["embedding"]
    mean = np.stack(means)
    deviation = np.stack(deviations)
    value = np.concatenate((z, mean, z - mean, np.abs(z - mean), deviation), axis=1)
    return value.astype(np.float32), encoded["base_logits"], encoded["labels"]


class ReferenceHead(torch.nn.Module):
    """Small correction head; only the bottleneck is personalized on device."""

    def __init__(self) -> None:
        super().__init__()
        self.project = torch.nn.Linear(EMBEDDING * 5, EMBEDDING)
        self.norm = torch.nn.LayerNorm(EMBEDDING)
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, EMBEDDING),
        )
        self.output = torch.nn.Linear(EMBEDDING, len(LABELS))

    def projected(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.project(fused)))

    def correction(self, projected: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(projected + self.adapter(projected)))

    def forward(self, fused: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.correction(self.projected(fused))


def train_backbone(
    model: Detector,
    parts: dict[str, list[Item]],
    cached: dict[str, tuple[Path, Path]],
    output: Path,
    epochs: int,
    patience: int,
    batch: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight(parts["train"], device))
    baseline_expected, baseline_logits = evaluate_detector(
        model, cached["valid"], batch, device
    )
    best = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(baseline_logits), torch.from_numpy(baseline_expected)
        )
    )
    history = [{"epoch": 0, "validation_loss": best}]
    print("backbone_baseline_valid_loss", best)
    torch.save(model.state_dict(), output / "backbone.pt")
    (output / "backbone-progress.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    stale, completed = 0, 0
    train_loader = feature_loader(cached["train"], batch, True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        progress = tqdm(train_loader, desc=f"backbone {epoch + 1}/{epochs}")
        for features, expected in progress:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = loss_fn(logits, expected.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{total / (progress.n + 1):.4f}")
        valid_expected, valid_logits = evaluate_detector(model, cached["valid"], batch, device)
        valid_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(valid_logits), torch.from_numpy(valid_expected)
        ))
        print("backbone_valid_loss", valid_loss)
        completed = epoch + 1
        history.append({"epoch": completed, "validation_loss": valid_loss})
        (output / "backbone-progress.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if valid_loss < best - 1.0e-4:
            best, stale = valid_loss, 0
            torch.save(model.state_dict(), output / "backbone.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(output / "backbone.pt", map_location=device, weights_only=True))
    return completed, best


def tensors(values: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device):
    return tuple(torch.from_numpy(value).to(device) for value in values)


def train_reference_head(
    head: ReferenceHead,
    train_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    valid_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    device: torch.device,
    epochs: int = 80,
) -> tuple[int, float]:
    head.to(device)
    train_tensors = tensors(train_values, device)
    valid_tensors = tensors(valid_values, device)
    weight = torch.tensor(
        np.clip((len(train_values[2]) - train_values[2].sum(axis=0)) / np.maximum(train_values[2].sum(axis=0), 1), 1, 8),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best, stale, completed = float("inf"), 0, 0
    count, batch = len(train_values[2]), 256
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch):
            selected = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_tensors[0][selected], train_tensors[1][selected])
            loss = loss_fn(logits, train_tensors[2][selected])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            value = float(loss_fn(head(valid_tensors[0], valid_tensors[1]), valid_tensors[2]).cpu())
        completed = epoch + 1
        if value < best - 1.0e-5:
            best, stale = value, 0
            torch.save(head.state_dict(), output / "reference-head.pt")
        else:
            stale += 1
            if stale >= 10:
                break
    head.load_state_dict(torch.load(output / "reference-head.pt", map_location=device, weights_only=True))
    return completed, best


def reference_logits(head: ReferenceHead, values, device: torch.device) -> np.ndarray:
    head.eval()
    fused, base, _ = tensors(values, device)
    with torch.no_grad():
        return head(fused, base).cpu().numpy()


def balanced_replay(head: ReferenceHead, values, limit: int, seed: int) -> dict[str, np.ndarray]:
    fused, base, labels = values
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels.sum(axis=1) == 0)]
    groups.extend(np.flatnonzero(labels[:, index] > 0.5) for index in range(len(LABELS)))
    quota = max(1, limit // len(groups))
    selected = np.unique(np.concatenate([
        rng.choice(group, min(quota, len(group)), replace=False) for group in groups if len(group)
    ]))
    device = next(head.parameters()).device
    with torch.no_grad():
        projected = head.projected(torch.from_numpy(fused[selected]).to(device)).cpu().numpy()
        teacher = head(
            torch.from_numpy(fused[selected]).to(device), torch.from_numpy(base[selected]).to(device)
        ).cpu().numpy()
    return {
        "projected": projected.astype(np.float16),
        "base_logits": base[selected].astype(np.float16),
        "labels": labels[selected].astype(np.float16),
        "teacher_logits": teacher.astype(np.float16),
    }


def audio_windows(path: Path) -> np.ndarray:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor).astype(np.float32)
    audio = np.nan_to_num(audio).clip(-4, 4)
    if len(audio) <= SAMPLES:
        return np.pad(audio, (0, max(0, SAMPLES - len(audio))))[:SAMPLES][None].astype(np.float32)
    starts = list(range(0, len(audio) - SAMPLES + 1, STEP))
    if starts[-1] != len(audio) - SAMPLES:
        starts.append(len(audio) - SAMPLES)
    return np.stack([audio[start : start + SAMPLES] for start in starts]).astype(np.float32)


def encode_windows(model: Detector, path: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    waveform = torch.from_numpy(audio_windows(path))
    embeddings, logits = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(waveform), 16):
            value = model.encode(mel(waveform[start : start + 16], device))
            embeddings.append(value.cpu().numpy())
            logits.append(model.head(value).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def user_fused(embedding: np.ndarray, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
    means = np.repeat(mean[None], len(embedding), axis=0)
    deviations = np.repeat(deviation[None], len(embedding), axis=0)
    return np.concatenate((embedding, means, embedding - means, np.abs(embedding - means), deviations), axis=1).astype(np.float32)


def personalize(
    base_head: ReferenceHead,
    replay: dict[str, np.ndarray],
    clean_projected: np.ndarray,
    clean_base: np.ndarray,
    device: torch.device,
    steps: int,
) -> tuple[ReferenceHead, float]:
    head = copy.deepcopy(base_head).to(device)
    for parameter in head.parameters():
        parameter.requires_grad = False
    for parameter in head.adapter.parameters():
        parameter.requires_grad = True
    head.output.bias.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad], lr=8.0e-4, weight_decay=1.0e-4
    )
    replay_values = {key: torch.from_numpy(value.astype(np.float32)).to(device) for key, value in replay.items()}
    user_p = torch.from_numpy(clean_projected).to(device)
    user_b = torch.from_numpy(clean_base).to(device)
    zeros = torch.zeros((len(user_p), len(LABELS)), device=device)
    generator = torch.Generator().manual_seed(20260828)
    started = time.perf_counter()
    for _ in range(steps):
        indices = torch.randint(
            len(replay_values["labels"]),
            (min(128, len(replay_values["labels"])),),
            generator=generator,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        replay_logits = replay_values["base_logits"][indices] + head.correction(replay_values["projected"][indices])
        user_logits = user_b + head.correction(user_p)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(replay_logits, replay_values["labels"][indices])
            + 1.5 * torch.nn.functional.binary_cross_entropy_with_logits(user_logits, zeros)
            + 0.25 * torch.nn.functional.mse_loss(replay_logits, replay_values["teacher_logits"][indices])
        )
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return head, time.perf_counter() - started


def external_report(
    model: Detector,
    head: ReferenceHead,
    reference: Path,
    directory: Path,
    scales: np.ndarray,
    biases: np.ndarray,
    replay: dict[str, np.ndarray],
    steps: int,
    device: torch.device,
) -> tuple[dict, ReferenceHead, dict[str, np.ndarray]]:
    reference_started = time.perf_counter()
    ref_embedding, ref_base = encode_windows(model, reference, device)
    reference_seconds = time.perf_counter() - reference_started
    mean, deviation = ref_embedding.mean(axis=0), ref_embedding.std(axis=0) + 1.0e-4
    ref_fused = user_fused(ref_embedding, mean, deviation)
    with torch.no_grad():
        ref_projected = head.projected(torch.from_numpy(ref_fused).to(device)).cpu().numpy()
    adapted, seconds = personalize(head, replay, ref_projected, ref_base, device, steps)
    adapted.eval()
    with torch.no_grad():
        clean_logits = ref_base + adapted.correction(torch.from_numpy(ref_projected).to(device)).cpu().numpy()
    clean_probabilities = calibrated(clean_logits, scales, biases)
    # Internal EGFxSet thresholds are over-confident on its fixed devices. A
    # personalized threshold must be derived from the user's negative support,
    # not be forced above that source-domain threshold.
    user_threshold = np.clip(clean_probabilities.max(axis=0) + 0.02, 0.05, 0.95)
    rows, truths, predictions = [], [], []
    for path in sorted(directory.glob("*.wav")):
        if path.resolve() == reference.resolve():
            continue
        embedding, base = encode_windows(model, path, device)
        fused = user_fused(embedding, mean, deviation)
        with torch.no_grad():
            logits = adapted(torch.from_numpy(fused).to(device), torch.from_numpy(base).to(device)).cpu().numpy()
        probabilities = calibrated(logits, scales, biases)
        count = min(2, len(probabilities))
        top2 = np.partition(probabilities, len(probabilities) - count, axis=0)[-count:].mean(axis=0)
        truth = external_expected(path)
        predicted = top2 >= user_threshold
        truths.append(truth)
        predictions.append(predicted)
        rows.append({
            "file": path.name,
            "expected": [LABELS[index] for index in np.flatnonzero(truth)],
            "predicted": [LABELS[index] for index in np.flatnonzero(predicted)],
            "top2_mean": dict(zip(LABELS, map(float, top2))),
        })
    expected_values, predicted_values = np.stack(truths), np.stack(predictions)
    report_metrics = metrics(expected_values, predicted_values.astype(np.float32), np.full(len(LABELS), 0.5, dtype=np.float32))
    # metrics() expects probabilities; boolean 0/1 values with a 0.5 threshold are exact here.
    report_metrics["clean_false_positive"] = None
    return {
        "role": "personalization-development; clean.wav is reference and excluded from scoring",
        "reference": str(reference.resolve()),
        "reference_windows": int(len(ref_embedding)),
        "reference_encode_seconds": reference_seconds,
        "query_files": int(len(rows)),
        "adapter_steps": steps,
        "adapter_seconds": seconds,
        "trainable_parameters": sum(p.numel() for p in adapted.parameters() if p.requires_grad),
        "threshold": dict(zip(LABELS, map(float, user_threshold))),
        "clean_false_positive_measurable": False,
        "clean_false_positive_note": "the only unambiguous clean file is the personalization reference and is excluded from scoring",
        "metrics": report_metrics,
        "results": rows,
    }, adapted, {"mean": mean.astype(np.float32), "standard_deviation": deviation.astype(np.float32)}


def acceptance_gate(calibration_report: dict, test_report: dict, external: dict) -> dict:
    failures = []
    for split_name, report in (("calibration", calibration_report), ("test", test_report)):
        if report["clean_false_positive"] > 0.05:
            failures.append(f"{split_name} clean false-positive rate exceeds 5%")
        for name in LABELS:
            if report[name]["recall"] < 0.80:
                failures.append(f"{split_name} {name} recall is below 80%")
    if not external["clean_false_positive_measurable"]:
        failures.append("personalized user-domain Clean FP is not independently measurable")
    for name in LABELS:
        if external["metrics"][name]["recall"] < 0.80:
            failures.append(f"personalized external {name} recall is below 80%")
    return {
        "passed": not failures,
        "requirements": {"clean_false_positive_max": 0.05, "per_class_recall_min": 0.80},
        "failures": failures,
    }


def paired_manifest(clean: list[Source], anchors: list[Item]) -> dict:
    clean_by_group = {source.group: source for source in clean if source.domain == "egfx"}
    counts: dict[str, int] = defaultdict(int)
    pairs = []
    for item in anchors:
        dry = clean_by_group.get(item.source.group)
        if dry is None:
            continue
        active = [LABELS[index] for index, value in enumerate(item.target or ()) if value > 0.5]
        role = "+".join(active) if active else "modulation-negative"
        counts[role] += 1
        pairs.append({
            "group": item.source.group,
            "split": partition(item.source),
            "label": role,
            "dry": str(dry.path.resolve()),
            "wet": str(item.source.path.resolve()),
        })
    payload = {"schema": 1, "source": "EGFxSet real hardware matched dry/wet", "counts": dict(sorted(counts.items())), "pairs": pairs}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


class EncoderExport(torch.nn.Module):
    def __init__(self, model: Detector) -> None:
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor):
        embedding = self.model.encode(value)
        return embedding, self.model.head(embedding)


def export(model: Detector, head: ReferenceHead, output: Path) -> None:
    model.cpu().eval()
    head.cpu().eval()
    mel_input = torch.randn(1, 1, MELS, FRAMES)
    torch.onnx.export(EncoderExport(model), mel_input, output / "reference-encoder.onnx", input_names=["mel"], output_names=["embedding", "base_logits"], opset_version=17, dynamo=False)
    fused = torch.randn(1, EMBEDDING * 5)
    base = torch.randn(1, len(LABELS))
    torch.onnx.export(head, (fused, base), output / "reference-head.onnx", input_names=["fused", "base_logits"], output_names=["logits"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output / "reference-encoder.onnx"))
    onnx.checker.check_model(onnx.load(output / "reference-head.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "train" / "runs" / "clean-reference-real-v1")
    parser.add_argument(
        "--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav"
    )
    parser.add_argument("--external", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=3.0e-4)
    parser.add_argument("--adapter-steps", type=int, default=250)
    parser.add_argument("--replay", type=int, default=4096)
    parser.add_argument("--cache", type=Path, default=ROOT / "train" / "cache" / "clean-reference-real-v1")
    parser.add_argument(
        "--guitar-effects-chains",
        action="store_true",
        help="include the CC-BY-4.0 DAFx25 archived five-effect-chain dataset",
    )
    parser.add_argument(
        "--aachen-rir",
        action="store_true",
        help="add CC-BY-4.0 measured-room reverb examples by ephemeral convolution",
    )
    parser.add_argument(
        "--tonetwist-big-muff-nc",
        action="store_true",
        help="include audited CC-BY-NC ToneTwisT Big Muff pairs for research only",
    )
    parser.add_argument("--tonetwist-train-repeats", type=int, default=8)
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="stop after cached-feature backbone fitting",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="initialize fitting from a license-compatible detector checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, help="skip backbone fitting and finalize from this checkpoint")
    parser.add_argument(
        "--stem-stride",
        type=int,
        choices=(1, 2),
        default=2,
        help="must match the architecture used to train --checkpoint",
    )
    parser.add_argument("--epochs-completed", type=int, default=0)
    parser.add_argument("--best-validation-loss", type=float, default=float("nan"))
    parser.add_argument("--experiment")
    parser.add_argument("--hypothesis")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    clean, anchors = discover(args.data)
    parts = real_only_split(clean, anchors)
    chain_items = {name: [] for name in parts}
    if args.guitar_effects_chains:
        chain_items = discover_guitar_effect_chains(args.data / "guitar-effects-chains")
        if not any(chain_items.values()):
            parser.error("--guitar-effects-chains found no archived audio")
        for name in parts:
            parts[name].extend(chain_items[name])
    if args.aachen_rir:
        rirs = discover_aachen_rirs(args.data / "aachen-chapel-rir")
        clean_paths = {source.path.resolve() for source in clean if eligible_clean(source)}
        clean_paths.update(
            item.source.path.resolve()
            for items in chain_items.values()
            for item in items
            if guitar_effect_chain_is_clean(item.source)
        )
        parts = add_rir_reverb(parts, rirs, clean_paths)
    tonetwist_items = {name: [] for name in parts}
    if args.tonetwist_big_muff_nc:
        tonetwist_items = discover_tonetwist_big_muff_nc(
            args.data / "tonetwist-nc", args.tonetwist_train_repeats
        )
        for name in parts:
            parts[name].extend(tonetwist_items[name])
    audit(parts)
    for name, items in parts.items():
        print(name, len(items), "clean", sum(not any(item.target or ()) for item in items))
    cache_started = time.perf_counter()
    cached = {name: cache_features(items, name, args.cache, args.batch) for name, items in parts.items()}
    cache_seconds = time.perf_counter() - cache_started
    device = target_device()
    # A stride-two stem cuts user and offline compute substantially while
    # retaining 8 x 14 positions at the deepest feature map. The historical
    # blind model keeps its default stride-one contract.
    model = Detector(stem_stride=args.stem_stride).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        completed, best = args.epochs_completed, args.best_validation_loss
    else:
        if args.initial_checkpoint:
            model.load_state_dict(
                torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
            )
        completed, best = train_backbone(
            model,
            parts,
            cached,
            args.output,
            args.epochs,
            args.patience,
            args.batch,
            device,
            args.backbone_lr,
        )
    if args.backbone_only:
        report = {
            "experiment": args.experiment or args.output.name,
            "checkpoint": str((args.output / "backbone.pt").resolve()),
            "initial_checkpoint": str(args.initial_checkpoint.resolve())
            if args.initial_checkpoint
            else None,
            "device": str(device),
            "epochs_completed": completed,
            "best_validation_loss": best,
            "backbone_learning_rate": args.backbone_lr,
            "feature_cache_seconds": cache_seconds,
            "split": {name: len(items) for name, items in parts.items()},
            "aachen_rir": args.aachen_rir,
            "guitar_effects_chains": args.guitar_effects_chains,
            "tonetwist_big_muff_nc": args.tonetwist_big_muff_nc,
            "tonetwist_items": {
                name: len(items) for name, items in tonetwist_items.items()
            },
        }
        (args.output / "backbone-metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    encoded = {
        name: encode_items(model, items, cached[name], args.batch, device)
        for name, items in parts.items()
    }
    fused = {name: fuse(values) for name, values in encoded.items()}
    head = ReferenceHead()
    head_epochs, head_best = train_reference_head(head, fused["train"], fused["valid"], args.output, device)
    calibrate_logits = reference_logits(head, fused["calibrate"], device)
    test_logits = reference_logits(head, fused["test"], device)
    scales, biases = fit_platt(fused["calibrate"][2], calibrate_logits)
    calibrate_probabilities = calibrated(calibrate_logits, scales, biases)
    threshold = thresholds(fused["calibrate"][2], calibrate_probabilities)
    calibration_report = metrics(fused["calibrate"][2], calibrate_probabilities, threshold)
    test_report = metrics(fused["test"][2], calibrated(test_logits, scales, biases), threshold)
    replay = balanced_replay(head, fused["train"], args.replay, args.seed)
    np.savez_compressed(args.output / "replay-embeddings.npz", **replay)
    external, personalized, clean_profile = external_report(
        model,
        head,
        args.reference,
        args.external,
        scales,
        biases,
        replay,
        args.adapter_steps,
        device,
    )
    torch.save(personalized.state_dict(), args.output / "personalized-reference-head.pt")
    torch.save(
        {
            "adapter": personalized.adapter.state_dict(),
            "output_bias": personalized.output.bias.detach().cpu(),
        },
        args.output / "personalization.pt",
    )
    np.savez_compressed(args.output / "clean-profile.npz", **clean_profile)
    pair_payload = paired_manifest(clean, anchors)
    (args.output / "real-dry-wet-pairs.json").write_text(json.dumps(pair_payload, indent=2, sort_keys=True) + "\n")
    remixer_contract = {
        "schema": 1,
        "reference_profile": "clean-profile.npz: 256-dimensional mean and standard deviation",
        "matched_pairs": "real-dry-wet-pairs.json: EGFxSet real hardware only",
        "restoration_input": "future complex-STFT or waveform wet audio conditioned by the clean profile",
        "suggested_frontend": {"rate": RATE, "fft": [1024, 4096], "hop": [256, 1024]},
        "suggested_losses": ["multi-resolution STFT", "complex spectral", "waveform L1", "clean-profile consistency"],
        "limitations": [
            "log-Mel detector features are not decoded into audio",
            "clipping and other nonlinear Drive processing are not exactly invertible",
            "the current public real pairs do not establish cross-device restoration quality",
        ],
    }
    (args.output / "remixer-contract.json").write_text(
        json.dumps(remixer_contract, indent=2, sort_keys=True) + "\n"
    )
    export(model, head, args.output)
    report = {
        "experiment": args.experiment or args.output.name,
        "hypothesis": args.hypothesis
        or "A frozen real-hardware backbone plus clean-domain prototype and embedding-only adapter improves user-domain precision without generated positive audio.",
        "architecture": {
            "backbone": f"compact-resnet18-16-32-64-128-stem-stride{args.stem_stride}",
            "backbone_parameters": parameters(model),
            "embedding": EMBEDDING,
            "reference_statistics": "mean+standard-deviation",
            "fused_features": ["query", "clean-mean", "query-minus-clean", "absolute-difference", "clean-standard-deviation"],
            "adapter_bottleneck": 32,
            "personalized_parameters": external["trainable_parameters"],
        },
        "data_policy": {
            "reference_head_positive_audio": (
                "EGFxSet real hardware plus CC-BY-4.0 DAFx25 archived chains"
                if args.guitar_effects_chains
                else "EGFxSet archived real hardware only"
            ),
            "backbone_checkpoint": (
                str(args.checkpoint.resolve()) if args.checkpoint else "trained in this run"
            ),
            "clean_audio": "archived clean/direct recordings only; Guitar-TECHS mic/amp excluded",
            "adapter_generated_audio": False,
            "backbone_effect_data": "inherited from --checkpoint when supplied",
            "initial_checkpoint": (
                str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None
            ),
            "guitar_effects_chains": {
                "enabled": args.guitar_effects_chains,
                "record": "https://zenodo.org/records/7871720"
                if args.guitar_effects_chains
                else None,
                "license": "CC-BY-4.0" if args.guitar_effects_chains else None,
                "items": {name: len(items) for name, items in chain_items.items()},
                "split_policy": "PRS+Les Paul train; Strat validation/calibration; Telecaster test",
            },
            "tonetwist_big_muff_nc": {
                "enabled": args.tonetwist_big_muff_nc,
                "license": "CC-BY-NC-4.0; non-commercial research only"
                if args.tonetwist_big_muff_nc
                else None,
                "records": [
                    "https://zenodo.org/records/10797916",
                    "https://zenodo.org/records/10891515",
                ]
                if args.tonetwist_big_muff_nc
                else [],
                "items": {
                    name: len(items) for name, items in tonetwist_items.items()
                },
                "split_policy": "DIY train only; published EHX train/validation/test preserved",
            },
            "user_side_dynamic_dsp": False,
        },
        "split": {name: len(items) for name, items in parts.items()},
        "training": {"device": str(device), "feature_cache_seconds": cache_seconds, "backbone_epochs": completed, "best_validation_loss": best, "head_epochs": head_epochs, "head_best_validation_loss": head_best},
        "calibration": {"scale": dict(zip(LABELS, map(float, scales))), "bias": dict(zip(LABELS, map(float, biases))), "threshold": dict(zip(LABELS, map(float, threshold)))},
        "calibrate": calibration_report,
        "calibration_gate_feasibility": gate_feasibility(fused["calibrate"][2], calibrate_probabilities),
        "test": test_report,
        "personalization_development": external,
        "quality_gate": acceptance_gate(calibration_report, test_report, external),
        "replay": {"items": int(len(replay["labels"])), "format": "float16 projected embeddings; no source audio"},
        "remixer_boundary": {
            "clean_profile": "256-dimensional mean/std reference profile is reusable by a future complex-STFT restoration model",
            "matched_real_pairs": len(pair_payload["pairs"]),
            "pair_manifest_sha256": pair_payload["sha256"],
            "exact_drive_inversion": False,
            "note": "Log-Mel embeddings are not an audio decoder; future restoration must use complex STFT or waveform input conditioned by this profile.",
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "calibration.json").write_text(json.dumps(report["calibration"], indent=2, sort_keys=True) + "\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
