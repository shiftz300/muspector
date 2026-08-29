#!/usr/bin/env python3
"""Evaluate an effect-specialized AFx-Rep teacher for Drive identity.

This is an offline research experiment, not a runtime model.  It keeps user
recordings evaluation-only and compares absolute and Clean-conditioned probes
before any compact-model distillation is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data import (
    Source,
    discover_guitar_effect_chains,
    discover_remfx_1_1,
    guitar_effect_chain_is_clean,
    partition,
)
from layout import CORPUS, PEDAL_IDENTITY_CACHE, PEDAL_IDENTITY_RUN
from model import SAMPLES
from pedal_identity import CATALOG, discover, select_dry, tone_devices, tone_source_key


SEED = 20260829
RATE = 48_000
WINDOW = 240_000
EMBEDDING = 512
LABELS = CATALOG["drive"]
UNKNOWN = 3
CHECKPOINT_SHA256 = "3587c4f3a1a8ecbc53b8023c480a0e6ff80719bcc26ce6ee6d08b8daf41d75d4"


@dataclass(frozen=True)
class Example:
    path: Path
    reference: Path
    label: int
    split: str
    category: str
    domain: str
    segment: int = 0
    segments: int = 1


class Probe(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None]
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(len(mean), 192),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(192, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp((value - self.mean) / self.deviation, -8.0, 8.0)
        return self.layers(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher(source: Path, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(source))
    from st_ito.models.panns import Cnn14

    model = Cnn14(
        embed_dim=EMBEDDING,
        sample_rate=float(RATE),
        window_size=2048,
        hop_size=1024,
        mel_bins=128,
        fmin=20.0,
        fmax=20_000.0,
        use_batchnorm=True,
        input_norm="minmax",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("encoder."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("encoder.")
    }
    model.load_state_dict(state)
    return model.eval()


def audio_segment(path: Path, segment: int, segments: int) -> np.ndarray:
    with soundfile.SoundFile(path) as stream:
        source_rate = int(stream.samplerate)
        source_window = int(round(WINDOW * source_rate / RATE))
        maximum = max(0, len(stream) - source_window)
        start = int(round(maximum * (segment + 0.5) / segments)) if segments > 1 else maximum // 2
        stream.seek(start)
        value = stream.read(source_window, dtype="float32", always_2d=True).mean(axis=1)
    if source_rate != RATE:
        divisor = math.gcd(source_rate, RATE)
        value = resample_poly(value, RATE // divisor, source_rate // divisor).astype(np.float32)
    value = np.pad(value, (0, max(0, WINDOW - len(value))))[:WINDOW]
    return np.nan_to_num(value).clip(-4.0, 4.0)


def cache_key(path: Path, segment: int, segments: int) -> str:
    try:
        name = path.relative_to(CORPUS).as_posix()
    except ValueError:
        name = str(path.resolve())
    return hashlib.sha256(f"{CHECKPOINT_SHA256}:{name}:{segment}/{segments}".encode()).hexdigest()[:24]


def embed_samples(
    model: torch.nn.Module,
    requests: set[tuple[Path, int, int]],
    cache: Path,
    batch: int,
) -> dict[tuple[Path, int, int], np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    result: dict[tuple[Path, int, int], np.ndarray] = {}
    missing = []
    for request in sorted(requests, key=lambda value: (str(value[0]), value[1], value[2])):
        output = cache / f"{cache_key(*request)}.npy"
        if output.exists():
            result[request] = np.load(output).astype(np.float32)
        else:
            missing.append((request, output))
    started = time.perf_counter()
    for offset in range(0, len(missing), batch):
        selected = missing[offset : offset + batch]
        audio = np.stack([audio_segment(*request) for request, _ in selected])
        value = torch.from_numpy(audio)[:, None]
        value /= value.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
        with torch.inference_mode():
            mid, _ = model(value)
            encoded = torch.nn.functional.normalize(mid, p=2, dim=1).cpu().numpy()
        for (request, output), embedding in zip(selected, encoded):
            np.save(output, embedding.astype(np.float16))
            result[request] = embedding.astype(np.float32)
        if offset % (batch * 10) == 0:
            print("afx_identity_cache", min(offset + batch, len(missing)), len(missing), flush=True)
    return result


def limited(values: list, limit: int) -> list:
    return sorted(values, key=lambda value: hashlib.sha256(str(value).encode()).hexdigest())[:limit]


def egfx_examples(root: Path) -> list[Example]:
    records = discover(root)
    clean: dict[tuple[str, str], list] = {}
    for record in records:
        if record.label == "Clean":
            clean.setdefault((record.split, record.pickup), []).append(record)
    by_role: dict[tuple[str, str], list] = {}
    for record in records:
        by_role.setdefault((record.split, record.label), []).append(record)
    result = []
    known = {name: index for index, name in enumerate(LABELS)}
    for (split, label), rows in sorted(by_role.items()):
        cap = 256 if split == "train" else 64
        if label not in known:
            cap //= 4
        for record in limited(rows, cap):
            candidates = [
                value
                for value in clean[(split, record.pickup)]
                if value.group != record.group
            ]
            index = int(hashlib.sha256(record.group.encode()).hexdigest()[:8], 16)
            reference = candidates[index % len(candidates)].path
            result.append(
                Example(
                    record.path,
                    reference,
                    known.get(label, UNKNOWN),
                    split,
                    "EGFx " + label,
                    "egfx-" + record.pickup,
                )
            )
    return result


def tone_split(path: Path, source_key: str) -> str:
    parts = {part.lower() for part in path.parts}
    if "test" in parts and "trainvaltest" not in parts:
        return "test"
    if source_key in {"idmt-gtr2", "nam", "trainvaltest"}:
        return "train"
    if source_key == "idmt-gtr4-sg":
        return "valid"
    if source_key == "prvt-gtr":
        return "calibrate"
    return partition(Source(path, "afx-teacher:" + source_key, "afx-teacher"))


def tonetwist_examples(root: Path) -> tuple[list[Example], list[dict]]:
    result = []
    inventory = []
    for device in tone_devices(root):
        dry_root = device.root if device.key == "ts9" else root / "tonetwist-pedal-identity-dry-nc"
        dry = sorted(
            path
            for path in dry_root.rglob("*.wav")
            if "dry" in {part.lower() for part in path.parts}
            or ".input." in path.name.lower()
            or path.name.lower().endswith("input.wav")
        )
        wet = sorted(
            path
            for path in device.root.rglob("*.target.wav")
            if "bass" not in tone_source_key(path)
        )
        count = 0
        for path in wet:
            reference = select_dry(path, dry)
            key = tone_source_key(path)
            for segment in range(3):
                result.append(
                    Example(
                        path,
                        reference,
                        LABELS.index(device.label),
                        tone_split(path, key),
                        "ToneTwist " + device.key,
                        "tonetwist-" + device.key + ":" + key,
                        segment,
                        3,
                    )
                )
                count += 1
        inventory.append({"device": device.key, "files": len(wet), "examples": count})
    return result, inventory


def big_muff_examples(root: Path) -> list[Example]:
    pairs = (
        ("train", "diy", root / "diy-big-muff/DRY/input.wav", root / "diy-big-muff/DIY-ElectroHarmonix-BigMuff/Vol=6_Tone=2_Sustain=5/target.wav"),
        ("train", "ehx-train", root / "ehx-big-muff/DRY/trainval/train.input.wav", root / "ehx-big-muff/ElectroHarmonix-BigMuff/trainval/S050_V100/S050_V100.train.target.wav"),
        ("valid", "ehx-valid", root / "ehx-big-muff/DRY/trainval/val.input.wav", root / "ehx-big-muff/ElectroHarmonix-BigMuff/trainval/S050_V100/S050_V100.val.target.wav"),
        ("calibrate", "ehx-valid", root / "ehx-big-muff/DRY/trainval/val.input.wav", root / "ehx-big-muff/ElectroHarmonix-BigMuff/trainval/S050_V100/S050_V100.val.target.wav"),
        ("test", "ehx-test", root / "ehx-big-muff/DRY/test/test.input.wav", root / "ehx-big-muff/ElectroHarmonix-BigMuff/test/S050_V100/S050_V100.test.target.wav"),
    )
    return [
        Example(wet, dry, UNKNOWN, split, "ToneTwist Big Muff", "big-muff-" + name, segment, 3)
        for split, name, dry, wet in pairs
        for segment in range(3)
    ]


def drive_open_set_examples(root: Path) -> tuple[list[Example], list[dict]]:
    """Load additional real Drive/Fuzz devices for identity research.

    Callers decide whether these named devices are catalog positives or
    open-set exposure.  The five Klon recordings have no upstream split, so
    they use a deterministic 2/1/1/1 recording-disjoint partition.
    """

    devices = (
        ("klon", "tonetwist-drive-open-set-klon-nc", True),
        ("metal-muff", "tonetwist-drive-open-set-metal-muff-nc", False),
        ("fuzzy-logic", "tonetwist-drive-open-set-fuzzy-logic-nc", False),
        ("silly-fuzz", "tonetwist-drive-open-set-silly-fuzz-nc", False),
    )
    result = []
    inventory = []
    for name, directory, external_dry in devices:
        device_root = root / directory
        if not device_root.exists():
            continue
        dry_root = device_root if external_dry else root / "tonetwist-pedal-identity-dry-nc"
        dry = sorted(
            path
            for path in dry_root.rglob("*.wav")
            if "dry" in {part.lower() for part in path.parts}
            or ".input." in path.name.lower()
            or path.name.lower().endswith("input.wav")
        )
        wet = sorted(
            path
            for path in device_root.rglob("*.target.wav")
            if "bass" not in tone_source_key(path)
        )
        unsplit = [path for path in wet if tone_source_key(path) == "trainvaltest"]
        unsplit_partition = {
            path: split
            for path, split in zip(
                unsplit,
                ["train"] * max(0, len(unsplit) - 3)
                + ["valid", "calibrate", "test"][-min(3, len(unsplit)) :],
            )
        }
        for path in wet:
            reference = select_dry(path, dry)
            source_key = tone_source_key(path)
            if source_key == "trainvaltest":
                split = unsplit_partition[path]
            else:
                split = tone_split(path, source_key)
            for segment in range(3):
                result.append(
                    Example(
                        path,
                        reference,
                        UNKNOWN,
                        split,
                        "ToneTwist open-set " + name,
                        "tonetwist-open-set:" + name + ":" + source_key,
                        segment,
                        3,
                    )
                )
        inventory.append(
            {"device": name, "files": len(wet), "examples": len(wet) * 3}
        )
    return result, inventory


def remfx_examples(root: Path) -> list[Example]:
    items = discover_remfx_1_1(root)
    clean = {item.source.group: item.source.path for item in items if item.target == (0.0, 0.0, 0.0)}
    by_split = {name: [] for name in ("train", "valid", "calibrate", "test")}
    for item in items:
        if item.target is not None and item.target[0] > 0.5:
            by_split[partition(item.source)].append(item)
    result = []
    for split, rows in by_split.items():
        cap = 128 if split == "train" else 32
        for item in limited(rows, cap):
            result.append(
                Example(item.source.path, clean[item.source.group], UNKNOWN, split, "RemFX distortion", "remfx")
            )
    return result


def dafx_examples(root: Path) -> list[Example]:
    parts = discover_guitar_effect_chains(root)
    result = []
    for split, items in parts.items():
        clean = {}
        for item in items:
            if guitar_effect_chain_is_clean(item.source):
                clean[item.source.group] = item.source.path
        wet = [
            item
            for item in items
            if item.target is not None and item.target[0] > 0.5 and item.source.group in clean
        ]
        cap = 128 if split == "train" else 32
        for item in limited(wet, cap):
            result.append(
                Example(item.source.path, clean[item.source.group], UNKNOWN, split, "DAFx non-catalog drive", item.source.domain)
            )
    return result


def features(
    examples: list[Example], embeddings: dict[tuple[Path, int, int], np.ndarray], mode: str
) -> np.ndarray:
    result = []
    reference_cache: dict[Path, np.ndarray] = {}
    for row in examples:
        query = embeddings[(row.path, row.segment, row.segments)]
        if row.reference not in reference_cache:
            keys = [key for key in embeddings if key[0] == row.reference]
            reference_cache[row.reference] = np.stack([embeddings[key] for key in keys]).mean(axis=0)
        reference = reference_cache[row.reference]
        relative = np.concatenate((query, reference, query - reference, np.abs(query - reference), query * reference))
        result.append(query if mode == "absolute" else np.concatenate((query, relative)))
    return np.stack(result).astype(np.float32)


def infer(model: Probe, values: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            result.append(model(torch.from_numpy(values[start : start + 512])).numpy())
    return np.concatenate(result)


def train_probe(
    values: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
    epochs: int,
) -> tuple[Probe, dict]:
    selected = splits == "train"
    mean = values[selected].mean(axis=0)
    deviation = values[selected].std(axis=0) + 1.0e-4
    model = Probe(mean, deviation)
    train_values = torch.from_numpy(values[selected])
    train_labels = torch.from_numpy(labels[selected])
    train_categories = categories[selected]
    weights = torch.zeros(len(train_labels))
    for label in range(4):
        label_mask = train_labels == label
        names = sorted(set(train_categories[label_mask.numpy()]))
        for name in names:
            mask = torch.logical_and(label_mask, torch.from_numpy(train_categories == name))
            weights[mask] = 1.0 / (4 * len(names) * mask.sum())
    loader = DataLoader(
        TensorDataset(train_values, train_labels),
        batch_size=128,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
    best = float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for batch_values, batch_labels in loader:
            loss = torch.nn.functional.cross_entropy(model(batch_values), batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        valid = splits == "valid"
        logits = infer(model, values[valid])
        loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels[valid])))
        print("afx_identity_epoch", epoch + 1, loss, flush=True)
        if loss < best - 1.0e-5:
            best = loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    return model, {
        "epochs": epoch + 1,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
    }


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def threshold_for(logits: np.ndarray, truth: np.ndarray) -> float:
    values = probabilities(logits)
    prediction = values[:, :3].argmax(axis=1)
    score = values[:, :3].max(axis=1)
    known = truth < 3
    best = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        accepted = np.logical_and(score >= threshold, values[:, 3] < score)
        false_accept = float(accepted[~known].mean()) if (~known).any() else 0.0
        correct = float(np.logical_and.reduce((accepted, known, prediction == truth)).sum() / max(known.sum(), 1))
        key = (false_accept <= 0.05, correct, -false_accept, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def report(logits: np.ndarray, truth: np.ndarray, categories: np.ndarray, threshold: float) -> dict:
    values = probabilities(logits)
    prediction = values[:, :3].argmax(axis=1)
    score = values[:, :3].max(axis=1)
    accepted = np.logical_and(score >= threshold, values[:, 3] < score)
    result = np.where(accepted, prediction, UNKNOWN)
    known = truth < 3
    per_class = {}
    for index, name in enumerate(LABELS):
        expected = truth == index
        actual = result == index
        tp = int(np.logical_and(expected, actual).sum())
        fp = int(np.logical_and(~expected, actual).sum())
        fn = int(np.logical_and(expected, ~actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": int(expected.sum())}
    return {
        "samples": len(truth),
        "closed_set_accuracy": float((prediction[known] == truth[known]).mean()),
        "correct_accept_rate": float((result[known] == truth[known]).mean()),
        "negative_false_accept_rate": float(accepted[~known].mean()),
        "per_class": per_class,
        "negative_category_false_accept": {
            name: float(accepted[np.logical_and(~known, categories == name)].mean())
            for name in sorted(set(categories[~known]))
        },
    }


def hardware(
    model: Probe,
    mode: str,
    threshold: float,
    embeddings: dict[tuple[Path, int, int], np.ndarray],
    directory: Path,
    reference: Path,
) -> dict:
    reference_values = np.stack([embeddings[(reference, segment, 3)] for segment in range(3)]).mean(axis=0)
    rows = []
    for path in sorted(directory.glob("*.wav")):
        if path == reference or not any(token in path.stem.lower() for token in ("drive", "fuzz", "rat")):
            continue
        query = np.stack([embeddings[(path, segment, 3)] for segment in range(3)])
        relative = np.concatenate(
            (
                query,
                np.broadcast_to(reference_values, query.shape),
                query - reference_values,
                np.abs(query - reference_values),
                query * reference_values,
            ),
            axis=1,
        )
        values = query if mode == "absolute" else np.concatenate((query, relative), axis=1)
        probability = probabilities(infer(model, values)).mean(axis=0)
        index = int(probability[:3].argmax())
        accepted = probability[index] >= threshold and probability[3] < probability[index]
        predicted = LABELS[index] if accepted else None
        expected = "RAT" if "rat" in path.stem.lower() else None
        rows.append(
            {
                "file": path.name,
                "expected": expected,
                "candidate": LABELS[index],
                "predicted": predicted,
                "score": float(probability[index]),
                "unknown_probability": float(probability[3]),
                "probability": {
                    name: float(probability[label_index])
                    for label_index, name in enumerate(LABELS)
                },
            }
        )
    rat = [row for row in rows if row["expected"] == "RAT"]
    unknown = [row for row in rows if row["expected"] is None]
    return {
        "rows": rows,
        "rat_recall": sum(row["predicted"] == "RAT" for row in rat) / len(rat),
        "noncatalog_false_accept": sum(row["predicted"] is not None for row in unknown) / len(unknown),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/private/tmp/muspector-st-ito"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/private/tmp/muspector-afx-rep.ckpt"))
    parser.add_argument("--cache", type=Path, default=PEDAL_IDENTITY_CACHE / "afx-rep-teacher")
    parser.add_argument("--output", type=Path, default=PEDAL_IDENTITY_RUN / "afx-rep-teacher")
    parser.add_argument("--development", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument("--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("AFx-Rep checkpoint hash does not match the audited official file")

    examples = egfx_examples(CORPUS / "egfxset")
    tone, inventory = tonetwist_examples(CORPUS)
    examples.extend(tone)
    examples.extend(big_muff_examples(CORPUS / "tonetwist-nc"))
    open_set, open_set_inventory = drive_open_set_examples(CORPUS)
    examples.extend(open_set)
    examples.extend(remfx_examples(CORPUS / "remfx-1-1"))
    examples.extend(dafx_examples(CORPUS / "guitar-effects-chains"))
    requests = {(row.path, row.segment, row.segments) for row in examples}
    for row in examples:
        reference_segments = 3 if soundfile.info(row.reference).duration > 10 else 1
        requests.update((row.reference, segment, reference_segments) for segment in range(reference_segments))
    hardware_paths = [
        path
        for path in args.development.glob("*.wav")
        if path == args.reference or any(token in path.stem.lower() for token in ("drive", "fuzz", "rat"))
    ]
    requests.update((path, segment, 3) for path in hardware_paths for segment in range(3))

    teacher = load_teacher(args.source, args.checkpoint)
    embeddings = embed_samples(teacher, requests, args.cache, args.batch)
    labels = np.asarray([row.label for row in examples], dtype=np.int64)
    splits = np.asarray([row.split for row in examples])
    categories = np.asarray([row.category for row in examples])
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in ("absolute", "hybrid"):
        values = features(examples, embeddings, mode)
        model, training = train_probe(values, labels, splits, categories, args.epochs)
        calibrate = splits == "calibrate"
        test = splits == "test"
        threshold = threshold_for(infer(model, values[calibrate]), labels[calibrate])
        test_report = report(infer(model, values[test]), labels[test], categories[test], threshold)
        hardware_report = hardware(model, mode, threshold, embeddings, args.development, args.reference)
        torch.save(model.state_dict(), args.output / f"{mode}-probe.pt")
        reports[mode] = {"training": training, "threshold": threshold, "test": test_report, "hardware_development": hardware_report}
    best = max(reports, key=lambda name: (reports[name]["hardware_development"]["rat_recall"], -reports[name]["hardware_development"]["noncatalog_false_accept"]))
    failures = []
    selected = reports[best]
    if selected["hardware_development"]["rat_recall"] < 0.5:
        failures.append("hardware RAT recall is below 50%")
    if selected["hardware_development"]["noncatalog_false_accept"] > 0.2:
        failures.append("hardware non-catalog false accept exceeds 20%")
    for name, value in selected["test"]["per_class"].items():
        if value["recall"] < 0.65:
            failures.append(name + " test recall is below 65%")
    if selected["test"]["negative_false_accept_rate"] > 0.05:
        failures.append("test negative false accept exceeds 5%")
    payload = {
        "experiment": "afx-rep-drive-identity-teacher",
        "architecture": {"teacher": "ST-ITO AFx-Rep Cnn14", "teacher_parameters": sum(value.numel() for value in teacher.parameters()), "embedding": EMBEDDING, "probes": ["absolute", "hybrid Clean-conditioned"], "user_gradient_updates": 0},
        "upstream": {"source": "https://github.com/csteinmetz1/st-ito", "source_revision": "ec50e0cc647cc3637d0cbbfc01d059c682c9fb27", "checkpoint": "https://huggingface.co/csteinmetz1/afx-rep", "checkpoint_sha256": CHECKPOINT_SHA256, "license": "Apache-2.0"},
        "data": {"tone_twist": inventory, "tone_twist_open_set": open_set_inventory, "examples": len(examples), "split_counts": {name: int((splits == name).sum()) for name in ("train", "valid", "calibrate", "test")}, "external_used_for_training_or_calibration": False},
        "reports": reports,
        "selected_probe": best,
        "teacher_gate": {"passed": not failures, "failures": failures},
        "runtime_eligible": False,
        "runtime_reason": "teacher is too large; a passing teacher must be distilled into the compact Inspector identity branch",
        "release_eligible": False,
        "release_reason": "ToneTwist and RemFX are non-commercial research data and no untouched physical-device final set exists",
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
