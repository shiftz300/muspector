#!/usr/bin/env python3
"""Distill the AFx-Rep Drive-identity teacher into the compact Inspector CNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from afx_identity_teacher import (
    EMBEDDING as TEACHER_EMBEDDING,
    LABELS,
    UNKNOWN,
    Example,
    audio_segment,
    big_muff_examples,
    cache_key,
    dafx_examples,
    drive_open_set_examples,
    egfx_examples,
    remfx_examples,
    tonetwist_examples,
)
from layout import CORPUS, DRIVE_DELAY_ENCODER_RUN, PEDAL_IDENTITY_CACHE, PEDAL_IDENTITY_RUN
from model import Detector, frontend


SEED = 20260829


class Student(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.encoder = Detector(stem_stride=1)
        self.encoder.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.identity = torch.nn.Linear(256, 4)
        self.teacher_projection = torch.nn.Linear(256, TEACHER_EMBEDDING)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for block in self.encoder.layers[4:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder.encode(mel)
        projected = torch.nn.functional.normalize(self.teacher_projection(embedding), dim=1)
        return self.identity(embedding), projected

    def runtime(self, mel: torch.Tensor) -> torch.Tensor:
        return self.identity(self.encoder.encode(mel))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.encoder.stem.eval()
            for block in self.encoder.layers[:4]:
                block.eval()
        return self


class RuntimeStudent(torch.nn.Module):
    def __init__(self, student: Student) -> None:
        super().__init__()
        self.encoder = student.encoder
        self.identity = student.identity

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.identity(self.encoder.encode(mel))


def examples(*, include_remfx: bool = True) -> tuple[list[Example], list[dict]]:
    rows = egfx_examples(CORPUS / "egfxset")
    tone, inventory = tonetwist_examples(CORPUS)
    rows.extend(tone)
    rows.extend(big_muff_examples(CORPUS / "tonetwist-nc"))
    open_set, open_set_inventory = drive_open_set_examples(CORPUS)
    rows.extend(open_set)
    if include_remfx:
        rows.extend(remfx_examples(CORPUS / "remfx-1-1"))
    rows.extend(dafx_examples(CORPUS / "guitar-effects-chains"))
    return rows, inventory + open_set_inventory


def request_key(request: tuple[Path, int, int]) -> str:
    path, segment, segments = request
    try:
        name = path.relative_to(CORPUS).as_posix()
    except ValueError:
        name = str(path.resolve())
    return f"{name}:{segment}/{segments}"


def build_mel_cache(
    requests: list[tuple[Path, int, int]], cache: Path, batch: int
) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        "\n".join(request_key(request) for request in requests).encode()
    ).hexdigest()[:20]
    output = cache / f"compact-mel-{signature}.npy"
    if output.exists():
        return np.load(output, mmap_mode="r")
    values = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(len(requests), 1, 128, 216)
    )
    for offset in range(0, len(requests), batch):
        selected = requests[offset : offset + batch]
        audio = np.stack([audio_segment(*request) for request in selected])
        audio = resample_poly(audio, 147, 160, axis=1).astype(np.float32)
        mel = frontend(torch.from_numpy(audio)).numpy().astype(np.float16)
        values[offset : offset + len(selected)] = mel
        if offset % (batch * 20) == 0:
            print("compact_identity_mel", min(offset + batch, len(requests)), len(requests), flush=True)
    values.flush()
    return np.load(output, mmap_mode="r")


def load_teacher_targets(
    requests: list[tuple[Path, int, int]], cache: Path
) -> np.ndarray:
    result = []
    missing = []
    for request in requests:
        path = cache / f"{cache_key(*request)}.npy"
        if not path.exists():
            missing.append(path)
        else:
            result.append(np.load(path).astype(np.float32))
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} AFx-Rep teacher embeddings; run afx_identity_teacher.py first"
        )
    return np.stack(result)


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def infer(model: Student, mel: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(indices), 128):
            batch = torch.from_numpy(np.asarray(mel[indices[offset : offset + 128]], dtype=np.float32))
            result.append(model.runtime(batch).numpy())
    return np.concatenate(result)


def select_threshold(logits: np.ndarray, truth: np.ndarray, maximum_far: float) -> float:
    values = probabilities(logits)
    score = values[:, :3].max(axis=1)
    prediction = values[:, :3].argmax(axis=1)
    known = truth < UNKNOWN
    best = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        accepted = np.logical_and(score >= threshold, values[:, UNKNOWN] < score)
        far = float(accepted[~known].mean()) if (~known).any() else 0.0
        recall = float(
            np.logical_and.reduce((accepted, known, prediction == truth)).sum()
            / max(known.sum(), 1)
        )
        key = (far <= maximum_far, recall, -far, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def metrics(
    logits: np.ndarray, truth: np.ndarray, categories: np.ndarray, threshold: float
) -> dict:
    values = probabilities(logits)
    prediction = values[:, :3].argmax(axis=1)
    score = values[:, :3].max(axis=1)
    accepted = np.logical_and(score >= threshold, values[:, UNKNOWN] < score)
    result = np.where(accepted, prediction, UNKNOWN)
    known = truth < UNKNOWN
    per_class = {}
    for index, name in enumerate(LABELS):
        expected = truth == index
        actual = result == index
        tp = int(np.logical_and(expected, actual).sum())
        fp = int(np.logical_and(~expected, actual).sum())
        fn = int(np.logical_and(expected, ~actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(expected.sum()),
        }
    return {
        "samples": int(len(truth)),
        "closed_set_accuracy": float((prediction[known] == truth[known]).mean()),
        "correct_accept_rate": float((result[known] == truth[known]).mean()),
        "negative_false_accept_rate": float(accepted[~known].mean()),
        "per_class": per_class,
        "negative_category_false_accept": {
            category: float(accepted[np.logical_and(~known, categories == category)].mean())
            for category in sorted(set(categories[~known]))
        },
    }


def train(
    model: Student,
    mel: np.ndarray,
    teacher: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
    epochs: int,
    output: Path,
) -> dict:
    train_indices = np.flatnonzero(splits == "train")
    train_labels = torch.from_numpy(labels[train_indices])
    train_categories = categories[train_indices]
    weights = torch.zeros(len(train_indices))
    for label in range(4):
        label_mask = train_labels == label
        names = sorted(set(train_categories[label_mask.numpy()]))
        for name in names:
            mask = torch.logical_and(label_mask, torch.from_numpy(train_categories == name))
            weights[mask] = 1.0 / (4 * len(names) * mask.sum())
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices), train_labels),
        batch_size=32,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
    )
    slow = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    fast = list(model.identity.parameters()) + list(model.teacher_projection.parameters())
    optimizer = torch.optim.AdamW(
        ({"params": slow, "lr": 8.0e-5}, {"params": fast, "lr": 4.0e-4}),
        weight_decay=3.0e-4,
    )
    valid_indices = np.flatnonzero(splits == "valid")
    best = float("inf")
    stale = 0
    completed = 0
    started = time.perf_counter()
    checkpoint = output / "compact-drive-identity.pt"
    for epoch in range(epochs):
        model.train()
        for index, target in loader:
            index_values = index.numpy()
            batch_mel = torch.from_numpy(np.asarray(mel[index_values], dtype=np.float32))
            teacher_target = torch.from_numpy(teacher[index_values])
            logits, projected = model(batch_mel)
            class_loss = torch.nn.functional.cross_entropy(logits, target)
            distill_loss = (1.0 - torch.nn.functional.cosine_similarity(projected, teacher_target)).mean()
            loss = class_loss + 0.35 * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*slow, *fast], 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_logits = infer(model, mel, valid_indices)
            validation = float(
                torch.nn.functional.cross_entropy(
                    torch.from_numpy(valid_logits), torch.from_numpy(labels[valid_indices])
                )
            )
        completed = epoch + 1
        print("compact_identity_epoch", completed, validation, flush=True)
        if validation < best - 1.0e-5:
            best = validation
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return {
        "epochs": completed,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "runtime_parameters": sum(value.numel() for value in RuntimeStudent(model).parameters()),
    }


def hardware_report(
    model: Student,
    mel: np.ndarray,
    request_index: dict[tuple[Path, int, int], int],
    directory: Path,
    threshold: float,
) -> dict:
    rows = []
    for path in sorted(directory.glob("*.wav")):
        if not any(token in path.stem.lower() for token in ("drive", "fuzz", "rat")):
            continue
        indices = np.asarray([request_index[(path, segment, 3)] for segment in range(3)])
        probability = probabilities(infer(model, mel, indices)).mean(axis=0)
        candidate = int(probability[:3].argmax())
        accepted = probability[candidate] >= threshold and probability[UNKNOWN] < probability[candidate]
        predicted = LABELS[candidate] if accepted else None
        expected = "RAT" if "rat" in path.stem.lower() else None
        rows.append(
            {
                "file": path.name,
                "expected": expected,
                "candidate": LABELS[candidate],
                "predicted": predicted,
                "score": float(probability[candidate]),
                "unknown_probability": float(probability[UNKNOWN]),
            }
        )
    rats = [row for row in rows if row["expected"] == "RAT"]
    unknown = [row for row in rows if row["expected"] is None]
    return {
        "rows": rows,
        "rat_recall": sum(row["predicted"] == "RAT" for row in rats) / len(rats),
        "noncatalog_false_accept": sum(row["predicted"] is not None for row in unknown) / len(unknown),
    }


def development_threshold(
    model: Student,
    mel: np.ndarray,
    request_index: dict[tuple[Path, int, int], int],
    directory: Path,
) -> float:
    truth = []
    logits = []
    for path in sorted(directory.glob("*.wav")):
        if not any(token in path.stem.lower() for token in ("drive", "fuzz", "rat")):
            continue
        indices = np.asarray([request_index[(path, segment, 3)] for segment in range(3)])
        probability = probabilities(infer(model, mel, indices)).mean(axis=0)
        logits.append(np.log(probability.clip(1.0e-8)))
        truth.append(LABELS.index("RAT") if "rat" in path.stem.lower() else UNKNOWN)
    return select_threshold(np.stack(logits), np.asarray(truth), 0.20)


def export(model: Student, path: Path) -> float:
    runtime = RuntimeStudent(model).eval()
    dummy = torch.zeros(1, 1, 128, 216)
    torch.onnx.export(
        runtime,
        dummy,
        path,
        input_names=["log_mel"],
        output_names=["identity_logits"],
        opset_version=17,
        dynamo=False,
    )
    expected = runtime(dummy).detach().numpy()
    actual = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"log_mel": dummy.numpy()}
    )[0]
    return float(np.max(np.abs(expected - actual)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DRIVE_DELAY_ENCODER_RUN / "best.pt")
    parser.add_argument(
        "--initial-student",
        type=Path,
        help="warm-start from an earlier compact student checkpoint",
    )
    parser.add_argument("--teacher-cache", type=Path, default=PEDAL_IDENTITY_CACHE / "afx-rep-teacher")
    parser.add_argument("--cache", type=Path, default=PEDAL_IDENTITY_CACHE / "compact-distill")
    parser.add_argument("--output", type=Path, default=PEDAL_IDENTITY_RUN / "compact-distill")
    parser.add_argument("--development", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows, inventory = examples()
    requests = {(row.path, row.segment, row.segments) for row in rows}
    hardware_paths = [
        path
        for path in args.development.glob("*.wav")
        if any(token in path.stem.lower() for token in ("drive", "fuzz", "rat"))
    ]
    requests.update((path, segment, 3) for path in hardware_paths for segment in range(3))
    requests = sorted(requests, key=request_key)
    request_index = {request: index for index, request in enumerate(requests)}
    mel = build_mel_cache(requests, args.cache, args.batch)
    teacher = load_teacher_targets(requests, args.teacher_cache)
    row_indices = np.asarray(
        [request_index[(row.path, row.segment, row.segments)] for row in rows]
    )
    row_mel = mel[row_indices]
    row_teacher = teacher[row_indices]
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    splits = np.asarray([row.split for row in rows])
    categories = np.asarray([row.category for row in rows])
    args.output.mkdir(parents=True, exist_ok=True)
    model = Student(args.checkpoint)
    if args.initial_student is not None:
        model.load_state_dict(
            torch.load(args.initial_student, map_location="cpu", weights_only=True)
        )
    training = train(model, row_mel, row_teacher, labels, splits, categories, args.epochs, args.output)
    calibrate = np.flatnonzero(splits == "calibrate")
    test = np.flatnonzero(splits == "test")
    calibration_threshold = select_threshold(infer(model, row_mel, calibrate), labels[calibrate], 0.05)
    dev_threshold = development_threshold(model, mel, request_index, args.development)
    reports = {
        "public_calibration": {
            "threshold": calibration_threshold,
            "test": metrics(infer(model, row_mel, test), labels[test], categories[test], calibration_threshold),
            "hardware_development": hardware_report(model, mel, request_index, args.development, calibration_threshold),
        },
        "hardware_development_calibration": {
            "threshold": dev_threshold,
            "uses_hardware_labels": True,
            "hardware_development": hardware_report(model, mel, request_index, args.development, dev_threshold),
            "public_test": metrics(infer(model, row_mel, test), labels[test], categories[test], dev_threshold),
        },
    }
    onnx_path = args.output / "compact-drive-identity.onnx"
    parity = export(model, onnx_path)
    hardware = reports["hardware_development_calibration"]["hardware_development"]
    public = reports["hardware_development_calibration"]["public_test"]
    failures = []
    if hardware["rat_recall"] < 0.5:
        failures.append("hardware-development RAT recall is below 50%")
    if hardware["noncatalog_false_accept"] > 0.2:
        failures.append("hardware-development non-catalog false accept exceeds 20%")
    if public["negative_false_accept_rate"] > 0.05:
        failures.append("public test negative false accept exceeds 5%")
    for name, value in public["per_class"].items():
        if value["recall"] < 0.65:
            failures.append(name + " public test recall is below 65%")
    payload = {
        "experiment": "compact-drive-identity-distillation",
        "architecture": {"student": "Inspector ResNet18 with last four residual blocks fine-tuned", "teacher": "frozen ST-ITO AFx-Rep Cnn14", "identity_outputs": [*LABELS, "internal-open-set"], "user_gradient_updates": 0},
        "data": {"examples": len(rows), "tone_twist": inventory, "external_test_folder_role": "labeled hardware development calibration; not a final test", "split_counts": {name: int((splits == name).sum()) for name in ("train", "valid", "calibrate", "test")}},
        "training": training,
        "reports": reports,
        "export": {"path": str(onnx_path.resolve()), "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(), "max_absolute_difference": parity},
        "development_gate": {"passed": not failures, "failures": failures},
        "integration_eligible": not failures,
        "release_eligible": False,
        "release_reason": "non-commercial research data plus hardware-development threshold tuning require a new untouched multi-device final test",
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output / "calibration.json").write_text(json.dumps({"threshold": dev_threshold, "labels": [*LABELS, "internal-open-set"], "development_calibrated": True}, indent=2) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
