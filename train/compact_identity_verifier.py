#!/usr/bin/env python3
"""Train the candidate-independent open-set verifier for compact Drive identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from afx_identity_teacher import LABELS, UNKNOWN as SOURCE_UNKNOWN
from compact_identity_distill import (
    SEED,
    Student,
    build_mel_cache,
    examples,
    request_key,
)
from layout import DRIVE_DELAY_ENCODER_RUN, PEDAL_IDENTITY_CACHE, PEDAL_IDENTITY_RUN


CATALOG = (
    *LABELS,
    "BigMuff",
    "MetalMuff",
    "FuzzyLogic",
    "SillyFuzz",
)
UNKNOWN = len(CATALOG)
FEATURES = 512
CATEGORY_CATALOG = {
    "ToneTwist Big Muff": "BigMuff",
    "ToneTwist open-set metal-muff": "MetalMuff",
    "ToneTwist open-set fuzzy-logic": "FuzzyLogic",
    "ToneTwist open-set silly-fuzz": "SillyFuzz",
}


class CatalogHead(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None]
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(FEATURES, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(128, len(CATALOG)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp((value - self.mean) / self.deviation, -8.0, 8.0)
        return self.layers(value)


class Verifier(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None]
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(FEATURES, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp((value - self.mean) / self.deviation, -8.0, 8.0)
        return self.layers(value).squeeze(1)


class Runtime(torch.nn.Module):
    def __init__(
        self, student: Student, catalog: CatalogHead, verifier: Verifier
    ) -> None:
        super().__init__()
        self.encoder = student.encoder
        self.teacher_projection = student.teacher_projection
        self.catalog = catalog
        self.verifier = verifier

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder.encode(mel)
        embedding = torch.nn.functional.normalize(
            self.teacher_projection(embedding), dim=1
        )
        return self.catalog(embedding), self.verifier(embedding)


def encode(model: Student, mel: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(indices), 128):
            value = torch.from_numpy(
                np.asarray(mel[indices[offset : offset + 128]], dtype=np.float32)
            )
            embedding = model.encoder.encode(value)
            embedding = torch.nn.functional.normalize(
                model.teacher_projection(embedding), dim=1
            )
            embeddings.append(embedding.numpy())
    return np.concatenate(embeddings)


def infer_catalog(model: CatalogHead, embeddings: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(embeddings), 512):
            result.append(model(torch.from_numpy(embeddings[offset : offset + 512])).numpy())
    return np.concatenate(result)


def train_catalog(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    epochs: int,
    output: Path,
) -> tuple[CatalogHead, dict]:
    selected = np.logical_and(splits == "train", labels < UNKNOWN)
    mean = embeddings[selected].mean(axis=0)
    deviation = embeddings[selected].std(axis=0) + 1.0e-4
    model = CatalogHead(mean, deviation)
    values = torch.from_numpy(embeddings[selected])
    truth = torch.from_numpy(labels[selected])
    weights = torch.zeros(len(truth))
    for label in range(len(CATALOG)):
        mask = truth == label
        weights[mask] = 1.0 / (len(CATALOG) * mask.sum())
    loader = DataLoader(
        TensorDataset(values, truth),
        batch_size=128,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
    valid = np.logical_and(splits == "valid", labels < UNKNOWN)
    checkpoint = output / "compact-drive-catalog.pt"
    best = float("inf")
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for batch_values, batch_truth in loader:
            loss = torch.nn.functional.cross_entropy(model(batch_values), batch_truth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logits = infer_catalog(model, embeddings[valid])
        validation = float(
            torch.nn.functional.cross_entropy(
                torch.from_numpy(logits), torch.from_numpy(labels[valid])
            )
        )
        print("compact_identity_catalog_epoch", epoch + 1, validation, flush=True)
        if validation < best - 1.0e-5:
            best = validation
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model, {
        "epochs": epoch + 1,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
    }


def infer_verifier(model: Verifier, embeddings: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(embeddings), 512):
            result.append(model(torch.from_numpy(embeddings[offset : offset + 512])).numpy())
    return np.concatenate(result)


def train_verifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
    epochs: int,
    output: Path,
) -> tuple[Verifier, dict]:
    selected = splits == "train"
    mean = embeddings[selected].mean(axis=0)
    deviation = embeddings[selected].std(axis=0) + 1.0e-4
    model = Verifier(mean, deviation)
    values = torch.from_numpy(embeddings[selected])
    known = torch.from_numpy((labels[selected] < UNKNOWN).astype(np.float32))
    names = categories[selected]
    weights = torch.zeros(len(known))
    for target in (0.0, 1.0):
        target_mask = known == target
        target_categories = sorted(set(names[target_mask.numpy()]))
        for category in target_categories:
            mask = torch.logical_and(target_mask, torch.from_numpy(names == category))
            weights[mask] = 1.0 / (2 * len(target_categories) * mask.sum())
    loader = DataLoader(
        TensorDataset(values, known),
        batch_size=128,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
    valid = splits == "valid"
    valid_truth = torch.from_numpy((labels[valid] < UNKNOWN).astype(np.float32))
    checkpoint = output / "compact-drive-verifier.pt"
    best = float("inf")
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for batch_values, batch_truth in loader:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch_values), batch_truth
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        valid_logits = infer_verifier(model, embeddings[valid])
        validation = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(valid_logits), valid_truth
            )
        )
        print("compact_identity_verifier_epoch", epoch + 1, validation, flush=True)
        if validation < best - 1.0e-5:
            best = validation
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model, {
        "epochs": epoch + 1,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
    }


def softmax(value: np.ndarray) -> np.ndarray:
    value = value - value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def scores(identity: np.ndarray, known: np.ndarray) -> np.ndarray:
    return softmax(identity) * sigmoid(known)[:, None]


def select_threshold(
    combined: np.ndarray, truth: np.ndarray, maximum_far: float
) -> float:
    prediction = combined.argmax(axis=1)
    confidence = combined.max(axis=1)
    positive = truth < UNKNOWN
    best = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        accepted = confidence >= threshold
        far = float(accepted[~positive].mean()) if (~positive).any() else 0.0
        recall = float(
            np.logical_and.reduce((accepted, positive, prediction == truth)).sum()
            / max(positive.sum(), 1)
        )
        key = (far <= maximum_far, recall, -far, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def metrics(
    combined: np.ndarray, truth: np.ndarray, categories: np.ndarray, threshold: float
) -> dict:
    prediction = combined.argmax(axis=1)
    confidence = combined.max(axis=1)
    accepted = confidence >= threshold
    result = np.where(accepted, prediction, UNKNOWN)
    positive = truth < UNKNOWN
    per_class = {}
    for index, name in enumerate(CATALOG):
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
        "closed_set_accuracy": float((prediction[positive] == truth[positive]).mean()),
        "correct_accept_rate": float((result[positive] == truth[positive]).mean()),
        "negative_false_accept_rate": float(accepted[~positive].mean()),
        "per_class": per_class,
        "negative_category_false_accept": {
            category: float(accepted[np.logical_and(~positive, categories == category)].mean())
            for category in sorted(set(categories[~positive]))
        },
    }


def aggregate_recordings(
    rows: list,
    combined: np.ndarray,
    truth: np.ndarray,
    categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match runtime by averaging all sampled segments from one recording."""

    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row.path), int(truth[index]), str(categories[index]))
        groups.setdefault(key, []).append(index)
    values = []
    labels = []
    names = []
    for (_, label, category), indices in groups.items():
        values.append(combined[indices].mean(axis=0))
        labels.append(label)
        names.append(category)
    return np.stack(values), np.asarray(labels), np.asarray(names)


def hardware_scores(
    student: Student,
    catalog: CatalogHead,
    verifier: Verifier,
    mel: np.ndarray,
    request_index: dict[tuple[Path, int, int], int],
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
        indices = np.asarray([request_index[(path, segment, 3)] for segment in range(3)])
        embedding = encode(student, mel, indices)
        identity = infer_catalog(catalog, embedding)
        known = infer_verifier(verifier, embedding)
        combined.append(scores(identity, known).mean(axis=0))
        truth.append(CATALOG.index("RAT") if "rat" in path.stem.lower() else UNKNOWN)
    return paths, np.stack(combined), np.asarray(truth)


def hardware_report(
    paths: list[Path], combined: np.ndarray, truth: np.ndarray, threshold: float
) -> dict:
    prediction = combined.argmax(axis=1)
    confidence = combined.max(axis=1)
    accepted = confidence >= threshold
    rows = []
    for index, path in enumerate(paths):
        predicted = CATALOG[prediction[index]] if accepted[index] else None
        expected = CATALOG[truth[index]] if truth[index] < UNKNOWN else None
        rows.append(
            {
                "file": path.name,
                "expected": expected,
                "candidate": CATALOG[prediction[index]],
                "predicted": predicted,
                "score": float(confidence[index]),
            }
        )
    positive = truth < UNKNOWN
    return {
        "rows": rows,
        "rat_recall": float((np.asarray([row["predicted"] for row in rows])[positive] == "RAT").mean()),
        "noncatalog_false_accept": float(accepted[~positive].mean()),
    }


def export(
    student: Student, catalog: CatalogHead, verifier: Verifier, path: Path
) -> float:
    runtime = Runtime(student, catalog, verifier).eval()
    dummy = torch.zeros(1, 1, 128, 216)
    torch.onnx.export(
        runtime,
        dummy,
        path,
        input_names=["log_mel"],
        output_names=["identity_logits", "known_logit"],
        opset_version=17,
        dynamo=False,
    )
    expected = runtime(dummy)
    actual = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"log_mel": dummy.numpy()}
    )
    return max(
        float(np.max(np.abs(expected[index].detach().numpy() - actual[index])))
        for index in range(2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=DRIVE_DELAY_ENCODER_RUN / "best.pt")
    parser.add_argument("--student-checkpoint", type=Path, default=PEDAL_IDENTITY_RUN / "compact-distill/compact-drive-identity.pt")
    parser.add_argument("--cache", type=Path, default=PEDAL_IDENTITY_CACHE / "compact-distill")
    parser.add_argument("--output", type=Path, default=PEDAL_IDENTITY_RUN / "compact-verifier")
    parser.add_argument("--development", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument("--epochs", type=int, default=60)
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
    row_indices = np.asarray(
        [request_index[(row.path, row.segment, row.segments)] for row in rows]
    )
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
    student = Student(args.base_checkpoint)
    student.load_state_dict(
        torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
    )
    row_embeddings = encode(student, mel, row_indices)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog, catalog_training = train_catalog(
        row_embeddings, labels, splits, args.epochs, args.output
    )
    row_identity = infer_catalog(catalog, row_embeddings)
    verifier, training = train_verifier(
        row_embeddings, labels, splits, categories, args.epochs, args.output
    )
    calibrate = splits == "calibrate"
    test = splits == "test"
    calibration_scores, calibration_truth, _ = aggregate_recordings(
        [row for row, selected in zip(rows, calibrate) if selected],
        scores(
            row_identity[calibrate],
            infer_verifier(verifier, row_embeddings[calibrate]),
        ),
        labels[calibrate],
        categories[calibrate],
    )
    calibration_threshold = select_threshold(
        calibration_scores, calibration_truth, 0.05
    )
    paths, development_scores, development_truth = hardware_scores(
        student, catalog, verifier, mel, request_index, args.development
    )
    development_threshold = select_threshold(
        development_scores, development_truth, 0.20
    )
    test_scores, test_truth, test_categories = aggregate_recordings(
        [row for row, selected in zip(rows, test) if selected],
        scores(row_identity[test], infer_verifier(verifier, row_embeddings[test])),
        labels[test],
        categories[test],
    )
    reports = {
        "public_calibration": {
            "threshold": calibration_threshold,
            "public_test": metrics(test_scores, test_truth, test_categories, calibration_threshold),
            "hardware_development": hardware_report(paths, development_scores, development_truth, calibration_threshold),
        },
        "hardware_development_calibration": {
            "threshold": development_threshold,
            "uses_hardware_labels": True,
            "public_test": metrics(test_scores, test_truth, test_categories, development_threshold),
            "hardware_development": hardware_report(paths, development_scores, development_truth, development_threshold),
        },
    }
    selected = reports["hardware_development_calibration"]
    failures = []
    hardware = selected["hardware_development"]
    public = selected["public_test"]
    if hardware["rat_recall"] < 0.5:
        failures.append("hardware-development RAT recall is below 50%")
    if hardware["noncatalog_false_accept"] > 0.2:
        failures.append("hardware-development non-catalog false accept exceeds 20%")
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
    onnx_path = args.output / "compact-drive-identity.onnx"
    parity = export(student, catalog, verifier, onnx_path)
    payload = {
        "experiment": "compact-drive-identity-open-set-verifier",
        "architecture": {"encoder": "distilled compact Inspector ResNet18", "identity_head": "independent 512-128 pedal-catalog MLP", "catalog": list(CATALOG), "verifier": "independent 512-64 knownness MLP", "user_gradient_updates": 0, "runtime_parameters": sum(value.numel() for value in Runtime(student, catalog, verifier).parameters())},
        "data": {"examples": len(rows), "tone_twist": inventory, "external_test_folder_role": "labeled hardware development calibration; not a final test"},
        "training": {"catalog": catalog_training, "verifier": training},
        "reports": reports,
        "development_gate": {"passed": not failures, "failures": failures},
        "integration_eligible": not failures,
        "release_eligible": False,
        "release_reason": "non-commercial research data and hardware-development calibration require a new untouched multi-device final test",
        "export": {"path": str(onnx_path.resolve()), "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(), "max_absolute_difference": parity},
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output / "calibration.json").write_text(json.dumps({"labels": list(CATALOG), "threshold": development_threshold, "development_calibrated": True}, indent=2) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
