#!/usr/bin/env python3
"""Train a candidate-conditioned temporal verifier for Reverb.

The frozen public-RIR encoder and clean-relative pair head remain the candidate
generator. The verifier sees ordered Mel-band energy plus the three calibrated
pair probabilities and is trained only around true Reverb or difficult
candidate/Drive examples. The public-only run never updates from private user
recordings. The optional hardware-replay run explicitly fine-tunes the compact
verifier on private development recordings with public replay regularization;
those results are resubstitution evidence, not an untouched final test.
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
from evaluate import expected as development_expected
from layout import (
    CORPUS,
    REVERB_CACHE,
    REVERB_ENCODER_RUN,
    REVERB_PAIR_RUN,
    REVERB_VERIFIER_PUBLIC_RUN,
)
from model import FRAMES, LABELS, MELS, Detector
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
from verifier import binary_metrics, load_encoded, top_two
from verifier_sequence import BANDS, path_sequences, sequence_cache


SEED = 20260829
DRIVE = LABELS.index("drive")
REVERB = LABELS.index("reverb")
PAIR_FLOOR = np.asarray([0.05, 0.55, 0.59], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class ReverbVerifier(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None, :, None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None, :, None]
        )
        self.temporal = torch.nn.Sequential(
            torch.nn.Conv1d(BANDS, 24, 9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(24, 32, 9, padding=16, dilation=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, 9, padding=48, dilation=12),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(64 + len(LABELS), 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(32, 1),
        )

    def forward(self, bands: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        value = torch.clamp((bands - self.mean) / self.deviation, -6.0, 6.0)
        value = self.temporal(value)
        pooled = torch.cat((value.mean(dim=2), value.amax(dim=2), pair), dim=1)
        return self.output(pooled).squeeze(1)


class CandidateDataset(Dataset):
    def __init__(
        self,
        sequence_path: Path,
        row_indices: np.ndarray,
        pair: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        pair_threshold: float,
    ) -> None:
        self.sequence = np.load(sequence_path, mmap_mode="r")
        truth = labels[:, REVERB] > 0.5
        drive_only = np.logical_and(labels[:, DRIVE] > 0.5, ~truth)
        difficult = pair[:, REVERB] >= 0.15
        selected = np.flatnonzero(np.logical_or.reduce((truth, drive_only, difficult)))
        self.row_indices = row_indices[selected]
        self.pair = pair[selected].astype(np.float32)
        self.truth = truth[selected].astype(np.float32)
        self.weight = np.ones(len(selected), dtype=np.float32)
        selected_domains = domains[selected].astype(str)
        selected_drive = drive_only[selected]
        selected_difficult = np.logical_and(
            ~truth[selected], self.pair[:, REVERB] >= pair_threshold
        )
        self.weight[selected_drive] *= 4.0
        self.weight[selected_difficult] *= 10.0
        self.weight[np.char.startswith(selected_domains, "egfx")] *= 3.0

    def __len__(self) -> int:
        return len(self.truth)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        bands = torch.from_numpy(
            np.array(self.sequence[self.row_indices[index]], dtype=np.float32, copy=True)
        )
        return (
            bands,
            torch.from_numpy(self.pair[index]),
            torch.tensor(self.truth[index]),
            torch.tensor(self.weight[index]),
        )


def logits(
    model: ReverbVerifier,
    sequence: np.ndarray,
    pair: np.ndarray,
    target: torch.device,
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(pair), 256):
            bands = torch.from_numpy(
                np.asarray(sequence[start : start + 256], dtype=np.float32).copy()
            ).to(target)
            probabilities = torch.from_numpy(pair[start : start + 256]).to(target)
            result.append(model(bands, probabilities).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def probabilities(values: np.ndarray, scale: float, bias: float) -> np.ndarray:
    value = values * scale + bias
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_calibration(expected: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    logits_tensor = torch.tensor(values, dtype=torch.float64)
    labels_tensor = torch.tensor(expected, dtype=torch.float64)
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_scale, bias), lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tensor * log_scale.exp() + bias, labels_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        float(log_scale.detach().exp().clamp(0.05, 20.0)),
        float(bias.detach().clamp(-20.0, 20.0)),
    )


def train_model(
    model: ReverbVerifier,
    train_data: CandidateDataset,
    valid_data: CandidateDataset,
    output: Path,
    epochs: int,
    patience: int,
    target: torch.device,
) -> tuple[int, float, float]:
    loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=256, shuffle=False, num_workers=0)
    positives = max(float(train_data.truth.sum()), 1.0)
    positive_weight = torch.tensor(
        np.clip((len(train_data) - positives) / positives, 1.0, 8.0),
        dtype=torch.float32,
        device=target,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2.0e-4)
    best = float("inf")
    stale = 0
    completed = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for bands, pair, truth, weight in loader:
            prediction = model(bands.to(target), pair.to(target))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction,
                truth.to(target),
                pos_weight=positive_weight,
                weight=weight.to(target),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for bands, pair, truth, _ in valid_loader:
                losses.append(
                    float(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            model(bands.to(target), pair.to(target)),
                            truth.to(target),
                            pos_weight=positive_weight,
                        )
                    )
                )
        valid_loss = float(np.mean(losses))
        completed = epoch + 1
        print("reverb_verifier_epoch", completed, "valid_loss", valid_loss, flush=True)
        if valid_loss < best - 1.0e-5:
            best = valid_loss
            stale = 0
            torch.save(model.state_dict(), output / "reverb-verifier.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(
        torch.load(output / "reverb-verifier.pt", map_location=target, weights_only=True)
    )
    return completed, best, time.perf_counter() - started


def adapt_on_development(
    model: ReverbVerifier,
    public_data: CandidateDataset,
    directory: Path,
    clean_directory: Path,
    reference: Path,
    encoder: Detector,
    head: RelativeHead,
    pair_scale: np.ndarray,
    pair_bias: np.ndarray,
    updates: int,
    target: torch.device,
) -> dict:
    reference_query, _ = encode_relative_windows(
        encoder, reference, "embedding-logits", torch.device("cpu")
    )
    count = reference_window_count(10, len(reference_query))
    selected = reference_query[:count]
    profile_mean = selected.mean(axis=0)
    profile_deviation = selected.std(axis=0) + 1.0e-4
    files = [
        path
        for path in sorted(directory.glob("*.wav"))
        if path.resolve() != reference.resolve()
    ]
    files.extend(sorted(clean_directory.glob("*.wav")))
    bands_by_file = []
    pair_by_file = []
    truth_by_file = []
    weight_by_file = []
    names = []
    for path in files:
        pair, bands = file_pair_and_sequence(
            path,
            encoder,
            head,
            profile_mean,
            profile_deviation,
            pair_scale,
            pair_bias,
        )
        truth = (
            float(development_expected(path)[REVERB])
            if path.parent.resolve() == directory.resolve()
            else 0.0
        )
        file_weight = np.full(len(pair), 1.0 / len(pair), dtype=np.float32)
        if truth:
            file_weight *= 2.0
        elif float(top_two(pair)[REVERB]) >= PAIR_FLOOR[REVERB]:
            file_weight *= 8.0
        bands_by_file.append(bands)
        pair_by_file.append(pair)
        truth_by_file.append(np.full(len(pair), truth, dtype=np.float32))
        weight_by_file.append(file_weight)
        names.append(path.name)
    bands = torch.from_numpy(np.concatenate(bands_by_file)).to(target)
    pair = torch.from_numpy(np.concatenate(pair_by_file).astype(np.float32)).to(target)
    truth = torch.from_numpy(np.concatenate(truth_by_file)).to(target)
    weight = torch.from_numpy(np.concatenate(weight_by_file)).to(target)
    weight = weight / weight.mean()
    generator = np.random.default_rng(SEED + 17)
    replay_indices = generator.choice(
        len(public_data), size=min(4_096, len(public_data)), replace=False
    )
    public_bands = torch.from_numpy(
        np.asarray(
            public_data.sequence[public_data.row_indices[replay_indices]],
            dtype=np.float32,
        ).copy()
    ).to(target)
    public_pair = torch.from_numpy(public_data.pair[replay_indices]).to(target)
    public_truth = torch.from_numpy(public_data.truth[replay_indices]).to(target)
    public_weight = torch.from_numpy(public_data.weight[replay_indices]).to(target)
    public_weight = public_weight / public_weight.mean()
    anchor = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=2.0e-4)
    model.train()
    started = time.perf_counter()
    for _ in range(updates):
        prediction = model(bands, pair)
        classification = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, truth, weight=weight
        )
        replay = torch.randint(0, len(public_truth), (512,), device=target)
        public_classification = torch.nn.functional.binary_cross_entropy_with_logits(
            model(public_bands[replay], public_pair[replay]),
            public_truth[replay],
            weight=public_weight[replay],
        )
        regularization = sum(
            (parameter - initial).pow(2).mean()
            for parameter, initial in zip(model.parameters(), anchor)
        )
        loss = classification + public_classification + 5.0e-4 * regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return {
        "updates": updates,
        "seconds": time.perf_counter() - started,
        "files": len(files),
        "wet_labelled_files": len(list(directory.glob("*.wav"))) - 1,
        "clean_files": len(list(clean_directory.glob("*.wav"))),
        "windows": int(len(truth)),
        "public_replay_windows": int(len(public_truth)),
        "file_names": names,
    }


def reverb_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    truth = expected.astype(bool)
    prediction = predicted.astype(bool)
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_full(
    expected: np.ndarray,
    pair: np.ndarray,
    pair_thresholds: np.ndarray,
    verifier: np.ndarray,
    verifier_threshold: float,
) -> dict:
    prediction = pair >= pair_thresholds
    prediction[:, REVERB] = np.logical_and(
        prediction[:, REVERB], verifier >= verifier_threshold
    )
    return binary_metrics(expected, prediction)


def load_models(base_run: Path, checkpoint: Path) -> tuple[Detector, RelativeHead, dict]:
    encoder = Detector(stem_stride=1)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder.eval()
    head = RelativeHead(259 * 5)
    head.load_state_dict(
        torch.load(base_run / "relative-head.pt", map_location="cpu", weights_only=True)
    )
    head.eval()
    report = json.loads((base_run / "metrics.json").read_text())
    return encoder, head, report


def file_pair_and_sequence(
    path: Path,
    encoder: Detector,
    head: RelativeHead,
    profile_mean: np.ndarray,
    profile_deviation: np.ndarray,
    pair_scale: np.ndarray,
    pair_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query, _ = encode_relative_windows(
        encoder, path, "embedding-logits", torch.device("cpu")
    )
    pair = calibrated(
        infer(
            head,
            fused_with_profile(query, profile_mean, profile_deviation),
            torch.device("cpu"),
        ),
        pair_scale,
        pair_bias,
    )
    return pair, path_sequences(path)


def development_rows(
    directory: Path,
    reference: Path,
    encoder: Detector,
    head: RelativeHead,
    model: ReverbVerifier,
    target: torch.device,
    pair_scale: np.ndarray,
    pair_bias: np.ndarray,
    verifier_scale: float,
    verifier_bias: float,
) -> tuple[
    list[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    reference_query, _ = encode_relative_windows(
        encoder, reference, "embedding-logits", torch.device("cpu")
    )
    count = reference_window_count(10, len(reference_query))
    selected = reference_query[:count]
    profile_mean = selected.mean(axis=0)
    profile_deviation = selected.std(axis=0) + 1.0e-4
    clean_pair = calibrated(
        infer(
            head,
            fused_with_profile(selected, profile_mean, profile_deviation),
            torch.device("cpu"),
        ),
        pair_scale,
        pair_bias,
    )
    pair_thresholds = np.minimum(
        np.maximum(clean_pair.max(axis=0) + 0.02, PAIR_FLOOR), 0.95
    )
    rows = []
    truths = []
    pairs = []
    verified = []
    for path in sorted(directory.glob("*.wav")):
        if path.resolve() == reference.resolve():
            continue
        pair_windows, sequence = file_pair_and_sequence(
            path,
            encoder,
            head,
            profile_mean,
            profile_deviation,
            pair_scale,
            pair_bias,
        )
        pair_file = top_two(pair_windows)
        verifier_windows = probabilities(
            logits(model, sequence, pair_windows, target), verifier_scale, verifier_bias
        )
        verifier_file = float(top_two(verifier_windows[:, None])[0])
        truth = development_expected(path)
        rows.append(
            {
                "file": path.name,
                "expected": [LABELS[index] for index in np.flatnonzero(truth)],
                "pair_probability": dict(zip(LABELS, map(float, pair_file))),
                "verifier_probability": verifier_file,
            }
        )
        truths.append(truth)
        pairs.append(pair_file)
        verified.append(verifier_file)
    return (
        rows,
        np.stack(truths),
        np.stack(pairs),
        np.asarray(verified),
        pair_thresholds,
        profile_mean,
        profile_deviation,
    )


def select_threshold(
    public_truth: np.ndarray,
    public_pair: np.ndarray,
    public_verifier: np.ndarray,
    pair_threshold: float,
    development_truth: np.ndarray,
    development_pair: np.ndarray,
    development_verifier: np.ndarray,
    development_pair_threshold: float,
) -> float:
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        public_prediction = np.logical_and(
            public_pair >= pair_threshold, public_verifier >= threshold
        )
        development_prediction = np.logical_and(
            development_pair >= development_pair_threshold,
            development_verifier >= threshold,
        )
        public = reverb_metrics(public_truth, public_prediction)
        development = reverb_metrics(development_truth, development_prediction)
        if public["recall"] >= 0.80 and development["recall"] >= 0.80:
            candidates.append(
                (development["f1"], development["precision"], public["f1"], threshold)
            )
    if not candidates:
        return 0.0
    return float(max(candidates)[-1])


def export(model: ReverbVerifier, output: Path) -> float:
    cpu = model.cpu().eval()
    bands = torch.randn(1, BANDS, FRAMES)
    pair = torch.rand(1, len(LABELS))
    torch.onnx.export(
        cpu,
        (bands, pair),
        output,
        input_names=["temporal_bands", "pair_probabilities"],
        output_names=["reverb_logit"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))
    expected = cpu(bands, pair).detach().numpy()
    actual = onnxruntime.InferenceSession(str(output)).run(
        None,
        {"temporal_bands": bands.numpy(), "pair_probabilities": pair.numpy()},
    )[0]
    return float(np.max(np.abs(expected - actual)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=REVERB_CACHE)
    parser.add_argument("--base-run", type=Path, default=REVERB_PAIR_RUN)
    parser.add_argument(
        "--checkpoint", type=Path, default=REVERB_ENCODER_RUN / "backbone.pt"
    )
    parser.add_argument("--output", type=Path, default=REVERB_VERIFIER_PUBLIC_RUN)
    parser.add_argument("--data", type=Path, default=CORPUS)
    parser.add_argument("--development", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument(
        "--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav"
    )
    parser.add_argument(
        "--clean-audit", type=Path, default=Path.home() / "Downloads/clean test"
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume-public", type=Path)
    parser.add_argument("--development-updates", type=int, default=0)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)
    target = target_device()

    encoded = {
        name: load_encoded(args.cache, name)
        for name in ("train", "valid", "calibrate", "test")
    }
    sequences = {name: sequence_cache(args.cache, name) for name in encoded}

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
            encoded[name],
            masks[name],
            "embedding-logits",
            SEED + index * 10_007,
            2 if name == "train" else 1,
        )
        for index, name in enumerate(("train", "valid", "calibrate", "test"))
    }

    encoder, head, base_metrics = load_models(args.base_run, args.checkpoint)
    pair_scale = np.asarray(
        [base_metrics["calibration"]["scale"][label] for label in LABELS],
        dtype=np.float32,
    )
    pair_bias = np.asarray(
        [base_metrics["calibration"]["bias"][label] for label in LABELS],
        dtype=np.float32,
    )
    pair_thresholds = np.asarray(
        [base_metrics["calibration"]["threshold"][label] for label in LABELS],
        dtype=np.float32,
    )
    pair = {
        name: calibrated(
            infer(head, episode[name][0], torch.device("cpu")), pair_scale, pair_bias
        )
        for name in episode
    }

    train_sequence = np.load(sequences["train"], mmap_mode="r")
    mean = np.asarray(train_sequence, dtype=np.float32).mean(axis=(0, 2))
    deviation = np.asarray(train_sequence, dtype=np.float32).std(axis=(0, 2)) + 1.0e-4
    model = ReverbVerifier(mean, deviation).to(target)
    train_data = CandidateDataset(
        sequences["train"],
        episode["train"][2],
        pair["train"],
        episode["train"][1],
        encoded["train"]["domain"][episode["train"][2]],
        pair_thresholds[REVERB],
    )
    valid_data = CandidateDataset(
        sequences["valid"],
        episode["valid"][2],
        pair["valid"],
        episode["valid"][1],
        encoded["valid"]["domain"][episode["valid"][2]],
        pair_thresholds[REVERB],
    )
    if args.resume_public is None:
        completed, best, seconds = train_model(
            model, train_data, valid_data, args.output, args.epochs, args.patience, target
        )
        public_source = str(args.output.resolve())
    else:
        model.load_state_dict(
            torch.load(
                args.resume_public / "reverb-verifier.pt",
                map_location=target,
                weights_only=True,
            )
        )
        completed, best, seconds = 0, None, 0.0
        public_source = str(args.resume_public.resolve())
    development_fit = None
    if args.development_updates:
        development_fit = adapt_on_development(
            model,
            train_data,
            args.development,
            args.clean_audit,
            args.reference,
            encoder,
            head,
            pair_scale,
            pair_bias,
            args.development_updates,
            target,
        )
    torch.save(model.state_dict(), args.output / "reverb-verifier.pt")

    verifier_logits = {
        name: logits(
            model,
            np.load(sequences[name], mmap_mode="r")[episode[name][2]],
            pair[name],
            target,
        )
        for name in ("calibrate", "test")
    }
    scale, bias = fit_calibration(
        episode["calibrate"][1][:, REVERB], verifier_logits["calibrate"]
    )
    verifier_probability = {
        name: probabilities(values, scale, bias) for name, values in verifier_logits.items()
    }

    (
        rows,
        development_truth,
        development_pair,
        development_verifier,
        development_pair_thresholds,
        development_profile_mean,
        development_profile_deviation,
    ) = development_rows(
        args.development,
        args.reference,
        encoder,
        head,
        model,
        target,
        pair_scale,
        pair_bias,
        scale,
        bias,
    )
    development_pair_threshold = float(development_pair_thresholds[REVERB])
    threshold = select_threshold(
        episode["calibrate"][1][:, REVERB] > 0.5,
        pair["calibrate"][:, REVERB],
        verifier_probability["calibrate"],
        pair_thresholds[REVERB],
        development_truth[:, REVERB] > 0.5,
        development_pair[:, REVERB],
        development_verifier,
        development_pair_threshold,
    )

    reports = {
        name: evaluate_full(
            episode[name][1],
            pair[name],
            pair_thresholds,
            verifier_probability[name],
            threshold,
        )
        for name in ("calibrate", "test")
    }
    development_prediction = development_pair >= development_pair_thresholds
    development_prediction[:, REVERB] = np.logical_and(
        development_prediction[:, REVERB], development_verifier >= threshold
    )
    development_report = binary_metrics(development_truth, development_prediction)
    for row, prediction in zip(rows, development_prediction):
        row["predicted"] = [LABELS[index] for index in np.flatnonzero(prediction)]

    clean_rows = []
    for path in sorted(args.clean_audit.glob("*.wav")):
        pair_windows, sequence = file_pair_and_sequence(
            path,
            encoder,
            head,
            development_profile_mean,
            development_profile_deviation,
            pair_scale,
            pair_bias,
        )
        verifier_file = float(
            top_two(
                probabilities(
                    logits(model, sequence, pair_windows, target), scale, bias
                )[:, None]
            )[0]
        )
        clean_rows.append(
            {
                "file": path.name,
                "pair_reverb_probability": float(top_two(pair_windows)[REVERB]),
                "verifier_probability": verifier_file,
                "passes_verifier": verifier_file >= threshold,
                "predicted_reverb": bool(
                    top_two(pair_windows)[REVERB] >= development_pair_threshold
                    and verifier_file >= threshold
                ),
            }
        )

    onnx_path = args.output / "reverb-verifier.onnx"
    parity = export(model, onnx_path)
    failures = []
    if reports["test"]["clean_false_positive"] > 0.05:
        failures.append("public test clean false-positive rate exceeds 5%")
    if reports["test"]["reverb"]["recall"] < 0.80:
        failures.append("public test Reverb recall is below 80%")
    if development_report["reverb"]["precision"] < 0.80:
        failures.append("development Reverb precision is below 80%")
    if development_report["reverb"]["recall"] < 0.80:
        failures.append("development Reverb recall is below 80%")

    report = {
        "experiment": args.output.name,
        "architecture": {
            "candidate": "frozen public-RIR clean-reference pair model",
            "verifier": "8x216 Mel bands -> dilated temporal Conv1d -> pair fusion -> Reverb",
            "parameters": sum(value.numel() for value in model.parameters()),
            "runtime_clean_import_gradient_updates": 0,
            "hardware_development_gradient_updates": (
                0 if development_fit is None else development_fit["updates"]
            ),
        },
        "data_policy": {
            "weights": (
                "CC-BY/CC0 grouped public train plus labelled hardware development"
                if development_fit
                else "CC-BY/CC0 grouped public train split only"
            ),
            "hard_negative_mining": "pair candidate or Drive-only windows",
            "threshold": "public calibration plus labelled hardware development set",
            "development_weights": 0 if development_fit is None else development_fit["windows"],
            "untouched_hardware_final": "not yet available",
            "excluded": ["IDMT", "RemFX", "Apple AU", "ToneTwisT"],
        },
        "training": {
            "device": str(target),
            "seed": SEED,
            "epochs": completed,
            "seconds": seconds,
            "best_validation_loss": best,
            "train_candidates": len(train_data),
            "valid_candidates": len(valid_data),
            "public_source": public_source,
            "development_fit": development_fit,
        },
        "calibration": {
            "scale": scale,
            "bias": bias,
            "threshold": threshold,
            "pair_reverb_threshold": float(pair_thresholds[REVERB]),
            "development_pair_reverb_threshold": development_pair_threshold,
        },
        "calibrate": reports["calibrate"],
        "test": reports["test"],
        "development": {
            "role": (
                "weight fitting and threshold selection; resubstitution, not a final test"
                if development_fit
                else "threshold selection only; not a final test"
            ),
            "metrics": development_report,
            "results": rows,
        },
        "clean_audit": {
            "role": "development clean-only check with the complete pair-plus-verifier gate",
            "recordings": len(clean_rows),
            "passes": sum(row["passes_verifier"] for row in clean_rows),
            "false_positives": sum(row["predicted_reverb"] for row in clean_rows),
            "results": clean_rows,
        },
        "export": {
            "onnx": str(onnx_path.resolve()),
            "sha256": sha256(onnx_path),
            "max_absolute_difference": parity,
        },
        "quality_gate": {
            "level": "development",
            "passed": not failures,
            "requirements": {
                "public_test_clean_false_positive_max": 0.05,
                "public_test_reverb_recall_min": 0.80,
                "development_reverb_precision_min": 0.80,
                "development_reverb_recall_min": 0.80,
            },
            "failures": failures,
        },
        "release_gate": {
            "passed": False,
            "failures": ["untouched device-disjoint labelled hardware test is not available"],
        },
    }
    (args.output / "calibration.json").write_text(
        json.dumps(report["calibration"], indent=2, sort_keys=True) + "\n"
    )
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
