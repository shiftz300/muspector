#!/usr/bin/env python3
"""Train global known-pedal identity heads on frozen Inspector encoders.

The heads are family-specific and use the same 1,295-value non-aligned
Clean-relative feature contract as Inspector pair inference. Clean, modulation,
and other-family recordings train an internal knownness gate; they are not a
semantic Unknown effect class. User recordings are evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data import Source, canonical, discover_remfx_1_1, partition
from evaluate import waveform, windows
from layout import (
    CORPUS,
    DRIVE_DELAY_CACHE,
    DRIVE_DELAY_ENCODER_RUN,
    PEDAL_IDENTITY_CACHE,
    PEDAL_IDENTITY_RUN,
    REVERB_CACHE,
    REVERB_ENCODER_RUN,
)
from model import RATE, SAMPLES, Detector, frontend
from relative import encode_relative_windows, reference_window_count


SEED = 20260829
QUERY = 259
INPUTS = QUERY * 5
IDENTITY_REFERENCE_WINDOWS = 3
CATALOG = {
    "drive": ("BluesDriver", "RAT", "TubeScreamer"),
    "delay": ("Digital-Delay", "Sweep-Echo", "TapeEcho"),
    "reverb": ("Hall-Reverb", "Plate-Reverb", "Spring-Reverb"),
}
MODULATION = ("Chorus", "Flanger", "Phaser")
ALL_EFFECTS = tuple(name for names in CATALOG.values() for name in names) + MODULATION
DISPLAY = {
    "BluesDriver": "Boss BD-2 / Blues Driver style",
    "TubeScreamer": "Ibanez Tube Screamer style",
    "RAT": "ProCo RAT / RAT style",
    "Digital-Delay": "Digital Delay",
    "Sweep-Echo": "Sweep Echo",
    "TapeEcho": "Tape Echo",
    "Hall-Reverb": "Hall Reverb",
    "Plate-Reverb": "Plate Reverb",
    "Spring-Reverb": "Spring Reverb",
}
FAMILY_INDEX = {"drive": 0, "delay": 1, "reverb": 2}
DAFX_SUFFIXES = ("-variable-parameters", "-variable-order", "-fixed", "-dry")


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    pickup: str
    group: str
    split: str


@dataclass(frozen=True)
class ToneDevice:
    key: str
    label: str
    root: Path
    record: str
    description: str


class IdentityHead(torch.nn.Module):
    def __init__(self, mean: np.ndarray, deviation: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32))[None])
        self.register_buffer(
            "deviation", torch.from_numpy(deviation.astype(np.float32))[None]
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(INPUTS, 192),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(192, 96),
            torch.nn.ReLU(),
        )
        self.identity = torch.nn.Linear(96, 3)
        self.known = torch.nn.Linear(96, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.clamp((value - self.mean) / self.deviation, -8.0, 8.0)
        value = self.trunk(value)
        return self.identity(value), self.known(value).squeeze(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_audio_embedding(
    model: Detector,
    path: Path,
    cache: Path,
    checkpoint_hash: str,
    target: torch.device,
) -> np.ndarray:
    relative = path.relative_to(CORPUS)
    key = hashlib.sha256(f"{checkpoint_hash}:{relative}".encode()).hexdigest()[:20]
    output = cache / "tonetwist" / f"{key}.npy"
    if output.exists():
        return np.load(output).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = stream_audio_embedding(model, path, target)
    np.save(output, values.astype(np.float16))
    return values


def stream_audio_embedding(
    model: Detector, path: Path, target: torch.device, batch_size: int = 16
) -> np.ndarray:
    """Encode long ToneTwist/RemFX files without materializing overlapping windows."""

    result: list[np.ndarray] = []
    with soundfile.SoundFile(path) as stream:
        rate = stream.samplerate
        source_window = int(round(SAMPLES * rate / RATE))
        source_hop = int(round((SAMPLES // 2) * rate / RATE))
        if len(stream) <= source_window:
            starts = [0]
        else:
            starts = list(range(0, len(stream) - source_window + 1, source_hop))
            if starts[-1] != len(stream) - source_window:
                starts.append(len(stream) - source_window)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(starts), batch_size):
                audio = []
                for offset in starts[start : start + batch_size]:
                    stream.seek(offset)
                    value = stream.read(
                        source_window, dtype="float32", always_2d=True
                    ).mean(axis=1)
                    if rate != RATE:
                        divisor = math.gcd(rate, RATE)
                        value = resample_poly(
                            value, RATE // divisor, rate // divisor
                        ).astype(np.float32)
                    value = np.pad(value, (0, max(0, SAMPLES - len(value))))[:SAMPLES]
                    audio.append(np.nan_to_num(value).clip(-4.0, 4.0))
                mel = frontend(torch.from_numpy(np.stack(audio)))
                embedding = model.encode(mel.to(target))
                result.append(
                    torch.cat((embedding, model.head(embedding)), dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
    return np.concatenate(result)


def target_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def discover(root: Path) -> list[Record]:
    result = []
    for label in ("Clean",) + ALL_EFFECTS:
        for path in sorted((root / label).rglob("*.wav")):
            pickup = path.parent.name.lower()
            group = f"egfx:{pickup}:{canonical(path.stem)}"
            result.append(
                Record(path, label, pickup, group, partition(Source(path, group, "egfx")))
            )
    if len(result) != 13 * 690:
        raise RuntimeError(f"expected 8,970 EGFx files, found {len(result)}")
    ownership: dict[str, str] = {}
    for record in result:
        previous = ownership.setdefault(record.group, record.split)
        if previous != record.split:
            raise RuntimeError(f"group leakage: {record.group}: {previous}/{record.split}")
    return result


def load_encoder(path: Path) -> Detector:
    model = Detector(stem_stride=1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def encode_batch(model: Detector, mel: torch.Tensor) -> np.ndarray:
    embedding = model.encode(mel)
    logits = model.head(embedding)
    return torch.cat((embedding, logits), dim=1).cpu().numpy().astype(np.float32)


def build_cache(
    records: list[Record],
    cache: Path,
    drive_delay_checkpoint: Path,
    reverb_checkpoint: Path,
    batch: int,
    target: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "frozen-encoder-queries.npz"
    expected_paths = np.asarray([str(record.path.relative_to(CORPUS)) for record in records])
    if path.exists():
        values = np.load(path)
        if np.array_equal(values["path"], expected_paths):
            return values["drive_delay"].astype(np.float32), values["reverb"].astype(np.float32)

    drive_delay = load_encoder(drive_delay_checkpoint).to(target)
    reverb = load_encoder(reverb_checkpoint).to(target)
    drive_delay_query = []
    reverb_query = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(records), batch):
            selected = records[start : start + batch]
            audio = np.stack([windows(waveform(record.path))[0] for record in selected])
            mel = frontend(torch.from_numpy(audio).to(target))
            drive_delay_query.append(encode_batch(drive_delay, mel))
            reverb_query.append(encode_batch(reverb, mel))
            if start % (batch * 20) == 0:
                print("pedal_identity_cache", min(start + batch, len(records)), len(records), flush=True)
    drive_delay_values = np.concatenate(drive_delay_query)
    reverb_values = np.concatenate(reverb_query)
    np.savez(
        path,
        path=expected_paths,
        label=np.asarray([record.label for record in records]),
        pickup=np.asarray([record.pickup for record in records]),
        group=np.asarray([record.group for record in records]),
        split=np.asarray([record.split for record in records]),
        drive_delay=drive_delay_values.astype(np.float16),
        reverb=reverb_values.astype(np.float16),
        seconds=np.asarray([time.perf_counter() - started]),
    )
    return drive_delay_values, reverb_values


def reference_indices(records: list[Record]) -> dict[tuple[str, str], list[int]]:
    result: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        if record.label == "Clean":
            result.setdefault((record.split, record.pickup), []).append(index)
    return result


def relative_features(
    family: str,
    records: list[Record],
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    references = reference_indices(records)
    labels = {name: index for index, name in enumerate(CATALOG[family])}
    count = IDENTITY_REFERENCE_WINDOWS
    features = np.empty((len(records), INPUTS), dtype=np.float32)
    truth = np.full(len(records), -1, dtype=np.int64)
    for index, record in enumerate(records):
        candidates = [
            value
            for value in references[(record.split, record.pickup)]
            if records[value].group != record.group
        ]
        if len(candidates) < count:
            raise RuntimeError(f"not enough non-aligned Clean references for {record.group}")
        seed = int(hashlib.sha256(f"{family}:{record.group}".encode()).hexdigest()[:8], 16)
        selected = [candidates[(seed + offset) % len(candidates)] for offset in range(count)]
        profile = queries[selected]
        mean = profile.mean(axis=0)
        deviation = profile.std(axis=0) + 1.0e-4
        query = queries[index]
        features[index] = np.concatenate(
            (query, mean, query - mean, np.abs(query - mean), deviation)
        )
        truth[index] = labels.get(record.label, -1)
    return (
        features,
        truth,
        np.asarray([record.split for record in records]),
        np.asarray([record.label for record in records]),
        np.asarray([record.pickup for record in records]),
    )


def encoded_split(cache: Path, split: str) -> dict[str, np.ndarray]:
    matches = sorted(cache.glob(f"{split}-relative-*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {split} encoded cache in {cache}, found {matches}")
    values = np.load(matches[0])
    required = {"embedding", "base_logits", "labels", "domain", "group"}
    if not required <= set(values.files):
        raise RuntimeError(f"encoded cache is missing {sorted(required - set(values.files))}")
    return {key: values[key] for key in values.files}


def dafx_domain_root(domain: str) -> str:
    for suffix in DAFX_SUFFIXES:
        if domain.endswith(suffix):
            return domain[: -len(suffix)]
    return domain


def dafx_unknown_features(
    family: str,
    cache: Path,
    limit: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build generic-effect rejection examples from public DAFx chain audio.

    The chain dataset names only effect families, never the catalog devices.
    Its active-family recordings are therefore open-set negatives. References
    are unprocessed recordings from a different improvisation on the same
    guitar, so paired/aligned content cannot leak into the identity head.
    """

    limits = limit or {"train": 4096, "valid": 1536, "calibrate": 1536, "test": 2048}
    all_features: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_split: list[np.ndarray] = []
    all_categories: list[np.ndarray] = []
    all_domains: list[np.ndarray] = []
    count = IDENTITY_REFERENCE_WINDOWS
    for split_name in ("train", "valid", "calibrate", "test"):
        encoded = encoded_split(cache, split_name)
        values = np.concatenate(
            (encoded["embedding"], encoded["base_logits"]), axis=1
        ).astype(np.float32)
        labels = encoded["labels"].astype(np.float32)
        domains = encoded["domain"].astype(str)
        groups = encoded["group"].astype(str)
        chain = np.char.startswith(domains, "guitar-effects-chains-")
        active = labels[:, FAMILY_INDEX[family]] > 0.5
        wet = np.flatnonzero(np.logical_and(chain, active))
        rng = np.random.default_rng(SEED + FAMILY_INDEX[family] * 100 + len(split_name))
        if len(wet) > limits[split_name]:
            wet = np.sort(rng.choice(wet, limits[split_name], replace=False))
        feature = np.empty((len(wet), INPUTS), dtype=np.float32)
        for offset, query_index in enumerate(wet):
            root = dafx_domain_root(domains[query_index])
            clean = np.flatnonzero(
                np.logical_and.reduce(
                    (
                        domains == root + "-dry",
                        labels.sum(axis=1) == 0.0,
                        groups != groups[query_index],
                    )
                )
            )
            if not len(clean):
                raise RuntimeError(f"no DAFx Clean reference for {domains[query_index]}")
            chosen = rng.choice(clean, count, replace=len(clean) < count)
            profile = values[chosen]
            mean = profile.mean(axis=0)
            deviation = profile.std(axis=0) + 1.0e-4
            query = values[query_index]
            feature[offset] = np.concatenate(
                (query, mean, query - mean, np.abs(query - mean), deviation)
            )
        all_features.append(feature)
        all_truth.append(np.full(len(wet), -1, dtype=np.int64))
        all_split.append(np.full(len(wet), split_name))
        all_categories.append(np.full(len(wet), "DAFx non-catalog " + family))
        all_domains.append(domains[wet])
    return tuple(
        np.concatenate(parts) for parts in (
            all_features,
            all_truth,
            all_split,
            all_categories,
            all_domains,
        )
    )


def append_examples(
    primary: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    extra: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.concatenate((left, right)) for left, right in zip(primary, extra))


def nonaligned_relative(
    query: np.ndarray, clean: np.ndarray, count: int, seed: int
) -> np.ndarray:
    """Match the runtime profile cardinality without aligned performances."""

    if not len(clean):
        raise RuntimeError("Clean profile has no windows")
    result = np.empty((len(query), INPUTS), dtype=np.float32)
    # Half-capture offset makes paired ToneTwist dry/wet windows deliberately
    # non-aligned; the coprime step varies the reference across query windows.
    base = seed + max(1, len(clean) // 2)
    for index, value in enumerate(query):
        selected = [clean[(base + index * 17 + offset * 31) % len(clean)] for offset in range(count)]
        profile = np.stack(selected)
        mean = profile.mean(axis=0)
        deviation = profile.std(axis=0) + 1.0e-4
        result[index] = np.concatenate(
            (value, mean, value - mean, np.abs(value - mean), deviation)
        )
    return result


def split_summary(
    family: str, truth: np.ndarray, split: np.ndarray, categories: np.ndarray
) -> dict:
    result = {}
    for name in ("train", "valid", "calibrate", "test"):
        selected = split == name
        counts = {
            CATALOG[family][index]: int(np.logical_and(selected, truth == index).sum())
            for index in range(3)
        }
        negative = np.logical_and(selected, truth < 0)
        result[name] = {
            "samples": int(selected.sum()),
            "identity": counts,
            "open_set_negative": int(negative.sum()),
            "negative_categories": {
                category: int(np.logical_and(negative, categories == category).sum())
                for category in sorted(set(categories[negative]))
            },
        }
    return result


def tone_devices(root: Path) -> tuple[ToneDevice, ...]:
    return (
        ToneDevice(
            "ts9",
            "TubeScreamer",
            root / "tonetwist-pedal-identity-ts9-nc",
            "https://zenodo.org/records/10797988",
            "physical Ibanez TS9, two settings, independent dry source",
        ),
        ToneDevice(
            "rodent",
            "RAT",
            root / "tonetwist-pedal-identity-rodent-nc",
            "https://zenodo.org/records/10796378",
            "physical Harley Benton Rodent, seven settings, RAT-style circuit",
        ),
        ToneDevice(
            "bdrive",
            "BluesDriver",
            root / "tonetwist-pedal-identity-bdrive-nc",
            "https://zenodo.org/records/10901417",
            "Multidrive Pedal Pro B-Drive, 21 settings, Blues Driver emulation",
        ),
    )


def tone_source_key(path: Path) -> str:
    stem = path.name.lower()
    for suffix in (".target.wav", ".input.wav"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.rsplit(".", 1)[-1].strip(".")


def select_dry(wet: Path, dry: list[Path]) -> Path:
    if len(dry) == 1:
        return dry[0]
    source_key = tone_source_key(wet)
    source_matches = [path for path in dry if tone_source_key(path) == source_key]
    if len(source_matches) == 1:
        return source_matches[0]
    lowered = wet.name.lower()
    for token in ("test", "train", "val"):
        if token in lowered:
            matches = [path for path in dry if token in path.name.lower()]
            if len(matches) == 1:
                return matches[0]
    raise RuntimeError(f"cannot select dry input for {wet} from {dry}")


def tonetwist_identity_features(
    encoder: Detector,
    cache: Path,
    checkpoint: Path,
    root: Path,
    target: torch.device,
    devices: tuple[ToneDevice, ...] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[dict]]:
    """Encode NC research-only device positives without user-side fitting.

    ToneTwist targets are long captures at several knob settings. All settings
    for the same 2.5-second source interval share one split group. The Clean
    input becomes one unordered global profile; query/reference performances
    are never aligned feature-by-feature.
    """

    checkpoint_hash = sha256(checkpoint)
    features: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    inventory: list[dict] = []
    for device in devices or tone_devices(root):
        dry_root = (
            device.root
            if device.key == "ts9"
            else root / "tonetwist-pedal-identity-dry-nc"
        )
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
        if not dry or not wet:
            raise RuntimeError(f"missing extracted ToneTwist audio for {device.key}")
        device_count = 0
        for wet_path in wet:
            dry_path = select_dry(wet_path, dry)
            clean = cached_audio_embedding(
                encoder, dry_path, cache, checkpoint_hash, target
            )
            query = cached_audio_embedding(
                encoder, wet_path, cache, checkpoint_hash, target
            )
            # Synchronization markers occupy the edges of ToneTwist captures.
            clean = clean[1:-1] if len(clean) > 4 else clean
            query = query[1:-1] if len(query) > 4 else query
            relative_seed = int(hashlib.sha256(str(wet_path).encode()).hexdigest()[:8], 16)
            value = nonaligned_relative(
                query, clean, IDENTITY_REFERENCE_WINDOWS, relative_seed
            )
            source_key = tone_source_key(wet_path)
            group_root = (
                "tonetwist-external-ts9"
                if device.key == "ts9"
                else "tonetwist-common:" + source_key
            )
            if "test" in {part.lower() for part in wet_path.parts} and "trainvaltest" not in {
                part.lower() for part in wet_path.parts
            }:
                window_splits = np.full(len(value), "test")
            else:
                window_splits = np.asarray(
                    [
                        partition(
                            Source(
                                wet_path,
                                f"{group_root}:{index:05d}",
                                f"tonetwist-{device.key}",
                            )
                        )
                        for index in range(len(value))
                    ]
                )
            features.append(value)
            truth.append(np.full(len(value), CATALOG["drive"].index(device.label), dtype=np.int64))
            splits.append(window_splits)
            categories.append(np.full(len(value), "ToneTwist " + device.key))
            domains.append(np.full(len(value), "tonetwist-" + device.key))
            device_count += len(value)
        inventory.append(
            {
                "key": device.key,
                "label": device.label,
                "record": device.record,
                "description": device.description,
                "wet_files": len(wet),
                "windows": device_count,
            }
        )
    return (
        tuple(
            np.concatenate(parts)
            for parts in (features, truth, splits, categories, domains)
        ),
        inventory,
    )


def tonetwist_big_muff_unknown_features(
    encoder: Detector,
    cache: Path,
    checkpoint: Path,
    root: Path,
    target: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use existing real Big Muff pairs as same-family open-set exposure."""

    pairs = (
        (
            "train",
            "diy",
            root / "diy-big-muff/DRY/input.wav",
            root
            / "diy-big-muff/DIY-ElectroHarmonix-BigMuff/Vol=6_Tone=2_Sustain=5/target.wav",
        ),
        (
            "train",
            "ehx-train",
            root / "ehx-big-muff/DRY/trainval/train.input.wav",
            root
            / "ehx-big-muff/ElectroHarmonix-BigMuff/trainval/S050_V100/S050_V100.train.target.wav",
        ),
        (
            "valid-calibrate",
            "ehx-valid",
            root / "ehx-big-muff/DRY/trainval/val.input.wav",
            root
            / "ehx-big-muff/ElectroHarmonix-BigMuff/trainval/S050_V100/S050_V100.val.target.wav",
        ),
        (
            "test",
            "ehx-test",
            root / "ehx-big-muff/DRY/test/test.input.wav",
            root
            / "ehx-big-muff/ElectroHarmonix-BigMuff/test/S050_V100/S050_V100.test.target.wav",
        ),
    )
    checkpoint_hash = sha256(checkpoint)
    result: list[list[np.ndarray]] = [[], [], [], [], []]
    for declared_split, device, dry_path, wet_path in pairs:
        if not dry_path.exists() or not wet_path.exists():
            raise RuntimeError(f"missing ToneTwist Big Muff pair: {dry_path} / {wet_path}")
        clean = cached_audio_embedding(
            encoder, dry_path, cache, checkpoint_hash, target
        )
        query = cached_audio_embedding(
            encoder, wet_path, cache, checkpoint_hash, target
        )
        clean = clean[1:-1] if len(clean) > 4 else clean
        query = query[1:-1] if len(query) > 4 else query
        relative_seed = int(hashlib.sha256(str(wet_path).encode()).hexdigest()[:8], 16)
        value = nonaligned_relative(
            query, clean, IDENTITY_REFERENCE_WINDOWS, relative_seed
        )
        if declared_split == "valid-calibrate":
            split = np.asarray(
                [
                    "valid"
                    if int(hashlib.sha256(f"{device}:{index}".encode()).hexdigest()[:8], 16)
                    % 2
                    == 0
                    else "calibrate"
                    for index in range(len(value))
                ]
            )
        else:
            split = np.full(len(value), declared_split)
        result[0].append(value)
        result[1].append(np.full(len(value), -1, dtype=np.int64))
        result[2].append(split)
        result[3].append(np.full(len(value), "ToneTwist Big Muff non-catalog"))
        result[4].append(np.full(len(value), "tonetwist-big-muff-" + device))
    return tuple(np.concatenate(parts) for parts in result)


def remfx_unknown_features(
    family: str,
    encoder: Detector,
    cache: Path,
    checkpoint: Path,
    root: Path,
    target: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build non-aligned same-family rejection examples from RemFX pairs."""

    items = discover_remfx_1_1(root)
    dry_by_group = {
        item.source.group: item
        for item in items
        if item.target is not None and not any(item.target)
    }
    wet = [
        item
        for item in items
        if item.target is not None and item.target[FAMILY_INDEX[family]] > 0.5
    ]
    dry_by_split: dict[str, list] = {name: [] for name in ("train", "valid", "calibrate", "test")}
    for item in dry_by_group.values():
        dry_by_split[partition(item.source)].append(item)
    checkpoint_hash = sha256(checkpoint)
    count = IDENTITY_REFERENCE_WINDOWS
    result: list[list[np.ndarray]] = [[], [], [], [], []]
    for item in wet:
        split_name = partition(item.source)
        candidates = [
            value
            for value in dry_by_split[split_name]
            if value.source.group != item.source.group
        ]
        if len(candidates) < count:
            raise RuntimeError(f"not enough RemFX Clean references for {item.source.group}")
        seed = int(
            hashlib.sha256(f"{family}:{item.source.group}".encode()).hexdigest()[:8], 16
        )
        selected = [candidates[(seed + offset) % len(candidates)] for offset in range(count)]
        clean = np.concatenate(
            [
                cached_audio_embedding(
                    encoder, value.source.path, cache, checkpoint_hash, target
                )
                for value in selected
            ]
        )
        query = cached_audio_embedding(
            encoder, item.source.path, cache, checkpoint_hash, target
        )
        value = nonaligned_relative(query, clean, count, seed)
        result[0].append(value)
        result[1].append(np.full(len(value), -1, dtype=np.int64))
        result[2].append(np.full(len(value), split_name))
        result[3].append(np.full(len(value), "RemFX non-catalog " + family))
        result[4].append(np.full(len(value), "remfx-1-1"))
    return tuple(np.concatenate(parts) for parts in result)


def infer(model: IdentityHead, values: np.ndarray, target: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    identities = []
    known = []
    with torch.no_grad():
        for start in range(0, len(values), 512):
            batch = torch.from_numpy(values[start : start + 512]).to(target)
            identity, knownness = model(batch)
            identities.append(identity.cpu().numpy())
            known.append(knownness.cpu().numpy())
    return np.concatenate(identities), np.concatenate(known)


def train_head(
    family: str,
    features: np.ndarray,
    truth: np.ndarray,
    split: np.ndarray,
    categories: np.ndarray,
    output: Path,
    target: torch.device,
    epochs: int,
    resume: bool,
) -> tuple[IdentityHead, dict]:
    train_mask = split == "train"
    valid_mask = split == "valid"
    mean = features[train_mask].mean(axis=0)
    deviation = features[train_mask].std(axis=0) + 1.0e-4
    model = IdentityHead(mean, deviation).to(target)
    train_values = torch.from_numpy(features[train_mask])
    train_truth = torch.from_numpy(truth[train_mask])
    train_categories = categories[train_mask].astype(str)
    sample_weight = torch.zeros(len(train_truth), dtype=torch.float32)
    for class_index in (-1, 0, 1, 2):
        class_mask = (train_truth < 0) if class_index < 0 else (train_truth == class_index)
        class_categories = sorted(set(train_categories[class_mask.numpy()]))
        for category in class_categories:
            group = torch.logical_and(
                class_mask, torch.from_numpy(train_categories == category)
            )
            sample_weight[group] = 1.0 / (4 * len(class_categories) * group.sum())
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    loader = DataLoader(
        TensorDataset(train_values, train_truth),
        batch_size=256,
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2.0e-4)
    best = float("inf")
    stale = 0
    completed = 0
    checkpoint = output / f"{family}-identity.pt"
    started = time.perf_counter()
    if resume and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=True))
        return model, {
            "epochs": 0,
            "best_validation_loss": None,
            "seconds": time.perf_counter() - started,
            "parameters": sum(value.numel() for value in model.parameters()),
            "resumed_checkpoint": str(checkpoint.resolve()),
        }
    for epoch in range(epochs):
        model.train()
        for values, labels in loader:
            values = values.to(target)
            labels = labels.to(target)
            identity, knownness = model(values)
            mask = labels >= 0
            known_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                knownness, mask.float()
            )
            identity_loss = (
                torch.nn.functional.cross_entropy(identity[mask], labels[mask])
                if mask.any()
                else torch.zeros((), device=target)
            )
            loss = known_loss + identity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        identity, knownness = infer(model, features[valid_mask], target)
        labels = truth[valid_mask]
        mask = labels >= 0
        validation = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(knownness), torch.from_numpy(mask.astype(np.float32))
            )
        )
        if mask.any():
            validation += float(
                torch.nn.functional.cross_entropy(
                    torch.from_numpy(identity[mask]), torch.from_numpy(labels[mask])
                )
            )
        completed = epoch + 1
        print("pedal_identity_epoch", family, completed, validation, flush=True)
        if validation < best - 1.0e-5:
            best = validation
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=True))
    return model, {
        "epochs": completed,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
    }


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def softmax(value: np.ndarray) -> np.ndarray:
    value = value - value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def select_threshold(identity: np.ndarray, knownness: np.ndarray, truth: np.ndarray) -> float:
    confidence = softmax(identity).max(axis=1) * sigmoid(knownness)
    prediction = identity.argmax(axis=1)
    positive = truth >= 0
    best = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        accepted = confidence >= threshold
        false_accept = float(accepted[~positive].mean()) if (~positive).any() else 0.0
        correct = float(
            np.logical_and.reduce((accepted, positive, prediction == truth)).sum()
            / max(positive.sum(), 1)
        )
        feasible = false_accept <= 0.05
        key = (feasible, correct, -false_accept, -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def metrics(
    family: str,
    identity: np.ndarray,
    knownness: np.ndarray,
    truth: np.ndarray,
    categories: np.ndarray,
    pickups: np.ndarray,
    threshold: float,
) -> dict:
    probability = softmax(identity)
    confidence = probability.max(axis=1) * sigmoid(knownness)
    prediction = identity.argmax(axis=1)
    accepted = confidence >= threshold
    result = np.where(accepted, prediction, -1)
    positive = truth >= 0
    closed = float((prediction[positive] == truth[positive]).mean())
    correct_accept = float((result[positive] == truth[positive]).mean())
    false_accept = float(accepted[~positive].mean())
    per_class = {}
    for index, name in enumerate(CATALOG[family]):
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
    negative_fars = {}
    for category in sorted(set(categories[~positive])):
        mask = categories == category
        negative_fars[category] = float(accepted[mask].mean())
    pickup_metrics = {}
    for pickup in sorted(set(pickups)):
        mask = pickups == pickup
        pickup_positive = np.logical_and(mask, positive)
        pickup_negative = np.logical_and(mask, ~positive)
        pickup_metrics[pickup] = {
            "correct_accept": float(
                (result[pickup_positive] == truth[pickup_positive]).mean()
            )
            if pickup_positive.any()
            else None,
            "negative_false_accept": float(accepted[pickup_negative].mean())
            if pickup_negative.any()
            else None,
        }
    return {
        "samples": int(len(truth)),
        "positive_samples": int(positive.sum()),
        "negative_samples": int((~positive).sum()),
        "closed_set_accuracy": closed,
        "correct_accept_rate": correct_accept,
        "negative_false_accept_rate": false_accept,
        "clean_false_positive_rate": float(
            accepted[np.logical_and(~positive, categories == "Clean")].mean()
        )
        if np.logical_and(~positive, categories == "Clean").any()
        else None,
        "per_class": per_class,
        "negative_category_false_accept": negative_fars,
        "pickup": pickup_metrics,
    }


def export(model: IdentityHead, path: Path) -> float:
    model = model.cpu().eval()
    dummy = torch.zeros(1, INPUTS)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["relative_features"],
        output_names=["identity_logits", "known_logit"],
        dynamic_axes=None,
        opset_version=17,
        dynamo=False,
    )
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    expected = model(dummy)
    actual = session.run(None, {"relative_features": dummy.numpy()})
    return max(
        float(np.max(np.abs(expected[index].detach().numpy() - actual[index])))
        for index in range(2)
    )


def file_features(
    path: Path,
    encoder: Detector,
    profile_mean: np.ndarray,
    profile_deviation: np.ndarray,
) -> np.ndarray:
    query, _ = encode_relative_windows(encoder, path, "embedding-logits", torch.device("cpu"))
    return np.concatenate(
        (
            query,
            np.broadcast_to(profile_mean, query.shape),
            query - profile_mean,
            np.abs(query - profile_mean),
            np.broadcast_to(profile_deviation, query.shape),
        ),
        axis=1,
    ).astype(np.float32)


def top_two(values: np.ndarray) -> np.ndarray:
    count = min(2, len(values))
    return np.partition(values, len(values) - count, axis=0)[-count:].mean(axis=0)


def external_development(
    directory: Path,
    reference: Path,
    models: dict[str, IdentityHead],
    encoders: dict[str, Detector],
    thresholds: dict[str, float],
) -> dict:
    if not directory.exists() or not reference.exists():
        return {"available": False}
    profiles = {}
    for family, encoder in encoders.items():
        query, _ = encode_relative_windows(
            encoder, reference, "embedding-logits", torch.device("cpu")
        )
        count = reference_window_count(10, len(query))
        selected = query[:count]
        profiles[family] = (selected.mean(axis=0), selected.std(axis=0) + 1.0e-4)
    rows = []
    for path in sorted(directory.glob("*.wav")):
        if path.resolve() == reference.resolve():
            continue
        lowered = path.stem.lower()
        active = {
            "drive": any(value in lowered for value in ("drive", "fuzz", "rat", "muff")),
            "delay": any(value in lowered for value in ("delay", "echo")),
            "reverb": any(
                value in lowered
                for value in ("ambience", "dream", "reverb", "room", "hall", "plate")
            ),
        }
        for family in CATALOG:
            if not active[family]:
                continue
            features = file_features(path, encoders[family], *profiles[family])
            identity, knownness = infer(models[family], features, torch.device("cpu"))
            probability = softmax(identity)
            combined = probability * sigmoid(knownness)[:, None]
            aggregate = top_two(combined)
            index = int(aggregate.argmax())
            accepted = float(aggregate[index]) >= thresholds[family]
            expected = "RAT" if family == "drive" and "rat" in lowered else None
            predicted = CATALOG[family][index] if accepted else None
            rows.append(
                {
                    "file": path.name,
                    "family": family,
                    "expected_identity": expected,
                    "predicted_identity": predicted,
                    "score": float(aggregate[index]),
                    "correct": predicted == expected,
                }
            )
    rat = [row for row in rows if row["expected_identity"] == "RAT"]
    reject = [row for row in rows if row["expected_identity"] is None]
    return {
        "available": True,
        "role": "identity development evaluation only; no weight or threshold updates",
        "rows": rows,
        "rat_recall": sum(row["predicted_identity"] == "RAT" for row in rat) / len(rat) if rat else None,
        "noncatalog_false_accept": sum(row["predicted_identity"] is not None for row in reject)
        / len(reject)
        if reject
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=CORPUS / "egfxset")
    parser.add_argument("--cache", type=Path, default=PEDAL_IDENTITY_CACHE)
    parser.add_argument("--output", type=Path, default=PEDAL_IDENTITY_RUN)
    parser.add_argument(
        "--drive-delay-checkpoint",
        type=Path,
        default=DRIVE_DELAY_ENCODER_RUN / "best.pt",
    )
    parser.add_argument(
        "--reverb-checkpoint",
        type=Path,
        default=REVERB_ENCODER_RUN / "backbone.pt",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-dafx-negatives",
        action="store_true",
        help="ablation: omit public cross-domain generic-effect rejection examples",
    )
    parser.add_argument(
        "--research-data",
        "--tonetwist-research",
        dest="research_data",
        action="store_true",
        help="include audited CC-BY-NC ToneTwist/RemFX data; never release-compatible",
    )
    parser.add_argument("--development", type=Path, default=Path.home() / "Downloads/test")
    parser.add_argument(
        "--reference", type=Path, default=Path.home() / "Downloads/test/clean.wav"
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    target = target_device()
    args.output.mkdir(parents=True, exist_ok=True)
    records = discover(args.data)
    drive_delay_query, reverb_query = build_cache(
        records,
        args.cache,
        args.drive_delay_checkpoint,
        args.reverb_checkpoint,
        args.batch,
        target,
    )
    reports = {}
    models = {}
    thresholds = {}
    encoders = {
        "drive": load_encoder(args.drive_delay_checkpoint),
        "delay": load_encoder(args.drive_delay_checkpoint),
        "reverb": load_encoder(args.reverb_checkpoint),
    }
    tonetwist = None
    tonetwist_unknown = None
    tonetwist_inventory: list[dict] = []
    if args.research_data:
        tonetwist, tonetwist_inventory = tonetwist_identity_features(
            encoders["drive"],
            args.cache,
            args.drive_delay_checkpoint,
            CORPUS,
            target,
        )
        tonetwist_unknown = tonetwist_big_muff_unknown_features(
            encoders["drive"],
            args.cache,
            args.drive_delay_checkpoint,
            CORPUS / "tonetwist-nc",
            target,
        )
    for family in CATALOG:
        queries = reverb_query if family == "reverb" else drive_delay_query
        features, truth, split, categories, pickups = relative_features(
            family, records, queries
        )
        if not args.no_dafx_negatives:
            cache = REVERB_CACHE if family == "reverb" else DRIVE_DELAY_CACHE
            features, truth, split, categories, pickups = append_examples(
                (features, truth, split, categories, pickups),
                dafx_unknown_features(family, cache),
            )
        if family == "drive" and tonetwist is not None:
            features, truth, split, categories, pickups = append_examples(
                (features, truth, split, categories, pickups), tonetwist
            )
            features, truth, split, categories, pickups = append_examples(
                (features, truth, split, categories, pickups), tonetwist_unknown
            )
        if args.research_data:
            checkpoint = (
                args.reverb_checkpoint if family == "reverb" else args.drive_delay_checkpoint
            )
            features, truth, split, categories, pickups = append_examples(
                (features, truth, split, categories, pickups),
                remfx_unknown_features(
                    family,
                    encoders[family],
                    args.cache,
                    checkpoint,
                    CORPUS / "remfx-1-1",
                    target,
                ),
            )
        model, training = train_head(
            family,
            features,
            truth,
            split,
            categories,
            args.output,
            target,
            args.epochs,
            args.resume,
        )
        calibrate = split == "calibrate"
        test = split == "test"
        calibration_identity, calibration_known = infer(
            model, features[calibrate], target
        )
        threshold = select_threshold(
            calibration_identity, calibration_known, truth[calibrate]
        )
        test_identity, test_known = infer(model, features[test], target)
        test_report = metrics(
            family,
            test_identity,
            test_known,
            truth[test],
            categories[test],
            pickups[test],
            threshold,
        )
        onnx_path = args.output / f"{family}-identity.onnx"
        parity = export(model, onnx_path)
        reports[family] = {
            "catalog": list(CATALOG[family]),
            "threshold": threshold,
            "split": split_summary(family, truth, split, categories),
            "training": training,
            "test": test_report,
            "export": {
                "path": str(onnx_path.resolve()),
                "sha256": sha256(onnx_path),
                "max_absolute_difference": parity,
            },
        }
        models[family] = model.cpu()
        thresholds[family] = threshold
    external = external_development(
        args.development, args.reference, models, encoders, thresholds
    )
    public_failures = []
    for family, report in reports.items():
        test = report["test"]
        if test["closed_set_accuracy"] < 0.80:
            public_failures.append(f"{family} closed-set accuracy is below 80%")
        if test["correct_accept_rate"] < 0.70:
            public_failures.append(f"{family} correct-accept rate is below 70%")
        if test["negative_false_accept_rate"] > 0.05:
            public_failures.append(f"{family} negative false-accept rate exceeds 5%")
        if (
            test["clean_false_positive_rate"] is not None
            and test["clean_false_positive_rate"] > 0.02
        ):
            public_failures.append(f"{family} Clean false-positive rate exceeds 2%")
        for identity, identity_metrics in test["per_class"].items():
            if identity_metrics["recall"] < 0.65:
                public_failures.append(f"{identity} recall is below 65%")
        dafx_far = test["negative_category_false_accept"].get(
            "DAFx non-catalog " + family
        )
        if dafx_far is not None and dafx_far > 0.05:
            public_failures.append(f"{family} DAFx non-catalog false accept exceeds 5%")
        big_muff_far = test["negative_category_false_accept"].get(
            "ToneTwist Big Muff non-catalog"
        )
        if big_muff_far is not None and big_muff_far > 0.10:
            public_failures.append("Drive ToneTwist Big Muff false accept exceeds 10%")
    development_failures = []
    if not external.get("available"):
        development_failures.append("hardware development recordings are unavailable")
    else:
        if external.get("rat_recall") is None or external["rat_recall"] < 0.50:
            development_failures.append("hardware RAT recall is below 50%")
        if (
            external.get("noncatalog_false_accept") is None
            or external["noncatalog_false_accept"] > 0.20
        ):
            development_failures.append("hardware non-catalog false accept exceeds 20%")
    payload = {
        "experiment": args.output.name,
        "architecture": {
            "backbones": "frozen routed Inspector encoders",
            "input": "1,295-value non-aligned Clean-relative feature",
            "identity_clean_profile": "three non-aligned windows (about ten seconds), separate from family-detector profile",
            "heads": "family-specific 192/96 MLP with identity and knownness outputs",
            "user_gradient_updates": 0,
            "seed": SEED,
            "requested_epochs": args.epochs,
            "head_batch_size": 256,
            "sampler": "balanced open-set plus three identity classes, then balanced source categories within class",
        },
        "data": {
            "sources": [
                "EGFxSet CC BY 4.0",
                "Guitar improvisations with chains of five effects CC BY 4.0",
            ]
            + (
                [
                    "ToneTwist pedal identity/Big Muff captures CC BY-NC 4.0",
                    "RemFX single-effect pairs CC-NC",
                ]
                if args.research_data
                else []
            ),
            "egfx_files": len(records),
            "catalog": CATALOG,
            "open_set_negatives": [
                "Clean",
                *MODULATION,
                "other target families",
                "cross-domain generic DAFx chain effects",
            ],
            "split": "performance-disjoint SHA-256 group partition",
            "limitation": (
                "ToneTwist adds cross-source/settings for Drive only; Delay/Reverb identities remain EGFx-only"
                if args.research_data
                else "identity positives remain single-device/single-setting in EGFxSet"
            ),
            "tonetwist_research": {
                "enabled": args.research_data,
                "license": "CC-BY-NC-4.0; non-commercial research only"
                if args.research_data
                else None,
                "inventory": tonetwist_inventory,
                "open_set_device": "real DIY/EHX Big Muff dry/wet pairs",
                "source_policy": "guitar-only; ToneTwist bass sources excluded",
                "user_recordings_used_for_gradients": False,
                "remfx_open_set": "generic Drive/Delay/Reverb single-effect pairs; non-aligned Clean references",
            },
        },
        "families": reports,
        "hardware_development": external,
        "public_gate": {
            "level": "public performance/device-split development",
            "passed": not public_failures,
            "failures": public_failures,
        },
        "integration_gate": {
            "passed": not public_failures and not development_failures,
            "failures": public_failures + development_failures,
        },
        "release_gate": {
            "passed": False,
            "failures": [
                "no untouched device-disjoint labelled pedal-identity test is available",
                *(
                    [
                        "ToneTwist CC-BY-NC positives and RemFX CC-NC open-set data cannot enter public release weights"
                    ]
                    if args.research_data
                    else []
                ),
            ],
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "calibration.json").write_text(
        json.dumps({family: {"threshold": value} for family, value in thresholds.items()}, indent=2)
        + "\n"
    )
    print(json.dumps(payload["integration_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
