"""Clean-source-first grouped data and dynamic blind-effect rendering."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile
import torch
from scipy.signal import fftconvolve, resample_poly

import effects
from model import LABELS, RATE, SAMPLES


TARGETS = {
    "BluesDriver": ("drive",),
    "TubeScreamer": ("drive",),
    "RAT": ("drive",),
    "Digital-Delay": ("delay",),
    "TapeEcho": ("delay",),
    "Sweep-Echo": ("delay",),
    "Plate-Reverb": ("reverb",),
    "Hall-Reverb": ("reverb",),
    "Spring-Reverb": ("reverb",),
    # Modulation families are explicit non-target hard negatives in this
    # narrow experiment. They must not be collapsed into an "unknown" class.
    "Chorus": (),
    "Flanger": (),
    "Phaser": (),
}


@dataclass(frozen=True)
class Source:
    path: Path
    group: str
    domain: str
    offset: float = 0.0


def eligible_clean(source: Source) -> bool:
    """Keep unambiguous direct/clean sources for reference and RIR fitting."""

    lowered = source.path.as_posix().lower()
    return not ("guitar-techs" in lowered and "/micamp/" in lowered)


@dataclass(frozen=True)
class Item:
    source: Source
    target: Optional[tuple[float, ...]]
    realization: int = 0
    augment: bool = True
    effect_ir: Optional[Path] = None
    effect_mix: float = 0.0


def vector(labels: Iterable[str]) -> tuple[float, ...]:
    selected = set(labels)
    return tuple(float(label in selected) for label in LABELS)


def discover(root: Path) -> tuple[list[Source], list[Item]]:
    clean = discover_egfx_clean(root / "egfxset")
    clean.extend(discover_techs(root / "guitar-techs"))
    clean.extend(discover_guitarset(root / "guitarset"))
    clean.extend(discover_guitarjam(root / "guitarjam"))
    anchors = discover_egfx_anchors(root / "egfxset")
    if not clean:
        raise RuntimeError(f"no clean training audio found below {root}")
    return clean, anchors


def discover_apple_au(root: Path) -> dict[str, list[Item]]:
    """Load deterministic AU captures without re-hashing their declared split."""

    result = {"train": [], "valid": [], "calibrate": [], "test": []}
    manifest = root / "manifest.json"
    if not manifest.exists():
        return result
    payload = json.loads(manifest.read_text())
    for record in payload["captures"]:
        name = record["split"]
        if name not in result:
            raise RuntimeError(f"invalid Apple AU split: {name}")
        path = root / record["path"]
        if not path.exists():
            raise RuntimeError(f"missing Apple AU capture: {path}")
        source = Source(path, record["source_group"], "apple-au")
        result[name].append(Item(source, vector((record["label"],))))
    return result


def discover_remfx_1_1(root: Path) -> list[Item]:
    """Load official RemFX single-effect pairs as training-only research data.

    The upstream renderer writes effected audio to ``input.wav`` and the
    untouched target to ``target.wav``.  Its public record description reverses
    those names, so this follows the pinned source implementation and verifies
    every five-element label tensor before accepting a pair.
    """

    result = []
    for label_path in sorted(root.rglob("wet_effects.pt")):
        directory = label_path.parent
        wet_path = directory / "input.wav"
        dry_path = directory / "target.wav"
        if not wet_path.exists() or not dry_path.exists():
            raise RuntimeError(f"incomplete RemFX pair: {directory}")
        labels = torch.load(label_path, map_location="cpu", weights_only=True)
        labels = torch.as_tensor(labels).flatten()
        if tuple(labels.shape) != (5,):
            raise RuntimeError(f"invalid RemFX label shape: {label_path}: {labels.shape}")
        active = labels > 0.5
        if int(active.sum()) != 1:
            raise RuntimeError(f"expected one RemFX effect: {label_path}: {labels.tolist()}")
        target = (
            float(active[3]),  # Distortion -> Drive
            float(active[2]),  # Delay
            float(active[0]),  # Reverb
        )
        pair_id = directory.relative_to(root).as_posix()
        group = f"remfx-1-1:{pair_id}"
        result.append(
            Item(Source(wet_path, group, "remfx-1-1"), target, augment=False)
        )
        result.append(
            Item(Source(dry_path, group, "remfx-1-1"), vector(()), augment=False)
        )
    return result


def guitar_effect_chain_split(stem: str) -> str:
    """Keep the DAFx chain benchmark guitar-disjoint from fitting.

    PRS and Les Paul supply fitting data. Stratocaster performances are split
    by whole improvisation between validation and calibration, while every
    Telecaster performance remains an unseen-device test.
    """

    performance = stem.split("__", 1)[0].lower()
    guitar = performance.split("_", 1)[0]
    if guitar in {"prs", "les"}:
        return "train"
    if guitar == "tele":
        return "test"
    if guitar == "strat":
        # The archive numbers each of bridge/neck x finger/pick from 01-25.
        # Split within every playing condition so validation and calibration
        # both cover all four conditions while whole improvisations stay intact.
        digits = "".join(character for character in performance if character.isdigit())
        return "valid" if int(digits or "0") <= 12 else "calibrate"
    raise RuntimeError(f"unknown guitar-effects-chains instrument: {stem}")


def guitar_effect_chain_target(path: Path) -> tuple[float, ...]:
    """Decode author-defined Overdrive/Chorus/Tremolo/Delay/Reverb bits."""

    if "__" not in path.stem:
        return vector(())
    bits = path.stem.rsplit("__", 1)[-1]
    if len(bits) != 5 or any(bit not in "01" for bit in bits):
        raise RuntimeError(f"invalid guitar-effects-chains label: {path}")
    return vector(
        label
        for label, enabled in (("drive", bits[0]), ("delay", bits[3]), ("reverb", bits[4]))
        if enabled == "1"
    )


def guitar_effect_chain_is_clean(source: Source) -> bool:
    if not source.domain.startswith("guitar-effects-chains-"):
        return False
    return "__" not in source.path.stem or source.path.stem.endswith("__00000")


def discover_guitar_effect_chains(root: Path) -> dict[str, list[Item]]:
    """Load the CC-BY-4.0 DAFx25 five-effect-chain dataset.

    Every archived recording is fixed data. Modulation-only examples remain
    target-negative hard cases, but only true dry/00000 files may serve as a
    user's non-aligned Clean reference during pair-head training.
    """

    result = {name: [] for name in ("train", "valid", "calibrate", "test")}
    for path in audio_files(root):
        stem = path.stem.lower()
        if "__" in stem:
            bits = stem.rsplit("__", 1)[-1]
            if len(bits) != 5 or any(bit not in "01" for bit in bits):
                continue
        performance = stem.split("__", 1)[0]
        guitar = performance.split("_", 1)[0]
        if guitar not in {"prs", "les", "strat", "tele"}:
            continue
        lowered_parts = "/".join(part.lower() for part in path.parts)
        if "position" in lowered_parts:
            variant = "variable-order"
        elif "param" in lowered_parts:
            variant = "variable-parameters"
        elif "__" in stem:
            variant = "fixed"
        else:
            variant = "dry"
        source = Source(
            path,
            f"guitar-effects-chains:{performance}",
            f"guitar-effects-chains-{guitar}-{variant}",
        )
        result[guitar_effect_chain_split(stem)].append(
            Item(source, guitar_effect_chain_target(path), augment=False)
        )
    audit(result)
    return result


def discover_tonetwist_big_muff_nc(
    root: Path, train_repeats: int = 1
) -> dict[str, list[Item]]:
    """Load audited ToneTwisT Big Muff dry/wet pairs for NC research.

    The DIY recording has no author-defined split and is training-only. The
    EHX recording's published train/validation/test boundaries are preserved.
    Dry and wet chunks share a group so an entire five-second performance
    interval remains in one partition. Wet recordings are Drive positives and
    explicit Delay/Reverb hard negatives.
    """

    if train_repeats < 1:
        raise ValueError("train_repeats must be positive")
    result = {name: [] for name in ("train", "valid", "calibrate", "test")}
    records = (
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
            "valid",
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
    for split_name, device, dry_path, wet_path in records:
        if not dry_path.exists() or not wet_path.exists():
            raise RuntimeError(f"missing ToneTwisT Big Muff pair: {dry_path} / {wet_path}")
        dry_info = soundfile.info(dry_path)
        wet_info = soundfile.info(wet_path)
        seconds = min(
            dry_info.frames / dry_info.samplerate,
            wet_info.frames / wet_info.samplerate,
        )
        chunks = max(1, int(seconds // 5.0))
        repeats = train_repeats if split_name == "train" else 1
        for chunk in range(chunks):
            group = f"tonetwist-nc:{device}:{chunk:04d}"
            domain = f"tonetwist-nc-{device}"
            for realization in range(repeats):
                result[split_name].append(
                    Item(
                        Source(dry_path, group, domain, chunk * 5.0),
                        vector(()),
                        realization,
                        augment=False,
                    )
                )
                result[split_name].append(
                    Item(
                        Source(wet_path, group, domain, chunk * 5.0),
                        vector(("drive",)),
                        realization,
                        augment=False,
                    )
                )
    audit(result)
    return result


def discover_aachen_rirs(root: Path) -> list[Path]:
    """Discover only the measured CC-BY-4.0 chapel impulse responses."""

    return sorted(
        path
        for path in audio_files(root)
        if "brir" in "/".join(part.lower() for part in path.parts)
    )


def add_rir_reverb(
    parts: dict[str, list[Item]], rirs: list[Path], clean_paths: set[Path]
) -> dict[str, list[Item]]:
    """Add fixed, split-disjoint measured-RIR reverb examples.

    The rendered waveform is ephemeral and reproducible. No derived audio
    corpus is written to disk. Each source keeps its existing group/split, and
    each split receives a disjoint subset of measured rooms/positions.
    """

    if len(rirs) < 8:
        raise RuntimeError(f"expected measured Aachen RIRs, found {len(rirs)}")
    boundaries = (int(len(rirs) * 0.60), int(len(rirs) * 0.76), int(len(rirs) * 0.88))
    rir_parts = {
        "train": rirs[: boundaries[0]],
        "valid": rirs[boundaries[0] : boundaries[1]],
        "calibrate": rirs[boundaries[1] : boundaries[2]],
        "test": rirs[boundaries[2] :],
    }
    result = {name: list(items) for name, items in parts.items()}
    for name, items in parts.items():
        available = rir_parts[name]
        clean_items = [
            item
            for item in items
            if item.target is not None
            and not any(item.target)
            and item.effect_ir is None
            and item.source.path.resolve() in clean_paths
        ]
        for item in clean_items:
            identity = stable(item.source.group)
            mix = (0.25, 0.40, 0.55, 0.70)[identity % 4]
            result[name].append(
                Item(
                    item.source,
                    vector(("reverb",)),
                    realization=identity % 1_000_003,
                    augment=False,
                    effect_ir=available[identity % len(available)],
                    effect_mix=mix,
                )
            )
    audit(result)
    return result


def discover_idmt_guitar(root: Path, manifest: Path) -> dict[str, list[Item]]:
    """Load the audited CC-BY-NC-ND IDMT guitar subset.

    Instrument settings 6/7 are training devices, setting 8 supplies grouped
    validation/calibration material, and setting 9 is a device-disjoint test.
    The manifest freezes a balanced sample without modifying upstream audio.
    """

    result = {name: [] for name in ("train", "valid", "calibrate", "test")}
    if not manifest.exists():
        return result
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != 1 or payload.get("doi") != "10.5281/zenodo.7544032":
        raise RuntimeError(f"invalid IDMT guitar manifest: {manifest}")
    target_mapping = {
        "clean": vector(()),
        "drive": vector(("drive",)),
        "delay": vector(("delay",)),
        "reverb": vector(("reverb",)),
    }
    owners: dict[str, str] = {}
    for record in payload["records"]:
        name = record["split"]
        if name not in result or record["target"] not in target_mapping:
            raise RuntimeError(f"invalid IDMT record: {record}")
        path = root / record["path"]
        if not path.exists():
            raise RuntimeError(f"missing IDMT guitar audio: {path}")
        group = f"idmt:{record['group']}"
        previous = owners.setdefault(group, name)
        if previous != name:
            raise RuntimeError(f"IDMT note group leaked across splits: {group}: {previous}/{name}")
        source = Source(
            path,
            group,
            f"idmt-setting-{record['instrument_setting']}",
        )
        result[name].append(Item(source, target_mapping[record["target"]]))
    return result


def discover_egfx_clean(root: Path) -> list[Source]:
    result = []
    for path in audio_files(root / "Clean"):
        pickup = path.parent.name.lower()
        result.append(Source(path, f"egfx:{pickup}:{canonical(path.stem)}", "egfx"))
    return result


def discover_egfx_anchors(root: Path) -> list[Item]:
    result = []
    for directory, labels in TARGETS.items():
        for path in audio_files(root / directory):
            pickup = path.parent.name.lower()
            source = Source(path, f"egfx:{pickup}:{canonical(path.stem)}", "egfx")
            result.append(Item(source, vector(labels)))
    return result


def discover_techs(root: Path) -> list[Source]:
    result = []
    for path in audio_files(root):
        lowered = "/".join(part.lower() for part in path.parts)
        name = path.stem.lower()
        # Both feeds are clean with respect to the four target effect families.
        # Keeping the real mic/amp capture as a zero-label nuisance is essential:
        # otherwise the network can mistake cabinet and room coloration for
        # drive, delay, or reverb. Pair both feeds under one performance group.
        if not any(token in lowered or token in name for token in ("directinput", "micamp")):
            continue
        player = next(
            (
                part.lower().split("_", 1)[0]
                for part in path.parts
                if part.lower().split("_", 1)[0] in {"p1", "p2", "p3"}
            ),
            "player",
        )
        info = soundfile.info(path)
        chunks = max(1, int(np.ceil((info.frames / info.samplerate) / 5.0)))
        group = f"techs:{player}:{canonical(path.stem)}"
        for chunk in range(chunks):
            result.append(Source(path, group, f"techs-{player}", chunk * 5.0))
    return result


def discover_guitarset(root: Path) -> list[Source]:
    """Pair GuitarSet microphone and pickup captures by performance."""

    result = []
    for path in audio_files(root):
        player = path.stem[:2]
        if not (len(player) == 2 and player.isdigit()):
            continue
        track = path.stem
        for suffix in ("_mic", "_mix", "_pickup", "_hex"):
            if track.endswith(suffix):
                track = track[: -len(suffix)]
        info = soundfile.info(path)
        chunks = max(1, int(np.ceil((info.frames / info.samplerate) / 5.0)))
        for chunk in range(chunks):
            result.append(
                Source(
                    path,
                    f"guitarset:{track}",
                    f"guitarset-{player}",
                    chunk * 5.0,
                )
            )
    return result


def discover_guitarjam(root: Path) -> list[Source]:
    # This corpus is one player, guitar, pickup setting, and recording chain.
    # It is useful as a real DI hard-negative domain, but cannot demonstrate
    # player/device generalization, so it is deliberately training-only.
    result = []
    for path in audio_files(root):
        info = soundfile.info(path)
        chunks = max(1, int(np.ceil((info.frames / info.samplerate) / 5.0)))
        for chunk in range(chunks):
            result.append(
                Source(path, "guitarjam:session", "guitarjam", chunk * 5.0)
            )
    return result


def partition(source: Source) -> str:
    if source.domain == "guitarjam":
        return "train"
    if source.domain.startswith("guitarset-"):
        player = source.domain.rsplit("-", 1)[-1]
        return {
            "00": "train",
            "01": "train",
            "02": "train",
            "03": "valid",
            "04": "calibrate",
            "05": "test",
        }[player]
    if source.domain == "techs-p1":
        return "train"
    if source.domain == "techs-p2":
        return "valid"
    if source.domain == "techs-p3":
        # Keep music performances out of fitting. A deterministic one-sixth
        # calibration slice exposes thresholds to this capture domain while
        # the remaining whole performances stay unseen test material.
        return "calibrate" if stable(source.group) % 6 == 2 else "test"
    bucket = int(hashlib.sha256(source.group.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "valid" if bucket == 7 else "calibrate" if bucket == 8 else "test"


def split(
    clean: list[Source],
    anchors: list[Item],
    train_realizations: int = 4,
    captures: Optional[dict[str, list[Item]]] = None,
    all_train_anchors: bool = False,
    fixed_clean_train: bool = False,
    anchors_evaluation_only: bool = False,
) -> dict[str, list[Item]]:
    result = {"train": [], "valid": [], "calibrate": [], "test": []}
    for source in clean:
        name = partition(source)
        repeats = train_realizations if name == "train" else 2
        for realization in range(repeats):
            if name == "train" and fixed_clean_train and realization == 0:
                # Preserve one genuinely untreated example from every training
                # source. The remaining realization(s) still receive dynamic
                # target effects, so capture domain and label evidence coexist.
                result[name].append(Item(source, vector(()), realization, False))
            else:
                result[name].append(Item(source, None, realization))
    for item in anchors:
        name = partition(item.source)
        # Validation/calibration/test retain every fixed hardware anchor. The
        # training policy is explicit so experiments can compare the historical
        # one-third subset with full partition-safe anchor coverage.
        if not anchors_evaluation_only and (
            name != "train"
            or all_train_anchors
            or stable(item.source.path.as_posix()) % 3 == 0
        ):
            result[name].append(item)
    if captures:
        for name, items in captures.items():
            result[name].extend(items)
    audit(result)
    return result


def egfx_hardware_test(clean: list[Source], anchors: list[Item]) -> list[Item]:
    """Build a performance-disjoint, fixed EGFx hardware benchmark."""

    result = [
        Item(source, vector(()), augment=False)
        for source in clean
        if source.domain == "egfx" and partition(source) == "test"
    ]
    result.extend(item for item in anchors if partition(item.source) == "test")
    audit({"test": result})
    return result


def audit(parts: dict[str, list[Item]]) -> None:
    owners: dict[str, str] = {}
    for name, items in parts.items():
        for item in items:
            previous = owners.setdefault(item.source.group, name)
            if previous != name:
                raise RuntimeError(
                    f"source group leaked across splits: {item.source.group}: {previous}/{name}"
                )


class Audio(torch.utils.data.Dataset):
    def __init__(
        self,
        items: list[Item],
        training: bool,
        seed: int,
        pedalboard_renderer: bool = False,
    ) -> None:
        self.items = items
        self.training = training
        self.seed = seed
        self.pedalboard_renderer = pedalboard_renderer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        rng = (
            random.Random(random.getrandbits(64))
            if self.training
            else random.Random(self.seed + stable(item.source.group) + item.realization * 104_729)
        )
        audio, target = render_item(
            item, rng, self.training, self.pedalboard_renderer
        )
        return torch.from_numpy(audio), torch.from_numpy(target)


def render_item(
    item: Item,
    rng: random.Random,
    training: bool,
    pedalboard_renderer: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    audio = load(item.source, rng, training)
    if item.target is None:
        if pedalboard_renderer:
            import effects_pedalboard

            return effects_pedalboard.chain(audio, rng)
        return effects.chain(audio, rng)
    target = np.asarray(item.target, dtype=np.float32)
    if item.effect_ir is not None:
        audio = convolve_reverb(audio, item.effect_ir, item.effect_mix)
    elif training and item.augment:
        audio = effects.capture(audio, rng, before=False)
    return audio, target


@lru_cache(maxsize=64)
def room_impulse(path: Path) -> np.ndarray:
    """Load and normalize a measured RIR once per worker process."""

    impulse, rate = soundfile.read(path, dtype="float32", always_2d=True)
    impulse = impulse[:, 0]
    if rate != RATE:
        divisor = np.gcd(rate, RATE)
        impulse = resample_poly(impulse, RATE // divisor, rate // divisor).astype(np.float32)
    impulse = np.nan_to_num(impulse, copy=False)
    peak = float(np.max(np.abs(impulse))) if len(impulse) else 0.0
    if peak <= 1.0e-8:
        raise RuntimeError(f"silent room impulse response: {path}")
    active = np.flatnonzero(np.abs(impulse) >= peak * 0.01)
    impulse = impulse[int(active[0]) :] if len(active) else impulse
    impulse = impulse[: RATE * 8]
    impulse /= np.sqrt(np.sum(impulse * impulse) + 1.0e-8)
    impulse.setflags(write=False)
    return impulse


def convolve_reverb(audio: np.ndarray, path: Path, mix: float) -> np.ndarray:
    """Apply one measured RIR while preserving a pedal-like dry component."""

    impulse = room_impulse(path)
    wet = fftconvolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)
    dry_rms = np.sqrt(np.mean(audio * audio) + 1.0e-8)
    wet_rms = np.sqrt(np.mean(wet * wet) + 1.0e-8)
    wet *= dry_rms / wet_rms
    result = (1.0 - mix) * audio + mix * wet
    maximum = float(np.max(np.abs(result)))
    if maximum > 1.0:
        result /= maximum
    return result.astype(np.float32, copy=False)


def load(source: Source, rng: random.Random, training: bool) -> np.ndarray:
    with soundfile.SoundFile(source.path) as stream:
        rate = stream.samplerate
        stream.seek(min(int(source.offset * rate), len(stream)))
        requested = int(SAMPLES * rate / RATE) + 16
        audio = stream.read(requested, dtype="float32", always_2d=True).mean(axis=1)
    if rate != RATE:
        divisor = np.gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor).astype(np.float32)
    if len(audio) >= SAMPLES:
        maximum = len(audio) - SAMPLES
        start = rng.randrange(maximum + 1) if training and maximum else maximum // 2
        audio = audio[start : start + SAMPLES]
    else:
        missing = SAMPLES - len(audio)
        offset = rng.randrange(missing + 1) if training else missing // 2
        audio = np.pad(audio, (offset, missing - offset))
    return np.nan_to_num(audio, copy=False).clip(-4.0, 4.0).astype(np.float32, copy=False)


def audio_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
        and path.suffix.lower() in {".wav", ".flac", ".aif", ".aiff"}
    )


def canonical(value: str) -> str:
    lowered = value.lower()
    for token in (
        "directinput",
        "micamp",
        "clean",
        "bluesdriver",
        "tube-screamer",
        "tubescreamer",
        "rat",
        "digital-delay",
        "digitaldelay",
        "tapeecho",
        "tape-echo",
        "sweep-echo",
        "sweepecho",
        "plate-reverb",
        "platereverb",
        "hall-reverb",
        "hallreverb",
        "spring-reverb",
        "springreverb",
        "chorus",
        "flanger",
        "phaser",
    ):
        lowered = lowered.replace(token, "")
    return lowered.strip("_- ")


def stable(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def label(target: Optional[tuple[float, ...]]) -> str:
    if target is None:
        return "dynamic"
    active = [LABELS[index] for index, value in enumerate(target) if value > 0.5]
    return "+".join(active) if active else "clean"
