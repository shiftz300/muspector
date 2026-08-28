"""Diverse on-the-fly effects using Spotify Pedalboard plus local DSP.

The renderer is an isolated non-commercial research dependency. It stores no
derived audio: every five-second realization is produced in memory from the
grouped clean source selected by ``data.py``.
"""

from __future__ import annotations

import random

import numpy as np
from pedalboard import Chorus, Clipping, Delay, Distortion, Pedalboard, Phaser, Reverb
from scipy.signal import fftconvolve

import effects
from model import LABELS, RATE


def process(plugin, audio: np.ndarray) -> np.ndarray:
    value = Pedalboard([plugin])(audio.astype(np.float32), RATE)
    return np.asarray(value, dtype=np.float32).reshape(-1)[: len(audio)]


def drive(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(3)
    if mode == 0:
        wet = process(Distortion(drive_db=rng.uniform(4.0, 38.0)), audio)
    elif mode == 1:
        wet = process(Clipping(threshold_db=rng.uniform(-24.0, -2.0)), audio)
    else:
        return effects.drive(audio, rng)
    wet = effects.tone(wet, rng)
    mix = rng.uniform(0.62, 1.0)
    return effects.match(audio * (1.0 - mix) + wet * mix, audio)


def delay(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    if mode >= 2:
        return effects.delay(audio, rng)
    if mode == 0:
        wet = process(
            Delay(
                delay_seconds=rng.uniform(0.055, 0.82),
                feedback=rng.uniform(0.12, 0.78),
                mix=rng.uniform(0.14, 0.52),
            ),
            audio,
        )
    else:
        first = process(
            Delay(
                delay_seconds=rng.uniform(0.075, 0.44),
                feedback=rng.uniform(0.18, 0.62),
                mix=rng.uniform(0.16, 0.42),
            ),
            audio,
        )
        wet = process(
            Delay(
                delay_seconds=rng.uniform(0.18, 0.74),
                feedback=rng.uniform(0.08, 0.40),
                mix=rng.uniform(0.08, 0.28),
            ),
            first,
        )
    if rng.random() < 0.55:
        wet = effects.lowpass(wet, rng.uniform(2_200.0, 9_000.0))
    return effects.match(wet, audio)


def fdn_reverb(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Schroeder-style comb bank, distinct from FreeVerb and sparse IRs."""

    base = rng.uniform(0.72, 1.35)
    delays_ms = (29.7, 37.1, 41.1, 43.7, 53.1, 61.7)
    feedback = rng.uniform(0.62, 0.88)
    impulse = np.zeros(min(len(audio), int(RATE * 4.0)), dtype=np.float32)
    impulse[0] = 1.0
    for index, milliseconds in enumerate(delays_ms):
        delay_samples = max(2, int(milliseconds * base * RATE / 1000.0))
        line_feedback = feedback * (0.97 - index * 0.018)
        for offset in range(delay_samples, len(impulse), delay_samples):
            tap = offset // delay_samples
            impulse[offset] += ((-1.0) ** (tap + index)) * line_feedback**tap
    impulse = effects.lowpass(impulse, rng.uniform(2_800.0, 10_000.0))
    impulse[0] = 1.0
    wet = fftconvolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)
    mix = rng.uniform(0.12, 0.46)
    return effects.match(audio * (1.0 - mix) + wet * mix, audio)


def reverb(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    if mode == 0:
        wet = process(
            Reverb(
                room_size=rng.uniform(0.15, 0.96),
                damping=rng.uniform(0.05, 0.92),
                wet_level=rng.uniform(0.12, 0.52),
                dry_level=rng.uniform(0.48, 0.92),
                width=rng.uniform(0.15, 1.0),
                freeze_mode=0.0,
            ),
            audio,
        )
        return effects.match(wet, audio)
    if mode == 1:
        return fdn_reverb(audio, rng)
    return effects.reverb(audio, rng)


def other(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    if mode == 0:
        return effects.match(
            process(
                Chorus(
                    rate_hz=rng.uniform(0.15, 3.8),
                    depth=rng.uniform(0.12, 0.85),
                    centre_delay_ms=rng.uniform(3.0, 18.0),
                    feedback=rng.uniform(-0.35, 0.35),
                    mix=rng.uniform(0.25, 0.75),
                ),
                audio,
            ),
            audio,
        )
    if mode == 1:
        return effects.match(
            process(
                Phaser(
                    rate_hz=rng.uniform(0.08, 4.0),
                    depth=rng.uniform(0.15, 0.95),
                    centre_frequency_hz=rng.uniform(250.0, 3_200.0),
                    feedback=rng.uniform(-0.55, 0.55),
                    mix=rng.uniform(0.25, 0.80),
                ),
                audio,
            ),
            audio,
        )
    return effects.other(audio, rng)


def render_family(audio: np.ndarray, label: str, rng: random.Random) -> np.ndarray:
    """Render one requested family for balanced personalization episodes."""

    value = effects.capture(audio, rng, before=True)
    if label == "drive":
        value = drive(value, rng)
    elif label == "delay":
        value = delay(value, rng)
    elif label == "reverb":
        value = reverb(value, rng)
    else:
        raise ValueError(f"unknown effect family: {label}")
    if rng.random() < 0.35:
        value = other(value, rng)
    if rng.random() < 0.55:
        value = effects.amp(value, rng)
    return effects.finish(effects.capture(value, rng, before=False))


def chain(audio: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    audio = effects.capture(audio, rng, before=True)
    complexity = rng.choices((0, 1, 2, 3), weights=(30, 50, 17, 3))[0]
    selected = rng.sample(list(LABELS), complexity)
    rng.shuffle(selected)
    for name in selected:
        if name == "drive":
            audio = drive(audio, rng)
        elif name == "delay":
            audio = delay(audio, rng)
        elif name == "reverb":
            audio = reverb(audio, rng)
        if rng.random() < 0.25:
            audio = effects.amp(audio, rng)
    if rng.random() < 0.35:
        audio = other(audio, rng)
    if rng.random() < 0.70:
        audio = effects.amp(audio, rng)
    audio = effects.capture(audio, rng, before=False)
    target = np.asarray(
        [float(label in selected) for label in LABELS], dtype=np.float32
    )
    return effects.finish(audio), target
