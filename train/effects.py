"""On-the-fly, dependency-light effect and capture-domain randomization."""

from __future__ import annotations

import random

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

from model import LABELS, RATE


def chain(audio: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Render zero to three target families plus non-target modulation."""

    audio = capture(audio, rng, before=True)
    # Isolated effects must dominate the positive pool. A combination-heavy
    # distribution lets the network use family co-occurrence as a shortcut
    # (for example, predicting Delay whenever Drive is present) instead of
    # learning each effect's own evidence.
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
            audio = amp(audio, rng)
    # Unsupported modulation is a nuisance condition, not a catch-all label.
    # Include it both alone and beside target effects to discourage shortcuts.
    if rng.random() < 0.30:
        audio = other(audio, rng)
    if rng.random() < 0.70:
        audio = amp(audio, rng)
    audio = capture(audio, rng, before=False)
    target = np.asarray([float(label in selected) for label in LABELS], dtype=np.float32)
    return finish(audio), target


def drive(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    gain = 10.0 ** rng.uniform(0.35, 1.65)
    value = audio * gain
    mode = rng.randrange(7)
    if mode == 0:
        wet = np.tanh(value)
    elif mode == 1:
        wet = np.clip(value, -rng.uniform(0.18, 0.60), rng.uniform(0.18, 0.60))
    elif mode == 2:
        wet = np.arctan(value * rng.uniform(1.0, 3.0)) * (2.0 / np.pi)
    elif mode == 3:
        positive = np.tanh(value * rng.uniform(0.8, 1.8))
        negative = np.tanh(value * rng.uniform(1.8, 4.0))
        wet = np.where(value >= 0.0, positive, negative)
    elif mode == 4:
        wet = value / (1.0 + np.abs(value))
    elif mode == 5:
        gate = rng.uniform(0.015, 0.08)
        wet = np.sign(value) * np.sqrt(np.maximum(np.abs(value) - gate, 0.0))
        wet = np.clip(wet, -1.0, 1.0)
    else:
        limited = np.clip(value, -1.0, 1.0)
        wet = limited - limited**3 / 3.0
    wet = tone(wet, rng)
    mix = rng.uniform(0.68, 1.0)
    return match(audio * (1.0 - mix) + wet * mix, audio)


def delay(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(5)
    if mode == 0:
        milliseconds = rng.uniform(45.0, 850.0)
        feedback = rng.uniform(0.15, 0.68)
        taps = 1 if milliseconds < 100.0 else rng.randint(2, 6)
    elif mode == 1:
        milliseconds = rng.uniform(180.0, 650.0)
        feedback = rng.uniform(0.30, 0.75)
        taps = rng.randint(3, 7)
    elif mode == 2:
        milliseconds = rng.uniform(55.0, 180.0)
        feedback = rng.uniform(0.05, 0.28)
        taps = rng.randint(1, 3)
    elif mode == 3:
        milliseconds = rng.uniform(240.0, 780.0)
        feedback = rng.uniform(0.25, 0.70)
        taps = rng.randint(3, 7)
    else:
        milliseconds = rng.uniform(120.0, 520.0)
        feedback = rng.uniform(0.18, 0.60)
        taps = rng.randint(2, 5)
    wet = np.zeros_like(audio)
    base = max(1, int(milliseconds * RATE / 1000.0))
    for tap in range(1, taps + 1):
        wobble = int(base * rng.uniform(-0.025, 0.025)) if mode in {1, 3} else 0
        offset = max(1, base * tap + wobble)
        if offset >= len(audio):
            break
        value = audio[:-offset] * feedback ** (tap - 1)
        if mode in {1, 3}:
            cutoff = rng.uniform(1_800.0, 7_000.0) / (1.0 + tap * 0.12)
            value = lowpass(value, cutoff)
        wet[offset:] += value
    mix = rng.uniform(0.16, 0.52)
    return match(audio + wet * mix, audio)


def reverb(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(6)
    decay = (0.35, 0.8, 1.4, 2.4, 3.6, 1.8)[mode] * rng.uniform(0.75, 1.35)
    length = min(len(audio), max(2_048, int(decay * RATE)))
    time = np.arange(length, dtype=np.float32) / RATE
    envelope = np.exp(-time * rng.uniform(3.5, 7.5) / max(decay, 0.1))
    density = (0.012, 0.025, 0.045, 0.08, 0.12, 0.035)[mode]
    generator = np.random.default_rng(rng.getrandbits(64))
    impulses = (generator.random(length) < density).astype(np.float32)
    signs = generator.uniform(-1.0, 1.0, length).astype(np.float32)
    impulse = impulses * signs * envelope
    predelay = int(rng.uniform(0.0, 0.08) * RATE)
    impulse[: min(predelay, length)] = 0.0
    impulse[0] = 1.0
    if mode in {1, 2, 3, 4}:
        impulse = lowpass(impulse, rng.uniform(2_500.0, 10_000.0))
    if mode == 5:
        impulse *= 0.65 + 0.35 * np.sin(2.0 * np.pi * rng.uniform(20.0, 55.0) * time)
    wet = fftconvolve(audio, impulse, mode="full")[: len(audio)]
    mix = rng.uniform(0.12, 0.48)
    return match(audio * (1.0 - mix) + wet * mix, audio)


def other(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(7)
    time = np.arange(len(audio), dtype=np.float32) / RATE
    if mode == 0:  # chorus
        wet = modulated_delay(audio, rng.uniform(12.0, 28.0), rng.uniform(0.2, 1.8), 5.0, time)
        return match(audio * 0.72 + wet * 0.45, audio)
    if mode == 1:  # flanger
        wet = modulated_delay(audio, rng.uniform(0.6, 4.5), rng.uniform(0.08, 0.9), 2.5, time)
        return match(audio + wet * rng.uniform(0.45, 0.85), audio)
    if mode == 2:  # phaser
        wet = audio.copy()
        for _ in range(rng.randint(3, 6)):
            coefficient = rng.uniform(0.35, 0.82)
            wet = lfilter([coefficient, -1.0], [1.0, -coefficient], wet).astype(np.float32)
        return match(audio * 0.65 + wet * 0.55, audio)
    if mode == 3:  # wah / resonant filter
        center = rng.uniform(450.0, 2_400.0)
        low = max(80.0, center / rng.uniform(1.5, 2.5))
        high = min(12_000.0, center * rng.uniform(1.5, 2.8))
        sos = butter(2, [low, high], btype="bandpass", fs=RATE, output="sos")
        wet = sosfilt(sos, audio).astype(np.float32) * rng.uniform(1.2, 2.8)
        return match(audio * 0.35 + wet, audio)
    if mode == 4:  # tremolo
        depth = rng.uniform(0.35, 0.85)
        lfo = 1.0 - depth / 2.0 + depth / 2.0 * np.sin(
            2.0 * np.pi * rng.uniform(2.0, 10.0) * time + rng.uniform(0.0, 6.28)
        )
        return finish(audio * lfo)
    if mode == 5:  # vibrato
        return match(
            modulated_delay(audio, rng.uniform(2.0, 7.0), rng.uniform(2.0, 7.0), 3.0, time),
            audio,
        )
    wet = modulated_delay(audio, rng.uniform(4.0, 12.0), rng.uniform(0.5, 3.0), 4.0, time)
    lfo = 0.75 + 0.25 * np.sin(2.0 * np.pi * rng.uniform(0.8, 5.0) * time)
    return match(audio * 0.55 + wet * lfo * 0.60, audio)


def capture(audio: np.ndarray, rng: random.Random, before: bool) -> np.ndarray:
    value = audio.astype(np.float32, copy=True)
    value *= 10.0 ** rng.uniform(-0.65, 0.35)
    if rng.random() < 0.35:
        # A real clean chain can be compressed or boosted without becoming a
        # target Drive effect. Random envelope compression prevents sustain
        # and crest factor from becoming drive shortcuts.
        attack = rng.uniform(0.003, 0.030)
        coefficient = np.exp(-1.0 / (attack * RATE))
        envelope = lfilter(
            [1.0 - coefficient], [1.0, -coefficient], np.abs(value)
        ).astype(np.float32)
        threshold = 10.0 ** (rng.uniform(-30.0, -12.0) / 20.0)
        ratio = rng.uniform(1.5, 6.0)
        compressed = threshold + np.maximum(envelope - threshold, 0.0) / ratio
        gain = np.minimum(1.0, compressed / np.maximum(envelope, 1.0e-5))
        value = match(value * gain, value)
    if rng.random() < 0.60:
        value = tone(value, rng)
    if rng.random() < 0.28:
        low = rng.uniform(25.0, 160.0)
        high = rng.uniform(5_500.0, 18_000.0)
        sos = butter(2, [low, min(high, RATE / 2.0 - 100.0)], btype="bandpass", fs=RATE, output="sos")
        value = sosfilt(sos, value).astype(np.float32)
    if rng.random() < 0.35:
        generator = np.random.default_rng(rng.getrandbits(64))
        white = generator.normal(0.0, 1.0, len(value)).astype(np.float32)
        if rng.random() < 0.45:
            white = lfilter([0.08], [1.0, -0.92], white).astype(np.float32)
        value += white * 10.0 ** rng.uniform(-5.2, -3.3)
    if rng.random() < 0.20:
        frequency = rng.choice((50.0, 60.0, 100.0, 120.0))
        time = np.arange(len(value), dtype=np.float32) / RATE
        value += np.sin(2.0 * np.pi * frequency * time + rng.uniform(0.0, 6.28)) * 10.0 ** rng.uniform(-4.8, -3.2)
    if rng.random() < 0.35:
        # Short early reflections and cabinet combing are capture conditions,
        # not a Reverb/Delay target. Keep every reflection below 80 ms so the
        # positive families still own audible repeats and late decay.
        impulse = np.zeros(int(RATE * 0.080) + 1, dtype=np.float32)
        impulse[0] = 1.0
        for _ in range(rng.randint(2, 8)):
            offset = rng.randint(int(RATE * 0.003), len(impulse) - 1)
            impulse[offset] += rng.uniform(-0.20, 0.25)
        value = match(fftconvolve(value, impulse, mode="full")[: len(value)], value)
    if rng.random() < 0.15:
        value += rng.uniform(-0.008, 0.008)
    if rng.random() < 0.18:
        ceiling = rng.uniform(0.65, 1.4) if before else rng.uniform(0.45, 1.0)
        value = np.clip(value, -ceiling, ceiling)
    if rng.random() < 0.12:
        factor = rng.choice((2, 3, 4))
        value = resample_poly(resample_poly(value, 1, factor), factor, 1)[: len(value)].astype(np.float32)
        if len(value) < len(audio):
            value = np.pad(value, (0, len(audio) - len(value)))
    return finish(value)


def amp(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild amp/cab coloration: deliberately a zero-label nuisance."""

    gain = rng.uniform(0.9, 3.4)
    colored = np.tanh(audio * gain) / max(np.tanh(gain), 1.0e-4)
    colored = lowpass(colored, rng.uniform(4_500.0, 13_000.0))
    if rng.random() < 0.55:
        sos = butter(1, rng.uniform(45.0, 140.0), btype="highpass", fs=RATE, output="sos")
        colored = sosfilt(sos, colored).astype(np.float32)
    mix = rng.uniform(0.35, 0.85)
    return match(audio * (1.0 - mix) + colored * mix, audio)


def tone(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    value = audio
    if rng.random() < 0.70:
        cutoff = rng.uniform(2_500.0, 15_000.0)
        low = lowpass(value, cutoff)
        amount = rng.uniform(-0.35, 0.65)
        value = value + amount * (low - value)
    if rng.random() < 0.45:
        cutoff = rng.uniform(70.0, 650.0)
        low = lowpass(value, cutoff)
        value = value + rng.uniform(-0.45, 0.45) * low
    return finish(value)


def lowpass(audio: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(2, min(cutoff, RATE / 2.0 - 100.0), btype="lowpass", fs=RATE, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def modulated_delay(
    audio: np.ndarray,
    base_ms: float,
    depth_ms: float,
    rate: float,
    time: np.ndarray,
) -> np.ndarray:
    delay = (base_ms + depth_ms * np.sin(2.0 * np.pi * rate * time)) * RATE / 1000.0
    source = np.arange(len(audio), dtype=np.float32) - delay.astype(np.float32)
    return np.interp(source, np.arange(len(audio), dtype=np.float32), audio, left=0.0).astype(np.float32)


def match(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source = float(np.sqrt(np.mean(reference * reference) + 1.0e-9))
    target = float(np.sqrt(np.mean(value * value) + 1.0e-9))
    return finish(value * min(4.0, source / max(target, 1.0e-5)))


def finish(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(audio, copy=False).clip(-4.0, 4.0).astype(np.float32, copy=False)
