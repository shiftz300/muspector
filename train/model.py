"""Fixed frontend and independently implemented compact blind Audio ResNet18."""

from __future__ import annotations

import math
from typing import Optional

import torch


RATE = 44_100
SECONDS = 5
SAMPLES = RATE * SECONDS
FFT = 2_048
HOP = 1_024
MELS = 128
FRAMES = SAMPLES // HOP + 1
FMIN = 30.0
FMAX = 16_000.0
LABELS = ("drive", "delay", "reverb")
EMBEDDING = 256


def hz_to_mel(frequency: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def mel_to_hz(value: float) -> float:
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def mel_filters(device: Optional[torch.device] = None) -> torch.Tensor:
    bins = FFT // 2 + 1
    lower = hz_to_mel(FMIN)
    upper = hz_to_mel(FMAX)
    points = torch.linspace(lower, upper, MELS + 2, device=device)
    frequencies = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    fft_frequencies = torch.linspace(0.0, RATE / 2.0, bins, device=device)
    filters = torch.zeros(MELS, bins, device=device)
    for band in range(MELS):
        left, center, right = frequencies[band : band + 3]
        rise = (fft_frequencies - left) / (center - left)
        fall = (right - fft_frequencies) / (right - center)
        filters[band] = torch.minimum(rise, fall).clamp_min(0.0)
        filters[band] *= 2.0 / (right - left)
    return filters


@torch.no_grad()
def frontend(waveform: torch.Tensor) -> torch.Tensor:
    """Convert `[batch, 220500]` waveforms to normalized `[B,1,128,216]`."""

    if waveform.ndim != 2 or waveform.shape[1] != SAMPLES:
        raise ValueError(f"expected [batch, {SAMPLES}], got {tuple(waveform.shape)}")
    window = torch.hann_window(FFT, periodic=True, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft=FFT,
        hop_length=HOP,
        win_length=FFT,
        window=window,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spectrum.abs().square()
    mel = torch.matmul(mel_filters(waveform.device), power).clamp_min(1.0e-10)
    mel = 10.0 * torch.log10(mel)
    peak = mel.amax(dim=(1, 2), keepdim=True)
    mel = torch.maximum(mel, peak - 80.0)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    deviation = mel.std(dim=(1, 2), keepdim=True).clamp_min(1.0e-5)
    return ((mel - mean) / deviation).unsqueeze(1)


def augment_features(value: torch.Tensor) -> torch.Tensor:
    """Apply deliberately light SpecAugment without erasing effect tails."""

    if torch.rand(()) >= 0.30:
        return value
    result = value.clone()
    for batch in range(result.shape[0]):
        width = int(torch.randint(0, 9, ()).item())
        if width:
            start = int(torch.randint(0, MELS - width + 1, ()).item())
            result[batch, :, start : start + width, :] = 0.0
        width = int(torch.randint(0, 11, ()).item())
        if width:
            start = int(torch.randint(0, FRAMES - width + 1, ()).item())
            result[batch, :, :, start : start + width] = 0.0
    return result


class Block(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            inputs, outputs, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(outputs)
        self.conv2 = torch.nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(outputs)
        self.skip = (
            torch.nn.Identity()
            if stride == 1 and inputs == outputs
            else torch.nn.Sequential(
                torch.nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(outputs),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = torch.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return torch.relu(value + residual)


class Detector(torch.nn.Module):
    def __init__(self, stem_stride: int = 1) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=stem_stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.layers = torch.nn.Sequential(
            Block(16, 16),
            Block(16, 16),
            Block(16, 32, 2),
            Block(32, 32),
            Block(32, 64, 2),
            Block(64, 64),
            Block(64, 128, 2),
            Block(128, 128),
        )
        self.dropout = torch.nn.Dropout(0.20)
        self.head = torch.nn.Linear(EMBEDDING, len(LABELS))

    def spatial(self, value: torch.Tensor) -> torch.Tensor:
        """Return the final time-frequency feature map before global pooling."""

        return self.layers(self.stem(value))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Return the stable clip embedding shared by blind and reference modes."""

        value = self.spatial(value)
        average = value.mean(dim=(2, 3))
        maximum = value.amax(dim=(2, 3))
        return torch.cat((average, maximum), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(value)))


def parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
