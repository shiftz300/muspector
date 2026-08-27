#!/usr/bin/env python3
"""Convert the released GFX Classifier PyTorch models to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as functional


class Settings(torch.nn.Module):
    """Run the released settings layers with their original 128 x 87 shape."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        condition = self.model.emb(label)
        condition = self.model.fc0(condition).reshape(-1, 1, 128, 87)
        value = torch.cat((audio, condition), dim=1)
        value = functional.max_pool2d(functional.relu(self.model.conv1(value)), 2, 2)
        value = functional.max_pool2d(functional.relu(self.model.conv2(value)), 2, 2)
        value = value.reshape(-1, 12 * 29 * 18)
        value = functional.relu(self.model.fc1(value))
        value = functional.relu(self.model.fc2(value))
        return functional.tanh(self.model.out(value))


def export(source: Path, code: Path, output: Path) -> None:
    sys.path.insert(0, str(code))
    model = torch.load(source, map_location="cpu", weights_only=False)
    model.eval()
    audio = torch.randn(1, 1, 128, 87)

    if hasattr(model, "n_classes"):
        inputs = (audio,)
        names = ["audio"]
        output_name = "logits"
    else:
        model = Settings(model)
        model.eval()
        inputs = (audio, torch.tensor([0], dtype=torch.int64))
        names = ["audio", "label"]
        output_name = "settings"

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        output,
        input_names=names,
        output_names=[output_name],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))

    with torch.no_grad():
        expected = model(*inputs).numpy()
    feed = {name: value.numpy() for name, value in zip(names, inputs)}
    actual = ort.InferenceSession(output).run([output_name], feed)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("code", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.code, args.output)


if __name__ == "__main__":
    main()
