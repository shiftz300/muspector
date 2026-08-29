#!/usr/bin/env python3
"""Export the frozen AFx-Rep encoder and global pedal heads as one ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from afx_identity_teacher import CHECKPOINT_SHA256, WINDOW, audio_segment, load_teacher, sha256
from compact_identity_verifier import CATALOG, CatalogHead, Verifier
from layout import PEDAL_IDENTITY_RUN


class Runtime(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        catalog: CatalogHead,
        verifier: Verifier,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.catalog = catalog
        self.verifier = verifier

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = waveform / waveform.abs().amax(dim=(1, 2), keepdim=True).clamp_min(
            1.0e-8
        )
        embedding, _ = self.encoder(waveform)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return self.catalog(embedding), self.verifier(embedding)


def load_catalog(path: Path) -> CatalogHead:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = CatalogHead(state["mean"][0].numpy(), state["deviation"][0].numpy())
    model.load_state_dict(state)
    return model.eval()


def load_verifier(path: Path) -> Verifier:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Verifier(state["mean"][0].numpy(), state["deviation"][0].numpy())
    model.load_state_dict(state)
    return model.eval()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/private/tmp/muspector-st-ito"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/private/tmp/muspector-afx-rep.ckpt")
    )
    parser.add_argument(
        "--catalog-run", type=Path, default=PEDAL_IDENTITY_RUN / "afx-rep-catalog"
    )
    parser.add_argument(
        "--output", type=Path, default=PEDAL_IDENTITY_RUN / "afx-rep-runtime"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path.home() / "Downloads/test/rat(mid gain).wav"
    )
    args = parser.parse_args()

    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("AFx-Rep checkpoint hash does not match the audited official file")
    encoder = load_teacher(args.source, args.checkpoint)
    catalog = load_catalog(args.catalog_run / "compact-drive-catalog.pt")
    verifier = load_verifier(args.catalog_run / "compact-drive-verifier.pt")
    runtime = Runtime(encoder, catalog, verifier).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "afx-pedal-identity.onnx"
    dummy = torch.zeros(1, 1, WINDOW)
    started = time.perf_counter()
    torch.onnx.export(
        runtime,
        dummy,
        model_path,
        input_names=["waveform_48khz"],
        output_names=["identity_logits", "known_logit"],
        opset_version=17,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - started

    audio = audio_segment(args.benchmark, 1, 3)[None, None]
    with torch.inference_mode():
        expected = runtime(torch.from_numpy(audio))
    session_started = time.perf_counter()
    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    session_seconds = time.perf_counter() - session_started
    session.run(None, {"waveform_48khz": audio})
    started = time.perf_counter()
    repetitions = 3
    actual = None
    for _ in range(repetitions):
        actual = session.run(None, {"waveform_48khz": audio})
    inference_seconds = (time.perf_counter() - started) / repetitions
    parity = max(
        float(
            np.max(
                np.abs(expected[index].detach().numpy() - actual[index])
            )
        )
        for index in range(2)
    )
    payload = {
        "artifact": str(model_path.resolve()),
        "sha256": digest(model_path),
        "bytes": model_path.stat().st_size,
        "input": {"rate": 48_000, "seconds": 5, "shape": [1, 1, WINDOW]},
        "outputs": {"catalog": list(CATALOG), "knownness": "single logit"},
        "user_gradient_updates": 0,
        "export_seconds": export_seconds,
        "onnxruntime_session_seconds": session_seconds,
        "onnxruntime_seconds_per_window": inference_seconds,
        "onnxruntime_max_absolute_difference": parity,
        "noncommercial_release_eligible": True,
        "commercial_release_eligible": False,
        "release_reason": (
            "the fitted heads use CC BY-NC 4.0 ToneTwist and Zenodo cc-nc RemFX "
            "recordings; redistribute only as the separately attributed "
            "non-commercial model component"
        ),
    }
    (args.output / "export.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    calibration = json.loads(
        (args.catalog_run / "calibration.json").read_text()
    )
    package = {
        "schema": 1,
        "labels": list(CATALOG),
        "display_labels": [
            "Blues Driver",
            "RAT",
            "Tube Screamer",
            "Big Muff",
            "Metal Muff",
            "Fuzzy Logic",
            "Silly Fuzz",
        ],
        "threshold": calibration["threshold"],
        "model_sha256": payload["sha256"],
        "license_scope": "non-commercial research",
        "user_gradient_updates": 0,
    }
    (args.output / "afx-pedal-identity.json").write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
