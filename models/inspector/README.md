# Muspector routed Inspector research candidate

Active artifacts use semantic names describing their role and training domain:

- `drive-delay-encoder.onnx`: compact 700,659-parameter ResNet18 encoder from
  the public-effects run.
- `drive-delay-head.onnx`: 348,931-parameter clean-relative pair head; only
  Drive and Delay outputs are used.
- `reverb-encoder.onnx`: release-compatible 700,659-parameter public-RIR
  encoder.
- `reverb-head.onnx`: release-compatible 348,931-parameter clean-relative pair
  head; only Reverb is used.
- `reverb-verifier.onnx`: 20,153-parameter temporal verifier which accepts
  8x216 pooled Mel bands and all three calibrated Reverb-branch pair
  probabilities. Reverb is active only when both the pair head and verifier
  pass their thresholds.
- `routed-device-profile.bin`: 1,036 little-endian float32 values containing
  Drive/Delay and Reverb mean/deviation statistics for the development Clean.
- `afx-pedal-identity.onnx`: 323.8 MB frozen AFx-Rep encoder plus seven-pedal
  catalog and open-set verifier, embedded only by the `embedded-identity`
  feature and distributed under the separate non-commercial notice.
- `afx-pedal-identity.json`: labels, calibrated threshold, model hash, and
  runtime/license contract for the embedded identity model.

Both branches use 44.1 kHz five-second windows, a 2048-point FFT, a 1024-sample
STFT hop, and 128 normalized log-Mel bins. Importing Clean Audio uses the first
Drive/Delay window and the first three overlapping Reverb windows, covering ten
seconds. The performance does not need to align with the inspected audio and
the user device performs zero gradient updates.

The `.musp-training` schema-4 format stores a name, two 518-float profiles, and
three routed thresholds. It contains no model weights, source audio, or
wet-labelled adapter. Older single-profile schemas are rejected; import the
original Clean Audio again to create a routed bundle.

## Current routed evidence

`train/runs/inspector-routed-public/metrics.json` reports:

- internal Drive/Delay/Reverb recall `99.42% / 85.93% / 99.80%`;
- conservative internal routed Clean FP upper bound `1.02%`;
- labelled hardware-development precision/recall `1.00/1.00`, `1.00/1.00`,
  and `.50/1.00` before the new Reverb verifier;
- zero routed Clean FP across strict leave-one-recording-out evaluation of six
  pickup/tone-state recordings at two windows per recording.

The bundled `reverb-verifier.onnx` comes from
`train/runs/reverb-verifier-hardware-replay`. It uses public candidate replay
plus the labelled hardware directory for development fitting. Its development
Reverb precision/recall is `1.00/1.00`; public test Reverb
precision/recall is `99.76%/99.32%`, and the six-recording Clean audit has zero
complete-gate false positives. The Rust runtime reproduces the complete gate:
the current 15-file hardware development fixture is `15/15` exact, including
all four former Reverb false positives removed. These recordings participated
in fitting, so this is a regression result, not a generalization claim. The
verifier passes its development gate; its release gate remains blocked until a
fully untouched device-disjoint labelled hardware set is evaluated.

Historical numbered experiment directories remain only as immutable audit
records. Active commands import semantic paths from `train/layout.py` and must
not create new sequence-number run names.

## SHA-256

- `drive-delay-encoder.onnx`: `cb77d22bf808b52bc85058c3a0bfdf4d51ade29144fd5c57f309bd0c6eeb040d`
- `drive-delay-head.onnx`: `77993b65940fa9ecc256aee7f7c924d56974ff5fc10a98becdca1cf904cbc78d`
- `reverb-encoder.onnx`: `b58bfbe3b8009cd1606534f89a023a7b3f95ba0cd1535269e590d86853d4bd76`
- `reverb-head.onnx`: `ecff1b524efc8b0c66b23e4ff7e9446a1d374877fbd905fa822b25f3ed3c417f`
- `routed-device-profile.bin`: `fc753c62c1c939fa63b34728ed442987d2f67e5129c37c06fdb4d4acec5f5ac4`
- bundled development Reverb verifier ONNX:
  `127a0180f081316fe640502feb7212de6867701e8ee1ee6216e0251fb0df5e80`
- `afx-pedal-identity.onnx`:
  `977c0e4a2f0ca4a61cf899df0c7a3fd03da41060db78a5aaee20ca7be32d8403`

The model recognizes effect families, not effect order, exact settings, or
Clean-audio reconstruction. See `LICENSES.md` before redistribution.
