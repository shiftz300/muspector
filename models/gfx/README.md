# GFX Classifier models

`fx.onnx` and `settings.onnx` are ONNX conversions of the official
`20201024_fxnet_mono_cont_best` and `20201020_setnetcond_mono_cont_best`
PyTorch releases. The network weights are unchanged; only their container
format is different.

Upstream code and models:

- https://github.com/mcomunita/gfx-classifier
- https://github.com/mcomunita/gfx_classifier_models_and_results

The models classify 13 overdrive, distortion, and fuzz units from a wet guitar
recording. They do not identify arbitrary effects or infer a complete chain.
The model input is a two-second, 22.05 kHz, 128-band power Mel spectrogram.

Checked-in SHA-256 values:

- `fx.onnx`: `478d1462ff70afa8b6a5afdeb56b1197651e4cc79e4e776c799f38f874106505`
- `settings.onnx`: `332b9ee328c507874cdeb8d162c024524d21af656492b90c6dfec9c8670232ba`

Recreate the checked-in files with `tools/gfx.py`; its Python dependencies are
conversion-time tools and are not required by Muspector at runtime.
