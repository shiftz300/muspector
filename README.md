# Muspector

Muspector is a compact GPUI audio inspector for studying guitar tones. The
current MVP performs local analysis, uses GFX Classifier to identify a likely
drive/fuzz unit and its controls, and proposes an editable hybrid effect chain.
Audio playback is intentionally deferred.

## Requirements

- macOS
- Rust 1.96 or newer

## Run

```sh
cargo dev
```

To inspect a file immediately at startup:

```sh
cargo dev -- /path/to/audio.mp3
```

`cargo dev` is a project-local alias for `cargo run`, defined in
`.cargo/config.toml`. Cargo does not provide a built-in `dev` command for native
applications.

Drop an audio file onto the Inspect panel or choose one with Open. Supported
containers and codecs include WAV/AIFF PCM, FLAC, MP3, AAC/ALAC in MP4, and Ogg
Vorbis.

## Test

Run the automated checks:

```sh
cargo fmt --all -- --check
cargo check
cargo test
cargo clippy --all-targets -- -D warnings
```

For a manual UI check, run `cargo dev` and verify that:

1. Inspect and Remix remain the same size when selected or hovered.
2. Tab changes and hover states fade smoothly in both directions.
3. Open launches the file picker and its hover state fades smoothly.
4. Dropping or opening a supported audio file displays an analysis report.
   The window should grow once to fit the complete report when the display has
   enough room.
5. Dropping another file works from every page, including Remix and results.
6. Opening an unsupported or invalid file shows a temporary toast without
   replacing the current page or locking the tabs.
7. Moving across Profile shows the corresponding time, peak, and RMS values.
8. Chain opens Remix with the inferred active effects in signal-flow order.
9. Effect switches, plus/minus controls, and scrolling over a knob update its
   parameters. Clicking a value allows direct numeric entry; Enter commits,
   Escape cancels, and Reset restores the inferred values.
10. Chain and effect cards fade without changing size. The window cannot shrink
    below the safe editor width, and overflowing pages show a vertical position
    indicator without horizontal reflow.

## Analysis

- format, sample rate, channels, and duration
- peak, RMS, crest factor, and near-full-scale sample count
- an interactive waveform and short-term RMS profile
- a 64-band logarithmic spectrum with low/mid/high energy, spectral centroid,
  and 85% rolloff markers
- bounded envelope correlation for delay and diffuse-tail evidence
- GFX blind classification for 13 drive/fuzz units and model-estimated Level,
  Gain/Drive/Distortion, and Tone/Filter controls
- heuristic Gate, Comp, EQ, Delay, and Reverb candidates with editable controls

Decoding runs on GPUI's background executor. Spectral analysis is streaming and
keeps only a 4096-sample FFT window plus accumulated bins in memory. Profile
analysis uses bounded streaming buckets and retains at most 192 display points.
The blind pass selects up to five high-energy two-second windows, converts them
to the model's original 22.05 kHz / 128-band power-Mel input, and averages their
predictions. Python is not required at runtime.

GFX was trained on isolated wet guitar and only covers overdrive, distortion,
and fuzz. A result from a full mix or a different effect family is out of
distribution. It does not recover a complete chain or its order, so the UI
labels the current result as GFX blind plus heuristic analysis. Paired
render-and-compare analysis is planned around StemFX after the blind workflow.

## Models and licenses

- GFX Classifier code and model weights: BSD-3-Clause. The converted weights,
  attribution, and upstream links are in `models/gfx`.
- `tract-onnx`: MIT or Apache-2.0; runs the checked-in ONNX models in Rust.
- `rubato`: MIT or Apache-2.0; performs fixed-ratio high-quality resampling.

`tools/gfx.py` reproduces the ONNX conversion from the official PyTorch files.
Its Python packages are conversion-only dependencies.

## Layout

- `app`: GPUI state and interface
- `analysis`: decoding and signal metrics
- `blind`: GFX preprocessing, inference, and pedal control mapping
- `chain`: effect inference and parameter models
- `models`: attributed inference weights
- `assets`: embedded interface icons
- `theme`: compact visual tokens
