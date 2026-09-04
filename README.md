# Muspector

Muspector is a compact GPUI desktop audio inspector for studying guitar tones.
It decodes audio locally, visualizes waveform, loudness, spectrum, and timeline
data, and proposes an editable signal chain from measured signal features.

Muspector is early-stage software. Its built-in analysis is usable without a
model runtime; learned inference and audio rendering will be supplied by the
separate `muspector-models` project.

## Features

- Local WAV, AIFF, FLAC, MP3, AAC, ALAC, Ogg, and Vorbis decoding
- Waveform, RMS, LUFS, spectrum, peak, crest, and frequency statistics
- Editable Gate, Compressor, Drive, EQ, Delay, and Reverb chain
- Smooth foreground and background inspections, limited to two concurrent jobs
- Selection, sample-accurate loop playback, history, lossless working edits, and project saves
- Persistent native output streams through CoreAudio, WASAPI, or Linux CPAL hosts
- User-selectable Auto (~10 ms when supported, otherwise device default), 128,
  256, 512, 1,024, or 2,048-frame buffers

The Model and Settings buttons in the upper-right toggle their floating panels.
Click the active button again, another panel button, or anywhere outside a panel
to close it.

Playback keeps the selected output stream open between files. Decode and format
conversion run on a background worker that feeds a bounded lock-free queue; the
audio callback only drains prepared frames. Seeks and loop changes invalidate
queued frames, and loop boundaries are tracked in output frames rather than by
the UI refresh timer.

## Requirements

- Rust 1.96 or newer
- macOS, Windows, or a Vulkan-capable X11/Wayland Linux desktop

## Run

```sh
cargo dev
cargo dev -- /path/to/audio.wav
```

`cargo dev` is the repository-local alias for `cargo run`. On macOS, build,
ad-hoc sign, and open an application bundle with:

```sh
./tools/app
./tools/app /path/to/audio.wav
```

Windows ASIO support remains opt-in and requires the Steinberg ASIO SDK:

```sh
cargo build --release --features asio
```

## Verify

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

CI runs the same format, lint, test, and dependency-audit gates before starting
the platform build matrix.

Pushes to `main` build optimized Linux x86-64, Windows x86-64, and macOS arm64
executables. A successful matrix replaces the assets and force-moves the
rolling `latest` tag and release to that commit.

## Model boundary

Muspector owns only the model-neutral contract in `src/remix.rs`. It defines
named physical controls, interleaved audio geometry, confidence, and a single
`ModelRuntime` trait. Its
`infer_segment` call accepts at most 480,000 frames. Real-time rendering is
negotiated once with `configure`, receives chain updates with an explicit
smoothing interval, and processes at most 2,048 frames into caller-owned output
buffers. The runtime also reports latency and resets state after transport
discontinuities; `process_block` may not allocate, lock, or perform I/O.

Training, evaluation, weights, downloads, manifests, model-store logic, and
package publication belong in the separate `muspector-models` repository. A
future Rust dependency connects through a small adapter implementing
`ModelRuntime`; the model crate does not depend on GPUI or Muspector UI state.

Muspector intentionally has no legacy model compatibility layer and ships no
embedded model weights. Until a model adapter is connected, it uses built-in
signal analysis and heuristic chain proposals only. Model package licenses are
independent from Muspector's Apache-2.0 source license and must permit the
intended use before loading.

## Layout

- `app`: GPUI state, interaction, and rendering
- `analysis`: streaming decode and signal metrics
- `audio`: playback, device discovery, and output routing
- `chain`: editable effect-chain types and heuristic proposal
- `clip`: selection copy, delete, paste, and float-WAV export
- `project`: version-4 project metadata and atomic save coordination
- `remix`: model-neutral inference and rendering contract
- `assets`: embedded interface resources
- `theme`: visual tokens

## License

Apache-2.0. See `LICENSE`.
