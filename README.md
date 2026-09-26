# Muspector

> Development paused / local project archived (2026-09-27). The model-training
> snapshot is in the sibling `muspector-models` repository at
> `ARCHIVE_STATUS_20260927.md`. No complete Wet-to-Clean model was promoted;
> `usable_model` remains null.

Muspector is a compact GPUI desktop audio inspector for studying guitar tones.
It decodes audio locally, visualizes waveform, loudness, spectrum, and timeline
data, and proposes an editable signal chain from measured signal features.

Muspector is early-stage software. Its analysis combines signal measurements
with an embedded GFX Classifier pass for Drive-family subtype and knob
estimation. General learned inference and trained restoration remain in the
separate `muspector-models` project; this app includes a development graybox
renderer so chain edits can already be auditioned end to end.

## Features

- Local WAV, AIFF, FLAC, MP3, AAC, ALAC, Ogg, and Vorbis decoding
- Waveform, RMS, LUFS, spectrum, peak, crest, and frequency statistics
- GFX blind classification for 13 overdrive, distortion, and fuzz units
- Editable Gate, Compressor, Drive, EQ, Delay, and Reverb chain
- Full-file Wet-to-Clean preview after reorder, bypass, and control changes
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

Chain changes render from the current source into a disposable float WAV on a
background worker. The player switches to the newest accepted preview at the
same position; stale jobs are cancelled and cannot replace newer edits. The
graybox executor traverses the displayed forward chain in reverse and currently
handles Drive, Compressor, EQ, and Delay. Gate is not claimed recoverable after
signal removal, and Reverb remains frozen rather than receiving a fake inverse.

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

General training, learned restoration, evaluation, downloads, manifests, model-store logic, and
package publication belong in the separate `muspector-models` repository. A
future Rust dependency connects its remix runtime through a small adapter
implementing `ModelRuntime`; the model crate does not depend on GPUI or
application UI state. The specialized GFX analyzer is already connected through
`tract-onnx`. It uses up to five high-energy two-second windows and reproduces
the model's 22.05 kHz, 128-band power-Mel input. Its closed-set result refines
the generic nonlinear family into Overdrive, Distortion, or Fuzz and estimates
the supported controls. It does not detect effect absence, arbitrary pedals,
order, or a complete chain; non-Drive effects remain heuristic until their
adapters are connected. Isolated wet guitar is the intended input domain.

The converted GFX weights are BSD-3-Clause and retain their notice and upstream
links in `models/gfx`. Other model package licenses are independent from
Muspector's Apache-2.0 source license and must permit the intended use before
loading.

### Live remix interaction

The existing effect cards are the live control surface; remixing must not open
a second editor. Dragging changes the displayed forward signal order, while the
runtime remains responsible for applying inverse stages in the order required
by the complete graph. Individual inverse models remain order-agnostic.

Every accepted reorder or bypass starts a fresh preview; control changes use a
short debounce so a rapid gesture does not launch one render per tick. Preview
jobs receive the complete chain and the executor alone chooses reverse graph
order. A future learned `ModelRuntime` adapter can replace individual graybox
processors without changing this UI contract. Unsupported stages pass through;
the app does not manufacture a recovered signal for an irreversible family.

Analysis and offline-render progress belongs directly on the waveform. A scan
advances from left to right across the affected range, using measured job
progress rather than a decorative timer. The selected range keeps its normal
highlight; during a scan only the unscanned portion is dimmed, and the cursor
reveals the highlighted waveform as it advances. The scan cursor is a single
solid line without glow;
its percentage follows directly above the cursor rather than occupying a fixed
corner badge. Selected-range Rescan and current full-file preview rendering use
the same interaction and smoothing.

## Layout

- `app`: GPUI state, interaction, and rendering
- `analysis`: streaming decode and signal metrics
- `audio`: playback, device discovery, and output routing
- `chain`: editable effect-chain types and heuristic proposal
- `clip`: selection copy, delete, paste, float-WAV export, and streaming preview I/O
- `gfx`: bounded GFX preprocessing, ONNX inference, subtype, and knob mapping
- `models/gfx`: attributed BSD-3-Clause inference weights
- `project`: version-4 project metadata and atomic save coordination
- `render`: reverse-order development graybox preview processors
- `remix`: model-neutral inference and rendering contract
- `assets`: embedded interface resources
- `theme`: visual tokens

## License

Apache-2.0. See `LICENSE`.
