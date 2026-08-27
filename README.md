# Muspector

Muspector is a compact GPUI audio inspector for studying guitar tones. The
current MVP performs local analysis, uses GFX Classifier to identify a likely
drive/fuzz unit and its controls, and proposes an editable hybrid effect chain.
The landscape workspace combines an RX-style analysis canvas with an
Ableton-inspired horizontal device chain. A lightweight local transport previews
the original audio without applying the proposed chain.

## Requirements

- Rust 1.96 or newer
- macOS, Windows, or a Vulkan-capable X11/Wayland Linux desktop

The CI artifacts currently target Apple silicon macOS, x86-64 Windows, and
x86-64 Linux.

## Run

```sh
cargo dev
```

To inspect a file immediately at startup:

```sh
cargo dev -- /path/to/audio.mp3
```

To launch the macOS app bundle with the Muspector Dock icon:

```sh
./tools/app /path/to/audio.mp3
```

`cargo dev` is a project-local alias for `cargo run`, defined in
`.cargo/config.toml`. Cargo does not provide a built-in `dev` command for native
applications.

Drop an audio file onto the workspace or choose one with Open. Supported
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

## CI executables

The `Build` workflow compiles native optimized executables on three GitHub-hosted
runners and uploads each result as a single, unwrapped artifact:

- `muspector-linux-x86_64`
- `muspector-windows-x86_64.exe`
- `muspector-macos-arm64`

Release builds use `opt-level=3`, fat LTO, one codegen unit, aborted panics, and
symbol stripping to keep model inference and visualization hot paths fast while
removing avoidable binary overhead. All GPUI assets and the application PNG are
compiled into the executable. Windows also embeds the icon and product metadata
in its native PE resources. macOS artifacts are ad-hoc signed; downloaded Linux
and macOS artifacts may need their executable permission restored with
`chmod +x` depending on the download client.

Successful `main` builds also update the rolling `latest` tag and Release. Its
three stable asset names are replaced on every successful run, so the Release
always points to the executables built from the current `main` commit.

For a manual UI check, run `cargo dev` and verify that:

1. Open launches the file picker and its hover state fades smoothly.
2. Dropping or opening a supported audio file displays a Profile-first analysis
   canvas beneath a full-file Overview strip, with Spectrum and a dense
   Inspector rail on the right. The window grows once to the safe landscape
   workspace size when the display has enough room.
3. The RX-style analysis workspace and editable Signal Chain remain visible
   together without tabs or page transitions.
4. Dropping another file works from the loaded workspace and replaces the
   previous analysis only after the new inspection succeeds.
5. Opening an unsupported or invalid file shows a temporary toast without
   replacing the current workspace or locking its controls.
6. Moving across the Workspace shows the corresponding time, peak, and RMS
   values. Dragging across it selects a time range; the Inspector switches Peak,
   RMS, and Crest to approximate metrics for that selected range.
7. A trackpad pinch zooms continuously around the gesture center. Command/Control
   + wheel provides the same anchored zoom, an ordinary two-finger scroll or
   wheel pans a zoomed view, and the middle mouse button drags the Workspace.
   The Overview viewport pans from its center and its `<` / `>` edges resize the
   visible range. Alt + wheel changes the vertical waveform scale.
   Command/Control + `+`, `-`, and `0` provide keyboard zoom controls.
8. The Workspace Play button and Space pause or resume the original audio. A
   click fixes the pink playhead at that time; dragging creates a selection, and
   the adjacent Loop button repeats that selected range.
9. The bottom Signal Chain is horizontally scrollable. Dragging anywhere on a
   device, including its expanded details, moves the complete card, leaves a
   muted placeholder, and reorders the surrounding devices as they make room.
   Reset restores the inferred order.
10. The compact pink or gray status dot and disclosure control share the top row
   of each narrow device label. Number, effect type, and confidence use a tight
   horizontal hierarchy; only a distinct detected model uses a rotated vertical
   label, with long names wrapped into adjacent columns. Expanding or collapsing
   full-width parameter controls is smooth.
11. Plus/minus controls and scrolling over a knob update its parameters.
   Clicking a value allows direct numeric entry; Enter commits, Escape cancels,
   and Reset restores the inferred values. Each knob shows the detected value
   as a muted scale marker; double-clicking the knob restores only that
   parameter to the marker.
12. Effect cards fade without changing title geometry when enabled or disabled.
13. The window cannot shrink below the safe editor width, and the device rack
    keeps readable parameter values while its horizontal indicator follows the
    visible position.
14. The upper-right CPU and RAM pressure strips update without interrupting the
    editor; inactive segments are gray and active pressure progresses through
    green, amber, and red.
15. The right-side History menu lists document-local chain changes. Clicking an
    entry jumps to that state; Command/Control + Z undoes and
    Command/Control + Shift + Z redoes. Consecutive knob adjustments and one
    card drag are each grouped into a single history step.

## Analysis

- format, sample rate, channels, and duration
- peak, RMS, integrated LUFS, crest factor, and near-full-scale sample count
- an interactive waveform with a fixed right-hand dBFS ruler, a subtle bottom
  baseline, short-term RMS, and
  momentary LUFS profiles
- a 64-band logarithmic spectrum with low/mid/high energy, spectral centroid,
  and 85% rolloff markers
- bounded envelope correlation for delay and diffuse-tail evidence
- GFX blind classification for 13 drive/fuzz units and model-estimated Level,
  Gain/Drive/Distortion, and Tone/Filter controls
- heuristic Gate, Comp, EQ, Delay, and Reverb candidates with editable controls

Decoding runs on GPUI's background executor. Spectral analysis is streaming and
keeps only a 4096-sample FFT window plus accumulated bins in memory. Profile
analysis uses bounded streaming buckets and retains at most 32,768 analysis
points, compacted to the available pixel width while rendering.
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
- `ebur128`: MIT; performs EBU R128 / ITU-R BS.1770 loudness analysis.
- `rodio`: MIT or Apache-2.0; provides native preview output while sharing the
  existing Symphonia 0.5.5 decoding stack.
- Lucide: ISC; supplies the small embedded interface icon set. Attribution and
  the license text are in `assets/icons/LUCIDE.md`.
- `sysinfo`: MIT; samples only system CPU and memory data for the header pressure
  strips, with its default multithreading disabled.

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
