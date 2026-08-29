# Muspector

Muspector is a compact GPUI audio inspector for studying guitar tones. The
current research build performs local analysis, uses the routed clean-reference
Inspector candidate to detect Drive, Delay, and Reverb, and
proposes an editable hybrid effect chain.
The landscape workspace combines an RX-style analysis canvas with an
Ableton-inspired horizontal device chain. A lightweight local transport previews
the original audio without applying the proposed chain.

## Requirements

- Rust 1.96 or newer
- macOS, Windows, or a Vulkan-capable X11/Wayland Linux desktop

The CI artifacts currently target Apple silicon macOS, x86-64 Windows, and
x86-64 Linux. Those rolling artifacts are non-commercial research builds
because they embed the separately licensed pedal-identity model.

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

Playback uses `rodio 0.22` over `cpal 0.17`: CoreAudio on macOS, WASAPI on a
normal Windows build, and the available ALSA/JACK host on Linux. The Settings
button beside the Wave button lists concrete output devices and persists the
selection; choosing System Default follows the operating-system route.

ASIO is intentionally opt-in because its Windows build requires the Steinberg
ASIO SDK and LLVM/libclang. Set `CPAL_ASIO_DIR` when needed and build on Windows
with:

```sh
cargo build --release --features asio
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
compiled into the executable. The rolling Release also compiles the 323.8 MB
pedal-identity ONNX into each executable via `embedded-identity`; ordinary
developer builds keep that feature optional. Windows also embeds the icon and
product metadata in its native PE resources. macOS artifacts are ad-hoc signed; downloaded Linux
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
4. Dropping or opening another file creates a pending tab and analyzes it in the
   background. Switching to another ready or pending tab does not cancel the
   job, and completion does not steal focus from the current workspace. Loading
   progress follows real decode, loudness, spectrum, chain inference, and blind
   model stages rather than an indeterminate timer.
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
   Right-clicking inside a selection opens Copy, Paste, Delete, Quick Export
   WAV, and Select All. Command/Control + C, V, E, and A provide the matching
   shortcuts; Backspace/Delete removes the selection. Audio edits are performed
   against a lossless temporary working file and do not overwrite the source.
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
15. The right-side History menu lists document-local signal-chain and audio
    edits. Clicking an entry restores its working audio, waveform, selection,
    playhead, and chain state; Command/Control + Z undoes and Command/Control +
    Shift + Z redoes. Consecutive knob adjustments and one card drag are each
    grouped into a single history step.
16. The Wave button at the upper right opens Inspector Training. Import Clean
    Audio builds and activates a new non-aligned clean reference; Import
    Training File replaces it with a portable `.musp-training` bundle; Export
    Current Training writes the active bundle. Re-scan open audio after changing
    the active training.
17. The adjacent Settings button opens the output-device menu. Switching output
    releases the current stream and applies the selected CoreAudio, WASAPI, or
    opt-in ASIO device on the next playback.
18. Closing a modified tab asks whether to Save, Don’t Save, or Cancel. Closing
    the application does the same for every modified tab and Save All completes
    before the window exits. Command/Control + S saves the active document.
    Saving writes a readable `<name>.muspector.json` project beside the source;
    when audio was edited, it also writes `<name>.muspector.wav` without
    overwriting the imported file. Temporary working audio and the playback
    stream are released during a normal exit.

## Analysis

- format, sample rate, channels, and duration
- peak, RMS, integrated LUFS, crest factor, and near-full-scale sample count
- an interactive waveform with a fixed right-hand dBFS ruler, a subtle bottom
  baseline, short-term RMS, and
  momentary LUFS profiles
- a 64-band logarithmic spectrum with low/mid/high energy, spectral centroid,
  and 85% rolloff markers
- bounded envelope correlation for delay and diffuse-tail evidence
- routed Inspector classification for Drive, Delay, and Reverb, with the detected
  confidence and evidence shown directly on the existing editable devices
- heuristic Gate, Comp, and EQ candidates plus editable controls for the
  model-classified effects

Decoding runs on GPUI's background executor. Spectral analysis is streaming and
keeps only a 4096-sample FFT window plus accumulated bins in memory. Profile
analysis uses bounded streaming buckets and retains at most 32,768 analysis
points, compacted to the available pixel width while rendering.
The model pass resamples to 44.1 kHz and evaluates overlapping five-second
windows using the training-compatible 2048-point FFT, 1024-sample hop, and 128
normalized log-Mel bins. The routed non-aligned Clean-reference model uses the
public-effects branch for Drive/Delay and the public-RIR branch for Reverb. Python is not
required at runtime.

The non-commercial research build can add concrete Drive model names when the
routed detector finds an isolated Drive without Delay or Reverb. It uses a frozen AFx-Rep Cnn14 encoder,
a seven-model global catalog, and an independent knownness verifier; it selects
at most three high-energy five-second windows at 48 kHz. This branch is
pretrained by the developer and does not use the imported Clean profile, train
on the user's computer, or register the user's pedals. If a developer build
omits the `embedded-identity` feature or rejects the recording, the card remains
the generic `Drive` family. The
rolling non-commercial Release embeds this model; its fitted heads use
non-commercial ToneTwist and RemFX data and are not covered by Apache-2.0.

The Wave menu makes the reference replaceable. A Clean import uses at most the
first ten seconds to compute separate Drive/Delay and Reverb frozen-encoder profiles and
routed clean-derived thresholds. It performs zero gradient updates, requires no
wet-labelled user recordings, and does not require aligned playing. A
`.musp-training` schema-4 bundle stores only the 1,036 profile statistics, thresholds,
and a name; the active bundle is restored from the platform application-support
directory on the next launch. Older bundles must be recreated from their Clean
audio because their single-profile contract is incompatible with the routed model.

The checked-in default profile targets the development guitar/rig represented
by `models/inspector/routed-device-profile.bin`; its routing was selected on the same
development set, so it is not an unbiased unseen-device claim. It does not
recover effect order or clean audio. Remixer remains a separate future model
boundary.

Reverb uses an additional compact temporal verifier after the clean-relative
pair head. The final decision is an AND gate: the pair score must pass the
profile-derived Reverb threshold and the verifier score must pass `0.342`.
The verifier is fixed at runtime; importing Clean Audio does not fine-tune it.
On the 15-file hardware development fixture the integrated Rust path is `15/15`
exact and removes all four former Reverb false positives. That fixture was used
during verifier fitting, so this is development regression evidence only; the
release gate still requires an untouched device-disjoint labelled hardware set.

## Models and licenses

- The Muspector source code is Apache-2.0. Model weights are separate artifacts
  and are not relicensed under the repository's Apache license. The public-RIR Reverb
  artifacts remain CC BY 4.0; the public-effects Drive/Delay artifacts are integrated as a
  local non-commercial research candidate pending a redistribution decision.
- The routed model uses the CC BY 4.0 DAFx25 chain, Aachen chapel RIR, EGFxSet,
  Guitar-TECHS, and GuitarSet sources plus CC0 GuitarJam. The Drive/Delay branch also
  uses in-memory renders produced by the GPL-3.0 Spotify Pedalboard training
  tool, which is not bundled or linked into the application. IDMT, RemFX,
  pretrained research weights, and Apple AU renders are excluded.
- A `.musp-training` schema-4 file contains the user's derived reference statistics
  and thresholds, not shared encoder/head weights or audio. Its use still
  depends on the user's rights to the imported recording.
- The AFx pedal-identity package is a separate non-commercial
  research artifact. Its upstream AFx-Rep encoder is Apache-2.0, while its
  fitted catalog/verifier heads depend on CC BY-NC 4.0 ToneTwist and Zenodo
  `cc-nc` RemFX recordings. GitHub rolling releases embed it under the separate
  notice in `models/inspector/AFX_PEDAL_IDENTITY_NOTICE.md`; this does not make
  the model or the combined research build commercially usable.
- Hashes, attribution, artifact boundaries, and upstream links are in
  `models/inspector/LICENSES.md`, `train/LICENSES.md`, and `NOTICE`.
- `tract-onnx`: MIT or Apache-2.0; runs the checked-in ONNX models in Rust.
- `rubato`: MIT or Apache-2.0; performs fixed-ratio high-quality resampling.
- `ebur128`: MIT; performs EBU R128 / ITU-R BS.1770 loudness analysis.
- `rodio`: MIT or Apache-2.0; provides native preview output while sharing the
  existing Symphonia 0.5.5 decoding stack.
- Lucide: ISC; supplies the small embedded interface icon set. Attribution and
  the license text are in `assets/icons/LUCIDE.md`.
- `sysinfo`: MIT; samples only system CPU and memory data for the header pressure
  strips, with its default multithreading disabled.

`train/reference.py`, `train/detect.py`, and `train/relative.py` reproduce the
semantic encoder/pair runs and ONNX exports; `train/route_relative_reports.py`
records label routing. `train/reverb_verifier.py` trains the candidate-conditioned
temporal Reverb verifier. The hardware-replay verifier is connected in the
local non-commercial research runtime, while its release gate remains failed
until an untouched device-disjoint hardware test passes.
Training dependencies are not required by the application. These notices
document provenance; they do not grant rights beyond the upstream licenses.
CC BY 4.0 attribution requirements still apply when redistributing the
Inspector artifacts.

## Layout

- `app`: GPUI state and interface
- `analysis`: decoding and signal metrics
- `audio`: rodio/CPAL playback, device discovery, and persisted output routing
- `clip`: streaming selection copy, delete, paste, and float-WAV export
- `project`: durable effect-chain metadata and edited-audio save coordination
- `blind`: routed Inspector preprocessing, dual-pair inference, and chain
  classification
- `identity`: optional fixed global pedal-catalog inference; no user training
- `chain`: effect inference and parameter models
- `models`: attributed inference weights
- `assets`: embedded interface icons
- `theme`: compact visual tokens
