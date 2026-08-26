# Muspector

Muspector is a compact GPUI audio inspector for studying guitar tones. The
current MVP focuses on fast local analysis. Effect-chain reconstruction,
parameter estimation, and playback are intentionally deferred.

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

## Analysis

- format, sample rate, channels, and duration
- peak, RMS, crest factor, and near-full-scale sample count
- an interactive waveform and short-term RMS profile
- a 64-band logarithmic spectrum with low/mid/high energy, spectral centroid,
  and 85% rolloff markers

Decoding runs on GPUI's background executor. Spectral analysis is streaming and
keeps only a 4096-sample FFT window plus accumulated bins in memory. Profile
analysis uses bounded streaming buckets and retains at most 192 display points.

## Layout

- `app`: GPUI state and interface
- `analysis`: decoding and signal metrics
- `assets`: embedded interface icons
- `theme`: compact visual tokens
