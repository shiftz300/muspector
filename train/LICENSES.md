# Training data licenses

Only the datasets listed here may contribute to release model weights. Raw
audio stays under `data/`, is ignored by Git, and is never redistributed with
Muspector.

## EGFxSet

- Record: https://zenodo.org/records/7044411
- License: CC BY 4.0
- Creators: Hegel Pedroza, Gerardo Meza, Iran R. Roman
- Use: real-hardware clean, drive, delay, and reverb examples; unsupported
  modulation recordings are non-target hard negatives, not an aggregate class.

## Guitar-TECHS

- Record: https://zenodo.org/records/14963133
- License: CC BY 4.0
- Creators: Hegel Pedroza Villalobos, Termeh Taheri, Wallace Abreu,
  Ryan Corey, Iran R. Roman
- Use: clean electric-guitar DI and paired microphone/amp captures from
  different players and hardware. Both are zero-label examples for the three
  target effect families; split grouping keeps the paired performance together.
  The v21-v24 relative experiments use the complete official P1/P2
  chords/scales/single-notes/techniques archives: P1 remains training-only,
  P2 remains validation-only, and P3 music remains calibration/test only.
  Only DirectInput recordings enter the clean-reference relative head; MicAmp
  recordings remain excluded because their room/amp coloration is ambiguous
  with a user's Clean reference.
  The `clean-reference-real-v1` experiment narrows this further to DI only:
  microphone/amp recordings are excluded because room and amp coloration are
  ambiguous with a user's Clean reference.

## GuitarSet

- Record: https://zenodo.org/records/3371780
- License: CC BY 4.0
- Creators: Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye,
  Juan P. Bello
- Use: paired mono microphone and pickup-mix captures. Players are assigned to
  disjoint train, validation, calibration, and test partitions.

## GuitarJam

- Repository: https://huggingface.co/datasets/Julian-br/GuitarJam
- Revision: `2d467bfec90af19301b01123494f4b2ba64c5a3a`
- License: CC0 1.0
- Creator: Julian-br
- Use: training-only clean electric-guitar DI hard negatives. The dataset has
  one player and one recording chain, so it is never used to claim
  cross-device validation.

## Guitar improvisations with chains of five effects

- Record: https://zenodo.org/records/7871720
- DOI: `10.5281/zenodo.7871720`
- License: CC BY 4.0
- Creator: Michele Rossi
- Use: fixed archived dry recordings and exhaustive effect-chain recordings
  covering Overdrive, Chorus, Tremolo, Delay, and Reverb. Muspector maps only
  Overdrive, Delay, and Reverb to target families; Chorus/Tremolo-only chains
  are explicit hard negatives.
- Provenance: 400 unique improvisations recorded with four guitars, two pickup
  positions, and pick/finger technique. The author generated three 12,800-file
  processed sets with fixed parameters, variable parameters, and variable
  effect order. Muspector does not regenerate or modify the source audio.
- Split policy: PRS and Les Paul train; Stratocaster whole performances are
  split between validation and calibration; every Telecaster performance is a
  device-disjoint test. All 97 dry/processed files from one improvisation stay
  in one split.

## Aachen Cathedral St. Nicholas Chapel room responses

- Record: https://zenodo.org/records/20428705
- DOI: `10.5281/zenodo.20428705`
- License: CC BY 4.0
- Creators: Martin Zerwas, Selin Kayku, FH Aachen
- Use: 46 measured four-channel B-format room impulse responses from the St.
  Nicholas Chapel. Muspector uses channel one to add release-compatible Reverb
  diversity to otherwise clean archived guitar recordings.
- Rendering policy: the response is resampled, onset-trimmed, energy-normalized,
  and convolved at a deterministic 25/40/55/70 percent wet mix during cache or
  training. No derived wet-audio corpus is written or redistributed.
- Split policy: the sorted RIR inventory is divided 60/16/12/12 percent across
  train, validation, calibration, and test; one measured response never crosses
  partitions. Guitar performances retain their pre-existing grouped split.

## Apple system Audio Units

- Components: `AUDistortion`, `AUDelay`, and `AUMatrixReverb` 1.6.0, supplied
  with macOS and enumerated by Apple's `auval` tool.
- License: proprietary Apple system software under the macOS Software License
  Agreement; no component binary, preset, impulse response, or rendered audio
  is redistributed by Muspector.
- Use: local experiment-only wet captures rendered from the licensed clean
  corpora above. Release eligibility of weights trained with these captures
  requires a separate legal review; this experiment does not establish it.
- Split policy: every capture inherits its clean performance's split. EGFxSet
  wet hardware recordings are excluded from training and calibration and used
  only as a test-partition device-disjoint benchmark.

## Spotify Pedalboard training renderer

- Repository: https://github.com/spotify/pedalboard
- Version: `0.9.24`
- License: GPL-3.0
- Use: local, training-time-only Drive, Delay, Reverb, Chorus, and Phaser DSP
  implementations in the public-effects research runs. The Python package is
  not linked into or distributed with the Rust application.
- Rendering policy: effects are rendered in memory from the release-compatible
  clean corpora above. No derived audio corpus, plugin binary, or Pedalboard
  source is copied into model artifacts.
- Release boundary: the routed public candidate contains no IDMT, RemFX,
  ToneTwisT, or AudioSet
  weights/data. Whether trained weights produced with a GPL training tool need
  additional notices is a separate redistribution review; this document does
  not relicense Pedalboard or provide legal advice. The public-RIR release
  line does not depend on Pedalboard.

The downloader verifies every official archive against the size and MD5
published by Zenodo. Any new dataset requires an explicit entry here and in
`sources.json` before it can enter training.

## Runtime and portable-profile boundary

The repository's Apache-2.0 license applies to training and runtime source code,
not automatically to datasets, checkpoints, or exported ONNX files. The
public-RIR Inspector is trained from scratch only from the release-compatible CC BY/CC0 sources
listed above. Its artifacts are distributed separately under CC BY 4.0 and
carry the required upstream attribution. No raw dataset audio or RIR is
committed or embedded in the application.

Muspector's `.musp-training` schema-4 format deliberately excludes all shared
encoders and heads. It stores only the user's routed clean-reference profiles,
thresholds, and a name. It contains no audio or wet-labelled device adapter.
The file does not grant rights to the recording used to create it.

The older IDMT/RemFX research artifacts are excluded from the runtime and repository
model directory. Their legal caveats remain relevant only to ignored local
research runs described below.

## Private hardware development recordings

The Reverb verifier development fit uses the user's private WAV recordings in
`~/Downloads/test` and `~/Downloads/clean test`. These
recordings are not committed, redistributed, or granted a public dataset
license. The resulting verifier remains a local non-commercial research
artifact. The labelled wet directory updates its weights and threshold, so it
is development training data rather than an external or final evaluation set.

## Investigated research artifacts excluded from release weights

These sources are relevant research controls but are not compatible with the
current release-weight policy. They have not entered release-model training;
where explicitly noted they may be evaluated only in isolated, ignored local
research runs:

- Sony Research Fx-Encoder++ code and weights:
  https://github.com/SonyResearch/Fx-Encoder_PlusPlus — CC BY-NC 4.0.
- Tencent Music RelFx source/checkpoint:
  https://huggingface.co/TMEGalaxyAudioEffect/relfx-ismir2026 — CC BY-NC-SA
  4.0; the public 405 MB checkpoint was evaluated only in an isolated local
  research run. Its benchmark code, checkpoint, embeddings, probes, and
  derived weights have been pruned and remain excluded from runtime and release
  artifacts. Only the small historical report is retained under
  `train/runs/archive`.
- ToneTwisT AFx real Big Muff dry/wet pairs:
  https://zenodo.org/records/10797916 and
  https://zenodo.org/records/10891515 — both audio records are CC BY-NC 4.0.
  The GitHub index is MIT, but that does not replace the audio-record licenses.
  Muspector preserves the published EHX train/validation/test boundary and uses
  the unsplit DIY recording only for training. Wet clips are Drive positives
  and explicit Delay/Reverb hard negatives in isolated non-commercial runs;
  they cannot contribute to the release-compatible v29 line.
- ToneTwisT Landlord Brewers Droop Chorus and Mooer Trelicopter records:
  https://zenodo.org/records/10796408 and
  https://zenodo.org/records/10796416 — CC BY-NC 4.0. These were investigated
  as possible modulation hard negatives but were not downloaded or used in
  v29-v46. The ToneTwisT GitHub index's MIT license covers the index code, not
  these separately licensed audio records. The upstream Plate/Spring Reverb
  data request remains open, so these records do not add the missing target
  device diversity.
- GuitarML Proteus and ToneLibrary:
  https://github.com/GuitarML/Proteus and
  https://github.com/GuitarML/ToneLibrary — GPL-3.0 code/model collection as
  published upstream. Proteus explicitly supports non-time-based amp,
  distortion, overdrive, and boost captures and excludes Reverb, Delay,
  Flange, and Phaser. No community capture or weight was downloaded or used;
  per-model provenance would require a separate audit before any future use.
- Fraunhofer IDMT-SMT-Audio-Effects:
  https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html and
  https://zenodo.org/records/7544032 — CC BY-NC-ND 4.0 and described by the
  publisher as evaluation-purpose data. Its four guitar subsets are used only
  by the explicit non-commercial `--idmt-guitar` experiment. Instrument/pickup
  settings 6/7 train, setting 8 supplies whole-group validation/calibration,
  and setting 9 remains device-disjoint test material. No audio is modified or
  redistributed, and resulting weights are excluded from release artifacts.
- ST-ITO source and AFx-Rep checkpoint:
  https://github.com/csteinmetz1/st-ito and
  https://huggingface.co/csteinmetz1/afx-rep — Apache-2.0 as published by the
  authors. The frozen checkpoint and locally fitted probes are isolated
  non-commercial research artifacts in this repository.
- RemFX source, official checkpoint, and evaluation pairs:
  https://github.com/mhrice/RemFx,
  https://zenodo.org/records/8218621, and
  https://zenodo.org/records/8187288 — source is Apache-2.0; the official
  evaluation data is CC-NC. `1-1.zip` may be used only in the explicit
  non-commercial research experiment and only as training data; it is not a
  held-out RemFX benchmark after that use.
- Dynamic-SUPERB SoundEffectDetection_RemFx derivative:
  https://huggingface.co/datasets/DynamicSuperb/SoundEffectDetection_RemFx —
  the dataset card does not declare a separate license. Its audio is treated
  under the upstream RemFX CC-NC boundary. The 600 rows have no auditable
  source-performance IDs, so their label-stratified split cannot establish
  content-disjoint generalization.
- EfficientAT source and `mn04_as` checkpoint:
  https://github.com/fschmid56/EfficientAT and its official GitHub Release —
  MIT as published by the authors. The checkpoint was pretrained on AudioSet.
  Google's AudioSet metadata is CC BY 4.0, but that does not grant rights to
  every underlying hosted media clip. Muspector uses the checkpoint only in
  isolated v43 research evaluation, does not redistribute it, and does not
  include the derived head in runtime or release artifacts.

Do not distill, fine-tune, redistribute, or include these artifacts in a
commercial Muspector release model without a separate product-license decision.
Current use is restricted to the user's non-commercial research runs.
