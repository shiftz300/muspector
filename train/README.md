# Inspector training

This directory contains only the active training and evaluation path for the
Muspector Inspector. The product target is a three-family multi-label detector:
`Drive`, `Delay`, and `Reverb`. Clean is the state where no family passes its
gate. Modulation effects are hard negatives, not an `Unknown` class.

The runtime is deliberately compact and CPU-friendly:

- 44.1 kHz, five-second windows;
- 2048-point FFT, 1024-sample hop, 128 normalized log-Mel bins;
- a clean-relative Drive/Delay encoder and pair head;
- a clean-relative Reverb encoder and pair head;
- a 20,153-parameter temporal Reverb verifier;
- zero user-side gradient updates when Clean Audio is imported.

Do not add Transformer or reconstruction work here. Remixer and clean-tone
reconstruction are a separate future model boundary.

## Directory policy

`layout.py` is the source of truth for active paths. New directories must use
semantic names in the form `{target}-{stage}-{data-domain}`. Sequence names
such as `v29` or `v51` are forbidden for active runs.

Active caches:

- `train/cache/reverb-pair-public-rir`
- `train/cache/drive-delay-pair-public-effects`
- `train/cache/pedal-identity-public-egfx`

Active runs:

- `train/runs/reverb-seed-public-chains`
- `train/runs/reverb-blind-public-rir`
- `train/runs/reverb-encoder-public-rir`
- `train/runs/reverb-pair-public-rir`
- `train/runs/drive-delay-encoder-public-effects`
- `train/runs/drive-delay-pair-public-effects`
- `train/runs/reverb-clean-audit`
- `train/runs/drive-delay-clean-audit`
- `train/runs/inspector-routed-public`
- `train/runs/reverb-verifier-hard-negative`
- `train/runs/reverb-verifier-hardware-replay`
- `train/runs/pedal-identity-noncommercial-release`

Everything else under `train/cache` is disposable. Historical run directories
may retain small JSON audit reports, but their checkpoints, virtual
environments, embeddings, and exported models are not active dependencies.

Preview and apply the same cleanup policy later with:

```sh
.venv310/bin/python train/cleanup.py --historical-runs --archive-runs
.venv310/bin/python train/cleanup.py --historical-runs --archive-runs --apply
```

The application loads only `models/inspector`. It never loads from
`train/cache` or `train/runs`.

## Environment

Python 3.10 is the supported training interpreter:

```sh
/opt/homebrew/bin/python3.10 -m venv .venv310
.venv310/bin/pip install -r train/requirements.txt
```

Essentia is not used.

## Public data

Download and audit the release-compatible sources with:

```sh
.venv310/bin/python train/download.py
.venv310/bin/python train/download.py guitar-effects-chains
.venv310/bin/python train/download.py aachen-chapel-rir
.venv310/bin/python train/audit_guitar_effect_chains.py
.venv310/bin/python train/audit_aachen_rir.py
```

The active routed family-detector path uses only the sources documented in `LICENSES.md` and
`sources.json`: DAFx25 guitar effect chains, Aachen chapel RIRs, EGFxSet,
Guitar-TECHS, GuitarSet, and GuitarJam. IDMT, RemFX, ToneTwisT, pretrained
research checkpoints, and Apple Audio Unit captures are excluded from those
weights. The pedal-identity component below is a separate non-commercial
release boundary.

Global pedal identity is a separate, explicitly non-commercial research line.
Download its CC BY-NC ToneTwist captures with:

```sh
.venv310/bin/python train/download.py \
  tonetwist-pedal-identity-dry-nc \
  tonetwist-pedal-identity-ts9-nc \
  tonetwist-pedal-identity-rodent-nc \
  tonetwist-pedal-identity-bdrive-nc \
  tonetwist-drive-open-set-klon-nc \
  tonetwist-drive-open-set-metal-muff-nc \
  tonetwist-drive-open-set-fuzzy-logic-nc \
  tonetwist-drive-open-set-silly-fuzz-nc
```

The quality-first developer pipeline uses the official Apache-2.0 ST-ITO
AFx-Rep checkpoint as a frozen 512-dimensional effect encoder, followed by a
small catalog head and an independent knownness verifier. It is not a user
training flow:

```sh
git clone https://github.com/csteinmetz1/st-ito /private/tmp/muspector-st-ito
curl -L https://huggingface.co/csteinmetz1/afx-rep/resolve/main/afx-rep.ckpt \
  -o /private/tmp/muspector-afx-rep.ckpt
.venv310/bin/python train/afx_identity_teacher.py
.venv310/bin/python train/afx_identity_catalog.py
.venv310/bin/python train/afx_identity_export.py
```

The exported fixed-input ONNX accepts one peak-normalized 48 kHz, five-second
waveform and returns seven identity logits plus one knownness logit. The catalog
is Blues Driver, RAT, Tube Screamer, Big Muff, Metal Muff, Fuzzy Logic, and
Silly Fuzz. Klon remains unsupported because its five-recording,
recording-disjoint identity test failed. A user imports only an unordered Clean
reference for the existing family detector; identity never consumes that Clean
profile and performs no local labels, backpropagation, or pedal registration.
Future user registration must remain an optional prototype layer and must not
change this base contract.

Developer builds without `embedded-identity` discover the optional research
package as `afx-pedal-identity.onnx` plus `afx-pedal-identity.json` in its
platform application-support `models` directory, or through
`MUSPECTOR_AFX_IDENTITY_ONNX`. If absent or rejected, Drive remains a generic
family result. The rolling non-commercial research Release builds with
`embedded-identity`; Git LFS supplies the 323.8 MB ONNX to CI and Rust embeds it
in the executable. ToneTwist is CC BY-NC 4.0 and the RemFX Zenodo record uses
the custom `cc-nc` identifier, so the model component must retain attribution
and must not be used in a commercial release.

Clean performances are grouped before splitting. Variants of one performance,
player, or device must not cross train, validation, calibration, and test.

## Training stages

The base stages are implemented by:

- `detect.py`: blind encoder training and export;
- `reference.py`: clean-reference encoder fitting;
- `relative.py`: non-aligned clean-relative pair-head fitting;
- `evaluate_multi_clean_profile.py`: pickup/tone-state clean audit;
- `route_relative_reports.py`: combine the Drive/Delay and Reverb branches;
- `reverb_verifier.py`: candidate-conditioned temporal Reverb verification.
- `pedal_identity.py`: frozen-backbone global known-pedal identity and
  open-set rejection; it never trains on user recordings.
- `afx_identity_teacher.py`: audited AFx-Rep feature cache and teacher control;
- `afx_identity_catalog.py`: global catalog head, knownness verifier, grouped
  evaluation, and development gate;
- `afx_identity_export.py`: one-file ONNX export, parity check, CPU benchmark,
  and package metadata;
- `compact_identity_distill.py` and `compact_identity_verifier.py`: compact
  student experiments; these remain rejected controls, not runtime artifacts.

Use each command's `--help` and write outputs only to paths declared by
`layout.py`. Every completed run must contain its configuration, split policy,
per-class metrics, Clean false-positive rate, cross-domain results, model
hashes, and explicit development/release gate status.

The current `afx-rep-catalog` development gate passes: public grouped test
closed-set accuracy and correct-accept rate are both `1.0`, public non-catalog
false accept is `0.0`, hardware-development RAT recall is `0.5`, and hardware
non-catalog false accept is `1/11`. Big Muff, Metal Muff, Fuzzy Logic, and Silly
Fuzz have only `1/3/3/3` independent public test recordings respectively, so
this is an integration candidate rather than a final generalization claim. It
passes the non-commercial integration gate, but an untouched multi-device test
is still required before claiming broad pedal-identification accuracy.

Runtime identity is limited to isolated Drive detections. The current catalog
was not trained to recover a pedal identity through a Delay/Reverb chain, and a
hardware development mixture produced an indistinguishable RAT score. Chain-
conditioned identity requires real multi-effect capture data before this gate
can be relaxed. The integrated Rust regression keeps the existing family result
at `15/15` exact, identifies `1/2` RAT recordings, and falsely names `1/11`
non-catalog Drive/Fuzz recordings. ONNX Runtime parity is `2.86e-6`; on the
development Mac, one five-second window takes about `38 ms` in ONNX Runtime and
`94 ms` in tract. The embedded ONNX is 323.8 MB.

## Reverb verifier

Train the public-only verifier:

```sh
.venv310/bin/python train/reverb_verifier.py
```

Fit the current hardware-development verifier with public replay:

```sh
.venv310/bin/python train/reverb_verifier.py \
  --resume-public train/runs/reverb-verifier-hard-negative \
  --development-updates 600 \
  --output train/runs/reverb-verifier-hardware-replay
```

The second command uses `$HOME/Downloads/test` and
`$HOME/Downloads/clean test` for weight fitting and threshold selection.
Those files are development data and must never be reported as an untouched
test set.

The bundled verifier uses this decision:

```text
top_two_mean(reverb_pair) >= clean_profile_reverb_threshold
and
top_two_mean(reverb_verifier) >= 0.342
```

## Current evidence

The final verifier report is
`train/runs/reverb-verifier-hardware-replay/metrics.json`:

- public grouped test: macro F1 `0.9820`, exact match `0.9588`, Clean FP
  `0.65%`;
- public Drive precision/recall `99.86% / 99.98%`;
- public Delay precision/recall `99.74% / 90.94%`;
- public Reverb precision/recall `99.76% / 99.32%`;
- hardware-development regression: Reverb precision/recall `1.00 / 1.00`;
- six hardware-development Clean recordings: complete-gate FP `0/6`;
- Rust integration regression: `15/15` exact with both the embedded profile and
  a profile imported from `test/clean.wav`.

The development gate passes. The release gate fails because no untouched,
device-disjoint labelled hardware final set exists. The `15/15` result is a
regression check, not a generalization result.

## User Clean contract

Clean import consumes at most ten seconds and creates two mean/deviation
profiles plus three thresholds. It performs no backpropagation, does not need
aligned playing, and does not store audio. The portable `.musp-training`
schema-4 file contains only 1,036 float32 statistics, thresholds, and a name.

## Artifact boundary

Runtime artifacts are copied to `models/inspector` only after their provenance,
hash, ONNX parity, and gate status are recorded. The current local research
runtime contains:

- `drive-delay-encoder.onnx`
- `drive-delay-head.onnx`
- `reverb-encoder.onnx`
- `reverb-head.onnx`
- `reverb-verifier.onnx`
- `routed-device-profile.bin`

See `LICENSES.md`, `sources.json`, and `models/inspector/LICENSES.md` before any
redistribution.
