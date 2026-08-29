# Inspector artifact notices

Muspector source code is Apache-2.0. Model artifacts retain separate provenance
and do not automatically inherit the repository license.

## Release-compatible public-RIR Reverb branch

`reverb-encoder.onnx` and `reverb-head.onnx` are CC BY 4.0 artifacts. They were
trained from scratch from the attributed CC BY/CC0 sources below and exclude
all restricted research datasets.

Copyright 2026 Muspector contributors.

License text: https://creativecommons.org/licenses/by/4.0/

## Public-effects Drive/Delay research branch

`drive-delay-encoder.onnx` and `drive-delay-head.onnx` use the same attributed
CC BY/CC0 guitar sources plus local in-memory effects rendered with Spotify
Pedalboard 0.9.24 (GPL-3.0). Pedalboard source/binaries and rendered audio are
not included or linked into Muspector. These two weights are integrated as a
local non-commercial research candidate; redistribution requires a separate
product-license decision. This notice does not relicense Pedalboard or provide
legal advice.

`reverb-verifier.onnx` is a local non-commercial research artifact initialized
from the public-only verifier and then fitted on private hardware-development
recordings with public replay. The private recordings are not included. Its
public inputs retain the attributions below; the artifact is not presented as
release-qualified until an untouched device-disjoint hardware evaluation is
available.

`routed-device-profile.bin` and portable `.musp-training` schema-4 files contain only
derived Clean-reference statistics, names, and thresholds. They contain no
model weights or audio. Users must have permission to process and export the
recording from which a profile is derived.

The routed model and verifier do not contain or derive from
IDMT-SMT-Audio-Effects, RemFX, ToneTwisT, Fx-Encoder++, RelFX, AFx-Rep,
AudioSet checkpoints, or Apple Audio Unit captures.

## Embedded AFx pedal identity

`afx-pedal-identity.onnx` is a distinct non-commercial research component. Its
AFx-Rep encoder is Apache-2.0; its fitted catalog and knownness heads use
ToneTwisT CC BY-NC 4.0 pedal captures and RemFX data whose official Zenodo
record declares `cc-nc`. It is not relicensed under Apache-2.0. Required
authors, links, modification notice, hash, and restrictions are recorded in
`AFX_PEDAL_IDENTITY_NOTICE.md` and `train/LICENSES.md`.

## Required upstream attribution

- Guitar improvisations with chains of five effects, Michele Rossi,
  CC BY 4.0, DOI `10.5281/zenodo.7871720`:
  https://zenodo.org/records/7871720
- Room acoustic measurement and simulation data of the St. Nicholas Chapel,
  Aachen Cathedral, Martin Zerwas, Selin Kayku, FH Aachen, CC BY 4.0,
  DOI `10.5281/zenodo.20428705`:
  https://zenodo.org/records/20428705
- EGFxSet, Hegel Pedroza, Gerardo Meza, Iran R. Roman, CC BY 4.0:
  https://zenodo.org/records/7044411
- Guitar-TECHS, Hegel Pedroza Villalobos, Termeh Taheri, Wallace Abreu,
  Ryan Corey, Iran R. Roman, CC BY 4.0:
  https://zenodo.org/records/14963133
- GuitarSet, Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye,
  Juan P. Bello, CC BY 4.0:
  https://zenodo.org/records/3371780
- GuitarJam, Julian-br, CC0 1.0:
  https://huggingface.co/datasets/Julian-br/GuitarJam
- Spotify Pedalboard 0.9.24, GPL-3.0:
  https://github.com/spotify/pedalboard

No source audio, room impulse response, generated wet corpus, or Pedalboard
binary is included. Complete provenance and excluded research sources are in
`train/LICENSES.md`.
