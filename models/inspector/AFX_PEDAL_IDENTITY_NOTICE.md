# AFx pedal-identity model notice

`afx-pedal-identity.onnx` is a separately licensed, non-commercial research
component embedded only in Muspector builds made with the `embedded-identity`
feature. It is not covered by the repository's Apache-2.0 software license and
must not be used or redistributed for a purpose primarily intended for
commercial advantage or monetary compensation.

The model combines:

- the AFx-Rep Cnn14 checkpoint by Christian J. Steinmetz, published under
  Apache-2.0: https://huggingface.co/csteinmetz1/afx-rep
- locally fitted catalog and knownness heads trained with ToneTwisT pedal
  captures published by Marco Comunità and contributors under CC BY-NC 4.0;
  https://creativecommons.org/licenses/by-nc/4.0/
  source records and per-device attribution are listed in `train/LICENSES.md`;
- RemFX `1-1.zip` evaluation recordings by Matthew Rice, Christian J.
  Steinmetz, Joshua D. Reiss, and György Fazekas, whose official Zenodo record
  declares the custom `cc-nc` license identifier:
  https://zenodo.org/records/8187288

Muspector modifies the upstream material by extracting AFx-Rep embeddings and
fitting a seven-class identity head plus an independent open-set verifier. No
source audio is included. The resulting model is distributed only for
non-commercial research, with no warranty and no claim that the listed data
licensors endorse Muspector.

This notice documents the project's release boundary; it is not legal advice.
