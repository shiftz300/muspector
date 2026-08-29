"""Canonical semantic paths for active Inspector training artifacts.

Run names describe their role, model, and data domain. Historical experiment
directories may retain old numeric identifiers for auditability, but active
commands must import these paths instead of embedding sequence numbers.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus"

REVERB_CACHE = ROOT / "train/cache/reverb-pair-public-rir"
DRIVE_DELAY_CACHE = ROOT / "train/cache/drive-delay-pair-public-effects"
PEDAL_IDENTITY_CACHE = ROOT / "train/cache/pedal-identity-public-egfx"

REVERB_SEED_RUN = ROOT / "train/runs/reverb-seed-public-chains"
REVERB_BLIND_RUN = ROOT / "train/runs/reverb-blind-public-rir"
REVERB_ENCODER_RUN = ROOT / "train/runs/reverb-encoder-public-rir"
REVERB_PAIR_RUN = ROOT / "train/runs/reverb-pair-public-rir"
DRIVE_DELAY_ENCODER_RUN = ROOT / "train/runs/drive-delay-encoder-public-effects"
DRIVE_DELAY_PAIR_RUN = ROOT / "train/runs/drive-delay-pair-public-effects"
REVERB_CLEAN_AUDIT_RUN = ROOT / "train/runs/reverb-clean-audit"
DRIVE_DELAY_CLEAN_AUDIT_RUN = ROOT / "train/runs/drive-delay-clean-audit"
ROUTED_INSPECTOR_RUN = ROOT / "train/runs/inspector-routed-public"
REVERB_VERIFIER_PUBLIC_RUN = ROOT / "train/runs/reverb-verifier-hard-negative"
REVERB_VERIFIER_RUN = ROOT / "train/runs/reverb-verifier-hardware-replay"
PEDAL_IDENTITY_RUN = ROOT / "train/runs/pedal-identity-noncommercial-release"

ACTIVE_CACHES = frozenset((REVERB_CACHE, DRIVE_DELAY_CACHE, PEDAL_IDENTITY_CACHE))
ACTIVE_RUNS = frozenset(
    (
        REVERB_SEED_RUN,
        REVERB_BLIND_RUN,
        REVERB_ENCODER_RUN,
        REVERB_PAIR_RUN,
        DRIVE_DELAY_ENCODER_RUN,
        DRIVE_DELAY_PAIR_RUN,
        REVERB_CLEAN_AUDIT_RUN,
        DRIVE_DELAY_CLEAN_AUDIT_RUN,
        ROUTED_INSPECTOR_RUN,
        REVERB_VERIFIER_PUBLIC_RUN,
        REVERB_VERIFIER_RUN,
        PEDAL_IDENTITY_RUN,
    )
)
