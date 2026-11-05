"""
Evasion stress test.

A metadata-only classifier is only as good as the attacker's laziness. Here
we take the *malicious* test files and apply cheap, realistic renaming /
repackaging tricks an adversary fully controls, then re-score them. The drop
in recall quantifies how brittle each model is to adaptive attackers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generation import _benign_stem
from .features import split_name


def _drop_double_ext(name: str) -> str:
    stem, exts = split_name(name)
    return f"{stem}.{exts[-1]}" if exts else name


def _benign_rename(name: str, rng) -> str:
    _, exts = split_name(name)
    return f"{_benign_stem(rng)}." + ".".join(exts) if exts else _benign_stem(rng)


def _archive_wrap(name: str, rng) -> str:
    stem, _ = split_name(name)
    return f"{stem}.{rng.choice(['zip', '7z', 'rar'])}"


ATTACKS = {
    "None (original)": "identity",
    "Drop decoy extension": "drop_double",
    "Benign-looking rename": "rename",
    "Pad size to 10-100 MB": "pad",
    "Wrap in archive": "archive",
    "Rename + wrap + pad": "combined",
}


def apply_attack(df_mal: pd.DataFrame, attack: str, seed: int = 0) -> pd.DataFrame:
    """Return a modified copy of the malicious rows under the named attack."""
    rng = np.random.default_rng(seed)
    out = df_mal.copy()
    kind = ATTACKS[attack]
    names, sizes = list(out["filename"]), out["size_bytes"].to_numpy().copy()
    if kind in ("drop_double", "rename", "combined"):
        names = [_drop_double_ext(n) for n in names]
    if kind in ("rename", "combined"):
        names = [_benign_rename(n, rng) for n in names]
    if kind in ("archive", "combined"):
        names = [_archive_wrap(n, rng) for n in names]
        sizes = (sizes * rng.uniform(0.6, 0.95, len(sizes))).astype(int)  # compression
    if kind in ("pad", "combined"):
        sizes = np.maximum(sizes, rng.uniform(10e6, 100e6, len(sizes)).astype(int))
    out["filename"], out["size_bytes"] = names, sizes
    return out
