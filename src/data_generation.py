"""
Synthetic dataset generator for filename/metadata-based file triage.

Every record is a (filename, size_bytes, label) triple, where label is
1 = likely malicious, 0 = benign. Nothing here touches real malware: the
generator only reproduces the *metadata patterns* that are well documented
in phishing / commodity-malware delivery (double extensions, script
droppers, random names, lure keywords, odd sizes) together with the
patterns of ordinary user and system files.

Realism is deliberately imperfect in both directions so the task is not
trivially separable:

* Benign "hard" cases: legitimate installers (.exe/.msi), admin scripts
  (.ps1/.bat/.sh/.js), hash/UUID-named cache files, legitimate compound
  extensions (backup.tar.gz, data.csv.bak) and words like "update" or
  "invoice" in genuine documents.
* Malicious "hard" cases: plain-looking lure documents (Q3_report.docm),
  weaponised containers (shipment.iso, invoice.zip) and exploit documents
  with ordinary names and sizes (report.pdf) that metadata cannot catch.
* A small amount of label noise (default 1%) models triage-label error.

All randomness flows from a single numpy Generator so a given seed yields
byte-identical output.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------

BENIGN_WORDS = [
    "report", "budget", "meeting", "notes", "minutes", "agenda", "draft",
    "final", "summary", "project", "proposal", "thesis", "chapter", "slides",
    "presentation", "lecture", "homework", "assignment", "resume", "cv",
    "letter", "contract", "schedule", "roadmap", "design", "spec", "plan",
    "team", "photo", "vacation", "family", "holiday", "birthday", "wedding",
    "recipe", "travel", "itinerary", "manual", "guide", "readme", "changelog",
    "data", "results", "analysis", "experiment", "figure", "table", "survey",
    "quarterly", "annual", "sales", "marketing", "q1", "q2", "q3", "q4",
    "backup", "export", "archive", "config", "settings", "main", "utils",
    "index", "style", "app", "test", "train", "model", "dataset", "invoice",
    "receipt", "statement", "tax", "payroll", "update", "release", "notes",
]

# Lure keywords commonly seen in malicious attachment / download names.
SUSPICIOUS_KEYWORDS = [
    "invoice", "payment", "urgent", "overdue", "receipt", "shipment",
    "shipping", "dhl", "fedex", "ups", "tracking", "scan", "scanned", "fax",
    "voicemail", "password", "reset", "account", "verify", "verification",
    "suspended", "security", "alert", "bank", "refund", "tax", "irs",
    "salary", "bonus", "crack", "keygen", "patch", "activator", "loader",
    "free", "hack", "cheat", "unlocker", "update", "flash", "codec", "setup",
    "install", "document", "confidential", "order", "quote", "remittance",
]

LEGIT_VENDORS = [
    "ChromeSetup", "FirefoxInstaller", "vlc-3.0.20-win64", "python-3.12.4-amd64",
    "Git-2.45.1-64-bit", "VSCodeUserSetup-x64-1.90.0", "zoom_installer",
    "SteamSetup", "npp.8.6.8.Installer.x64", "7z2406-x64", "OBS-Studio-30.1.2",
    "node-v20.14.0-x64", "Docker Desktop Installer", "SlackSetup", "TeamsSetup",
    "AcroRdrDC2400220759_en_US", "WinRAR-x64-701", "gimp-2.10.38-setup",
    "LibreOffice_24.2.4_Win_x86-64", "Spotify-Setup", "putty-64bit-0.81-installer",
]

# --------------------------------------------------------------------------
# Extension groups and typical (log-normal) size parameters.
# Size parameters are (median_bytes, sigma_of_ln). Medians follow common
# real-world magnitudes (e.g. phone photos ~3 MB, Office docs ~100 KB).
# --------------------------------------------------------------------------

BENIGN_EXT = {
    # ext: (sampling weight, median bytes, sigma)
    "pdf": (10, 400e3, 1.2), "docx": (8, 80e3, 1.0), "xlsx": (6, 60e3, 1.1),
    "pptx": (4, 2.5e6, 1.0), "txt": (5, 6e3, 1.4), "csv": (4, 150e3, 1.8),
    "md": (2, 5e3, 1.0), "jpg": (10, 2.8e6, 0.7), "png": (8, 600e3, 1.2),
    "gif": (1, 900e3, 1.2), "heic": (2, 2.0e6, 0.5), "mp4": (4, 80e6, 1.3),
    "mp3": (3, 5e6, 0.6), "mov": (1, 150e6, 1.0), "wav": (1, 30e6, 1.0),
    "zip": (4, 20e6, 1.6), "gz": (1, 15e6, 1.6), "7z": (1, 40e6, 1.5),
    "py": (3, 8e3, 1.1), "html": (2, 40e3, 1.2), "css": (1, 15e3, 1.0),
    "json": (3, 20e3, 1.6), "c": (1, 12e3, 1.0), "java": (1, 10e3, 1.0),
    "log": (2, 200e3, 2.0), "tmp": (1, 50e3, 2.0), "ics": (1, 4e3, 0.6),
}

# Benign executables / scripts: legitimate but superficially "suspicious".
BENIGN_RISKY_EXT = {
    "exe": (6, 60e6, 1.0), "msi": (3, 80e6, 0.9), "dmg": (2, 120e6, 0.8),
    "ps1": (2, 6e3, 1.0), "bat": (1, 2e3, 1.0), "sh": (2, 3e3, 1.0),
    "js": (3, 60e3, 1.5), "dll": (1, 2e6, 1.2), "iso": (1, 3e9, 0.6),
    "lnk": (1, 2e3, 0.3),
}

# Executable / script extensions used as the *outer* extension in malicious
# delivery.  Sizes are droppers/loaders: typically small.
MAL_EXEC_EXT = {
    "exe": (10, 300e3, 1.4), "scr": (6, 250e3, 1.2), "com": (1, 60e3, 1.0),
    "pif": (1, 60e3, 1.0), "vbs": (6, 8e3, 1.2), "js": (8, 15e3, 1.3),
    "jse": (2, 15e3, 1.2), "wsf": (2, 10e3, 1.2), "hta": (3, 20e3, 1.2),
    "bat": (3, 3e3, 1.0), "cmd": (2, 3e3, 1.0), "ps1": (4, 8e3, 1.2),
    "lnk": (4, 3e3, 0.8), "dll": (3, 500e3, 1.2), "jar": (2, 400e3, 1.2),
}

# Containers and macro documents used to smuggle payloads.
MAL_CONTAINER_EXT = {
    "zip": (5, 400e3, 1.3), "rar": (3, 400e3, 1.3), "iso": (4, 2e6, 1.4),
    "img": (2, 2e6, 1.4), "docm": (4, 120e3, 0.9), "xlsm": (4, 150e3, 0.9),
    "doc": (2, 200e3, 0.9), "xls": (1, 200e3, 0.9), "one": (2, 300e3, 1.0),
    "html": (3, 30e3, 1.0), "pdf": (3, 200e3, 1.0),
}

# Decoy "inner" extensions used in double-extension tricks.
DECOY_EXT = ["pdf", "doc", "docx", "xlsx", "jpg", "png", "txt", "mp4", "mp3", "zip"]


def _weighted_choice(rng: np.random.Generator, table: dict) -> tuple[str, float, float]:
    """Pick an extension from a {ext: (weight, median, sigma)} table."""
    exts = list(table)
    w = np.array([table[e][0] for e in exts], dtype=float)
    ext = exts[rng.choice(len(exts), p=w / w.sum())]
    _, median, sigma = table[ext]
    return ext, median, sigma


def _lognormal_size(rng: np.random.Generator, median: float, sigma: float) -> int:
    """Draw a file size (bytes) from a log-normal with the given median."""
    return int(max(16, rng.lognormal(np.log(median), sigma)))


def _rand_token(rng: np.random.Generator, n: int, alphabet: str) -> str:
    return "".join(rng.choice(list(alphabet), size=n))


# --------------------------------------------------------------------------
# Benign name templates
# --------------------------------------------------------------------------

def _benign_stem(rng: np.random.Generator) -> str:
    """Return a stem that looks like an ordinary user/system file name."""
    r = rng.random()
    if r < 0.35:  # word_word[_v2|_final|_2024]
        k = rng.integers(1, 4)
        words = list(rng.choice(BENIGN_WORDS, size=k, replace=False))
        sep = rng.choice(["_", "-", " ", ""])
        if sep == "":
            words = [w.capitalize() for w in words]
        stem = sep.join(words)
        tail = rng.random()
        if tail < 0.25:
            stem += f"{sep or '_'}v{rng.integers(1, 6)}"
        elif tail < 0.45:
            stem += f"{sep or '_'}{rng.integers(2015, 2027)}"
        elif tail < 0.55:
            stem += f"{sep or '_'}final"
        return stem
    if r < 0.50:  # camera naming
        return rng.choice(["IMG_", "DSC_", "DSCN", "PXL_", "GOPR"]) + str(rng.integers(1000, 99999))
    if r < 0.60:  # screenshots / dated exports
        y, m, d = rng.integers(2018, 2027), rng.integers(1, 13), rng.integers(1, 29)
        return rng.choice([
            f"Screenshot {y}-{m:02d}-{d:02d} at {rng.integers(0, 24)}.{rng.integers(0, 60):02d}.{rng.integers(0, 60):02d}",
            f"export_{y}{m:02d}{d:02d}",
            f"{y}-{m:02d}-{d:02d}_{rng.choice(BENIGN_WORDS)}",
        ])
    if r < 0.68:  # hash / uuid cache-style names (legitimately high entropy)
        if rng.random() < 0.5:
            return _rand_token(rng, int(rng.integers(8, 33)), "0123456789abcdef")
        h = "0123456789abcdef"
        return "-".join(_rand_token(rng, n, h) for n in (8, 4, 4, 4, 12))
    if r < 0.80:  # copy / download duplicates
        base = rng.choice(BENIGN_WORDS)
        return rng.choice([f"{base} ({rng.integers(1, 5)})", f"Copy of {base}", f"{base}_copy"])
    # short single word or code-ish module names
    return rng.choice(BENIGN_WORDS) + rng.choice(["", "", "_utils", "_test", "2", "_old"])


def generate_benign(rng: np.random.Generator) -> tuple[str, int, str]:
    """One benign record -> (filename, size, generation_template)."""
    r = rng.random()
    if r < 0.08:  # legitimate installers
        ext = rng.choice(["exe", "msi", "exe", "dmg"])
        _, median, sigma = BENIGN_RISKY_EXT[ext]
        return f"{rng.choice(LEGIT_VENDORS)}.{ext}", _lognormal_size(rng, median, sigma), "benign_installer"
    if r < 0.15:  # admin / dev scripts and other "risky" types used legitimately
        ext, median, sigma = _weighted_choice(rng, BENIGN_RISKY_EXT)
        return f"{_benign_stem(rng)}.{ext}", _lognormal_size(rng, median, sigma), "benign_risky_type"
    if r < 0.19:  # legitimate compound extensions
        stem = _benign_stem(rng)
        compound = rng.choice(["tar.gz", "tar.bz2", "sql.gz", "csv.bak", "json.gz", "log.1", "min.js", "d.ts"])
        size = _lognormal_size(rng, 2e6 if "gz" in compound or "bz2" in compound else 50e3, 1.5)
        return f"{stem}.{compound}", size, "benign_compound"
    ext, median, sigma = _weighted_choice(rng, BENIGN_EXT)
    return f"{_benign_stem(rng)}.{ext}", _lognormal_size(rng, median, sigma), "benign_common"


# --------------------------------------------------------------------------
# Malicious name templates
# --------------------------------------------------------------------------

def _lure_stem(rng: np.random.Generator) -> str:
    """Social-engineering stem built from lure keywords + reference numbers."""
    k = rng.integers(1, 4)
    words = list(rng.choice(SUSPICIOUS_KEYWORDS, size=k, replace=False))
    style = rng.random()
    if style < 0.3:
        words = [w.upper() for w in words]
    elif style < 0.6:
        words = [w.capitalize() for w in words]
    stem = rng.choice(["_", "-", " ", ""]).join(words)
    if rng.random() < 0.6:  # reference / order numbers
        stem += rng.choice(["_", "#", "-", " "]) + str(rng.integers(1000, 99999999))
    return stem


def generate_malicious(rng: np.random.Generator) -> tuple[str, int, str]:
    """One malicious record -> (filename, size, generation_template)."""
    r = rng.random()
    if r < 0.24:  # double extension: invoice.pdf.exe, photo.jpg   .scr
        stem = _lure_stem(rng) if rng.random() < 0.6 else _benign_stem(rng)
        outer, median, sigma = _weighted_choice(rng, MAL_EXEC_EXT)
        pad = " " * int(rng.integers(4, 40)) if rng.random() < 0.15 else ""
        return (f"{stem}.{rng.choice(DECOY_EXT)}{pad}.{outer}",
                _lognormal_size(rng, median, sigma), "mal_double_ext")
    if r < 0.42:  # random / high-entropy loader names
        n = int(rng.integers(6, 24))
        alphabet = rng.choice([string.ascii_letters + string.digits,
                               string.ascii_lowercase + string.digits,
                               string.ascii_uppercase])
        outer, median, sigma = _weighted_choice(rng, MAL_EXEC_EXT)
        return f"{_rand_token(rng, n, alphabet)}.{outer}", _lognormal_size(rng, median, sigma), "mal_random_name"
    if r < 0.62:  # misleading lure name + executable / script
        outer, median, sigma = _weighted_choice(rng, MAL_EXEC_EXT)
        return f"{_lure_stem(rng)}.{outer}", _lognormal_size(rng, median, sigma), "mal_lure_exec"
    if r < 0.80:  # weaponised container or macro document with lure name
        ext, median, sigma = _weighted_choice(rng, MAL_CONTAINER_EXT)
        stem = _lure_stem(rng) if rng.random() < 0.55 else _benign_stem(rng)
        return f"{stem}.{ext}", _lognormal_size(rng, median, sigma), "mal_container"
    if r < 0.88:  # size anomalies: padded binaries (>100 MB) or tiny "documents"
        if rng.random() < 0.5:
            outer = rng.choice(["exe", "scr", "dll"])
            name = f"{_lure_stem(rng) if rng.random() < 0.5 else _benign_stem(rng)}.{outer}"
            return name, int(rng.uniform(150e6, 750e6)), "mal_size_anomaly"
        ext = rng.choice(["pdf", "docx", "xlsx", "jpg"])
        return f"{_benign_stem(rng)}.{ext}", int(rng.integers(200, 4000)), "mal_size_anomaly"
    # stealthy: ordinary-looking name, ordinary extension and size
    stealth_table = {k: BENIGN_EXT[k] for k in ("pdf", "docx", "xlsx", "zip", "html")}
    ext, median, sigma = _weighted_choice(rng, stealth_table)
    return f"{_benign_stem(rng)}.{ext}", _lognormal_size(rng, median, sigma), "mal_stealthy"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate_dataset(n_samples: int = 20000, malicious_fraction: float = 0.25,
                     label_noise: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """
    Build the synthetic triage dataset.

    Parameters
    ----------
    n_samples : total number of files.
    malicious_fraction : prior probability that a file is malicious.
    label_noise : fraction of labels flipped uniformly at random.
    seed : RNG seed (fully determines the output).

    Returns
    -------
    DataFrame with columns: filename, size_bytes, template, label.
    """
    rng = np.random.default_rng(seed)
    n_mal = int(round(n_samples * malicious_fraction))
    rows = [(*generate_benign(rng), 0) for _ in range(n_samples - n_mal)]
    rows += [(*generate_malicious(rng), 1) for _ in range(n_mal)]
    df = pd.DataFrame(rows, columns=["filename", "size_bytes", "template", "label"])

    # Label noise: flip a small random subset (records the clean label too).
    df["label_clean"] = df["label"]
    flip = rng.random(len(df)) < label_noise
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

    # Shuffle so classes are interleaved.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
