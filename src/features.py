"""
Static metadata feature extraction.

Input is only what a mail gateway, proxy or EDR agent sees *before* opening
a file: its name and its size. No content bytes, hashes or sandbox output
are used, so extraction costs microseconds per file.

Feature groups
--------------
Lexical (name):   name_length, stem_length, name_entropy, digit_ratio,
                  upper_ratio, special_ratio, num_dots, has_space_padding,
                  suspicious_keyword
Extension:        double_extension, ext_cat_<category> (one-hot)
Size:             size_bucket_<k> (one-hot, 7 log-scale buckets)
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .data_generation import DECOY_EXT, SUSPICIOUS_KEYWORDS

# Extension -> coarse category. Unknown extensions map to "other".
EXT_CATEGORIES = {
    "document": ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv",
                 "md", "rtf", "odt", "ics", "one"],
    "macro_document": ["docm", "xlsm", "pptm", "dotm"],
    "image": ["jpg", "jpeg", "png", "gif", "heic", "bmp", "svg"],
    "media": ["mp4", "mp3", "mov", "wav", "avi", "mkv"],
    "archive": ["zip", "rar", "7z", "gz", "bz2", "tar"],
    "disk_image": ["iso", "img", "dmg", "vhd"],
    "code_data": ["py", "c", "java", "css", "json", "log", "tmp", "ts", "1", "bak", "sql"],
    "web": ["html", "htm", "hta"],
    "executable": ["exe", "msi", "scr", "com", "pif", "dll", "jar"],
    "script": ["js", "jse", "vbs", "vbe", "wsf", "bat", "cmd", "ps1", "sh"],
    "shortcut": ["lnk", "url"],
}
EXT_TO_CAT = {e: c for c, exts in EXT_CATEGORIES.items() for e in exts}
CATEGORY_NAMES = list(EXT_CATEGORIES) + ["other"]

# Outer extensions that make a double extension dangerous.
RISKY_OUTER = set(EXT_CATEGORIES["executable"] + EXT_CATEGORIES["script"]
                  + EXT_CATEGORIES["shortcut"] + ["hta"])

# Size buckets (upper bounds in bytes); last bucket is open-ended.
SIZE_EDGES = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
SIZE_LABELS = ["<1KB", "1-10KB", "10-100KB", "100KB-1MB", "1-10MB", "10-100MB", ">100MB"]

_KEYWORD_RE = re.compile("|".join(sorted(SUSPICIOUS_KEYWORDS, key=len, reverse=True)))


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character of string s (0 for empty)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def split_name(filename: str) -> tuple[str, list[str]]:
    """
    Split a filename into (stem, [extensions...]) with trailing whitespace
    padding preserved in the stem side. 'a.pdf   .exe' -> ('a', ['pdf', 'exe']).
    """
    parts = filename.split(".")
    if len(parts) == 1:
        return filename, []
    stem = parts[0]
    exts = [p.strip().lower() for p in parts[1:]]
    return stem, exts


def extract_one(filename: str, size_bytes: int) -> dict:
    """Compute the feature dict for a single file."""
    stem, exts = split_name(filename)
    ext = exts[-1] if exts else ""
    n = max(len(filename), 1)

    # Double extension: an inner decoy (document/media) extension followed by
    # an executable/script outer extension. Legit compounds (tar.gz) don't fire.
    double_ext = int(len(exts) >= 2 and exts[-2] in DECOY_EXT and ext in RISKY_OUTER)

    feats = {
        "name_length": len(filename),
        "stem_length": len(stem),
        "name_entropy": shannon_entropy(stem),
        "digit_ratio": sum(ch.isdigit() for ch in filename) / n,
        "upper_ratio": sum(ch.isupper() for ch in filename) / n,
        "special_ratio": sum(not ch.isalnum() and ch != "." for ch in filename) / n,
        "num_dots": filename.count("."),
        "has_space_padding": int("   " in filename),
        "suspicious_keyword": int(bool(_KEYWORD_RE.search(stem.lower()))),
        "double_extension": double_ext,
    }
    cat = EXT_TO_CAT.get(ext, "other")
    for c in CATEGORY_NAMES:
        feats[f"ext_cat_{c}"] = int(cat == c)
    bucket = int(np.searchsorted(SIZE_EDGES, size_bytes, side="right"))
    for i, lab in enumerate(SIZE_LABELS):
        feats[f"size_bucket_{lab}"] = int(bucket == i)
    return feats


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorise a DataFrame with `filename` and `size_bytes` columns."""
    return pd.DataFrame([extract_one(f, s) for f, s in zip(df["filename"], df["size_bytes"])],
                        index=df.index)
