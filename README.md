# file-triage-ml

**Lightweight ML triage of suspicious files from filename and metadata features.**

A small, fully reproducible research project: classify files as *likely-malicious* vs *benign*
using only what is visible before a file is opened, namely its **name** and its **size**.
The intended use is a microsecond-cost **first-pass filter** that decides which files are worth
sending to expensive analysis (AV scanning, sandbox detonation, analyst review). It is not meant to
work as a detector on its own.

The full write-up is in [`paper/paper.md`](paper/paper.md).

| | |
|---|---|
| Data | 20,000 synthetic file records (25.5 % malicious), generated in `src/data_generation.py` |
| Features | 29 static metadata features (lexical, extension, size), in `src/features.py` |
| Models | rule baseline, logistic regression, random forest, histogram gradient boosting (scikit-learn) |
| Runtime | about 55 s for the whole pipeline on a CPU (5-seed stability run included) |
| Best test ROC-AUC | 0.940 (gradient boosting); random forest F1 0.849 |

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

To skip the 5-seed stability run:

```bash
python run_pipeline.py --quick
```

Each run regenerates `data/synthetic_files.csv` and everything in `results/`. For a fixed seed the
data and metrics are deterministic. Only the timing figures change between machines.

## Repository layout

```
run_pipeline.py            one-command reproducible pipeline
src/data_generation.py     synthetic dataset generator (documented templates, seeded)
src/features.py            static metadata feature extractor
src/models.py              rule baseline + sklearn model zoo
src/evaluate.py            metrics + all figures
src/evasion.py             attacker transformations for the evasion stress test
data/synthetic_files.csv   generated dataset (filename, size_bytes, template, label, label_clean)
results/                   CSV/JSON metrics, pipeline_log.txt, figures/
paper/paper.md             research paper
```

## Dataset design

Real labelled corpora of filenames with ground truth are scarce and often sensitive, so the data is
**synthetic**. Each record is produced by one of ten documented templates (`template` column):

| Template | Label | What it models |
|---|---|---|
| `benign_common` | 0 | word-based names, camera names (`IMG_1234`), screenshots, dated exports, hash/UUID cache names, "Copy of" duplicates, with common extensions and log-normal sizes typical for the type |
| `benign_installer` | 0 | legitimate vendor installers (`python-3.12.4-amd64.exe`, `SteamSetup.msi`), large |
| `benign_risky_type` | 0 | legitimate scripts/executables (`.ps1`, `.bat`, `.sh`, `.js`, `.dll`, `.lnk`, `.iso`) |
| `benign_compound` | 0 | legitimate multi-dot names (`backup.tar.gz`, `data.csv.bak`, `app.min.js`) |
| `mal_double_ext` | 1 | `invoice.pdf.exe`, `photo.jpg     .scr` (15 % with whitespace padding) |
| `mal_random_name` | 1 | high-entropy loader names (`iHKgosdAPTgN.exe`) |
| `mal_lure_exec` | 1 | social-engineering names on scripts/executables (`DHL_Shipment_4471.vbs`) |
| `mal_container` | 1 | macro documents, ISO/IMG/ZIP/RAR, `.one`, HTML smuggling |
| `mal_size_anomaly` | 1 | executables padded to 150–750 MB, or "documents" of a few hundred bytes |
| `mal_stealthy` | 1 | ordinary name, ordinary extension, ordinary size: **cannot be caught from metadata** |

The benign side contains hard negatives on purpose (installers, admin scripts, hash names, words
such as "invoice" or "update" in real documents). The malicious side includes a stealthy class
(12 % of malicious files) that sets a ceiling on achievable recall. A further 1 % of labels are
flipped at random to model triage-label noise. `label_clean` keeps the label before flipping.

## Features

| Group | Features |
|---|---|
| Lexical | `name_length`, `stem_length`, `name_entropy` (Shannon, bits/char of the stem), `digit_ratio`, `upper_ratio`, `special_ratio`, `num_dots`, `has_space_padding`, `suspicious_keyword` |
| Extension | `double_extension` (decoy inner extension followed by an executable/script outer extension), one-hot `ext_cat_*` over 12 categories |
| Size | one-hot `size_bucket_*` over 7 log-scale buckets (<1 KB … >100 MB) |

Feature extraction takes about 26 µs per file in pure Python.

## Threat model

* **Defender:** a mail gateway, web proxy, file-share scanner or EDR agent that sees a file's name
  and size before the file is opened, and needs to rank or route files cheaply.
* **What the model is for:** deciding which files to prioritise for deeper inspection. A low score
  must never mean "safe". It only means "not prioritised by this filter".
* **Attacker in scope:** commodity or phishing operators who reuse well-known delivery tricks such
  as double extensions, script droppers, lure keywords, random loader names and containers.
* **Attacker out of scope (and why):** an adaptive attacker controls the filename and the size
  completely. `src/evasion.py` shows that renaming the payload, wrapping it in an archive and
  padding it together bring recall to ≈0 % for every model. The defence against such attackers has
  to come from content analysis, provenance (Mark-of-the-Web, sender reputation) and behaviour.
  Metadata triage only lowers the cost of handling the lazy majority.
* **Data assumption:** the synthetic distribution stands in for a real one. Absolute numbers are
  not claims about production traffic. The relative findings (which features carry signal, how
  models compare, how evasion behaves) are the contribution.

## Results (seed 42, held-out test set of 5,000 files, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TPR @ 1 % FPR |
|---|---|---|---|---|---|---|
| Rule baseline | 0.813 | 0.605 | 0.771 | 0.678 | 0.826 | 0.098 |
| Logistic regression | 0.910 | 0.908 | 0.723 | 0.805 | 0.922 | 0.666 |
| Random forest | 0.930 | 0.940 | 0.775 | **0.849** | 0.935 | 0.753 |
| Gradient boosting | **0.930** | **0.950** | 0.767 | 0.849 | **0.940** | **0.760** |

* Across 5 independent seeds: GB F1 0.845 ± 0.005, ROC-AUC 0.945 ± 0.003. RF F1 0.842 ± 0.007.
  LR F1 0.793 ± 0.007 (`results/seed_stability.csv`).
* Double-extension and lure-executable files are caught 100 % of the time. Stealthy files are
  caught 0–1.4 % of the time, as intended by the design.
* The tree models flag 0.4–0.7 % of legitimate installers, against 87 % for the hand-written rule.
* Evasion: a benign-looking rename alone drops RF recall from 0.80 to 0.29. Rename + archive +
  padding drops it to 0.00 (`results/evasion_recall.csv`).

Figures are in `results/figures/`: `confusion_matrices.png`, `roc_curve.png`,
`feature_importance.png`, `entropy_distribution.png`, `evasion_recall.png`.
