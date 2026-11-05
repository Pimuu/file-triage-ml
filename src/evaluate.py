"""
Metrics and figures.

Colours follow a fixed, colour-vision-deficiency-checked categorical order
(blue, orange, aqua) on a near-white surface; benign is always blue and
malicious always orange so identity never changes between figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled
BENIGN_C, MAL_C = SERIES[0], SERIES[1]
MODEL_COLORS = {"Logistic regression": SERIES[0], "Random forest": SERIES[1],
                "Gradient boosting": SERIES[2], "Rule baseline": INK_2}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 100,
    })


_style()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Standard classification metrics plus two triage operating points."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "fpr": fp / (fp + tn),
        # Triage operating points read off the test curves:
        "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "precision_at_90pct_recall": float(prec[rec >= 0.90].max()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_confusion_matrices(cms: dict, path):
    """Grid of confusion-matrix heatmaps (counts + row %), one per model."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        row_pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap=SEQ_BLUE, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                dark = row_pct[i, j] > 0.55
                ax.text(j, i, f"{cm[i, j]:,}\n{row_pct[i, j]:.1%}", ha="center", va="center",
                        color="white" if dark else INK, fontsize=10)
        ax.set_xticks([0, 1], ["benign", "malicious"])
        ax.set_yticks([0, 1], ["benign", "malicious"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on the held-out test set (threshold 0.5)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc(curves: dict, path, rule_point=None):
    """ROC curves for the learned models; the rule baseline as a single point."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, zoom in zip(axes, (False, True)):
        for name, (fpr, tpr, auc) in curves.items():
            ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[name], label=f"{name} (AUC {auc:.3f})")
        if rule_point is not None:
            ax.plot(*rule_point, "o", ms=8, color=INK_2, mec=SURFACE, mew=2,
                    label="Rule baseline (operating point)")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRID)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        if zoom:
            ax.set_xlim(0, 0.10)
            ax.set_ylim(0.5, 1.0)
            ax.set_title("Zoom: low-FPR triage region")
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_title("ROC curve, held-out test set")
            ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(imp: pd.DataFrame, path, top: int = 15):
    """
    Two panels: random-forest permutation importance (drop in ROC-AUC) and
    logistic-regression standardized coefficients.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    rf = imp.sort_values("rf_permutation_mean", ascending=True).tail(top)
    axes[0].barh(rf.index, rf["rf_permutation_mean"], xerr=rf["rf_permutation_std"],
                 color=SERIES[1], height=0.6, error_kw={"ecolor": INK_2, "lw": 1})
    axes[0].set_title("Random forest: permutation importance")
    axes[0].set_xlabel("Mean drop in test ROC-AUC when feature is shuffled")
    axes[0].grid(axis="y", visible=False)

    lr = imp.reindex(imp["lr_coef"].abs().sort_values().index).tail(top)
    colors = [MAL_C if c > 0 else BENIGN_C for c in lr["lr_coef"]]
    axes[1].barh(lr.index, lr["lr_coef"], color=colors, height=0.6)
    axes[1].axvline(0, color=INK_2, lw=1)
    axes[1].set_title("Logistic regression: standardized coefficients")
    axes[1].set_xlabel("Coefficient (orange pushes toward malicious, blue toward benign)")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_distribution(feats: pd.DataFrame, y, path):
    """Histogram of filename-stem entropy, benign vs malicious (density)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, feats["name_entropy"].max() + 0.1, 45)
    for lab, name, c in ((0, "benign", BENIGN_C), (1, "malicious", MAL_C)):
        vals = feats.loc[np.asarray(y) == lab, "name_entropy"]
        ax.hist(vals, bins=bins, density=True, alpha=0.55, color=c, edgecolor=SURFACE, lw=0.6,
                label=f"{name} (n={len(vals):,}, median {vals.median():.2f})")
    ax.set_xlabel("Shannon entropy of filename stem (bits / character)")
    ax.set_ylabel("Density")
    ax.set_title("Filename entropy: benign vs malicious")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_evasion(ev: pd.DataFrame, path):
    """Grouped horizontal bars: recall on malicious test files under each attack."""
    models = [m for m in MODEL_COLORS if m in ev.columns]
    attacks = list(ev.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    h = 0.8 / len(models)
    y = np.arange(len(attacks))
    for k, m in enumerate(models):
        ax.barh(y + (k - (len(models) - 1) / 2) * h, ev[m], height=h * 0.85,
                color=MODEL_COLORS[m], label=m)
    ax.set_yticks(y, attacks)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall on malicious test files (threshold 0.5)")
    ax.set_title("Evasion stress test: recall after attacker-controlled changes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(models), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
