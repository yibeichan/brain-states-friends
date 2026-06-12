#!/usr/bin/env python
"""
08d_plots.py - cross-subject / cross-model aggregate panels for 08d results.

Loads per-lag D1 checkpoints produced by ``08d_transformer_depth.py`` and
emits three manuscript-oriented panel families:

* **P1** - per-subject depth curves (6-subject 2×3 small multiples), one file
  per model. Solid = D1 main at best lag (excluding lag 0); dashed = D1
  neg-control at the same lag; gray band = 95% permutation null CI.
* **P3** - peak-layer summary scatter: peak-layer fraction of network depth
  per (subject, model).
* **P5** - main vs neg-control accuracy/chance ratio scatter, with a caveat
  annotation about the heterogeneous ``run_onset_anchored`` label set
  (see ``the design notes``).

All panels save both PDF and PNG (dpi=300) under
``{SCRATCH_DIR}/output/08d_transformer_depth/_plots/`` by default. Runs on
CPU in under a few minutes for all 3 models × 6 subjects.

Gracefully handles in-flight partial coverage (llama sub-01/02/03/05 still
partial at time of writing): missing (lag, layer) cells are NaN, peak
extraction uses ``np.nanmax``, P1 annotates completion %, and P3/P5 skip
subject×model cells with no usable data.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.plot_style import (  # noqa: E402
    SUBJECT_MARKERS,
    TR_SECONDS,
    apply_publication_style,
)
from utils.transformer_analysis import load_d1_per_lag_matrix  # noqa: E402

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise RuntimeError("SCRATCH_DIR must be set in the environment / .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("08d_plots")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
ALL_MODELS = ["dinov2-large", "w2v-bert-2.0", "llama-3.2-3b"]
ALL_PANELS = ["P1", "P3", "P5"]

MODEL_N_LAYERS = {"dinov2-large": 24, "w2v-bert-2.0": 24, "llama-3.2-3b": 28}

#: Pretty labels for axis titles / legends
MODEL_LABELS = {
    "dinov2-large":  "DINOv2-large (video)",
    "w2v-bert-2.0":  "w2v-bert 2.0 (audio)",
    "llama-3.2-3b":  "LLaMA 3.2 3B (text)",
}

#: tab10[0:3] - consistent with matplotlib defaults; readable on white
MODEL_COLORS = {
    "dinov2-large":  "#1f77b4",
    "w2v-bert-2.0":  "#ff7f0e",
    "llama-3.2-3b":  "#2ca02c",
}

N_LAGS = 9
PEAK_LAG_EXCLUDE = {0}  # synchronous / autocorrelation diagnostic, not HRF peak

NEG_CONTROL_CAVEAT = (
    "Neg-control = run_onset_anchored states (heterogeneous: ab-common + a-anchored + b-anchored).\n"
    "a-anchored subcategory is contaminated by Friends theme-song stereotypy.\n"
    "Partial-effect variant (D1_neg_v2) pending - see 2026-04-20_neg_control_redesign.md."
)


# ---------------------------------------------------------------------------
# Data loading + peak extraction
# ---------------------------------------------------------------------------


def partials_dir_for(sub_id: str, model: str, parcellation: str) -> Path:
    return (
        Path(SCRATCH_DIR) / "output" / "08d_transformer_depth"
        / parcellation / sub_id / f"friends_{model}" / "partials"
    )


def load_sub_model(
    sub_id: str, model: str, parcellation: str,
) -> dict | None:
    """Load D1 main + neg-control matrices for one (subject, model). Return None if neither exists."""
    pdir = partials_dir_for(sub_id, model, parcellation)
    if not pdir.exists():
        return None
    n_layers = MODEL_N_LAYERS[model]
    main = load_d1_per_lag_matrix(str(pdir), "D1_main", N_LAGS, n_layers)
    neg = load_d1_per_lag_matrix(str(pdir), "D1_neg_control", N_LAGS, n_layers)
    if main["n_total"] == 0 and neg["n_total"] == 0:
        return None
    return {
        "sub": sub_id, "model": model, "n_layers": n_layers,
        "main": main, "neg": neg,
    }


def find_peak(matrices: dict, exclude_lag0: bool = True) -> dict | None:
    """Return (best_lag, best_layer, acc, chance, ratio, effect_size) at peak.

    Uses accuracy / chance as the peak criterion (comparable across
    subjects/models with different n_classes). Returns None if no usable cells.
    """
    acc = matrices["balanced_accuracy"]
    chance = matrices["chance_level"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = acc / chance
    if exclude_lag0:
        ratio_search = ratio.copy()
        ratio_search[0, :] = np.nan
    else:
        ratio_search = ratio
    if np.all(np.isnan(ratio_search)):
        return None
    flat_idx = np.nanargmax(ratio_search)
    lag, layer = np.unravel_index(flat_idx, ratio_search.shape)
    return {
        "lag": int(lag),
        "layer": int(layer),
        "acc": float(acc[lag, layer]),
        "chance": float(chance[lag, layer]),
        "ratio": float(ratio[lag, layer]),
        "effect_size": float(matrices["normalized_effect_size"][lag, layer]),
        "p_perm": float(matrices["p_perm"][lag, layer]),
    }


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------


def save_fig(fig, out_path: Path) -> None:
    """Save figure as both PDF and PNG. ``out_path`` is the stem (no suffix)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use string concatenation rather than Path.with_suffix - model names
    # like "w2v-bert-2.0" or "llama-3.2-3b" contain dots that with_suffix
    # would strip as a fake extension.
    pdf_path = out_path.parent / f"{out_path.name}.pdf"
    png_path = out_path.parent / f"{out_path.name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    logger.info("wrote %s.{pdf,png}", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# P1 - per-subject depth curves, one file per model
# ---------------------------------------------------------------------------


def plot_p1(
    results_by_sub: dict[str, dict],
    model: str,
    out_dir: Path,
) -> None:
    n_layers = MODEL_N_LAYERS[model]
    fig, axes = plt.subplots(
        2, 3, figsize=(11, 6.5), sharex=True, sharey=True,
    )
    fig.suptitle(
        f"D1 depth profile - {MODEL_LABELS[model]} (content_eligible)",
        y=1.00, fontsize=12,
    )
    layers_x = np.arange(n_layers)

    for ax, sub in zip(axes.flat, ALL_SUBJECTS):
        res = results_by_sub.get(sub)
        if res is None:
            ax.set_title(f"{sub} - no data", fontsize=9)
            ax.set_xticks([0, n_layers // 2, n_layers - 1])
            continue

        main_peak = find_peak(res["main"])
        if main_peak is None:
            ax.set_title(f"{sub} - no usable cells", fontsize=9)
            continue

        lag_star = main_peak["lag"]
        main_acc = res["main"]["balanced_accuracy"][lag_star]
        main_ch = res["main"]["chance_level"][lag_star]
        null_mean = res["main"]["null_mean"][lag_star]
        null_std = res["main"]["null_std"][lag_star]
        neg_acc = res["neg"]["balanced_accuracy"][lag_star]
        neg_ch = res["neg"]["chance_level"][lag_star]

        with np.errstate(invalid="ignore", divide="ignore"):
            main_ratio = main_acc / main_ch
            neg_ratio = neg_acc / neg_ch
            null_hi = (null_mean + 1.96 * null_std) / main_ch
            null_lo = (null_mean - 1.96 * null_std) / main_ch

        # Null CI band (skip where null fields missing)
        valid_band = ~np.isnan(null_hi) & ~np.isnan(null_lo)
        if valid_band.any():
            ax.fill_between(
                layers_x, null_lo, null_hi,
                where=valid_band, step=None,
                color="gray", alpha=0.2, linewidth=0,
                label="95% null CI",
            )
        # Chance = 1.0 in ratio space
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

        ax.plot(
            layers_x, main_ratio,
            color=MODEL_COLORS[model], linewidth=1.8,
            label="content_eligible",
        )
        if not np.all(np.isnan(neg_ratio)):
            ax.plot(
                layers_x, neg_ratio,
                color=MODEL_COLORS[model], linewidth=1.2,
                linestyle="--", alpha=0.75,
                label="run_onset_anchored",
            )

        # Mark peak layer
        ax.axvline(
            main_peak["layer"], color=MODEL_COLORS[model],
            linewidth=0.7, alpha=0.4,
        )

        title = (
            f"{sub} · lag={lag_star} TRs ({lag_star * TR_SECONDS:.1f}s) · "
            f"peak L{main_peak['layer']}/{n_layers}"
        )
        ax.set_title(title, fontsize=9)

        # Coverage annotation (bottom-right)
        n_done = res["main"]["n_total"]
        n_expected = N_LAGS * n_layers
        if n_done < n_expected:
            ax.text(
                0.98, 0.03,
                f"{n_done}/{n_expected} layers",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="firebrick",
            )

    for i, ax in enumerate(axes.flat):
        if i >= 3:
            ax.set_xlabel("Layer index")
        if i % 3 == 0:
            ax.set_ylabel("Accuracy / chance")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    # One shared legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02), frameon=False,
        )

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_fig(fig, out_dir / f"P1_depth_curves_{model}")


# ---------------------------------------------------------------------------
# P3 - peak-layer summary scatter (peak layer as fraction of network depth)
# ---------------------------------------------------------------------------


def plot_p3(results_all: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    sub_x = {sub: i for i, sub in enumerate(ALL_SUBJECTS)}

    model_fractions = {m: [] for m in ALL_MODELS}
    jitter = {"dinov2-large": -0.18, "w2v-bert-2.0": 0.0, "llama-3.2-3b": 0.18}

    for model in ALL_MODELS:
        n_layers = MODEL_N_LAYERS[model]
        for sub in ALL_SUBJECTS:
            res = results_all.get((sub, model))
            if res is None:
                continue
            peak = find_peak(res["main"])
            if peak is None:
                continue
            frac = peak["layer"] / max(n_layers - 1, 1)
            model_fractions[model].append(frac)
            ax.scatter(
                sub_x[sub] + jitter[model],
                frac,
                marker=SUBJECT_MARKERS[sub],
                s=70,
                color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.6,
                zorder=3,
            )

    # Median horizontal lines per model
    for model in ALL_MODELS:
        fracs = model_fractions[model]
        if not fracs:
            continue
        med = float(np.median(fracs))
        ax.axhline(
            med, color=MODEL_COLORS[model],
            linestyle=":", linewidth=1.0, alpha=0.55,
        )
        ax.text(
            len(ALL_SUBJECTS) - 0.4, med,
            f" med={med:.2f}",
            color=MODEL_COLORS[model],
            fontsize=7, va="center",
        )

    ax.set_xticks(range(len(ALL_SUBJECTS)))
    ax.set_xticklabels(ALL_SUBJECTS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Peak layer / (n_layers − 1)")
    ax.set_title(
        "P3 · Peak layer for content_eligible decoding, normalized by network depth"
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    # Legends: model colors + subject markers
    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in ALL_MODELS if model_fractions[m]
    ]
    coverage_note = []
    for m in ALL_MODELS:
        k = len(model_fractions[m])
        if k < len(ALL_SUBJECTS):
            coverage_note.append(f"{m.split('-')[0]}: {k}/{len(ALL_SUBJECTS)}")
    if coverage_note:
        fig.text(
            0.99, 0.01, "partial coverage → " + ", ".join(coverage_note),
            ha="right", va="bottom", fontsize=7, color="dimgray",
        )

    ax.legend(
        handles=model_handles, loc="lower left",
        fontsize=8, frameon=False,
    )

    fig.tight_layout()
    save_fig(fig, out_dir / "P3_peak_layer_summary")


# ---------------------------------------------------------------------------
# P5 - main vs neg-control ratio scatter (with caveat)
# ---------------------------------------------------------------------------


def plot_p5(results_all: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    xs, ys = [], []
    for model in ALL_MODELS:
        for sub in ALL_SUBJECTS:
            res = results_all.get((sub, model))
            if res is None:
                continue
            main_peak = find_peak(res["main"])
            neg_peak = find_peak(res["neg"])
            if main_peak is None or neg_peak is None:
                continue
            x = neg_peak["ratio"]
            y = main_peak["ratio"]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xs.append(x); ys.append(y)
            ax.scatter(
                x, y,
                marker=SUBJECT_MARKERS[sub],
                s=85,
                color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.6,
                zorder=3,
            )

    if xs:
        lo = min(min(xs), min(ys), 1.0) * 0.9
        hi = max(max(xs), max(ys)) * 1.05
        diag = np.array([lo, hi])
        ax.plot(diag, diag, color="gray", linestyle="--",
                linewidth=1.0, alpha=0.7, zorder=1, label="y = x")
        ax.axhline(1.0, color="lightgray", linewidth=0.5, zorder=0)
        ax.axvline(1.0, color="lightgray", linewidth=0.5, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("neg-control peak accuracy / chance")
    ax.set_ylabel("main peak accuracy / chance")
    ax.set_title(
        "P5 · content_eligible vs run_onset_anchored decodability (ratio form)"
    )
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Legend: models - place outside the axes on the right so it doesn't
    # occlude data points in the upper-left region.
    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in ALL_MODELS
    ]
    ax.legend(
        handles=model_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=False,
    )

    # Caveat box - below the plot area (outside axes) to keep data visible.
    fig.text(
        0.5, -0.04, NEG_CONTROL_CAVEAT,
        ha="center", va="top",
        fontsize=7, color="dimgray",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="lightgray", alpha=0.95,
        ),
    )

    fig.tight_layout()
    save_fig(fig, out_dir / "P5_main_vs_neg_scatter")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="08d aggregate plots (P1/P3/P5) across subjects and models.",
    )
    p.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS,
                   help=f"Subjects (default: {' '.join(ALL_SUBJECTS)})")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, help="Models to plot.")
    p.add_argument("--panels", nargs="+", default=ALL_PANELS,
                   choices=ALL_PANELS, help="Which panels to emit.")
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95", help="Kept for future VT-aware paths; not used.")
    p.add_argument(
        "--output_dir", default=None,
        help="Output dir (default: $SCRATCH_DIR/output/08d_transformer_depth/_plots/).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    apply_publication_style()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(SCRATCH_DIR) / "output" / "08d_transformer_depth" / "_plots"
    )
    logger.info("output dir: %s", out_dir)

    # Load every (sub, model) combo requested, even those with partial data.
    results_all: dict[tuple[str, str], dict] = {}
    for model in args.models:
        for sub in args.subjects:
            res = load_sub_model(sub, model, args.parcellation)
            if res is None:
                logger.warning("%s × %s: no data - skipped", sub, model)
                continue
            main_done = res["main"]["n_total"]
            neg_done = res["neg"]["n_total"]
            expected = N_LAGS * MODEL_N_LAYERS[model]
            logger.info(
                "%s × %s: main %d/%d, neg %d/%d",
                sub, model, main_done, expected, neg_done, expected,
            )
            results_all[(sub, model)] = res

    if not results_all:
        logger.error("no data loaded for any subject×model - exiting")
        sys.exit(1)

    if "P1" in args.panels:
        for model in args.models:
            by_sub = {
                sub: results_all[(sub, model)]
                for sub in args.subjects if (sub, model) in results_all
            }
            if not by_sub:
                logger.warning("P1 %s: no usable subjects", model)
                continue
            plot_p1(by_sub, model, out_dir)

    if "P3" in args.panels:
        plot_p3(results_all, out_dir)

    if "P5" in args.panels:
        plot_p5(results_all, out_dir)

    logger.info("done - wrote %s", out_dir)


if __name__ == "__main__":
    main()
