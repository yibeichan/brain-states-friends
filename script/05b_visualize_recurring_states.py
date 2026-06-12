#!/usr/bin/env python
"""
05b: Visualize Top Recurring Brain States with yabplot (Cortical + Subcortical)

Creates a multi-panel figure showing the top recurring brain states with
cortical AND subcortical visualizations side-by-side, using yabplot for
headless PyVista-based rendering.

Layout per state row:
    Col 0 (width 2.5): Cortical surface (yab.plot_cortical, schaefer_100)
    Col 1 (width 1.5): Subcortical 3D   (yab.plot_subcortical, 4s156_subcortical)
    Col 2 (width 1.0): Metrics text (recurrence score, run spread)
    Col 3 (width 0.8): Top runs horizontal bar chart

Shared color range: 95th percentile of absolute z-scores across all top-N states,
applied identically to both cortical and subcortical panels.

Prerequisites:
  - 05a_recurrence_analysis.py completed (recurrence_summary.json, fractional_occupancy.pkl)
  - 04_combined_hdphmm.py (select mode) completed (state_means_parcel.npy)
  - yabplot installed (declared as a git dependency; see https://github.com/yibeichan/yabplot)
  - Subcortical VTK meshes built via yabplot's tools/build_4s156_subcortical.py

Subcortical interpretation note:
  Subcortical parcels are anatomically-defined (CIT168, HCP), not functionally-
  derived like the Schaefer-100 cortical parcels. Hippocampus and amygdala have
  ~50-70% lower tSNR due to susceptibility artifacts; interpret those conservatively.
"""

import os
import sys
import pickle
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from dotenv import load_dotenv

# Headless PyVista must be configured BEFORE importing yabplot
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.viz_yabplot import (
    setup_yabplot_headless,
    load_parcel_labels,
    pattern_to_cortical_dict,
    pattern_to_subcortical_dict,
    get_cortical_atlas_dir,
    get_subcortical_atlas_dir,
    render_cortical_to_image,
    render_subcortical_to_image,
)
from utils.common import normalize_parcellation_name
from utils.plot_style import apply_publication_style

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

apply_publication_style()


def _select_top_state_keys(recurring_states: Dict, n_states: int) -> List[str]:
    """Return top-N recurring state keys sorted by recurrence score."""
    if n_states < 1:
        raise ValueError(f"--n_states must be >= 1, got {n_states}")
    if not recurring_states:
        raise ValueError("No recurring states found in input summary.")

    sorted_keys = sorted(
        recurring_states.keys(),
        key=lambda k: recurring_states[k]["recurrence_score"],
        reverse=True,
    )[:n_states]
    if not sorted_keys:
        raise ValueError("No recurring states available after top-N selection.")
    return sorted_keys


def _shared_color_range(recurring_states: Dict, sorted_keys: List[str]) -> Tuple[float, float]:
    """Compute robust symmetric color range from the 95th percentile of |z|."""
    all_values: List[float] = []
    for key in sorted_keys:
        pattern = np.asarray(recurring_states[key]["pattern"]).ravel()
        all_values.extend(pattern.tolist())
    if not all_values:
        raise ValueError("Cannot compute color range: selected states contain no pattern values.")

    vmax = float(np.percentile(np.abs(all_values), 95))
    if (not np.isfinite(vmax)) or vmax <= 0:
        logger.warning("Computed non-positive/invalid vmax (%.6f); using epsilon=1e-6", vmax)
        vmax = 1e-6
    return (-vmax, vmax)


def validate_cortical_label_alignment(labels_df, parcellation: str) -> None:
    """Fail fast if TSV cortical labels do not match the custom schaefer100 LUT."""
    match = re.search(r"atlas-4S(\d+)Parcels", parcellation)
    if not match:
        raise ValueError(
            f"Expected parcellation format 'atlas-4S<N>Parcels', got {parcellation!r}"
        )
    total_parcels = int(match.group(1))
    cortical_threshold = total_parcels - 56
    if cortical_threshold <= 0:
        raise ValueError(
            f"Invalid cortical threshold ({cortical_threshold}) for parcellation {parcellation!r}"
        )

    tsv_labels = set(
        labels_df.loc[labels_df["index"] <= cortical_threshold, "label_7network"].astype(str).tolist()
    )
    if not tsv_labels:
        raise ValueError("No cortical labels found in atlas TSV.")

    import yabplot as yab
    # Read labels from the custom atlas LUT (built from the same CIFTI, so names match TSV)
    atlas_labels = set(
        yab.get_atlas_regions(
            atlas="schaefer100_cortical",
            category="cortical",
            custom_atlas_path=get_cortical_atlas_dir(),
        )
    )
    missing = sorted(tsv_labels - atlas_labels)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Cortical label mismatch with custom schaefer100 atlas LUT. "
            f"Missing {len(missing)} labels (first 10: {preview}). "
            "Rebuild the atlas with yabplot's tools/build_schaefer100_cortical.py"
        )

    logger.info(
        "Validated cortical label alignment: %d TSV labels match custom schaefer100 LUT.",
        len(tsv_labels),
    )


def create_multipanel_figure(
    brain_patterns: Dict,
    labels_df,
    parcellation: str,
    subcortical_atlas_dir: str,
    output_path: str,
    subject_id: str,
    n_states: int = 5,
) -> None:
    """
    Create a multi-panel figure (top-N recurring states, 4 columns per row).

    Shared color range computed from 95th percentile of |values| across all states.
    """
    recurring_states = brain_patterns["active_states"]
    sorted_keys = _select_top_state_keys(recurring_states, n_states)
    n_rows = len(sorted_keys)
    color_range = _shared_color_range(recurring_states, sorted_keys)
    vmax = color_range[1]
    logger.info(f"Shared color range: ({-vmax:.3f}, {vmax:.3f})")

    fig_height = 4.0 * n_rows
    fig = plt.figure(figsize=(20, fig_height))

    gs = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        width_ratios=[2.5, 1.5, 1.0, 0.8],
        height_ratios=[1] * n_rows,
        hspace=0.25, wspace=0.15,
    )

    for idx, key in enumerate(sorted_keys):
        state_data = recurring_states[key]
        pattern = state_data["pattern"]
        recurrence_score = state_data["recurrence_score"]
        run_spread = state_data["run_spread"]
        top_runs = state_data.get("top_runs", [])
        state_id = state_data.get("state_id", "N/A")

        logger.info(f"Rendering state {idx + 1}/{n_rows}: {key} (ID {state_id})")

        # Build named-key dicts (resolves ordering mismatch risk - stats agent)
        cortical_dict = pattern_to_cortical_dict(pattern, labels_df, parcellation)
        subcortical_dict = pattern_to_subcortical_dict(pattern, labels_df, parcellation)

        # --- Column 0: Cortical surface ---
        ax_cortical = fig.add_subplot(gs[idx, 0])
        try:
            cortical_img = render_cortical_to_image(cortical_dict, color_range)
            ax_cortical.imshow(cortical_img)
        except Exception as e:
            logger.error(f"Cortical render failed for {key}: {e}")
            ax_cortical.text(
                0.5, 0.5, f"Cortical render\nerror:\n{e}",
                transform=ax_cortical.transAxes, ha="center", va="center",
                fontsize=8, color="red",
            )
        ax_cortical.axis("off")
        ax_cortical.text(
            0.02, 0.95, f"#{idx + 1}", transform=ax_cortical.transAxes,
            fontsize=18, fontweight="bold", color="black",
            verticalalignment="top",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="gold",
                      edgecolor="black", linewidth=2),
        )

        # --- Column 1: Subcortical 3D ---
        ax_subcortical = fig.add_subplot(gs[idx, 1])
        try:
            subcortical_img = render_subcortical_to_image(
                subcortical_dict, color_range, subcortical_atlas_dir
            )
            ax_subcortical.imshow(subcortical_img)
        except Exception as e:
            logger.error(f"Subcortical render failed for {key}: {e}")
            ax_subcortical.text(
                0.5, 0.5, f"Subcortical render\nerror:\n{e}",
                transform=ax_subcortical.transAxes, ha="center", va="center",
                fontsize=8, color="red",
            )
        ax_subcortical.axis("off")
        ax_subcortical.set_title("Subcortical*", fontsize=8, color="gray", pad=2)

        # --- Column 2: Metrics ---
        ax_metrics = fig.add_subplot(gs[idx, 2])
        ax_metrics.axis("off")
        metrics_text = (
            f"HMM State: {state_id}\n\n"
            f"Recurrence:\n{recurrence_score:.1%}\n\n"
            f"Run spread:\n{run_spread} runs"
        )
        ax_metrics.text(
            0.1, 0.85, metrics_text, transform=ax_metrics.transAxes,
            fontsize=12, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray",
                      alpha=0.3, edgecolor="gray"),
        )

        # --- Column 3: Top runs bar chart ---
        ax_runs = fig.add_subplot(gs[idx, 3])
        if top_runs:
            runs = [ep for ep, _ in top_runs[:5]]
            fo_pct = [val * 100 for _, val in top_runs[:5]]
            y_pos = range(len(runs))
            bars = ax_runs.barh(y_pos, fo_pct, color="steelblue", alpha=0.8)
            ax_runs.set_yticks(y_pos)
            ax_runs.set_yticklabels(runs, fontsize=8)
            ax_runs.set_xlabel("FO (%)", fontsize=9)
            ax_runs.set_title("Top Runs", fontsize=10, fontweight="bold")
            ax_runs.invert_yaxis()
            for bar, pct in zip(bars, fo_pct):
                ax_runs.text(
                    bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8,
                )
        else:
            ax_runs.text(
                0.5, 0.5, "No run\ndata",
                transform=ax_runs.transAxes, ha="center", va="center",
                fontsize=10, color="gray",
            )
            ax_runs.axis("off")

    # Main title
    fig.suptitle(
        f"Top {n_rows} States by Recurrence Score - {subject_id}\n"
        f"Cortical (Schaefer-100, functionally-derived) | "
        f"Subcortical* (anatomically-defined: CIT168, HCP thalamus, Hippo/Amyg, Cerebellum)",
        fontsize=13, fontweight="bold", y=0.998,
    )

    # Shared colorbar
    cax = fig.add_axes([0.12, 0.01, 0.28, 0.012])
    import matplotlib.cm as cm
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=color_range[0], vmax=color_range[1]),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Activation (z-score)", fontsize=11, fontweight="bold")

    # Supplementary figure footnote.
    # Marks this as a supplementary diagnostic (not a primary interpretation figure)
    # and states the parcellation logic difference, as required by the 05b review plan.
    fig.text(
        0.5, 0.005,
        "Supplementary figure - for exploration and manual review, not a primary state-interpretation result. "
        "Cortical parcels (Schaefer-100) are functionally-derived; subcortical parcels (CIT168, HCP) are "
        "anatomically-defined. The two systems differ in parcellation logic and reliability and should not "
        "be treated as directly rank-comparable on a shared activation scale. "
        "*Hippo/amygdala: ~50–70% lower tSNR due to susceptibility artifacts; interpret conservatively.",
        ha="center", fontsize=7.5, color="gray", style="italic",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    logger.info(f"Saved multi-panel figure: {output_path}")
    logger.info(f"Saved PDF: {pdf_path}")


def create_individual_state_plots(
    brain_patterns: Dict,
    labels_df,
    parcellation: str,
    subcortical_atlas_dir: str,
    output_dir: str,
    subject_id: str,
    n_states: int = 10,
) -> None:
    """
    Save individual high-resolution cortical+subcortical plots per recurring state.
    """
    recurring_states = brain_patterns["active_states"]
    sorted_keys = _select_top_state_keys(recurring_states, n_states)
    color_range = _shared_color_range(recurring_states, sorted_keys)

    for idx, key in enumerate(sorted_keys):
        state_data = recurring_states[key]
        pattern = state_data["pattern"]
        recurrence_score = state_data["recurrence_score"]
        run_spread = state_data["run_spread"]
        state_id = state_data.get("state_id", idx)

        logger.info(f"Individual plot {idx + 1}/{len(sorted_keys)}: {key}")

        cortical_dict = pattern_to_cortical_dict(pattern, labels_df, parcellation)
        subcortical_dict = pattern_to_subcortical_dict(pattern, labels_df, parcellation)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Cortical
        try:
            cortical_img = render_cortical_to_image(
                cortical_dict, color_range, figsize=(1200, 500)
            )
            axes[0].imshow(cortical_img)
            axes[0].set_title("Cortical (Schaefer-100)", fontsize=12)
        except Exception as e:
            logger.error(f"Cortical render failed: {e}")
            axes[0].text(0.5, 0.5, f"Render error: {e}", ha="center", va="center")
        axes[0].axis("off")

        # Subcortical
        try:
            subcortical_img = render_subcortical_to_image(
                subcortical_dict, color_range, subcortical_atlas_dir,
                figsize=(800, 600)
            )
            axes[1].imshow(subcortical_img)
            axes[1].set_title(
                "Subcortical* (CIT168, HCP thalamus, Hippo/Amyg, Cerebellum)",
                fontsize=11,
            )
        except Exception as e:
            logger.error(f"Subcortical render failed: {e}")
            axes[1].text(0.5, 0.5, f"Render error: {e}", ha="center", va="center")
        axes[1].axis("off")

        fig.suptitle(
            f"Recurring State #{idx + 1} (ID {state_id}) - {subject_id}\n"
            f"Recurrence: {recurrence_score:.1%} | Runs: {run_spread}\n"
            f"Supplementary. Cortical: functionally-derived (Schaefer-100); "
            f"Subcortical*: anatomically-defined (CIT168, HCP) - not rank-comparable on a shared scale.\n"
            f"*Hippo/amygdala: lower tSNR, interpret conservatively.",
            fontsize=10, fontweight="bold",
        )

        out_file = os.path.join(
            output_dir, f"recurring_state_{idx + 1:02d}_state{state_id}.png"
        )
        fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved: {out_file}")


def _save_render_image(image: np.ndarray, output_path: str, figsize: Tuple[float, float]) -> None:
    """Save a rendered RGB array to disk without axes or extra padding."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _save_metrics_panel(
    output_path: str,
    state_rank: int,
    state_id: str,
    recurrence_score: float,
    run_spread: int,
) -> None:
    """Save the metrics text box as a standalone high-resolution panel."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.axis("off")
    metrics_text = (
        f"Top recurring state #{state_rank}\n\n"
        f"HMM state ID: {state_id}\n\n"
        f"Recurrence:\n{recurrence_score:.1%}\n\n"
        f"Run spread:\n{run_spread} runs"
    )
    ax.text(
        0.08, 0.92, metrics_text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="gray", linewidth=1.5),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _save_top_runs_panel(output_path: str, top_runs: List[Tuple[str, float]]) -> None:
    """Save the top runs horizontal bar chart as a standalone panel."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if top_runs:
        runs = [ep for ep, _ in top_runs[:5]]
        fo_pct = [val * 100 for _, val in top_runs[:5]]
        y_pos = np.arange(len(runs))
        bars = ax.barh(y_pos, fo_pct, color="steelblue", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(runs, fontsize=12)
        ax.set_xlabel("Fractional occupancy (%)", fontsize=13)
        ax.set_title("Top Runs", fontsize=15, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        xmax = max(fo_pct) * 1.15 if fo_pct else 1.0
        ax.set_xlim(0, xmax)
        for bar, pct in zip(bars, fo_pct):
            ax.text(
                bar.get_width() + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=12,
            )
    else:
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No run data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="gray",
        )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _save_shared_colorbar(output_path: str, color_range: Tuple[float, float]) -> None:
    """Save the shared activation colorbar as a standalone panel."""
    fig, ax = plt.subplots(figsize=(8, 1.4))
    sm = cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=color_range[0], vmax=color_range[1]),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Activation (z-score)", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def export_poster_components(
    brain_patterns: Dict,
    labels_df,
    parcellation: str,
    subcortical_atlas_dir: str,
    output_dir: str,
    n_states: int = 5,
) -> None:
    """
    Export each multipanel component as a standalone high-resolution file.

    Outputs are written flat into `output_dir` for easy poster assembly.
    """
    recurring_states = brain_patterns["active_states"]
    sorted_keys = _select_top_state_keys(recurring_states, n_states)
    color_range = _shared_color_range(recurring_states, sorted_keys)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_shared_colorbar(os.path.join(output_dir, "05b_shared_colorbar.png"), color_range)

    for rank, key in enumerate(sorted_keys, start=1):
        state_data = recurring_states[key]
        pattern = state_data["pattern"]
        recurrence_score = float(state_data["recurrence_score"])
        run_spread = int(state_data["run_spread"])
        top_runs = state_data.get("top_runs", [])
        state_id = str(state_data.get("state_id", key))

        logger.info("Exporting poster components for %s (rank %d, state ID %s)", key, rank, state_id)

        cortical_dict = pattern_to_cortical_dict(pattern, labels_df, parcellation)
        subcortical_dict = pattern_to_subcortical_dict(pattern, labels_df, parcellation)
        file_stub = f"05b_state{rank:02d}_id{state_id}"

        cortical_img = render_cortical_to_image(
            cortical_dict,
            color_range,
            figsize=(2400, 900),
        )
        _save_render_image(
            cortical_img,
            os.path.join(output_dir, f"{file_stub}_cortical.png"),
            figsize=(16, 6),
        )

        subcortical_img = render_subcortical_to_image(
            subcortical_dict,
            color_range,
            subcortical_atlas_dir,
            figsize=(1800, 1400),
        )
        _save_render_image(
            subcortical_img,
            os.path.join(output_dir, f"{file_stub}_subcortical.png"),
            figsize=(10, 8),
        )

        _save_metrics_panel(
            os.path.join(output_dir, f"{file_stub}_metrics.png"),
            state_rank=rank,
            state_id=state_id,
            recurrence_score=recurrence_score,
            run_spread=run_spread,
        )

        _save_top_runs_panel(
            os.path.join(output_dir, f"{file_stub}_top_runs.png"),
            top_runs=top_runs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top recurring brain states with cortical + subcortical panels (yabplot)"
    )
    parser.add_argument("--sub_id", type=str, required=True, help="Subject ID (e.g. sub-01)")
    parser.add_argument("--parcellation", type=str, required=True,
                        help="Parcellation name (e.g. atlas-4S156Parcels or 4S156)")
    parser.add_argument("--n_states", type=int, default=5,
                        help="Number of top recurring states to visualize (default: 5)")
    parser.add_argument(
        "--poster_output_dir",
        type=str,
        default=None,
        help="Optional flat output folder for poster-ready standalone components.",
    )
    parser.add_argument(
        "--vt", type=str, default=None,
        help="Variance threshold subdirectory under final/ (e.g., 0.99). "
             "Reads from final/vt{VT}/. If omitted, reads from final/ directly (legacy path).",
    )
    args = parser.parse_args()
    if args.n_states < 1:
        raise ValueError(f"--n_states must be >= 1, got {args.n_states}")

    # Configure headless rendering before any yabplot/pyvista imports
    setup_yabplot_headless()

    parcellation = normalize_parcellation_name(args.parcellation)
    SCRATCH_DIR = os.getenv("SCRATCH_DIR")
    if not SCRATCH_DIR:
        raise EnvironmentError("SCRATCH_DIR not set. Source .env or set the variable.")

    # Input paths - must match 05a's vt-aware output layout
    recurrence_dir = os.path.join(
        SCRATCH_DIR, "output", "05a_recurrence_analysis", parcellation, args.sub_id
    )
    if args.vt is not None:
        recurrence_dir = os.path.join(recurrence_dir, f"vt{args.vt}")
    summary_file = os.path.join(recurrence_dir, "recurrence_summary.json")
    fo_file = os.path.join(recurrence_dir, "fractional_occupancy.pkl")

    if args.vt is not None:
        hmm_dir = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation, args.sub_id,
            "final", f"vt{args.vt}"
        )
    else:
        hmm_dir = os.path.join(
            SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation, args.sub_id, "final"
        )
    state_means_file = os.path.join(hmm_dir, "state_means_parcel.npy")

    output_dir = os.path.join(
        SCRATCH_DIR, "output", "05b_recurring_states_visualization", parcellation, args.sub_id
    )
    if args.vt is not None:
        output_dir = os.path.join(output_dir, f"vt{args.vt}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("05b: RECURRING BRAIN STATES - yabplot (cortical + subcortical)")
    logger.info("=" * 70)
    logger.info(f"Subject:       {args.sub_id}")
    logger.info(f"Parcellation:  {parcellation}")
    logger.info(f"Top N states:  {args.n_states}")
    logger.info(f"Output dir:    {output_dir}")
    logger.info("=" * 70)

    # Validate inputs
    for path, label in [
        (summary_file, "recurrence_summary.json"),
        (fo_file, "fractional_occupancy.pkl"),
        (state_means_file, "state_means_parcel.npy"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{label} not found: {path}")
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load inputs
    logger.info("Loading recurrence summary and brain state patterns...")
    with open(summary_file, "r") as f:
        summary = json.load(f)
    with open(fo_file, "rb") as f:
        fo = pickle.load(f)
    state_means = np.load(state_means_file)  # (n_states, n_parcels)

    all_scores = summary["recurrence_scores"]
    n_runs = summary["n_runs"]
    # Active states: any state with recurrence score > 0
    active_indices = [i for i, s in enumerate(all_scores) if s > 0]
    logger.info(f"Found {len(active_indices)} active states (score > 0), {n_runs} runs")

    # Build brain_patterns dict for visualization
    brain_patterns = {"active_states": {}}
    for idx in active_indices:
        score = float(all_scores[idx])
        spread = int(round(score * n_runs))
        pattern = state_means[idx]  # shape (n_parcels,)

        state_fos = [(run_id, fo[run_id][idx]) for run_id in fo]
        state_fos.sort(key=lambda x: x[1], reverse=True)
        top_runs = [(run_id, val) for run_id, val in state_fos[:5] if val > 0]

        brain_patterns["active_states"][f"state_{idx}"] = {
            "pattern": pattern,
            "recurrence_score": score,
            "run_spread": spread,
            "state_id": str(idx),
            "top_runs": top_runs,
        }

    # Load atlas parcel labels
    logger.info("Loading parcel labels...")
    labels_df = load_parcel_labels(parcellation)
    validate_cortical_label_alignment(labels_df, parcellation)

    # Get pre-built subcortical atlas dir
    logger.info("Locating subcortical atlas VTK files...")
    subcortical_atlas_dir = get_subcortical_atlas_dir()

    # Multi-panel figure
    logger.info(f"Creating multi-panel figure (top {args.n_states} states)...")
    multipanel_path = os.path.join(
        output_dir, f"{args.sub_id}_top{args.n_states}_recurring_states.png"
    )
    create_multipanel_figure(
        brain_patterns, labels_df, parcellation, subcortical_atlas_dir,
        multipanel_path, args.sub_id, n_states=args.n_states,
    )

    # Individual state plots (2× n_states)
    logger.info("Creating individual state plots...")
    individual_dir = os.path.join(output_dir, "individual_states")
    Path(individual_dir).mkdir(parents=True, exist_ok=True)
    create_individual_state_plots(
        brain_patterns, labels_df, parcellation, subcortical_atlas_dir,
        individual_dir, args.sub_id, n_states=args.n_states * 2,
    )

    if args.poster_output_dir:
        logger.info("Exporting standalone poster components...")
        export_poster_components(
            brain_patterns,
            labels_df,
            parcellation,
            subcortical_atlas_dir,
            args.poster_output_dir,
            n_states=args.n_states,
        )

    logger.info("=" * 70)
    logger.info("05b visualization complete!")
    logger.info(f"Multi-panel figure: {multipanel_path}")
    logger.info(f"Individual plots:   {individual_dir}/")
    if args.poster_output_dir:
        logger.info(f"Poster components: {args.poster_output_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
