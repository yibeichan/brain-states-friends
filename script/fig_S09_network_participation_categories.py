#!/usr/bin/env python3
"""Supplementary Figure S9 - all-category canonical-network participation.

Applies the *same* metric definitions as the main content-eligible Figure 2C
(``fig_F2_network_participation.py``) to every recurrence-screening category in
``state_flags.csv``. The output is descriptive provenance/context, not a new main
scientific claim.

Scope and framing (see plan
``docs/supplementary/2026-06-14_network_participation_categories_plan.md``):
  - Canonical networks are an annotation frame for parcel-space fitted state maps.
  - Categories are provenance/screening labels, not cognitive state classes.
  - Do not interpret unused/rare/low-confidence/season-temporal categories
    biologically.
  - Metrics use ``abs(state_mean)`` -> participation magnitude, not polarity.
  - "Entropy" is spread across canonical network labels, not temporal entropy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.network_participation import (  # noqa: E402
    compute_network_participation_metrics,
    plot_network_participation_by_category,
    save_network_participation_outputs,
    summarize_network_participation,
)
from utils.plot_style import (  # noqa: E402
    NETWORK_ORDER,
    SUBJECT_MARKERS,
    apply_publication_style,
    display_network,
    load_parcel_networks,
)

# Display the screening categories in a stable, readable order: content-eligible
# first, then provenance/QC buckets. Labels are descriptive only.
CATEGORY_ORDER = [
    "eligible_for_content_analysis",
    "run_onset_anchored",
    "low_confidence",
    "unused",
    "rare",
    "season_temporal",
]
CATEGORY_LABELS = {
    "eligible_for_content_analysis": "Content-eligible",
    "run_onset_anchored": "Run-onset anchored",
    "low_confidence": "Low confidence",
    "unused": "Unused",
    "rare": "Rare",
    "season_temporal": "Season/temporal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the supplementary all-category network-participation figure."
    )
    parser.add_argument("--parcellation", default="atlas-4S156Parcels")
    parser.add_argument("--vt", default="vt0.95")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"sub-0{i}" for i in range(1, 7)],
    )
    parser.add_argument(
        "--expected-states",
        type=int,
        default=300,
        help="Set to -1 to disable the all-category state-count check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    apply_publication_style()

    scratch_dir = Path(os.environ["SCRATCH_DIR"])
    flags_dir = scratch_dir / "output" / "05e_temporal_trend_a4" / args.parcellation
    model_dir = scratch_dir / "output" / "04_combined_hdphmm" / args.parcellation
    out_dir = (
        scratch_dir
        / "output"
        / "manuscript_figures"
        / "supp_network_participation_categories"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state_flags = {
        subject: pd.read_csv(flags_dir / subject / args.vt / "state_flags.csv")
        for subject in args.subjects
    }
    state_means = {
        subject: np.load(
            model_dir / subject / "final" / args.vt / "state_means_parcel.npy"
        )
        for subject in args.subjects
    }
    parcel_networks = load_parcel_networks(args.parcellation)
    if parcel_networks is None:
        raise RuntimeError(f"Could not load parcel networks for {args.parcellation}")

    # summary_categories=None keeps every recurrence-screening category.
    metrics, _ = compute_network_participation_metrics(
        state_flags=state_flags,
        state_means=state_means,
        subjects=args.subjects,
        parcel_networks=parcel_networks,
        network_order=NETWORK_ORDER,
        summary_categories=None,
    )
    if args.expected_states >= 0 and len(metrics) != args.expected_states:
        raise RuntimeError(
            f"Expected {args.expected_states} states across all categories, "
            f"found {len(metrics)}"
        )

    summary = summarize_network_participation(metrics, NETWORK_ORDER)
    # Report category counts in the JSON top level for quick provenance checks.
    summary["category_counts"] = {
        str(cat): int((metrics["summary_category"] == cat).sum())
        for cat in metrics["summary_category"].unique()
    }
    summary["n_states_total"] = int(len(metrics))

    metrics_path, summary_path = save_network_participation_outputs(
        metrics,
        summary,
        out_dir,
        prefix="figS09_network_participation_categories",
    )
    plot_network_participation_by_category(
        metrics,
        summary,
        out_dir / "figS09_network_participation_categories",
        args.subjects,
        NETWORK_ORDER,
        SUBJECT_MARKERS,
        display_network=display_network,
        category_labels=CATEGORY_LABELS,
        category_order=CATEGORY_ORDER,
    )

    print(f"saved: {metrics_path}")
    print(f"saved: {summary_path}")
    print(
        f"saved: {out_dir / 'figS09_network_participation_categories'}.{{pdf,png,svg}}"
    )
    print(f"n_states_total={len(metrics)} n_subjects={metrics['subject'].nunique()}")
    print("category counts:")
    for cat in CATEGORY_ORDER:
        n = int((metrics["summary_category"] == cat).sum())
        if n:
            cat_summary = summary["by_category"].get(cat, {})
            med = cat_summary.get("metric_medians", {})
            combo = cat_summary.get("unordered_top3_combination_counts", [])
            top_combo = (
                f"{'|'.join(combo[0]['networks'])} ({combo[0]['n_states']}/{n})"
                if combo
                else "n/a"
            )
            print(
                f"  {cat}: n={n} "
                f"top1_med={med.get('top1_share', float('nan')):.3f} "
                f"entropy_med={med.get('normalized_network_entropy', float('nan')):.3f} "
                f"top-combo={top_combo}"
            )


if __name__ == "__main__":
    main()
