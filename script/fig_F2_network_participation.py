#!/usr/bin/env python3
"""Render Figure 2C network-participation outputs (content-eligible states).

Figure 2C remains the *main* network-participation panel and is restricted to
content-eligible recurrent states. All recurrence-screening categories are
summarized separately in the supplementary figure
(``fig_S_network_participation_categories.py``).
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
    ELIGIBLE_CATEGORY,
    compute_network_participation_metrics,
    plot_network_legend,
    plot_network_pie_mosaic,
    save_network_participation_outputs,
)
from utils.plot_style import (  # noqa: E402
    NETWORK_COLORS,
    NETWORK_ORDER,
    apply_publication_style,
    display_network,
    load_parcel_networks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2C network-participation CSV/JSON/plots."
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
        default=159,
        help="Set to -1 to disable the content-eligible state-count check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    apply_publication_style()

    scratch_dir = Path(os.environ["SCRATCH_DIR"])
    flags_dir = scratch_dir / "output" / "05e_temporal_trend_a4" / args.parcellation
    model_dir = scratch_dir / "output" / "04_combined_hdphmm" / args.parcellation
    out_dir = scratch_dir / "output" / "manuscript_figures" / "fig2"
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

    metrics, summary = compute_network_participation_metrics(
        state_flags=state_flags,
        state_means=state_means,
        subjects=args.subjects,
        parcel_networks=parcel_networks,
        network_order=NETWORK_ORDER,
        summary_categories=[ELIGIBLE_CATEGORY],
    )
    if args.expected_states >= 0 and summary["n_states"] != args.expected_states:
        raise RuntimeError(
            f"Expected {args.expected_states} content-eligible states, "
            f"found {summary['n_states']}"
        )

    metrics_path, summary_path = save_network_participation_outputs(
        metrics, summary, out_dir
    )

    # Main Panel C: sub-01 per-state pie mosaic + shared network legend.
    # Other subjects replicate into fig2/supp/<sub>/.
    main_subject = "sub-01"
    n_main = plot_network_pie_mosaic(
        metrics, main_subject, NETWORK_ORDER, NETWORK_COLORS,
        out_dir / f"fig2_C_{main_subject}_network_pies",
    )
    plot_network_legend(
        NETWORK_ORDER, NETWORK_COLORS, out_dir / "fig2_C_legend_networks",
        display_fn=display_network, ncol=2, fig_w=2.6,
    )
    print(f"saved: fig2_C_{main_subject}_network_pies.{{png,svg}} ({n_main} states)")
    print("saved: fig2_C_legend_networks.{png,svg}")
    for subject in [s for s in args.subjects if s != main_subject]:
        supp_dir = out_dir / "supp" / subject
        supp_dir.mkdir(parents=True, exist_ok=True)
        n_supp = plot_network_pie_mosaic(
            metrics, subject, NETWORK_ORDER, NETWORK_COLORS,
            supp_dir / f"fig2_C_{subject}_network_pies",
        )
        print(f"saved: supp/{subject}/fig2_C_{subject}_network_pies.{{png,svg}} "
              f"({n_supp} states)")

    medians = summary["metric_medians"]
    print(f"saved: {metrics_path}")
    print(f"saved: {summary_path}")
    print(f"n_states={summary['n_states']} n_subjects={summary['n_subjects']}")
    print(
        "medians: "
        f"top1={medians['top1_share']:.3f}, "
        f"top3={medians['top3_share']:.3f}, "
        f"networks_ge_10pct={medians['n_networks_ge_10pct']:.0f}, "
        f"entropy={medians['normalized_network_entropy']:.3f}"
    )
    largest_combo = summary["unordered_top3_combination_counts"][0]
    print(
        "largest unordered top-3 combination: "
        f"{'|'.join(largest_combo['networks'])} "
        f"({largest_combo['n_states']}/{summary['n_states']})"
    )


if __name__ == "__main__":
    main()
