#!/usr/bin/env python3
"""
04rc_reliability_fc.py - Empirical FC for LOSO and split-half reliability folds.

Computes state-conditioned Ledoit-Wolf FC for each LOSO fold and split-half,
using the same methodology as 05f_state_fc.py but pointing at fold/half
decoded states instead of the primary model.

Prerequisites:
    - 04_combined_hdphmm.py (mode: loso_fit / split_half_fit) completed
    - 02_extract_parcel_ts.py completed

Outputs per fold/half:
    state_empirical_corr.npy   # (K, n_parcels, n_parcels)
    fc_metadata.json           # n_trs_per_state, shrinkage, etc.

Usage:
    # Single LOSO fold
    python script/04rc_reliability_fc.py --sub_id sub-01 --mode loso --fold 1

    # Single split-half
    python script/04rc_reliability_fc.py --sub_id sub-01 --mode split_half --half A

    # All folds for a subject (SLURM array: 0-5 for LOSO, 0-1 for split-half)
    sbatch script/04rc_reliability_fc.sh
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.state_fc import (
    load_matched_data,
    compute_empirical_state_fc,
)

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

ATLAS_BASE = os.getenv("ATLAS_DIR")
if ATLAS_BASE is None:
    raise ValueError("ATLAS_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_n_parcels(parcellation):
    """Get number of atlas parcels from TSV."""
    import pandas as pd
    tsv_path = os.path.join(ATLAS_BASE, parcellation, f"{parcellation}_dseg.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    return len(df)


def compute_fold_fc(sub_id, parcellation, decoded_states_path, results_path,
                    output_dir, min_trs=30, force=False):
    """Compute and save empirical FC for one fold/half.

    Args:
        sub_id:               Subject ID.
        parcellation:         Full parcellation name.
        decoded_states_path:  Path to decoded_states.pkl.
        results_path:         Path to JSON with selected_config.n_components.
        output_dir:           Where to save outputs.
        min_trs:              Min TRs per state for reliable FC.
        force:                Overwrite existing outputs.
    """
    corr_path = os.path.join(output_dir, "state_empirical_corr.npy")
    meta_path = os.path.join(output_dir, "fc_metadata.json")

    if os.path.exists(corr_path) and os.path.exists(meta_path) and not force:
        logger.info("FC already computed: %s", corr_path)
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load decoded states
    with open(decoded_states_path, "rb") as f:
        decoded_states = pickle.load(f)

    # Get K from results JSON
    with open(results_path) as f:
        results = json.load(f)
    K = results["selected_config"]["n_components"]

    # Get atlas parcel count
    n_parcels = get_n_parcels(parcellation)

    # Load matched data
    parcel_ts, viterbi, n_runs = load_matched_data(
        sub_id, parcellation, decoded_states, n_parcels,
        scratch_dir=SCRATCH_DIR,
    )

    # Compute FC
    corr_parcel, n_trs_per_state, reliable, shrinkage_alpha = (
        compute_empirical_state_fc(parcel_ts, viterbi, K, min_trs=min_trs)
    )

    # Save
    np.save(corr_path, corr_parcel)

    metadata = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "K": K,
        "n_parcels": int(corr_parcel.shape[1]),
        "n_runs": n_runs,
        "total_trs": int(len(viterbi)),
        "min_trs": min_trs,
        "n_reliable_states": int(reliable.sum()),
        "n_trs_per_state": n_trs_per_state.tolist(),
        "shrinkage_alpha_per_state": [
            float(a) if np.isfinite(a) else None for a in shrinkage_alpha
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved FC (%d, %d, %d) to %s", *corr_parcel.shape, corr_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute empirical FC for LOSO / split-half reliability folds."
    )
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    parser.add_argument("--mode", required=True, choices=["loso", "split_half"])
    parser.add_argument("--fold", type=int, default=None,
                        help="LOSO season number (1-based)")
    parser.add_argument("--half", type=str, default=None, choices=["A", "B"],
                        help="Split-half identifier")
    parser.add_argument("--min_trs", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parc = normalize_parcellation_name(args.parcellation)
    hmm_base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, args.sub_id,
    )

    if args.mode == "loso":
        # Auto-detect from SLURM_ARRAY_TASK_ID if --fold not given
        fold = args.fold
        if fold is None:
            slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_task is not None:
                fold = int(slurm_task) + 1  # 0-indexed → 1-indexed
            else:
                parser.error("--fold required in loso mode (or set SLURM_ARRAY_TASK_ID)")

        fold_dir = os.path.join(hmm_base, "loso", f"season_{fold}")
        ds_path = os.path.join(fold_dir, "decoded_states.pkl")
        results_path = os.path.join(fold_dir, "loso_results.json")
        output_dir = fold_dir  # Save FC alongside existing LOSO outputs

        logger.info("=" * 60)
        logger.info("04rc Reliability FC — LOSO season %d", fold)
        logger.info("Subject: %s | Parcellation: %s", args.sub_id, parc)
        logger.info("=" * 60)

        compute_fold_fc(
            args.sub_id, parc, ds_path, results_path,
            output_dir, min_trs=args.min_trs, force=args.force,
        )

    elif args.mode == "split_half":
        half = args.half
        if half is None:
            slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_task is not None:
                half = "A" if int(slurm_task) == 0 else "B"
            else:
                parser.error("--half required in split_half mode (or set SLURM_ARRAY_TASK_ID)")

        half_dir = os.path.join(hmm_base, "split_half", half)
        ds_path = os.path.join(half_dir, "decoded_states.pkl")
        results_path = os.path.join(half_dir, "split_half_results.json")
        output_dir = half_dir  # Save FC alongside existing split-half outputs

        logger.info("=" * 60)
        logger.info("04rc Reliability FC — Split-half %s", half)
        logger.info("Subject: %s | Parcellation: %s", args.sub_id, parc)
        logger.info("=" * 60)

        compute_fold_fc(
            args.sub_id, parc, ds_path, results_path,
            output_dir, min_trs=args.min_trs, force=args.force,
        )


if __name__ == "__main__":
    main()
