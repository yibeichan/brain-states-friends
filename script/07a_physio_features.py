#!/usr/bin/env python3
"""
07a_physio_features.py - Extract TR-aligned physiological features per run.

Supports both Friends and Movie10 stimuli via the --stimulus flag.
Core extraction logic is identical; only I/O paths differ.

This script is fully independent of the brain pipeline — it discovers runs
from the physprep directory and derives TR counts from the physio recording
length.  Downstream scripts (07b, 07c) truncate to min(len(features),
len(states)) at load time to guarantee alignment.

Features (7 columns per TR):
    0: HR (bpm)         — from r_peak_corrected, interpolated
    1: HRV_RMSSD (ms)   — 30s rolling window
    2: breathing_rate    — from inhale_max, interpolated
    3: RVT              — respiratory volume per time
    4: EDA_tonic        — mean per TR
    5: EDA_phasic       — mean per TR (log-transformed before z-score)
    6: SCR_binary       — any SCR onset in TR window (0/1)

QC handling: per-channel NaN masking (ECG fail → cols 0,1 = NaN, etc.).
PPG used as ECG backup when ECG fails but PPG passes.
Normalization: within-run z-score on cols 0-5; SCR_binary (col 6) not z-scored.

Prerequisites:
    Physprep data available for the chosen stimulus.
    No brain pipeline outputs required.

Outputs:
    {SCRATCH_DIR}/output/{output_dir}/{sub_id}/
        {run_id}_physio_features.npy       — shape (n_trs, 7), z-scored
        {run_id}_physio_features_raw.npy   — shape (n_trs, 7), pre-normalization
        {run_id}_physio_quality.json       — per-channel QC status
        feature_columns.json               — column name mapping
        extraction_summary.json            — run counts, skip reasons, QC stats
"""

import os
import sys
import json
import gzip
import logging
import argparse

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.physio_io import (
    FEATURE_COLUMNS,
    SAMPLES_PER_TR,
    build_physio_inventory,
    load_quality,
    load_preproc_physio,
    extract_physio_features,
    normalize_features_within_run,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
DATA_DIR = os.getenv("DATA_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")
if DATA_DIR is None:
    raise ValueError("DATA_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Stimulus configuration ─────────────────────────────────────────────────

STIMULUS_CONFIG = {
    "friends": {
        "physprep_dir": os.path.join(DATA_DIR, "friends.physprep"),
        "output_dir": "07a_physio_features",
    },
    "movie10": {
        "physprep_dir": os.path.join(DATA_DIR, "movie10.physprep"),
        "output_dir": "m10_07a_physio_features",
    },
    "harrypotter": {
        "physprep_dir": os.path.join(DATA_DIR, "harrypotter.physprep"),
        "output_dir": "hp_07a_physio_features",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract TR-aligned physiological features per run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Friends (default)
  python script/07a_physio_features.py --sub_id sub-01

  # Movie10
  python script/07a_physio_features.py --sub_id sub-01 --stimulus movie10

  # Single run (debugging)
  python script/07a_physio_features.py --sub_id sub-01 --run_id s01e02a
        """,
    )
    parser.add_argument(
        "--sub_id", type=str, required=True, help='Subject ID (e.g., "sub-01")'
    )
    parser.add_argument(
        "--stimulus",
        type=str,
        default="friends",
        choices=["friends", "movie10", "harrypotter"],
        help="Stimulus dataset (default: friends)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Process a single run (for debugging). If omitted, processes all runs.",
    )
    return parser.parse_args()


def _n_trs_from_preproc(preproc_path):
    """Derive TR count from physio recording length.

    n_trs = n_samples // SAMPLES_PER_TR (floor division).
    Any trailing samples shorter than a full TR are discarded.
    """
    with gzip.open(preproc_path, "rt") as f:
        # Count lines minus header
        n_lines = sum(1 for _ in f) - 1
    return n_lines // SAMPLES_PER_TR


def _make_safe_filename(run_key: str) -> str:
    """Convert a physio inventory key to a safe output filename.

    Friends task keys: 's01e02a' → 's01e02a'
    Movie10 task keys: 'bourne01' → 'bourne01', 'figures05_run-01' → 'figures05_run-01'
    """
    return run_key.replace("/", "_").replace(" ", "_")


def main():
    args = parse_args()
    sub_id = args.sub_id
    stimulus = args.stimulus
    config = STIMULUS_CONFIG[stimulus]

    physprep_dir = config["physprep_dir"]
    out_dir = os.path.join(SCRATCH_DIR, "output", config["output_dir"], sub_id)
    os.makedirs(out_dir, exist_ok=True)

    # ── Build physio inventory (no decoded_states needed) ─────────────
    inventory = build_physio_inventory(
        sub_id, physprep_dir, decoded_state_keys=None, stimulus=stimulus
    )
    logger.info("Physio inventory: %d runs discovered", len(inventory))

    # Filter to single run if requested
    if args.run_id:
        if args.run_id in inventory:
            inventory = {args.run_id: inventory[args.run_id]}
        else:
            # Try matching against task entity for flexibility
            matched = {k: v for k, v in inventory.items() if args.run_id in k}
            if matched:
                inventory = matched
            else:
                logger.error("Run %s not found in inventory", args.run_id)
                sys.exit(1)

    # ── Extract features per run ──────────────────────────────────────
    summary = {
        "sub_id": sub_id,
        "stimulus": stimulus,
        "total_runs": len(inventory),
        "processed": 0,
        "skipped_no_preproc": 0,
        "skipped_no_events": 0,
        "skipped_error": 0,
        "channel_fail_counts": {"ECG": 0, "RSP": 0, "EDA": 0, "PPG": 0},
        "cardiac_usable": 0,
        "run_details": {},
    }

    for run_key in sorted(inventory.keys()):
        files = inventory[run_key]

        # Check required files
        preproc_path = files.get("preproc_physio")
        events_path = files.get("events")
        quality_path = files.get("quality")

        if preproc_path is None:
            logger.warning("Skipping %s: no preproc physio file", run_key)
            summary["skipped_no_preproc"] += 1
            summary["run_details"][run_key] = {"status": "skipped", "reason": "no_preproc"}
            continue

        if events_path is None:
            logger.warning("Skipping %s: no events file", run_key)
            summary["skipped_no_events"] += 1
            summary["run_details"][run_key] = {"status": "skipped", "reason": "no_events"}
            continue

        # Derive TR count from physio recording length
        n_trs = _n_trs_from_preproc(preproc_path)
        if n_trs < 1:
            logger.warning("Skipping %s: recording too short (%d TRs)", run_key, n_trs)
            summary["skipped_error"] += 1
            summary["run_details"][run_key] = {"status": "skipped", "reason": "too_short"}
            continue

        # Load QC
        channel_qc, quality_source = load_quality(quality_path)
        for ch, passed in channel_qc.items():
            if not passed:
                summary["channel_fail_counts"][ch] += 1
        if channel_qc.get("ECG", False):
            cardiac_source = "ecg"
            summary["cardiac_usable"] += 1
        elif channel_qc.get("PPG", False):
            cardiac_source = "ppg"
            summary["cardiac_usable"] += 1
        else:
            cardiac_source = "none"

        # Extract features
        try:
            features_raw = extract_physio_features(
                preproc_path, events_path, n_trs, channel_qc
            )
            features = normalize_features_within_run(features_raw)

            # Save feature arrays
            safe_name = _make_safe_filename(run_key)

            # Normalized features (z-scored within run) — for state profiles, lags, TTAs
            feat_path = os.path.join(out_dir, f"{safe_name}_physio_features.npy")
            np.save(feat_path, features)

            # Raw features (pre-normalization) — for run-level arousal proxies
            raw_path = os.path.join(out_dir, f"{safe_name}_physio_features_raw.npy")
            np.save(raw_path, features_raw)

            # Save per-run quality
            qc_path = os.path.join(out_dir, f"{safe_name}_physio_quality.json")
            with open(qc_path, "w") as f:
                json.dump(
                    {ch: "Pass" if ok else "Fail" for ch, ok in channel_qc.items()},
                    f,
                    indent=2,
                )

            summary["processed"] += 1
            summary["run_details"][run_key] = {
                "status": "processed",
                "n_trs": n_trs,
                "n_nan_trs": int(np.any(np.isnan(features), axis=1).sum()),
                "output_file": os.path.basename(feat_path),
                "cardiac_source": cardiac_source,
                "quality_source": quality_source,
            }
            logger.info(
                "Processed %s: %d TRs, shape %s", run_key, n_trs, features.shape
            )

        except Exception as e:
            logger.error("Error processing %s: %s", run_key, e, exc_info=True)
            summary["skipped_error"] += 1
            summary["run_details"][run_key] = {
                "status": "error",
                "error": str(e),
            }

    # ── Save metadata ─────────────────────────────────────────────────
    # Feature columns mapping
    cols_path = os.path.join(out_dir, "feature_columns.json")
    with open(cols_path, "w") as f:
        json.dump(
            {str(i): name for i, name in enumerate(FEATURE_COLUMNS)},
            f,
            indent=2,
        )

    # Extraction summary
    summary_path = os.path.join(out_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Done: %d/%d runs processed, %d skipped, %d errors",
        summary["processed"],
        summary["total_runs"],
        summary["skipped_no_preproc"] + summary["skipped_no_events"],
        summary["skipped_error"],
    )
    logger.info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
