#!/usr/bin/env python3
"""
08a_content_features.py - Extract TR-level content features from narrative annotations.

Converts sentence and scene annotations from te-charnet into TR-aligned feature
arrays for each run. Independent of brain pipeline — only requires decoded_states
to determine n_trs per run.

Content features are stimulus-level (same video for all subjects), so output is
stored without sub_id in the path. Any subject's decoded_states can be used to
determine n_trs per run.

16 features per TR (see content_io.CONTENT_FEATURE_COLUMNS):
    Dialogue structure (0-5):
        0: speech_presence, 1: dialogue_rate, 2: n_speakers,
        3: speaker_change, 4: silence_duration_s, 5: utterance_duration_s
    Scene features (6-8):
        6: scene_boundary, 7: n_scene_speakers, 8: scene_duration_s
    Character features (9-15):
        9-14: monica/ross/rachel/chandler/joey/phoebe_speaking
        15: n_main_in_scene

Prerequisites:
    - 04_combined_hdphmm.py (mode: select) completed → decoded_states.pkl (for n_trs)
    - te-charnet annotations at $ANNOTATION_DIR (see .env.example)

Outputs:
    {SCRATCH_DIR}/output/08a_content_features/{parcellation}/
        {run_id}_content_features.npy     # (n_trs, 16) raw features
        {run_id}_content_features_norm.npy # (n_trs, 16) z-scored (continuous only)
        feature_summary.json              # Column names, coverage stats
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.content_io import (
    CONTENT_FEATURE_COLUMNS,
    MAIN_CHARACTERS,
    N_FEATURES,
    build_content_inventory,
    extract_content_features,
    normalize_content_features,
    validate_timestamp_alignment,
)
from utils.common import normalize_parcellation_name

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract TR-level content features from narrative annotations.",
    )
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--parcellation", type=str, default="atlas-4S156Parcels")
    parser.add_argument(
        "--vt", type=str, default=None,
        help="Variance threshold subdirectory (e.g., 0.99).",
    )
    parser.add_argument(
        "--annotation_dir", type=str,
        default=os.getenv("ANNOTATION_DIR"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    logger.info("=" * 60)
    logger.info("08a - Content Feature Extraction")
    logger.info("=" * 60)
    logger.info("Subject: %s, Parcellation: %s", sub_id, parc)

    # ── Load decoded_states for n_trs per run ────────────────────────────
    vt_subdir = f"vt{args.vt}" if args.vt else ""
    ds_path = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id,
        "final", vt_subdir, "decoded_states.pkl",
    ).replace("//", "/")

    if not os.path.exists(ds_path):
        logger.error("decoded_states.pkl not found: %s", ds_path)
        sys.exit(1)

    with open(ds_path, "rb") as f:
        decoded_states = pickle.load(f)
    logger.info("Loaded decoded_states: %d runs", len(decoded_states))

    # ── Build content inventory ──────────────────────────────────────────
    run_ids = sorted(decoded_states.keys())
    inventory = build_content_inventory(run_ids, args.annotation_dir)

    # ── Output directory ─────────────────────────────────────────────────
    # Content features are stimulus-level (same for all subjects), so no sub_id
    # or vt in path. We only need decoded_states to know n_trs per run.
    out_dir = os.path.join(
        SCRATCH_DIR, "output", "08a_content_features", parc,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Extract features per run ─────────────────────────────────────────
    summary_stats = {
        "sub_id": sub_id,
        "parcellation": parc,
        "feature_columns": CONTENT_FEATURE_COLUMNS,
        "n_features": len(CONTENT_FEATURE_COLUMNS),
        "runs": {},
    }

    n_success = 0
    n_skipped = 0

    for run_id in run_ids:
        n_trs = len(decoded_states[run_id])
        inv = inventory.get(run_id, {})
        sent_path = inv.get("sentences")
        scene_path = inv.get("scenes")

        if sent_path is None:
            logger.warning("No sentence annotation for %s — skipping", run_id)
            n_skipped += 1
            continue

        # Validate alignment BEFORE extraction (returns loaded DataFrame to avoid re-read)
        try:
            offset_s, sent_df = validate_timestamp_alignment(sent_path, n_trs)
        except ValueError as e:
            logger.error("%s: timestamp alignment failed: %s — skipping", run_id, e)
            n_skipped += 1
            continue

        if offset_s > 0:
            logger.info("%s: applying %.1fs timestamp offset", run_id, offset_s)

        # Extract features (with validated offset and pre-loaded DataFrame)
        features = extract_content_features(
            sent_path, scene_path, n_trs, offset_s=offset_s, sent_df=sent_df,
        )

        # Save raw features
        raw_path = os.path.join(out_dir, f"{run_id}_content_features.npy")
        np.save(raw_path, features)

        # Save normalized features
        norm_features = normalize_content_features(features)
        norm_path = os.path.join(out_dir, f"{run_id}_content_features_norm.npy")
        np.save(norm_path, norm_features)

        # Per-run summary
        speech_frac = float(np.nanmean(features[:, 0]))  # fraction of TRs with speech
        mean_dialogue_rate = float(np.nanmean(features[:, 1]))
        speaker_change_frac = float(np.nanmean(features[:, 3]))  # sparsity diagnostic
        has_scenes = not np.all(np.isnan(features[:, 6]))

        # Character speaking fractions (cols 9-14)
        char_fracs = {}
        for ci, cname in enumerate(MAIN_CHARACTERS):
            char_fracs[cname.lower()] = round(float(np.nanmean(features[:, 9 + ci])), 3)

        fmri_duration_s = n_trs * 1.49
        summary_stats["runs"][run_id] = {
            "n_trs": int(n_trs),
            "fmri_duration_s": round(fmri_duration_s, 1),
            "offset_s": round(offset_s, 2),
            "speech_fraction": round(speech_frac, 3),
            "mean_dialogue_rate": round(mean_dialogue_rate, 3),
            "speaker_change_fraction": round(speaker_change_frac, 3),
            "has_scene_annotations": has_scenes,
            "character_speaking_fractions": char_fracs,
        }

        n_success += 1
        if n_success % 50 == 0:
            logger.info("  Processed %d / %d runs...", n_success, len(run_ids))

    # ── Save summary ─────────────────────────────────────────────────────
    summary_stats["n_runs_total"] = len(run_ids)
    summary_stats["n_runs_processed"] = n_success
    summary_stats["n_runs_skipped"] = n_skipped

    # Global coverage stats
    if n_success > 0:
        all_speech_fracs = [
            v["speech_fraction"] for v in summary_stats["runs"].values()
        ]
        summary_stats["global_speech_fraction_mean"] = round(
            float(np.mean(all_speech_fracs)), 3
        )
        summary_stats["global_speech_fraction_std"] = round(
            float(np.std(all_speech_fracs)), 3
        )
        n_with_scenes = sum(
            1 for v in summary_stats["runs"].values()
            if v["has_scene_annotations"]
        )
        summary_stats["n_runs_with_scenes"] = n_with_scenes

    summary_path = os.path.join(out_dir, "feature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info(
        "Done! %d runs processed, %d skipped. Output: %s",
        n_success, n_skipped, out_dir,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
