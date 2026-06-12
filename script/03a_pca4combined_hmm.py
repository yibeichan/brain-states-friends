#!/usr/bin/env python3
"""
03a_pca4combined_hmm.py - PCA preparation for combined (cross-season) HDP-HMM.

This is Step 1 of the combined-HMM pipeline. For each subject it:
  1. Creates a season-stratified 70/15/15 train/valid/test split (primary)
  2. Creates 6 LOSO (leave-one-season-out) splits
  3. Fits PCA on the primary training data only
  4. Fits a separate PCA on each LOSO fold's training data
  5. Projects all splits through their respective PCA model
  6. Saves everything 04_combined_hdphmm.py needs

This script does NOT fit the HMM (that is 04_combined_hdphmm.py).

PCA is fit on training data only to prevent data leakage. Test and validation
runs are projected through the train-fitted PCA. For LOSO, each fold's PCA is
fit on the 5-season training pool, then the held-out season is projected through
that fold's PCA.

Outputs (saved to {SCRATCH_DIR}/output/03a_pca4combined_hmm/{parcellation}/{sub_id}/):
    pca_model.pkl                      Primary PCA (fitted on primary train data)
    n_pcs_lookup.json                  {"0.80": n, "0.85": n, "0.90": n}
    pca_variance_summary.json          cumvar array, n_pcs per threshold, participation ratio
    projected/
        train/{run_id}.npy             shape (n_trs, n_pcs_max)
        valid/{run_id}.npy
        test/{run_id}.npy
    splits/
        primary.json                   train/valid/test run ID lists (primary split)
        loso_season_1.json             LOSO fold 1: train/valid/test run ID lists
        ...
        loso_season_6.json
    loso/
        season_1/
            pca_model.pkl              PCA fitted on S2-S6 training data (fold 1)
            n_pcs_lookup.json
            pca_variance_summary.json
            projected/
                train/{run_id}.npy
                valid/{run_id}.npy
                test/{run_id}.npy      S1 runs projected through S2-S6 PCA
        ...
    summary.json

Usage:
    python script/03a_pca4combined_hmm.py --sub_id sub-01
    python script/03a_pca4combined_hmm.py --sub_id sub-01 --parcellation atlas-4S156Parcels

    # All subjects:
    for sub in 01 02 03 04 05 06; do
        python script/03a_pca4combined_hmm.py --sub_id sub-${sub} --parcellation atlas-4S156Parcels
    done
"""

import os
import sys
import json
import pickle
import glob
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
BASE_DIR = os.getenv('BASE_DIR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Variance thresholds for n_pcs lookup.
# components (66 and 124 PCs respectively), defeating dimensionality reduction
# and undersampling per-state covariance parameters.
VARIANCE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]

# Base random seed for reproducibility.
# Each season uses seed_base + season; each LOSO (held_out, season) pair uses
# seed_base + held_out * 10 + season. No collisions for seasons 1-6.
SPLIT_RANDOM_SEED = 100

TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
TEST_FRAC  = 0.15

# Fraction of each remaining season used for LOSO validation
LOSO_VALID_FRAC = 0.20

# Fraction of each half used for validation in split-half mode
SPLIT_HALF_VALID_FRAC = 0.20

# sub-04 has data only through Season 4
SUB04_MAX_SEASON = 4

# Minimum episodes per season to attempt a 70/15/15 split
MIN_EPISODES_FOR_SPLIT = 6


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='PCA preparation for combined (cross-season) HDP-HMM.'
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help='Subject ID (e.g., "sub-01")')
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--split_mode', type=str, default=None,
                        choices=['split_half'],
                        help='Additional split mode to run (e.g., split_half). '
                             'Primary + LOSO always run; this adds extra splits.')
    return parser.parse_args()


# =============================================================================
# Episode utilities  [copied verbatim from 03b_pca4hmm.py]
# =============================================================================

def load_episode_ids(episode_file):
    """Load episode IDs from a text file (one per line)."""
    if not os.path.exists(episode_file):
        raise FileNotFoundError(f"Episode file not found: {episode_file}")
    with open(episode_file, 'r') as f:
        episode_ids = [line.strip() for line in f if line.strip()]
    if not episode_ids:
        raise ValueError(f"Episode file is empty: {episode_file}")
    return episode_ids


def group_episodes_by_season(episode_ids):
    """Group episode IDs by season number. Returns dict: season (int) -> sorted list."""
    pattern = re.compile(r'^s(\d+)e(\d+)([a-z])$')
    season_episodes = {}
    for ep_id in episode_ids:
        match = pattern.match(ep_id)
        if not match:
            raise ValueError(f"Episode ID '{ep_id}' doesn't match pattern 's##e##[a-z]'")
        season = int(match.group(1))
        if season not in season_episodes:
            season_episodes[season] = []
        season_episodes[season].append(ep_id)
    for season in season_episodes:
        season_episodes[season].sort()
    return season_episodes


def get_episode_base(ep_id):
    """Extract episode base: 's01e02a' -> 's01e02'."""
    match = re.match(r'^(s\d+e\d+)[a-z]$', ep_id)
    if not match:
        raise ValueError(f"Cannot parse episode base from '{ep_id}'")
    return match.group(1)


def group_runs_by_episode(run_ids):
    """Group run IDs by episode base. Returns dict: episode_base -> sorted list of run_ids."""
    episodes = {}
    for run_id in run_ids:
        ep_base = get_episode_base(run_id)
        if ep_base not in episodes:
            episodes[ep_base] = []
        episodes[ep_base].append(run_id)
    for ep_base in episodes:
        episodes[ep_base].sort()
    return episodes


# =============================================================================
# Data loading  [copied verbatim from 03b_pca4hmm.py]
# =============================================================================

def load_single_run(subject_id, run_id, data_dir, parcellation):
    """
    Load a single run's parcel time series.

    Returns:
        Tuple of (data, file_path) where data has shape (n_trs, n_parcels)
        with background column already removed. Returns (None, None) if not found.
    """
    glob_pattern = os.path.join(
        data_dir, parcellation, subject_id,
        f"{subject_id}*task-{run_id}*parcel_avg.npy"
    )
    matching_files = glob.glob(glob_pattern)

    if not matching_files:
        return None, None

    file_path = matching_files[0]
    data = np.load(file_path)

    # Drop background column (column 0)
    original_cols = data.shape[1]
    if original_cols in (157, 457, 557, 657, 757, 857, 957, 1057):
        data = data[:, 1:]

    # Validate finite values
    if not np.all(np.isfinite(data)):
        n_bad = (~np.isfinite(data)).sum()
        pct = n_bad / data.size * 100
        if pct < 0.1:
            logger.warning(
                f"Run {run_id}: {n_bad} non-finite values ({pct:.4f}%), replacing with 0"
            )
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(
                f"Run {run_id}: {n_bad} non-finite values ({pct:.2f}%) - too many"
            )

    return data, file_path


# =============================================================================
# PCA utilities  [copied verbatim from 03b_pca4hmm.py]
# =============================================================================

def compute_n_pcs_lookup(cumulative_var, thresholds):
    """
    For each variance threshold, find the number of PCs needed.

    Returns:
        Dict mapping threshold string (e.g., "0.90") to n_pcs (int)
    """
    lookup = {}
    for thresh in thresholds:
        idx = np.searchsorted(cumulative_var, thresh)
        n_pcs = int(idx + 1) if idx < len(cumulative_var) else len(cumulative_var)
        key = f"{thresh:.2f}"
        lookup[key] = n_pcs
        logger.info(f"  {thresh:.0%} variance -> {n_pcs} PCs (cumvar={cumulative_var[n_pcs-1]:.4f})")
    return lookup


# =============================================================================
# Diagnostic plots
# =============================================================================

def save_scree_plot(explained_var_ratio, cumvar, n_pcs_lookup, out_dir, label='primary'):
    """
    Two-panel scree plot: individual variance per PC (bar) and cumulative (line).
    Vertical lines mark each variance threshold from n_pcs_lookup.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(len(explained_var_ratio), max(n_pcs_lookup.values()) + 10)
    x = np.arange(1, n_show + 1)

    # Left: individual variance
    ax1.bar(x, explained_var_ratio[:n_show], color='steelblue', alpha=0.7, width=0.8)
    ax1.set_xlabel('PC index')
    ax1.set_ylabel('Explained variance ratio')
    ax1.set_title(f'{label}: per-component variance')

    # Right: cumulative variance with threshold markers
    ax2.plot(x, cumvar[:n_show], color='steelblue', linewidth=2)
    colors = plt.cm.Set2(np.linspace(0, 1, len(n_pcs_lookup)))
    for i, (thresh_str, n_pcs) in enumerate(sorted(n_pcs_lookup.items())):
        if n_pcs <= n_show:
            ax2.axvline(n_pcs, color=colors[i], linestyle='--', alpha=0.8,
                        label=f'{thresh_str} → {n_pcs} PCs')
            ax2.axhline(float(thresh_str), color=colors[i], linestyle=':', alpha=0.4)
    ax2.set_xlabel('PC index')
    ax2.set_ylabel('Cumulative variance')
    ax2.set_title(f'{label}: cumulative variance')
    ax2.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'pca_scree_{label}.png'), dpi=150)
    plt.close(fig)
    logger.info(f"[{label}] Saved scree plot to {out_dir}/pca_scree_{label}.png")


def save_cumvar_comparison(primary_cumvar, loso_cumvars, n_pcs_lookup, out_dir):
    """
    Overlay cumulative variance curves for primary PCA and all LOSO folds.
    Checks PCA stability across cross-validation folds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_show = min(len(primary_cumvar), max(n_pcs_lookup.values()) + 10)
    x_primary = np.arange(1, min(len(primary_cumvar), n_show) + 1)
    ax.plot(x_primary, primary_cumvar[:n_show], color='black', linewidth=2.5,
            label='primary', zorder=10)

    cmap = plt.cm.tab10
    for i, (season, cv) in enumerate(sorted(loso_cumvars.items())):
        x_loso = np.arange(1, min(len(cv), n_show) + 1)
        ax.plot(x_loso, cv[:n_show], color=cmap(i), linewidth=1.2, alpha=0.7,
                label=f'LOSO S{season}')

    # Threshold markers
    for thresh_str, n_pcs in sorted(n_pcs_lookup.items()):
        if n_pcs <= n_show:
            ax.axvline(n_pcs, color='grey', linestyle=':', alpha=0.4)
            ax.text(n_pcs + 0.3, float(thresh_str) - 0.02, f'{thresh_str}',
                    fontsize=7, color='grey')

    ax.set_xlabel('PC index')
    ax.set_ylabel('Cumulative variance')
    ax.set_title('PCA stability: primary vs LOSO folds')
    ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_cumvar_comparison.png'), dpi=150)
    plt.close(fig)
    logger.info(f"Saved cumulative variance comparison to {out_dir}/pca_cumvar_comparison.png")


# =============================================================================
# Split creation
# =============================================================================

def create_stratified_split(season_episodes, seed_base=SPLIT_RANDOM_SEED):
    """
    Season-stratified 70/15/15 split across all seasons.

    For each season independently: shuffle episodes with a season-specific seed,
    split 70/15/15 at the episode level (multi-part episode runs stay together),
    then expand episodes to run IDs. Union across seasons gives the final
    train/valid/test sets.

    Episode-level splitting prevents temporal continuity leakage: both parts
    of a two-part Friends episode (a, b) always land in the same split.

    Args:
        season_episodes: dict  season (int) -> sorted list of run IDs
        seed_base: int         base seed; each season uses seed_base + season

    Returns:
        train_runs, valid_runs, test_runs: sorted lists of run IDs
    """
    train_all, valid_all, test_all = [], [], []

    for season in sorted(season_episodes.keys()):
        run_ids = season_episodes[season]
        episodes = group_runs_by_episode(run_ids)
        episode_bases = sorted(episodes.keys())
        n_ep = len(episode_bases)

        if n_ep < MIN_EPISODES_FOR_SPLIT:
            raise ValueError(
                f"Season {season}: only {n_ep} episodes, need >= {MIN_EPISODES_FOR_SPLIT} "
                f"for a valid 70/15/15 split."
            )

        rng = np.random.RandomState(seed_base + season)
        shuffled = episode_bases.copy()
        rng.shuffle(shuffled)

        n_valid = max(2, round(n_ep * VALID_FRAC))
        n_test  = max(2, round(n_ep * TEST_FRAC))
        n_train = n_ep - n_valid - n_test

        if n_train < 4:
            raise ValueError(
                f"Season {season}: only {n_train} train episodes after split "
                f"(from {n_ep} total). Need at least 4."
            )

        train_ep = sorted(shuffled[:n_train])
        valid_ep = sorted(shuffled[n_train:n_train + n_valid])
        test_ep  = sorted(shuffled[n_train + n_valid:])

        train_season = sorted(r for ep in train_ep for r in episodes[ep])
        valid_season = sorted(r for ep in valid_ep for r in episodes[ep])
        test_season  = sorted(r for ep in test_ep  for r in episodes[ep])

        train_all += train_season
        valid_all += valid_season
        test_all  += test_season

        logger.info(
            f"  Season {season}: {n_ep} episodes -> "
            f"train={len(train_ep)} eps / {len(train_season)} runs, "
            f"valid={len(valid_ep)} eps / {len(valid_season)} runs, "
            f"test={len(test_ep)} eps / {len(test_season)} runs"
        )

    return sorted(train_all), sorted(valid_all), sorted(test_all)


def create_loso_splits(season_episodes, loso_seed_base=SPLIT_RANDOM_SEED):
    """
    Leave-One-Season-Out splits: for each season s, hold out ALL of season s
    as the test set. From the remaining seasons, take LOSO_VALID_FRAC (20%)
    per-season as validation and the rest as training.

    Multi-part episodes within the remaining seasons stay together.

    The seed for each (held_out, remaining_season) pair is:
        loso_seed_base + held_out * 10 + remaining_season
    This is collision-free for seasons 1-6 (seeds range 111-165).

    Args:
        season_episodes: dict  season (int) -> sorted list of run IDs
        loso_seed_base:  int   base seed

    Returns:
        dict: held_out_season (int) -> {
            'train': sorted list,
            'valid': sorted list,
            'test':  sorted list,
            'held_out_season': int,
        }
    """
    all_seasons = sorted(season_episodes.keys())
    loso = {}

    for held_out in all_seasons:
        test_runs = sorted(season_episodes[held_out])
        remaining_seasons = {s: season_episodes[s] for s in all_seasons if s != held_out}

        # Stratified valid/train split across each remaining season
        train_pool, valid_pool = [], []
        for s, run_ids in sorted(remaining_seasons.items()):
            episodes = group_runs_by_episode(run_ids)
            ep_bases = sorted(episodes.keys())
            n_ep = len(ep_bases)

            # Unique seed per (held_out, remaining_season) pair; no collisions for S1-S6
            rng = np.random.RandomState(loso_seed_base + held_out * 10 + s)
            shuffled = ep_bases.copy()
            rng.shuffle(shuffled)

            n_valid = max(2, round(n_ep * LOSO_VALID_FRAC))
            valid_ep = sorted(shuffled[:n_valid])
            train_ep = sorted(shuffled[n_valid:])

            valid_pool += sorted(r for ep in valid_ep for r in episodes[ep])
            train_pool += sorted(r for ep in train_ep for r in episodes[ep])

        loso[held_out] = {
            'train': sorted(train_pool),
            'valid': sorted(valid_pool),
            'test':  test_runs,
            'held_out_season': held_out,
        }

        logger.info(
            f"  LOSO held_out=S{held_out}: "
            f"train={len(loso[held_out]['train'])} runs, "
            f"valid={len(loso[held_out]['valid'])} runs, "
            f"test={len(test_runs)} runs"
        )

    return loso


def create_split_half_splits(season_episodes, seed_base=SPLIT_RANDOM_SEED):
    """
    Split-half reliability splits: interleaved odd/even episodes within each season.

    For each season, sort episodes by episode number, assign odd-indexed to
    half A and even-indexed to half B (0-indexed, so first episode -> A).
    Multi-part episodes (a/b) stay together as a unit.

    Within each half, further split into train/valid (80/20) for HMM fitting.

    Seed for train/valid split within each half:
        seed_base + 300 + half_idx * 10 + season
    (collision-free with primary 101-106 and LOSO 111-166)

    Args:
        season_episodes: dict  season (int) -> sorted list of run IDs
        seed_base:       int   base seed

    Returns:
        dict: half_label ('A' or 'B') -> {
            'train': sorted list of run IDs,
            'valid': sorted list of run IDs,
            'all': sorted list of all run IDs in this half,
        }
    """
    halves = {'A': [], 'B': []}

    for season in sorted(season_episodes.keys()):
        run_ids = season_episodes[season]
        episodes = group_runs_by_episode(run_ids)
        ep_bases = sorted(episodes.keys())

        for i, ep_base in enumerate(ep_bases):
            half = 'A' if i % 2 == 0 else 'B'
            halves[half].extend(episodes[ep_base])

    result = {}
    for half_idx, (half_label, run_ids) in enumerate(sorted(halves.items())):
        run_ids = sorted(run_ids)

        # Split into train/valid within this half (season-stratified)
        half_season_eps = group_episodes_by_season(run_ids)
        train_pool, valid_pool = [], []

        for s, s_runs in sorted(half_season_eps.items()):
            eps = group_runs_by_episode(s_runs)
            ep_bases = sorted(eps.keys())
            n_ep = len(ep_bases)

            rng = np.random.RandomState(seed_base + 300 + half_idx * 10 + s)
            shuffled = ep_bases.copy()
            rng.shuffle(shuffled)

            n_valid = max(1, round(n_ep * SPLIT_HALF_VALID_FRAC))
            valid_ep = sorted(shuffled[:n_valid])
            train_ep = sorted(shuffled[n_valid:])

            valid_pool += sorted(r for ep in valid_ep for r in eps[ep])
            train_pool += sorted(r for ep in train_ep for r in eps[ep])

        result[half_label] = {
            'train': sorted(train_pool),
            'valid': sorted(valid_pool),
            'all': run_ids,
        }

        logger.info(
            f"  Split-half {half_label}: {len(run_ids)} total runs "
            f"(train={len(train_pool)}, valid={len(valid_pool)})"
        )

    return result


# =============================================================================
# PCA fitting and projection
# =============================================================================

def fit_pca_and_project(train_runs, all_splits, data_dir, parcellation,
                        sub_id, out_dir, label='primary'):
    """
    Fit PCA on concatenated training runs, project all splits, save .npy files.
    Used for both the primary split and each LOSO fold.

    Training data is loaded once and cached to avoid redundant disk reads when
    projecting the training split.

    PCA is fit on training data only (no leakage): test/validation runs are
    centered with the train mean and projected through train-fitted components.

    Args:
        train_runs:   list of run IDs to fit PCA on
        all_splits:   dict  split_name -> list of run IDs to project
                      e.g. {'train': [...], 'valid': [...], 'test': [...]}
        data_dir:     root dir containing 02_parcel_ts_avg/{parcellation}/
        parcellation: normalized parcellation name
        sub_id:       subject ID string
        out_dir:      dir to write pca_model.pkl, projected/, n_pcs_lookup.json
        label:        string label for log messages (e.g. 'primary', 'loso_s1')

    Returns:
        pca:          fitted sklearn PCA object
        n_pcs_lookup: dict  threshold_str -> n_pcs (int)
        cumvar:       np.ndarray  cumulative explained variance ratios (length n_components)
    """
    # -------------------------------------------------------------------------
    # 1. Load and concatenate training data; cache for reuse during projection
    # -------------------------------------------------------------------------
    train_cache = {}  # run_id -> np.ndarray (n_trs, n_parcels)
    X_list = []
    missing_train_runs = []
    for run_id in train_runs:
        data, _ = load_single_run(sub_id, run_id, data_dir, parcellation)
        if data is None:
            missing_train_runs.append(run_id)
            continue
        train_cache[run_id] = data
        X_list.append(data)

    if missing_train_runs:
        raise FileNotFoundError(
            f"[{label}] Missing {len(missing_train_runs)} training run files for PCA fit. "
            f"Examples: {missing_train_runs[:5]}"
        )

    if not X_list:
        raise RuntimeError(f"[{label}] No training data loaded - cannot fit PCA")

    X_train = np.vstack(X_list)
    n_trs_train, n_parcels = X_train.shape
    logger.info(
        f"[{label}] Training data: {n_trs_train} TRs × {n_parcels} parcels "
        f"({len(X_list)}/{len(train_runs)} runs loaded)"
    )

    # -------------------------------------------------------------------------
    # 2. Fit PCA on training data only
    #    n_components = min(n_trs, n_parcels) = n_parcels for typical sizes
    #    (e.g., ~88k TRs >> 156 parcels), giving a full-rank PCA basis.
    # -------------------------------------------------------------------------
    n_components = min(X_train.shape)
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # Verify PCA components shape: sklearn returns (n_components, n_features)
    assert pca.components_.shape == (n_components, n_parcels), (
        f"[{label}] Unexpected PCA components shape: {pca.components_.shape}, "
        f"expected ({n_components}, {n_parcels})"
    )

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs_lookup = compute_n_pcs_lookup(cumvar, VARIANCE_THRESHOLDS)
    n_pcs_max = max(n_pcs_lookup.values())

    logger.info(
        f"[{label}] PCA fitted: {n_components} components; "
        f"n_pcs at 90%: {n_pcs_lookup.get('0.90', 'N/A')} "
        f"(cumvar={cumvar[n_pcs_lookup.get('0.90', 1) - 1]:.4f})"
    )

    # -------------------------------------------------------------------------
    # 3. Save PCA model (protocol=4 for broad Python 3.x compatibility)
    # -------------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    pca_path = os.path.join(out_dir, 'pca_model.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f, protocol=4)

    # -------------------------------------------------------------------------
    # 4. Save n_pcs lookup
    # -------------------------------------------------------------------------
    with open(os.path.join(out_dir, 'n_pcs_lookup.json'), 'w') as f:
        json.dump(n_pcs_lookup, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Save variance summary
    #    Participation ratio (effective dimensionality):
    #        PR = (Σλᵢ)² / Σλᵢ²   where λᵢ = pca.explained_variance_ (eigenvalues)
    #    Represents how many components carry the signal; PR ≈ k for uniform eigenvalues.
    # -------------------------------------------------------------------------
    eigenvalues = pca.explained_variance_
    participation_ratio = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())

    variance_summary = {
        'n_components_total': int(n_components),
        'n_parcels': int(n_parcels),
        'n_train_trs': int(n_trs_train),
        'n_train_runs': int(len(X_list)),
        'n_pcs_lookup': n_pcs_lookup,
        'n_pcs_max': int(n_pcs_max),
        'participation_ratio': round(participation_ratio, 3),
        'cumvar': [round(float(v), 6) for v in cumvar],
        'explained_variance_ratio': [round(float(v), 6) for v in pca.explained_variance_ratio_],
    }
    with open(os.path.join(out_dir, 'pca_variance_summary.json'), 'w') as f:
        json.dump(variance_summary, f, indent=2)

    # -------------------------------------------------------------------------
    # 6. Project all splits and save .npy files
    #    Training runs reuse the in-memory cache to avoid redundant disk reads.
    # -------------------------------------------------------------------------
    for split_name, run_ids in all_splits.items():
        split_dir = os.path.join(out_dir, 'projected', split_name)
        os.makedirs(split_dir, exist_ok=True)
        n_projected = 0

        for run_id in run_ids:
            # Reuse cached training data (already preprocessed) to avoid redundant disk read
            if run_id in train_cache:
                data = train_cache[run_id]
            else:
                data, _ = load_single_run(sub_id, run_id, data_dir, parcellation)
                if data is None:
                    raise FileNotFoundError(
                        f"[{label}] split={split_name}: missing run file for {run_id}. "
                        "Failing early to prevent downstream partial outputs."
                    )

            # Project through PCA; save n_pcs_max columns (downstream scripts select k)
            projected = pca.transform(data)[:, :n_pcs_max]
            np.save(os.path.join(split_dir, f'{run_id}.npy'), projected)
            n_projected += 1

        assert n_projected == len(run_ids), (
            f"[{label}] split={split_name}: projected {n_projected}/{len(run_ids)} runs"
        )
        logger.info(
            f"[{label}] split={split_name}: projected {n_projected}/{len(run_ids)} runs"
        )

    return pca, n_pcs_lookup, cumvar


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()
    args = parse_args()

    # Validate environment
    if not SCRATCH_DIR:
        raise RuntimeError("SCRATCH_DIR not set - check .env file")
    if not BASE_DIR:
        raise RuntimeError("BASE_DIR not set - check .env file")

    parcellation = normalize_parcellation_name(args.parcellation)

    logger.info("=" * 70)
    logger.info("PCA Preparation for Combined HDP-HMM")
    logger.info("=" * 70)
    logger.info(f"Subject:       {args.sub_id}")
    logger.info(f"Parcellation:  {parcellation}")
    logger.info(f"Seed base:     {SPLIT_RANDOM_SEED}")
    logger.info(f"Var thresholds: {VARIANCE_THRESHOLDS}")
    logger.info("=" * 70)

    output_base = os.path.join(
        SCRATCH_DIR, 'output', '03a_pca4combined_hmm', parcellation, args.sub_id
    )
    splits_dir = os.path.join(output_base, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    data_dir = os.path.join(SCRATCH_DIR, 'output', '02_parcel_ts_avg')

    # =========================================================================
    # [2/5] Load episode IDs and build season_episodes dict
    # =========================================================================
    logger.info("[2/5] Loading episode IDs and building season map...")

    episode_file = os.path.join(BASE_DIR, f'{args.sub_id}_episode_ids.txt')
    all_episode_ids = load_episode_ids(episode_file)
    season_episodes = group_episodes_by_season(all_episode_ids)

    # sub-04: exclude seasons 5 and 6 (no data available beyond season 4)
    if args.sub_id == 'sub-04':
        for s in sorted(list(season_episodes.keys())):
            if s > SUB04_MAX_SEASON:
                logger.warning(
                    f"  Excluding season {s} for sub-04 "
                    f"(max season = {SUB04_MAX_SEASON})"
                )
                del season_episodes[s]

    available_seasons = sorted(season_episodes.keys())
    logger.info(f"  Available seasons: {available_seasons}")
    for s in available_seasons:
        logger.info(f"    Season {s}: {len(season_episodes[s])} runs")

    # =========================================================================
    # [3/5] Create primary season-stratified split and fit primary PCA
    # =========================================================================
    logger.info("[3/5] Creating primary season-stratified split and fitting PCA...")

    train_runs, valid_runs, test_runs = create_stratified_split(season_episodes)

    # Hard correctness checks: no overlap between any split pair
    assert not (set(train_runs) & set(valid_runs)), "Primary split: train/valid overlap!"
    assert not (set(train_runs) & set(test_runs)),  "Primary split: train/test overlap!"
    assert not (set(valid_runs) & set(test_runs)),  "Primary split: valid/test overlap!"

    logger.info(
        f"  Primary split totals: train={len(train_runs)}, "
        f"valid={len(valid_runs)}, test={len(test_runs)} runs"
    )

    # Save primary split JSON
    primary_split = {
        'split_type': 'stratified',
        'seed_base': SPLIT_RANDOM_SEED,
        'train_frac': TRAIN_FRAC,
        'valid_frac': VALID_FRAC,
        'test_frac': TEST_FRAC,
        'train': train_runs,
        'valid': valid_runs,
        'test': test_runs,
        'n_train': len(train_runs),
        'n_valid': len(valid_runs),
        'n_test': len(test_runs),
        'seasons_covered': available_seasons,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(splits_dir, 'primary.json'), 'w') as f:
        json.dump(primary_split, f, indent=2)
    logger.info(f"  Saved splits/primary.json")

    # Fit PCA on primary train data; project all three primary splits
    primary_pca, primary_n_pcs_lookup, primary_cumvar = fit_pca_and_project(
        train_runs=train_runs,
        all_splits={'train': train_runs, 'valid': valid_runs, 'test': test_runs},
        data_dir=data_dir,
        parcellation=parcellation,
        sub_id=args.sub_id,
        out_dir=output_base,
        label='primary',
    )

    n_pcs_max = max(primary_n_pcs_lookup.values())

    save_scree_plot(primary_pca.explained_variance_ratio_, primary_cumvar,
                    primary_n_pcs_lookup, output_base, label='primary')

    # Shape sanity check on a sample of projected primary train files
    logger.info("  Verifying projected file shapes (primary train sample)...")
    train_proj_dir = os.path.join(output_base, 'projected', 'train')
    for run_id in train_runs[:3]:
        proj_path = os.path.join(train_proj_dir, f'{run_id}.npy')
        if os.path.exists(proj_path):
            arr = np.load(proj_path)
            assert arr.shape[1] == n_pcs_max, (
                f"Primary train {run_id}: expected {n_pcs_max} PCs, got {arr.shape[1]}"
            )
    logger.info("  Primary shape checks passed.")

    # =========================================================================
    # [4/5] Create LOSO splits and fit per-fold PCA
    # =========================================================================
    logger.info("[4/5] Creating LOSO splits and fitting per-fold PCA...")

    loso_splits = create_loso_splits(season_episodes)
    loso_summary = {}
    loso_cumvars = {}  # season -> cumvar array, for comparison plot

    # Regex to extract season number from a run_id (e.g. 's01e01a' -> 1)
    _season_re = re.compile(r'^s(\d+)')

    for held_out_season, fold in sorted(loso_splits.items()):
        logger.info(f"  --- LOSO fold: held_out=S{held_out_season} ---")

        # Correctness assertions for this LOSO fold
        test_seasons_found = set()
        for r in fold['test']:
            m = _season_re.match(r)
            if m:
                test_seasons_found.add(int(m.group(1)))
        assert test_seasons_found == {held_out_season}, (
            f"LOSO S{held_out_season}: test contains unexpected seasons: {test_seasons_found}"
        )

        non_test_seasons_in_test = set()
        for r in fold['train']:
            m = _season_re.match(r)
            if m and int(m.group(1)) == held_out_season:
                non_test_seasons_in_test.add(r)
        assert not non_test_seasons_in_test, (
            f"LOSO S{held_out_season}: held-out season found in train set!"
        )

        assert not (set(fold['train']) & set(fold['test'])),  \
            f"LOSO S{held_out_season}: train/test overlap!"
        assert not (set(fold['valid']) & set(fold['test'])),  \
            f"LOSO S{held_out_season}: valid/test overlap!"
        assert not (set(fold['train']) & set(fold['valid'])), \
            f"LOSO S{held_out_season}: train/valid overlap!"

        # Save LOSO split JSON
        remaining_seasons = [s for s in available_seasons if s != held_out_season]
        loso_split_record = {
            'split_type': 'loso',
            'held_out_season': held_out_season,
            'loso_valid_frac': LOSO_VALID_FRAC,
            'train': fold['train'],
            'valid': fold['valid'],
            'test': fold['test'],
            'n_train': len(fold['train']),
            'n_valid': len(fold['valid']),
            'n_test': len(fold['test']),
            'pca_fitted_on': f"train (seasons {remaining_seasons})",
            'timestamp': datetime.now().isoformat(),
        }
        loso_json_path = os.path.join(splits_dir, f'loso_season_{held_out_season}.json')
        with open(loso_json_path, 'w') as f:
            json.dump(loso_split_record, f, indent=2)
        logger.info(f"  Saved splits/loso_season_{held_out_season}.json")

        # Fit PCA on this fold's training pool; project train/valid/test
        loso_out_dir = os.path.join(output_base, 'loso', f'season_{held_out_season}')
        loso_pca, loso_n_pcs_lookup, loso_cumvar = fit_pca_and_project(
            train_runs=fold['train'],
            all_splits={'train': fold['train'], 'valid': fold['valid'], 'test': fold['test']},
            data_dir=data_dir,
            parcellation=parcellation,
            sub_id=args.sub_id,
            out_dir=loso_out_dir,
            label=f'loso_s{held_out_season}',
        )

        loso_n_pcs_max = max(loso_n_pcs_lookup.values())

        # Shape check on a sample of LOSO test files (held-out season projected through other-season PCA)
        loso_test_dir = os.path.join(loso_out_dir, 'projected', 'test')
        for run_id in fold['test'][:3]:
            proj_path = os.path.join(loso_test_dir, f'{run_id}.npy')
            if os.path.exists(proj_path):
                arr = np.load(proj_path)
                assert arr.shape[1] == loso_n_pcs_max, (
                    f"LOSO S{held_out_season} test {run_id}: "
                    f"expected {loso_n_pcs_max} PCs, got {arr.shape[1]}"
                )

        loso_cumvars[held_out_season] = loso_cumvar

        loso_summary[held_out_season] = {
            'n_train': len(fold['train']),
            'n_valid': len(fold['valid']),
            'n_test': len(fold['test']),
            'n_pcs_max': loso_n_pcs_max,
            'n_pcs_lookup': loso_n_pcs_lookup,
        }

    # Diagnostic: cumulative variance comparison across all PCA fits
    if loso_cumvars:
        save_cumvar_comparison(primary_cumvar, loso_cumvars,
                              primary_n_pcs_lookup, output_base)

    # =========================================================================
    # [4b/5] Create split-half splits and fit per-half PCA (if requested)
    # =========================================================================
    split_half_summary = {}
    if args.split_mode == 'split_half':
        logger.info("[4b/5] Creating split-half splits and fitting per-half PCA...")

        split_half_splits = create_split_half_splits(season_episodes)

        for half_label, half_data in sorted(split_half_splits.items()):
            logger.info(f"  --- Split-half {half_label} ---")

            # No test set for split-half - use all runs as train+valid for HMM,
            # then 04rb compares the two halves. But for fit_pca_and_project,
            # we need train/valid/test. Use train+valid for PCA, project all runs.
            # For 04 split_half_fit: train on 'train', validate on 'valid'.
            # "test" in split context = the other half's runs (handled by 04rb, not here).
            # Save splits with empty test so 04 can load them uniformly.

            split_half_record = {
                'split_type': 'split_half',
                'half': half_label,
                'split_half_valid_frac': SPLIT_HALF_VALID_FRAC,
                'train': half_data['train'],
                'valid': half_data['valid'],
                'test': [],  # no within-half test; cross-half comparison done by 04rb
                'n_train': len(half_data['train']),
                'n_valid': len(half_data['valid']),
                'n_total_runs': len(half_data['all']),
                'timestamp': datetime.now().isoformat(),
            }
            split_json_path = os.path.join(splits_dir, f'split_half_{half_label}.json')
            with open(split_json_path, 'w') as f:
                json.dump(split_half_record, f, indent=2)
            logger.info(f"  Saved splits/split_half_{half_label}.json")

            # Fit PCA on this half's training data; project train+valid
            half_out_dir = os.path.join(output_base, 'split_half', half_label)
            half_pca, half_n_pcs_lookup, half_cumvar = fit_pca_and_project(
                train_runs=half_data['train'],
                all_splits={
                    'train': half_data['train'],
                    'valid': half_data['valid'],
                },
                data_dir=data_dir,
                parcellation=parcellation,
                sub_id=args.sub_id,
                out_dir=half_out_dir,
                label=f'split_half_{half_label}',
            )

            split_half_summary[half_label] = {
                'n_train': len(half_data['train']),
                'n_valid': len(half_data['valid']),
                'n_total': len(half_data['all']),
                'n_pcs_max': max(half_n_pcs_lookup.values()),
                'n_pcs_lookup': half_n_pcs_lookup,
            }

    # =========================================================================
    # [5/5] Save summary.json
    # =========================================================================
    logger.info("[5/5] Saving summary...")

    elapsed = time.time() - start_time
    summary = {
        'sub_id': args.sub_id,
        'parcellation': parcellation,
        'seasons_available': available_seasons,
        'n_seasons': len(available_seasons),
        'primary': {
            'n_train': len(train_runs),
            'n_valid': len(valid_runs),
            'n_test': len(test_runs),
            'n_pcs_max': n_pcs_max,
            'n_pcs_lookup': primary_n_pcs_lookup,
        },
        'loso': {str(k): v for k, v in loso_summary.items()},
        'n_loso_folds': len(loso_splits),
        'split_half': split_half_summary if split_half_summary else None,
        'output_dir': output_base,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat(),
    }
    summary_path = os.path.join(output_base, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"COMPLETED: PCA preparation for {args.sub_id}")
    logger.info(
        f"  Primary: train={len(train_runs)}, valid={len(valid_runs)}, "
        f"test={len(test_runs)} runs; n_pcs_max={n_pcs_max}"
    )
    logger.info(f"  LOSO folds: {len(loso_splits)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Output: {output_base}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
