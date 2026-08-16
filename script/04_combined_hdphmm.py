#!/usr/bin/env python3
"""
04_combined_hdphmm.py - Fit one combined weak-limit HMM per subject across all seasons.

Two-stage model selection (2026-03-13 improvement):

  fit mode --stage 1 (SLURM-parallelizable):
    Stage 1 - select variance threshold (PCA dimension).
    Sweep vt ∈ {0.80..0.99} with fixed nc=60, gamma=5, cov=diag.
    5 configs, each with N seeds. Metric: per-dimension validation LL.
    SLURM array: 0-4 (5 configs).

  fit mode --stage 2 (SLURM-parallelizable):
    Stage 2 - select K, gamma, covariance at the vt chosen in Stage 1.
    Sweep nc × gamma × cov. Metric: BIC (training LL, effective K).
    SLURM array: 0-(N_stage2-1) where N depends on selected vt's cov rules.

  select mode (after all fit tasks complete):
    Orchestrates two-stage selection → refit on train+valid → test eval → decode.

  select_seed / select_finalize: parallelized refit variants.

  loso_fit mode (secondary validation, after select):
    Leave-One-Season-Out refit for cross-season generalization.

Prerequisites:
    - 03a_pca4combined_hmm.py completed for the subject
    - Output at {SCRATCH_DIR}/output/03a_pca4combined_hmm/{parcellation}/{sub_id}/

Usage:
    # Stage 1 fit (5 vt configs):
    python script/04_combined_hdphmm.py \\
        --sub_id sub-01 --mode fit --stage 1 --task_index 0

    # Stage 2 fit (after Stage 1 select):
    python script/04_combined_hdphmm.py \\
        --sub_id sub-01 --mode fit --stage 2 --task_index 0

    # Select mode (two-stage orchestration):
    python script/04_combined_hdphmm.py \\
        --sub_id sub-01 --mode select --n_final_seeds 10

    # LOSO fit mode (one fold, run for each season):
    python script/04_combined_hdphmm.py \\
        --sub_id sub-01 --mode loso_fit --loso_season 1
"""

import os
import sys
import json
import pickle
import time
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
# JAX GPU backend: auto-detect and use if available
try:
    import jax as _jax
    if _jax.default_backend() == "gpu":
        from utils.hdphmm_jax import WeakLimitHMM_JAX as WeakLimitHMM
        from utils.hdphmm import infer_n_active_states  # utility stays numpy
        logging.getLogger(__name__).info(
            "Using JAX GPU backend on %s", _jax.devices("gpu")[0]
        )
    else:
        from utils.hdphmm import WeakLimitHMM, infer_n_active_states
except (ImportError, RuntimeError):
    from utils.hdphmm import WeakLimitHMM, infer_n_active_states
from utils.common import normalize_parcellation_name, _get_season
from utils.hmm_io import (load_split, load_n_pcs_lookup, get_projected_dir,
                           load_projected_runs, decode_all_runs,
                           back_project_states)
from config.combined_hmm_config import (
    build_config_grid, build_stage1_grid, build_stage2_grid,
    config_name, legacy_config_name,
    FIXED_PARAMS, VARIANCE_THRESHOLDS, COVARIANCE_RULES,
    N_COMPONENTS_OPTIONS, GAMMA_OPTIONS, N_CONFIGS, N_CONFIGS_STAGE1,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
BASE_DIR = os.getenv('BASE_DIR')
if not SCRATCH_DIR:
    raise RuntimeError(
        "SCRATCH_DIR not set in environment. "
        "Check .env file or export SCRATCH_DIR before running."
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default parallel jobs for E-step (forward/backward across sequences).
# Set via --n_jobs CLI arg. Use cpus-per-task in SLURM to match.
DEFAULT_N_JOBS = 8

# Sub-04: S5 has only 4 runs (~1,888 TRs); S6 has 0. Use S1-S4 only.
SUB04_MAX_SEASON = 4



# =============================================================================
# Path helpers
# =============================================================================

def get_combined_base(sub_id, parcellation):
    """Root output directory of 03a_pca4combined_hmm for this subject."""
    return os.path.join(
        SCRATCH_DIR, 'output', '03a_pca4combined_hmm', parcellation, sub_id
    )


def get_output_base(sub_id, parcellation):
    """Root output directory for 04_combined_hdphmm results."""
    return os.path.join(
        SCRATCH_DIR, 'output', '04_combined_hdphmm', parcellation, sub_id
    )


def get_final_dir(output_base, vt):
    """Final output directory for a given variance threshold.

    Returns path like: {output_base}/final/vt{vt:.2f}/
    """
    return os.path.join(output_base, 'final', f'vt{vt:.2f}')





# =============================================================================
# Fit mode
# =============================================================================

def fit_single_config(config, sub_id, parcellation, n_fit_seeds, n_jobs=1):
    """Fit one combined cross-season HMM config with multiple seeds.

    For each seed:
      1. Create WeakLimitHMM with the config parameters
      2. Fit on full cross-season training data (multi-sequence via lengths)
      3. Score on training and validation data
      4. Score per-season validation LL (early warning for non-stationarity)
      5. Save per-seed JSON + model pkl (enables resume on SLURM timeout)

    Config summary (aggregated across seeds) saved at end.
    """
    cfg_name = config_name(config)
    logger.info(f"Config: {cfg_name}")

    combined_base = get_combined_base(sub_id, parcellation)
    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    vt_key = f"{config['variance_threshold']:.2f}"
    n_pcs = n_pcs_lookup[vt_key]
    logger.info(f"Variance threshold {vt_key} -> {n_pcs} PCs")

    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)

    logger.info(
        f"Loading train ({len(split['train'])} runs) and "
        f"valid ({len(split['valid'])} runs)..."
    )
    X_train, lengths_train = load_projected_runs(
        split['train'], projected_dir, n_pcs, 'train'
    )
    X_valid, lengths_valid = load_projected_runs(
        split['valid'], projected_dir, n_pcs, 'valid'
    )
    n_train = sum(lengths_train)
    n_valid = sum(lengths_valid)
    logger.info(
        f"Train: {n_train} TRs x {n_pcs} PCs, {len(lengths_train)} sequences"
    )
    logger.info(
        f"Valid: {n_valid} TRs x {n_pcs} PCs, {len(lengths_valid)} sequences"
    )

    # Group validation runs by season for per-season LL diagnostic
    season_valid_runs = {}
    for run_id in split['valid']:
        s = _get_season(run_id)
        season_valid_runs.setdefault(s, []).append(run_id)

    # Cache per-season validation arrays once per config to avoid repeated I/O
    # in each seed loop.
    season_valid_cache = {}
    for s in sorted(season_valid_runs.keys()):
        sv_runs = season_valid_runs[s]
        if not sv_runs:
            continue
        try:
            X_sv, lengths_sv = load_projected_runs(
                sv_runs, projected_dir, n_pcs, 'valid'
            )
            season_valid_cache[s] = (X_sv, lengths_sv)
        except Exception as e_sv:
            logger.warning(f"Season {s} validation cache load failed: {e_sv}")

    output_dir = os.path.join(
        get_output_base(sub_id, parcellation), 'configs', cfg_name
    )
    os.makedirs(output_dir, exist_ok=True)

    seed_results = []

    for seed_idx in range(n_fit_seeds):
        seed = seed_idx
        seed_result_path = os.path.join(output_dir, f'seed_{seed}.json')

        # Resume: skip seeds already completed (allows SLURM restart after timeout)
        if os.path.exists(seed_result_path):
            try:
                with open(seed_result_path, 'r') as f:
                    result = json.load(f)
                if result.get('status') in ('success', 'failed'):
                    seed_results.append(result)
                    logger.info(f"Seed {seed}: already done, loading result")
                    continue
                else:
                    logger.warning(f"Seed {seed}: incomplete result, refitting")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Seed {seed}: corrupt JSON ({e}), refitting")

        logger.info(f"Seed {seed}/{n_fit_seeds - 1}: fitting...")
        start_time = time.time()

        try:
            np.random.seed(seed)
            random.seed(seed)

            model = WeakLimitHMM(
                n_components=config['n_components'],
                covariance_type=config['covariance_type'],
                alpha=config['alpha'],
                gamma=config['gamma'],
                kappa=config['kappa'],
                rho=config['rho'],
                n_iter=config['n_iter'],
                tol=config['tol'],
                min_state_usage=config['min_state_usage'],
                random_state=seed,
                verbose=False,
                n_jobs=n_jobs,
            )

            model.fit(X_train, lengths=lengths_train)

            train_ll = model.score(X_train, lengths=lengths_train)
            valid_ll = model.score(X_valid, lengths=lengths_valid)
            train_ll_ps = train_ll / n_train
            valid_ll_ps = valid_ll / n_valid

            n_active = -1
            if hasattr(model, 'history') and model.history:
                usage = model.history.get('state_usage')
                if usage:
                    n_active = int(
                        np.sum(np.array(usage[-1]) > config['min_state_usage'])
                    )

            elapsed = time.time() - start_time
            converged = getattr(model, 'converged_', False)
            n_iters = (
                len(model.history.get('log_likelihood', []))
                if hasattr(model, 'history') and model.history else -1
            )

            # Per-season validation LL (heuristic non-stationarity diagnostic).
            # With N=6 seasons, IQR is unreliable as a statistical test.
            # This is a qualitative warning only - LOSO (loso_fit mode) provides
            # the definitive cross-season generalization test.
            valid_ll_per_season = {}
            for s in sorted(season_valid_cache.keys()):
                try:
                    X_sv, lengths_sv = season_valid_cache[s]
                    sv_ll = model.score(X_sv, lengths=lengths_sv) / sum(lengths_sv)
                    valid_ll_per_season[f's{s}'] = float(sv_ll)
                except Exception as e_sv:
                    logger.warning(f"  Season {s} valid LL scoring failed: {e_sv}")

            # Tukey fence on per-season LL values (heuristic only; N=6 makes
            # IQR unstable - treat any flagged season as a signal for LOSO, not
            # a definitive finding)
            if len(valid_ll_per_season) >= 3:
                season_lls = np.array(list(valid_ll_per_season.values()))
                season_keys = list(valid_ll_per_season.keys())
                median_ll = np.median(season_lls)
                iqr_ll = (
                    np.percentile(season_lls, 75) - np.percentile(season_lls, 25)
                )
                lower_bound = median_ll - 1.5 * iqr_ll
                for s_key, ll in zip(season_keys, season_lls):
                    logger.info(f"  {s_key} valid LL/TR = {ll:.4f}")
                    if iqr_ll > 0 and ll < lower_bound:
                        logger.warning(
                            f"  HEURISTIC ALERT: {s_key} valid LL ({ll:.4f}) below "
                            f"Tukey fence (median={median_ll:.4f}, IQR={iqr_ll:.4f}). "
                            "LOSO analysis will quantify cross-season generalization."
                        )

            result = {
                'seed': seed,
                'train_ll': float(train_ll),
                'valid_ll': float(valid_ll),
                'train_ll_per_sample': float(train_ll_ps),
                'valid_ll_per_sample': float(valid_ll_ps),
                'valid_ll_per_season': valid_ll_per_season,
                'n_train_samples': n_train,
                'n_valid_samples': n_valid,
                'n_active_states': n_active,
                'converged': converged,
                'n_iterations': n_iters,
                'elapsed_seconds': round(elapsed, 2),
                'status': 'success',
            }

            model_path = os.path.join(output_dir, f'seed_{seed}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=4)
            del model  # free memory before next seed

            logger.info(
                f"Seed {seed}: train_ll/s={train_ll_ps:.4f}, "
                f"valid_ll/s={valid_ll_ps:.4f}, "
                f"active_states={n_active}, {elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Seed {seed} failed: {type(e).__name__}: {e}")
            result = {
                'seed': seed,
                'train_ll': None,
                'valid_ll': None,
                'train_ll_per_sample': None,
                'valid_ll_per_sample': None,
                'valid_ll_per_season': {},
                'n_train_samples': n_train,
                'n_valid_samples': n_valid,
                'n_active_states': -1,
                'converged': False,
                'n_iterations': -1,
                'elapsed_seconds': round(elapsed, 2),
                'status': 'failed',
                'error': f'{type(e).__name__}: {str(e)}',
            }

        with open(seed_result_path, 'w') as f:
            json.dump(result, f, indent=2)
        seed_results.append(result)

    # Aggregate across seeds
    successful = [r for r in seed_results if r['status'] == 'success']
    summary = {
        'config': config,
        'config_name': cfg_name,
        'sub_id': sub_id,
        'parcellation': parcellation,
        'n_pcs': n_pcs,
        'n_seeds_total': n_fit_seeds,
        'n_seeds_successful': len(successful),
        'n_seeds_failed': n_fit_seeds - len(successful),
        'seed_results': seed_results,
        'timestamp': datetime.now().isoformat(),
    }

    if successful:
        valid_lls = [r['valid_ll_per_sample'] for r in successful]
        train_lls = [r['train_ll_per_sample'] for r in successful]
        best_train_idx = int(np.argmax(train_lls))
        best_valid_idx = int(np.argmax(valid_lls))
        summary['best_seed_by_train'] = successful[best_train_idx]['seed']
        summary['best_seed_by_valid'] = successful[best_valid_idx]['seed']
        summary['best_train_ll_per_sample'] = float(max(train_lls))
        summary['best_seed_valid_ll_per_sample'] = float(successful[best_train_idx]['valid_ll_per_sample'])
        summary['mean_valid_ll_per_sample'] = float(np.mean(valid_lls))
        summary['std_valid_ll_per_sample'] = float(np.std(valid_lls))
        summary['best_valid_ll_per_sample'] = float(max(valid_lls))
    else:
        logger.error("All seeds failed for this config!")

    summary_path = os.path.join(output_dir, 'config_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Config summary saved to {summary_path}")
    if successful:
        logger.info(
            f"Best seed (by train LL): {summary['best_seed_by_train']}, "
            f"that seed's valid_ll/s={summary['best_seed_valid_ll_per_sample']:.4f}"
        )
    return summary


# =============================================================================
# Select mode
# =============================================================================

def _collect_config_scores(configs_dir, config_grid, selection_metric):
    """Collect config summaries and compute scores for a grid of configs.

    Args:
        configs_dir: path to configs/ directory containing per-config subdirs
        config_grid: list of config dicts
        selection_metric: scoring metric name

    Returns:
        List of score dicts, sorted by score descending.
    """
    config_scores = []
    for cfg_idx, cfg in enumerate(config_grid):
        cfg_name_str = config_name(cfg)
        summary_path = os.path.join(configs_dir, cfg_name_str, 'config_summary.json')
        if not os.path.exists(summary_path):
            logger.warning(f"  {cfg_name_str}: summary not found, skipping")
            continue
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        if summary.get('n_seeds_successful', 0) == 0:
            logger.warning(f"  {cfg_name_str}: no successful seeds, skipping")
            continue
        config_scores.append({
            'config_index': cfg_idx,
            'config_name': cfg_name_str,
            'config': cfg,
            'mean_valid_ll': summary['mean_valid_ll_per_sample'],
            'best_valid_ll': summary['best_valid_ll_per_sample'],
            'std_valid_ll': summary['std_valid_ll_per_sample'],
            'best_seed_valid_ll': summary.get('best_seed_valid_ll_per_sample'),
            'n_successful': summary['n_seeds_successful'],
            'score': _compute_selection_score(summary, selection_metric),
        })
    config_scores.sort(key=lambda x: x['score'], reverse=True)
    return config_scores


def _run_stage1_selection(sub_id, parcellation):
    """Stage 1: select variance threshold by per-dimension validation LL.

    Reads Stage 1 fit results, selects best vt, saves stage1_result.json.

    Returns:
        float: selected variance threshold
    """
    output_base = get_output_base(sub_id, parcellation)
    configs_dir = os.path.join(output_base, 'configs')

    logger.info("\n--- Stage 1: Variance Threshold Selection ---")
    stage1_grid = build_stage1_grid()
    stage1_scores = _collect_config_scores(configs_dir, stage1_grid, 'll_per_dim')

    if not stage1_scores:
        logger.error("No Stage 1 configs found! Run --mode fit --stage 1 first.")
        sys.exit(1)

    best_s1 = stage1_scores[0]
    selected_vt = best_s1['config']['variance_threshold']

    logger.info(f"Stage 1 best: {best_s1['config_name']} (score={best_s1['score']:.6f})")
    logger.info("Stage 1 rankings:")
    for i, cs in enumerate(stage1_scores):
        logger.info(
            f"  {i + 1}. {cs['config_name']}: "
            f"mean_valid_ll={cs['mean_valid_ll']:.6f}, "
            f"ll_per_dim={cs['score']:.6f}"
        )

    # Save Stage 1 result
    stage1_result = {
        'selected_vt': selected_vt,
        'rankings': [
            {
                'rank': i + 1,
                'config_name': cs['config_name'],
                'vt': cs['config']['variance_threshold'],
                'mean_valid_ll': cs['mean_valid_ll'],
                'score_ll_per_dim': cs['score'],
            }
            for i, cs in enumerate(stage1_scores)
        ],
        'timestamp': datetime.now().isoformat(),
    }
    stage1_path = os.path.join(output_base, 'stage1_result.json')
    with open(stage1_path, 'w') as f:
        json.dump(stage1_result, f, indent=2)
    logger.info(f"Stage 1 result saved: selected_vt={selected_vt}")

    return selected_vt


def select_and_refit(sub_id, parcellation, n_final_seeds, force_refit=False, n_jobs=1,
                     selection_metric='bic', fixed_vt=None, select_config=None):
    """Model selection, final refit, decode, and back-projection.

    Steps:
      1. Select config (manual override via select_config, or two-stage auto)
      2. Refit on train+valid combined (n_final_seeds seeds)
      3. Evaluate on test set; report vs. random-assignment baseline
      4. Decode all episodes (train + valid + test)
      5. Back-project state means to parcel space
      6. Save all final outputs
    """
    combined_base = get_combined_base(sub_id, parcellation)
    output_base = get_output_base(sub_id, parcellation)
    configs_dir = os.path.join(output_base, 'configs')

    logger.info(f"\n{'=' * 60}")
    logger.info(f"COMBINED MODEL - Model Selection: {sub_id}")
    logger.info(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # [1] Config selection (manual override or two-stage auto)
    # -------------------------------------------------------------------------
    if select_config is not None:
        config_scores = _resolve_config_override(configs_dir, select_config, selection_metric)
        best = config_scores[0]
        selected_vt = best['config']['variance_threshold']
        logger.info(f"\nManual config override: {select_config} (vt={selected_vt})")
    else:
        if fixed_vt is not None:
            selected_vt = fixed_vt
            logger.info(f"\nUsing fixed variance threshold (bypassing Stage 1): vt={selected_vt}")
        else:
            selected_vt = _load_stage1_result(sub_id, parcellation)
            if selected_vt is None:
                selected_vt = _run_stage1_selection(sub_id, parcellation)
            logger.info(f"\nUsing variance threshold from Stage 1: vt={selected_vt}")

        logger.info(f"\n--- Stage 2: K/gamma/cov Selection (metric={selection_metric}) ---")
        stage2_grid = build_stage2_grid(selected_vt)
        config_scores = _collect_config_scores(configs_dir, stage2_grid, selection_metric)

        if not config_scores:
            logger.error(
                "No Stage 2 configs found! Run --mode fit --stage 2 first."
            )
            sys.exit(1)

        best = config_scores[0]

    final_dir = get_final_dir(output_base, selected_vt)
    final_result_path = os.path.join(final_dir, 'final_results.json')
    if os.path.exists(final_result_path) and not force_refit:
        logger.info(f"Final results exist at {final_dir}. Use --force_refit to redo.")
        return

    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)

    logger.info(f"Best config: {best['config_name']}")
    logger.info(f"Selection metric: {selection_metric}")
    logger.info(
        f"Mean valid LL/sample: {best['mean_valid_ll']:.6f} "
        f"+/- {best['std_valid_ll']:.6f} (score={best['score']:.6f})"
    )

    # Warn if top-2 are statistically indistinguishable
    # For BIC, std_valid_ll is on the wrong scale (validation LL vs BIC-penalized
    # training LL). Use a relative threshold instead: if score diff < 1% of
    # the absolute score range, the configs are effectively indistinguishable.
    if len(config_scores) >= 2:
        top1 = config_scores[0]
        top2 = config_scores[1]
        diff = abs(top1['score'] - top2['score'])
        if selection_metric == 'bic':
            # Relative threshold: 1% of score magnitude
            threshold = 0.01 * max(abs(top1['score']), abs(top2['score']), 1e-10)
        else:
            threshold = max(top1['std_valid_ll'], top2['std_valid_ll'])
        if threshold > 0 and diff < threshold:
            logger.warning(
                f"STATISTICAL WARNING: Top-2 configs indistinguishable - "
                f"score diff={diff:.6f} < threshold={threshold:.6f}. "
                "Prefer the simpler config (fewer states or diagonal cov)."
            )

    logger.info("\nTop 5 configs:")
    for i, cs in enumerate(config_scores[:5]):
        logger.info(
            f"  {i + 1}. {cs['config_name']}: "
            f"mean={cs['mean_valid_ll']:.6f} +/- {cs['std_valid_ll']:.6f} "
            f"(score={cs['score']:.6f})"
        )

    # -------------------------------------------------------------------------
    # [3] Refit on train+valid combined
    # -------------------------------------------------------------------------
    selected_config = best['config']
    vt_key = f"{selected_config['variance_threshold']:.2f}"
    n_pcs = n_pcs_lookup[vt_key]

    logger.info(
        f"\nRefitting on train+valid "
        f"({len(split['train'])} + {len(split['valid'])} = "
        f"{len(split['train']) + len(split['valid'])} runs, {n_pcs} PCs)..."
    )

    train_data, lengths_train = load_projected_runs(
        split['train'], projected_dir, n_pcs, 'train'
    )
    valid_data, lengths_valid = load_projected_runs(
        split['valid'], projected_dir, n_pcs, 'valid'
    )
    X_trainval = np.vstack([train_data, valid_data])
    lengths_trainval = lengths_train + lengths_valid
    n_trainval = sum(lengths_trainval)

    X_test, lengths_test = load_projected_runs(
        split['test'], projected_dir, n_pcs, 'test'
    )
    n_test = sum(lengths_test)
    if n_test <= 0:
        logger.error("No test data available!")
        sys.exit(1)

    os.makedirs(final_dir, exist_ok=True)

    best_final_model = None
    best_final_ll = -np.inf
    best_final_seed = -1
    final_seed_results = []

    seeds_dir = os.path.join(final_dir, 'seeds')
    os.makedirs(seeds_dir, exist_ok=True)

    expected_config_name = best.get('config_name', '')

    for seed in range(n_final_seeds):
        seed_result_path = os.path.join(seeds_dir, f'seed_{seed}.json')
        seed_model_path = os.path.join(seeds_dir, f'seed_{seed}_model.pkl')

        if os.path.exists(seed_result_path) and not force_refit:
            try:
                with open(seed_result_path, 'r') as f:
                    res = json.load(f)
                if res.get('status') in ('success', 'failed'):
                    saved_config = res.get('config_name', '')
                    if saved_config and saved_config != expected_config_name:
                        logger.warning(
                            f"  Seed {seed}: config mismatch "
                            f"(saved={saved_config}, expected={expected_config_name}), refitting")
                    else:
                        logger.info(f"  Final refit seed {seed}/{n_final_seeds - 1}: already done, loading result ({res['status']})")
                        final_seed_results.append(res)
                        if res.get('status') == 'success' and res['trainval_ll_per_sample'] > best_final_ll:
                            best_final_ll = res['trainval_ll_per_sample']
                            best_final_seed = res['seed']
                            best_final_model = None  # Will be loaded later if this remains best
                        continue
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"  Seed {seed}: corrupt JSON ({e}), refitting")

        logger.info(f"  Final refit seed {seed}/{n_final_seeds - 1}...")
        start_time = time.time()
        np.random.seed(seed)
        random.seed(seed)

        model = WeakLimitHMM(
            n_components=selected_config['n_components'],
            covariance_type=selected_config['covariance_type'],
            alpha=selected_config['alpha'],
            gamma=selected_config['gamma'],
            kappa=selected_config['kappa'],
            rho=selected_config['rho'],
            n_iter=selected_config['n_iter'],
            tol=selected_config['tol'],
            min_state_usage=selected_config['min_state_usage'],
            random_state=seed,
            verbose=False,
            n_jobs=n_jobs,
        )

        try:
            model.fit(X_trainval, lengths=lengths_trainval)
            trainval_ll = model.score(X_trainval, lengths=lengths_trainval)
            trainval_ll_ps = trainval_ll / n_trainval
            elapsed = time.time() - start_time

            with open(seed_model_path, 'wb') as f:
                pickle.dump(model, f, protocol=4)

            res = {
                'seed': seed,
                'config_name': expected_config_name,
                'trainval_ll_per_sample': float(trainval_ll_ps),
                'elapsed_seconds': round(elapsed, 2),
                'status': 'success',
            }
            with open(seed_result_path, 'w') as f:
                json.dump(res, f, indent=2)

            final_seed_results.append(res)

            if trainval_ll_ps > best_final_ll:
                best_final_ll = trainval_ll_ps
                best_final_model = model
                best_final_seed = seed

            logger.info(f"    trainval_ll/s={trainval_ll_ps:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  Final refit seed {seed} failed: {e}")
            res = {
                'seed': seed,
                'status': 'failed',
                'error': str(e),
                'elapsed_seconds': round(elapsed, 2),
            }
            with open(seed_result_path, 'w') as f:
                json.dump(res, f, indent=2)
            final_seed_results.append(res)

    if best_final_seed != -1 and best_final_model is None:
        best_model_path = os.path.join(seeds_dir, f'seed_{best_final_seed}_model.pkl')
        logger.info(f"  Loading best model from {best_model_path}")
        with open(best_model_path, 'rb') as f:
            best_final_model = pickle.load(f)

    if best_final_model is None:
        logger.error("All final refit seeds failed!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # [4] Evaluate on test set
    # -------------------------------------------------------------------------
    test_ll = best_final_model.score(X_test, lengths=lengths_test)
    test_ll_ps = test_ll / n_test
    n_states = int(best_final_model.n_components)
    n_active_states = max(
        1,
        infer_n_active_states(
            best_final_model,
            min_state_usage=selected_config['min_state_usage'],
        ),
    )

    # Report both baselines for schema/backward compatibility:
    # total-state baseline (lenient) and active-state baseline (stricter).
    baseline_ll_per_tr_total = np.log(1.0 / n_states)
    baseline_ll_per_tr_active = np.log(1.0 / n_active_states)
    generalizes_total = bool(test_ll_ps > baseline_ll_per_tr_total)
    generalizes_active = bool(test_ll_ps > baseline_ll_per_tr_active)
    overfit_gap = float(best_final_ll - test_ll_ps)

    logger.info(f"\nTEST LL/sample: {test_ll_ps:.6f} (seed {best_final_seed})")
    logger.info(
        f"Baseline(total states) LL/sample: {baseline_ll_per_tr_total:.6f} "
        f"(log(1/{n_states}))"
    )
    logger.info(
        f"Baseline(active states) LL/sample: {baseline_ll_per_tr_active:.6f} "
        f"(log(1/{n_active_states}))"
    )
    logger.info(f"Generalizes vs total-state baseline: {generalizes_total}")
    logger.info(f"Generalizes vs active-state baseline: {generalizes_active}")
    logger.info(f"Overfit gap (trainval - test): {overfit_gap:.6f}")

    # Per-season test LL
    test_by_season = {}
    for run_id in split['test']:
        s = _get_season(run_id)
        test_by_season.setdefault(s, []).append(run_id)

    test_ll_per_season = {}
    for s, s_test_runs in sorted(test_by_season.items()):
        try:
            X_st, lengths_st = load_projected_runs(
                s_test_runs, projected_dir, n_pcs, 'test'
            )
            st_ll = best_final_model.score(X_st, lengths=lengths_st) / sum(lengths_st)
            test_ll_per_season[f's{s}'] = float(st_ll)
            logger.info(f"  Season {s} test LL/TR: {st_ll:.6f}")
        except Exception as e_ts:
            logger.warning(f"  Season {s} test LL failed: {e_ts}")

    # -------------------------------------------------------------------------
    # [5] Decode all episodes (train + valid + test)
    # -------------------------------------------------------------------------
    logger.info("\nDecoding all episodes (train + valid + test)...")
    decoded_states = decode_all_runs(best_final_model, split, projected_dir, n_pcs)
    logger.info(f"Decoded {len(decoded_states)} runs")

    # -------------------------------------------------------------------------
    # [6] Back-project state means + covariances to parcel space
    # -------------------------------------------------------------------------
    pca_path = os.path.join(combined_base, 'pca_model.pkl')
    state_means_parcel = None
    state_covars_parcel = None
    if os.path.exists(pca_path):
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        state_means_parcel, state_covars_parcel = back_project_states(
            best_final_model, pca, n_pcs
        )
        # Copy PCA model to final dir for self-contained reference
        with open(os.path.join(final_dir, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f, protocol=4)
    else:
        logger.warning(f"PCA model not found at {pca_path}; skipping back-projection")

    # -------------------------------------------------------------------------
    # [7] Save all final outputs
    # -------------------------------------------------------------------------
    with open(os.path.join(final_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_final_model, f, protocol=4)

    with open(os.path.join(final_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded_states, f, protocol=4)

    np.save(os.path.join(final_dir, 'state_means_pca.npy'), best_final_model.means_)

    if state_means_parcel is not None:
        np.save(os.path.join(final_dir, 'state_means_parcel.npy'), state_means_parcel)

    if state_covars_parcel is not None:
        np.save(os.path.join(final_dir, 'state_covars_parcel.npy'), state_covars_parcel)

    np.save(os.path.join(final_dir, 'state_covars.npy'), best_final_model.covars_)

    final_result = {
        'sub_id': sub_id,
        'parcellation': parcellation,
        'selected_config': selected_config,
        'selected_config_name': best['config_name'],
        'selection_metric': selection_metric,
        'all_config_rankings': [
            {
                'rank': i + 1,
                'config_name': cs['config_name'],
                'mean_valid_ll': cs['mean_valid_ll'],
                'std_valid_ll': cs['std_valid_ll'],
                'score': cs['score'],
            }
            for i, cs in enumerate(config_scores[:10])
        ],
        'final_refit': {
            'n_seeds': n_final_seeds,
            'best_seed': best_final_seed,
            'trainval_ll_per_sample': float(best_final_ll),
            'test_ll_per_sample': float(test_ll_ps),
            'test_ll_per_season': test_ll_per_season,
            # Explicit dual-baseline schema.
            'baseline_ll_per_sample_total_states': float(baseline_ll_per_tr_total),
            'baseline_ll_per_sample_active_states': float(baseline_ll_per_tr_active),
            'generalizes_vs_baseline_total_states': generalizes_total,
            'generalizes_vs_baseline_active_states': generalizes_active,
            'n_total_states': int(n_states),
            'n_active_states': int(n_active_states),
            'overfit_gap': overfit_gap,
            'seed_results': final_seed_results,
        },
        'data_info': {
            'n_train_runs': len(split['train']),
            'n_valid_runs': len(split['valid']),
            'n_test_runs': len(split['test']),
            'n_trainval_samples': n_trainval,
            'n_test_samples': n_test,
            'n_pcs': n_pcs,
            'variance_threshold': selected_config['variance_threshold'],
        },
        'model_info': {
            'n_states': n_states,
        },
        'model_selection': {
            'method': 'manual_override' if select_config else 'two_stage',
            'select_config_override': select_config,
            'stage1_metric': 'll_per_dim',
            'stage1_selected_vt': selected_config['variance_threshold'],
            'stage2_metric': selection_metric,
        },
        'schema_info': {
            'schema_version': '3.1',
            'baseline_primary': 'active_states',
            'legacy_keys_preserved': False,
        },
        'n_decoded_runs': len(decoded_states),
        'timestamp': datetime.now().isoformat(),
    }

    with open(final_result_path, 'w') as f:
        json.dump(final_result, f, indent=2)
    logger.info(f"\nFinal results saved to {final_dir}")


# =============================================================================
# Select Seed and Finalize modes (Parallelized select)
# =============================================================================

def _compute_selection_score(summary, selection_metric):
    """Compute config selection score from a config_summary dict.

    Metrics:
      'll_per_sample': raw mean validation LL/sample (legacy default)
      'll_per_dim': per-dimension validation LL = valid_ll/sample / n_pcs
                    (Stage 1 metric - NOTE: biased toward higher vt because
                    low-variance PCs with λ<1 contribute positive -½·log(λ) terms,
                    inflating LL/dim. Use --fixed_vt to bypass Stage 1 with an
                    externally validated vt instead.)
      'bic': BIC on training LL with effective-K parameter counting
             (Stage 2 metric - principled complexity penalty)

    LOSO is the definitive generalization test and takes precedence.
    """
    n_pcs = summary['n_pcs']
    mean_ll = summary['mean_valid_ll_per_sample']

    if selection_metric == 'll_per_sample':
        return mean_ll

    elif selection_metric == 'll_per_dim':
        return mean_ll / n_pcs

    elif selection_metric == 'll_per_sample_per_pc':
        # Legacy alias for ll_per_dim
        return mean_ll / n_pcs

    elif selection_metric == 'bic':
        successful = [r for r in summary['seed_results'] if r.get('status') == 'success']
        if not successful:
            return -np.inf

        # BIC uses training LL (not validation LL) - BIC is derived as an
        # approximation to the marginal likelihood under training data.
        # Using validation LL would double-count regularization.
        #
        # Self-consistency: use best seed's train LL AND K_eff together.
        # Averaging LL across seeds but taking K_eff from best seed is
        # inconsistent when seeds find different local optima with
        # different active state counts.
        best_seed = max(
            [r for r in successful
             if r.get('train_ll_per_sample') is not None
             and r.get('n_active_states', -1) > 0],
            key=lambda r: r['train_ll_per_sample'],
            default=None,
        )
        if best_seed is None:
            return -np.inf

        best_train_ll = float(best_seed['train_ll_per_sample'])
        K = int(best_seed['n_active_states'])

        n_bic = int(best_seed.get('n_train_samples', 0))
        if n_bic <= 1:
            return -np.inf

        D = n_pcs
        cov = summary['config']['covariance_type']
        # Parameter counting for BIC penalty:
        #   means + covariances + transition rows + initial state distribution.
        #   HDP beta weights excluded (prior on transitions, not likelihood params).
        #   startprob_ adds K-1 free params (~1.3% of total for typical K).
        if cov == 'diag':
            n_params = 2 * K * D + K * (K - 1) + (K - 1)
        else:  # full
            n_params = K * D + K * D * (D + 1) // 2 + K * (K - 1) + (K - 1)
        bic_penalty_per_sample = n_params * np.log(n_bic) / (2 * n_bic)
        return best_train_ll - bic_penalty_per_sample

    return mean_ll  # fallback


def _get_best_config(sub_id, parcellation, selection_metric='bic', fixed_vt=None,
                     select_config=None):
    """Get best Stage 2 config using two-stage selection or manual override.

    Loads Stage 1 result (selected vt), then scores Stage 2 configs.
    If fixed_vt is provided, bypasses Stage 1 result loading entirely.
    If select_config is provided, bypasses all auto-selection and uses
    the specified config directly (loads its config_summary.json).
    """
    output_base = get_output_base(sub_id, parcellation)
    configs_dir = os.path.join(output_base, 'configs')

    if select_config is not None:
        return _resolve_config_override(configs_dir, select_config, selection_metric)

    if fixed_vt is not None:
        selected_vt = fixed_vt
        logger.info(f"Using fixed variance threshold (bypassing Stage 1): vt={selected_vt}")
    else:
        selected_vt = _load_stage1_result(sub_id, parcellation)
        if selected_vt is None:
            selected_vt = _run_stage1_selection(sub_id, parcellation)

    stage2_grid = build_stage2_grid(selected_vt)
    config_scores = _collect_config_scores(configs_dir, stage2_grid, selection_metric)

    if not config_scores:
        logger.error("No valid Stage 2 configs found!")
        sys.exit(1)
    return config_scores


def _resolve_config_override(configs_dir, select_config, selection_metric):
    """Resolve a manual config override by loading its config_summary.json.

    Returns a config_scores-compatible list with a single entry.
    """
    summary_path = os.path.join(configs_dir, select_config, 'config_summary.json')
    if not os.path.isfile(summary_path):
        available = sorted([
            d for d in os.listdir(configs_dir)
            if os.path.isfile(os.path.join(configs_dir, d, 'config_summary.json'))
        ])
        logger.error(f"Config not found: {summary_path}")
        logger.error(f"Available configs: {available[:20]}")
        sys.exit(1)

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    if summary.get('n_seeds_successful', 0) == 0:
        logger.error(f"Config {select_config} has no successful seeds!")
        sys.exit(1)

    config = summary['config']
    logger.info(f"Manual config override: {select_config}")
    logger.info(f"  nc={config['n_components']}, gamma={config['gamma']}, "
                f"cov={config['covariance_type']}, vt={config['variance_threshold']}")

    return [{
        'config_index': 0,
        'config_name': select_config,
        'config': config,
        'mean_valid_ll': summary['mean_valid_ll_per_sample'],
        'best_valid_ll': summary['best_valid_ll_per_sample'],
        'std_valid_ll': summary['std_valid_ll_per_sample'],
        'best_seed_valid_ll': summary.get('best_seed_valid_ll_per_sample'),
        'n_successful': summary['n_seeds_successful'],
        'score': _compute_selection_score(summary, selection_metric),
        'selection_method': 'manual_override',
    }]

def select_seed(sub_id, parcellation, seed_index, force_refit=False, n_jobs=1,
                selection_metric='bic', fixed_vt=None, select_config=None):
    combined_base = get_combined_base(sub_id, parcellation)
    output_base = get_output_base(sub_id, parcellation)

    config_scores = _get_best_config(sub_id, parcellation, selection_metric=selection_metric,
                                     fixed_vt=fixed_vt, select_config=select_config)
    best = config_scores[0]
    selected_config = best['config']

    # Extract vt from the selected config (critical for correct output dir)
    effective_vt = selected_config['variance_threshold']
    selected_vt = _resolve_vt(sub_id, parcellation, fixed_vt=effective_vt)
    final_dir = get_final_dir(output_base, selected_vt)
    seeds_dir = os.path.join(final_dir, 'seeds')
    os.makedirs(seeds_dir, exist_ok=True)

    seed_result_path = os.path.join(seeds_dir, f'seed_{seed_index}.json')
    seed_model_path = os.path.join(seeds_dir, f'seed_{seed_index}_model.pkl')

    expected_config_name = best.get('config_name', '')
    if os.path.exists(seed_result_path) and not force_refit:
        try:
            with open(seed_result_path, 'r') as f: res = json.load(f)
            if res.get('status') in ('success', 'failed'):
                saved_config = res.get('config_name', '')
                if saved_config and saved_config != expected_config_name:
                    logger.warning(
                        f"Seed {seed_index}: config mismatch "
                        f"(saved={saved_config}, expected={expected_config_name}), refitting")
                else:
                    logger.info(f"Seed {seed_index} already done ({res['status']})")
                    return
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Seed {seed_index}: corrupt JSON ({e}), refitting")

    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    vt_key = f"{selected_config['variance_threshold']:.2f}"
    n_pcs = n_pcs_lookup[vt_key]

    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)

    train_data, lengths_train = load_projected_runs(split['train'], projected_dir, n_pcs, 'train')
    valid_data, lengths_valid = load_projected_runs(split['valid'], projected_dir, n_pcs, 'valid')
    X_trainval = np.vstack([train_data, valid_data])
    lengths_trainval = lengths_train + lengths_valid
    n_trainval = sum(lengths_trainval)
    
    logger.info(f"Fitting final refit seed {seed_index}...")
    start_time = time.time()
    np.random.seed(seed_index)
    random.seed(seed_index)
    
    model = WeakLimitHMM(
        n_components=selected_config['n_components'],
        covariance_type=selected_config['covariance_type'],
        alpha=selected_config['alpha'], gamma=selected_config['gamma'],
        kappa=selected_config['kappa'], rho=selected_config['rho'],
        n_iter=selected_config['n_iter'], tol=selected_config['tol'],
        min_state_usage=selected_config['min_state_usage'],
        random_state=seed_index, verbose=False, n_jobs=n_jobs)
        
    try:
        model.fit(X_trainval, lengths=lengths_trainval)
        trainval_ll = model.score(X_trainval, lengths=lengths_trainval)
        trainval_ll_ps = trainval_ll / n_trainval
        elapsed = time.time() - start_time
        
        with open(seed_model_path, 'wb') as f: pickle.dump(model, f, protocol=4)
        res = {'seed': seed_index, 'config_name': expected_config_name, 'trainval_ll_per_sample': float(trainval_ll_ps), 'elapsed_seconds': round(elapsed, 2), 'status': 'success'}
        logger.info(f"Seed {seed_index} trainval_ll/s={trainval_ll_ps:.4f} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Seed {seed_index} failed: {e}")
        res = {'seed': seed_index, 'config_name': expected_config_name, 'status': 'failed', 'error': str(e), 'elapsed_seconds': round(elapsed, 2)}
        
    with open(seed_result_path, 'w') as f: json.dump(res, f, indent=2)

def select_finalize(sub_id, parcellation, n_final_seeds, force_refit=False, n_jobs=1,
                    selection_metric='bic', fixed_vt=None, select_config=None):
    combined_base = get_combined_base(sub_id, parcellation)
    output_base = get_output_base(sub_id, parcellation)

    config_scores = _get_best_config(sub_id, parcellation, selection_metric=selection_metric,
                                     fixed_vt=fixed_vt, select_config=select_config)
    best = config_scores[0]
    selected_config = best['config']

    # Extract vt from the selected config (critical for correct output dir)
    effective_vt = selected_config['variance_threshold']
    selected_vt = _resolve_vt(sub_id, parcellation, fixed_vt=effective_vt)
    final_dir = get_final_dir(output_base, selected_vt)
    seeds_dir = os.path.join(final_dir, 'seeds')

    final_result_path = os.path.join(final_dir, 'final_results.json')
    if os.path.exists(final_result_path) and not force_refit:
        logger.info(f"Final results exist at {final_dir}. Use --force_refit to redo.")
        return

    n_pcs_lookup = load_n_pcs_lookup(combined_base)
    vt_key = f"{selected_config['variance_threshold']:.2f}"
    n_pcs = n_pcs_lookup[vt_key]
    split = load_split(combined_base, split_type='primary')
    projected_dir = get_projected_dir(combined_base)
    
    X_test, lengths_test = load_projected_runs(split['test'], projected_dir, n_pcs, 'test')
    n_test = sum(lengths_test)
    if n_test <= 0:
        logger.error("No test data available!")
        sys.exit(1)

    best_final_model, best_final_ll, best_final_seed = None, -np.inf, -1
    final_seed_results = []
    
    for seed in range(n_final_seeds):
        seed_result_path = os.path.join(seeds_dir, f'seed_{seed}.json')
        if not os.path.exists(seed_result_path):
            logger.error(f"Missing result for seed {seed}!")
            sys.exit(1)
        with open(seed_result_path, 'r') as f: res = json.load(f)
        final_seed_results.append(res)
        if res.get('status') == 'success' and res['trainval_ll_per_sample'] > best_final_ll:
            best_final_ll = res['trainval_ll_per_sample']
            best_final_seed = res['seed']
            
    if best_final_seed == -1:
        logger.error("No successful seeds found!")
        sys.exit(1)
        
    best_model_path = os.path.join(seeds_dir, f'seed_{best_final_seed}_model.pkl')
    with open(best_model_path, 'rb') as f: best_final_model = pickle.load(f)
    logger.info(f"Loaded best model from seed {best_final_seed}")
    
    test_ll_ps = best_final_model.score(X_test, lengths=lengths_test) / n_test
    n_states = int(best_final_model.n_components)
    n_active_states = max(1, infer_n_active_states(best_final_model, min_state_usage=selected_config['min_state_usage']))
    
    baseline_ll_per_tr_total = np.log(1.0 / n_states)
    baseline_ll_per_tr_active = np.log(1.0 / n_active_states)
    test_by_season = {}
    for r in split['test']: test_by_season.setdefault(_get_season(r), []).append(r)
    test_ll_per_season = {}
    for s, s_test_runs in sorted(test_by_season.items()):
        X_st, lengths_st = load_projected_runs(s_test_runs, projected_dir, n_pcs, 'test')
        test_ll_per_season[f's{s}'] = float(best_final_model.score(X_st, lengths=lengths_st) / sum(lengths_st))
        
    logger.info("Decoding all episodes...")
    decoded_states = decode_all_runs(best_final_model, split, projected_dir, n_pcs)
    
    pca_path = os.path.join(combined_base, 'pca_model.pkl')
    state_means_parcel = None
    state_covars_parcel = None
    if os.path.exists(pca_path):
        with open(pca_path, 'rb') as f: pca = pickle.load(f)
        state_means_parcel, state_covars_parcel = back_project_states(
            best_final_model, pca, n_pcs
        )
        with open(os.path.join(final_dir, 'pca_model.pkl'), 'wb') as f: pickle.dump(pca, f, protocol=4)

    with open(os.path.join(final_dir, 'best_model.pkl'), 'wb') as f: pickle.dump(best_final_model, f, protocol=4)
    with open(os.path.join(final_dir, 'decoded_states.pkl'), 'wb') as f: pickle.dump(decoded_states, f, protocol=4)
    np.save(os.path.join(final_dir, 'state_means_pca.npy'), best_final_model.means_)
    if state_means_parcel is not None: np.save(os.path.join(final_dir, 'state_means_parcel.npy'), state_means_parcel)
    if state_covars_parcel is not None: np.save(os.path.join(final_dir, 'state_covars_parcel.npy'), state_covars_parcel)
    np.save(os.path.join(final_dir, 'state_covars.npy'), best_final_model.covars_)
    
    train_data, lengths_train = load_projected_runs(split['train'], projected_dir, n_pcs, 'train')
    valid_data, lengths_valid = load_projected_runs(split['valid'], projected_dir, n_pcs, 'valid')
    
    final_result = {
        'sub_id': sub_id, 'parcellation': parcellation,
        'selected_config': selected_config, 'selected_config_name': best['config_name'],
        'selection_metric': selection_metric,
        'all_config_rankings': [{'rank': i+1, 'config_name': cs['config_name'], 'mean_valid_ll': cs['mean_valid_ll'], 'std_valid_ll': cs['std_valid_ll'], 'score': cs['score']} for i, cs in enumerate(config_scores[:10])],
        'final_refit': {
            'n_seeds': n_final_seeds, 'best_seed': best_final_seed,
            'trainval_ll_per_sample': float(best_final_ll), 'test_ll_per_sample': float(test_ll_ps),
            'test_ll_per_season': test_ll_per_season,
            'baseline_ll_per_sample_total_states': float(baseline_ll_per_tr_total),
            'baseline_ll_per_sample_active_states': float(baseline_ll_per_tr_active),
            'generalizes_vs_baseline_total_states': bool(test_ll_ps > baseline_ll_per_tr_total),
            'generalizes_vs_baseline_active_states': bool(test_ll_ps > baseline_ll_per_tr_active),
            'n_total_states': int(n_states), 'n_active_states': int(n_active_states),
            'overfit_gap': float(best_final_ll - test_ll_ps),
            'seed_results': final_seed_results
        },
        'data_info': {
            'n_train_runs': len(split['train']),
            'n_valid_runs': len(split['valid']),
            'n_test_runs': len(split['test']),
            'n_trainval_samples': sum(lengths_train) + sum(lengths_valid),
            'n_test_samples': n_test, 'n_pcs': n_pcs, 'variance_threshold': selected_config['variance_threshold']
        },
        'model_info': {'n_states': n_states},
        'model_selection': {
            'method': 'manual_override' if select_config else 'two_stage',
            'select_config_override': select_config,
            'stage1_metric': 'll_per_dim',
            'stage1_selected_vt': selected_config['variance_threshold'],
            'stage2_metric': selection_metric,
        },
        'schema_info': {'schema_version': '3.1', 'baseline_primary': 'active_states', 'legacy_keys_preserved': False},
        'n_decoded_runs': len(decoded_states),
        'timestamp': datetime.now().isoformat()
    }
    with open(final_result_path, 'w') as f: json.dump(final_result, f, indent=2)
    logger.info(f"Final results saved to {final_dir}")

# =============================================================================
# LOSO fit mode
# =============================================================================

def _refit_fold(selected_config, config_name_str, pca, n_pcs, split,
                projected_dir, output_dir, fold_label, n_seeds,
                n_jobs=1, force_refit=False, test_split_name='test'):
    """Generic multi-seed refit, decode, and save for an independent fold.

    Shared by loso_fit() and split_half_fit(). Handles:
    - Multi-seed refit with per-seed JSON checkpointing
    - Best model selection by trainval LL
    - Test LL evaluation with dual baselines
    - Decoding of test runs
    - Back-projection to parcel space
    - Saving common artifacts

    Args:
        selected_config:  Config dict from primary final_results.json
        config_name_str:  Expected config name string (for checkpointing validation)
        pca:              Fold-specific PCA model
        n_pcs:            Number of PCs to use
        split:            Dict with 'train', 'valid', 'test' run ID lists
        projected_dir:    Path to fold's projected data
        output_dir:       Where to save fold outputs
        fold_label:       Human-readable label for logging (e.g. "LOSO S3", "Half A")
        n_seeds:          Number of seeds for refit
        n_jobs:           Parallel jobs for E-step
        force_refit:      If True, redo even if results exist
        test_split_name:  Subdirectory name for loading test data (default 'test').
                          For split-half, pass 'valid' because the validation runs
                          are reused as proxy test data and live in projected/valid/.

    Returns:
        dict with keys: best_model, best_seed, trainval_ll_per_sample,
        test_ll_per_sample, n_states, n_active_states, baselines,
        decoded_states, seed_results, data_info
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load train+valid data
    train_data, lengths_train = load_projected_runs(
        split['train'], projected_dir, n_pcs, 'train'
    )
    valid_data, lengths_valid = load_projected_runs(
        split['valid'], projected_dir, n_pcs, 'valid'
    )
    X_trainval = np.vstack([train_data, valid_data])
    lengths_trainval = lengths_train + lengths_valid
    n_trainval = sum(lengths_trainval)

    # Load test data
    X_test, lengths_test = load_projected_runs(
        split['test'], projected_dir, n_pcs, test_split_name
    )
    n_test = sum(lengths_test)

    logger.info(
        f"{fold_label} split: train={len(split['train'])}, "
        f"valid={len(split['valid'])}, test={len(split['test'])} runs "
        f"({n_trainval} trainval TRs, {n_test} test TRs)"
    )

    if n_test <= 0:
        logger.error(f"{fold_label}: No test data available!")
        sys.exit(1)

    # Multi-seed refit with checkpointing
    seeds_dir = os.path.join(output_dir, 'seeds')
    os.makedirs(seeds_dir, exist_ok=True)

    best_model = None
    best_ll = -np.inf
    best_seed = -1
    seed_results = []

    for seed in range(n_seeds):
        seed_result_path = os.path.join(seeds_dir, f'seed_{seed}.json')
        seed_model_path = os.path.join(seeds_dir, f'seed_{seed}_model.pkl')

        if os.path.exists(seed_result_path) and not force_refit:
            try:
                with open(seed_result_path, 'r') as f:
                    res = json.load(f)
                if res.get('status') in ('success', 'failed'):
                    saved_config = res.get('config_name', '')
                    if saved_config and saved_config != config_name_str:
                        logger.warning(
                            f"  {fold_label} seed {seed}: config mismatch "
                            f"(saved={saved_config}, expected={config_name_str}), refitting")
                    else:
                        logger.info(
                            f"  {fold_label} seed {seed}/{n_seeds - 1}: "
                            f"already done ({res['status']})")
                        seed_results.append(res)
                        if (res.get('status') == 'success'
                                and res['trainval_ll_per_sample'] > best_ll):
                            best_ll = res['trainval_ll_per_sample']
                            best_seed = res['seed']
                            best_model = None  # lazy load
                        continue
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"  Seed {seed}: corrupt JSON ({e}), refitting")

        logger.info(f"  {fold_label} seed {seed}/{n_seeds - 1}...")
        start_time = time.time()
        np.random.seed(seed)
        random.seed(seed)

        model = WeakLimitHMM(
            n_components=selected_config['n_components'],
            covariance_type=selected_config['covariance_type'],
            alpha=selected_config['alpha'],
            gamma=selected_config['gamma'],
            kappa=selected_config['kappa'],
            rho=selected_config['rho'],
            n_iter=selected_config['n_iter'],
            tol=selected_config['tol'],
            min_state_usage=selected_config['min_state_usage'],
            random_state=seed,
            verbose=False,
            n_jobs=n_jobs,
        )

        try:
            model.fit(X_trainval, lengths=lengths_trainval)
            trainval_ll = model.score(X_trainval, lengths=lengths_trainval)
            trainval_ll_ps = trainval_ll / n_trainval
            elapsed = time.time() - start_time

            with open(seed_model_path, 'wb') as f:
                pickle.dump(model, f, protocol=4)

            res = {
                'seed': seed,
                'config_name': config_name_str,
                'trainval_ll_per_sample': float(trainval_ll_ps),
                'elapsed_seconds': round(elapsed, 2),
                'status': 'success',
            }
            with open(seed_result_path, 'w') as f:
                json.dump(res, f, indent=2)

            seed_results.append(res)

            if trainval_ll_ps > best_ll:
                best_ll = trainval_ll_ps
                best_seed = seed
                best_model = model

            logger.info(f"    trainval_ll/s={trainval_ll_ps:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  {fold_label} seed {seed} failed: {e}")
            res = {
                'seed': seed,
                'status': 'failed',
                'error': str(e),
                'elapsed_seconds': round(elapsed, 2),
            }
            with open(seed_result_path, 'w') as f:
                json.dump(res, f, indent=2)
            seed_results.append(res)

    # Load best model if not already in memory
    if best_seed != -1 and best_model is None:
        best_model_path = os.path.join(seeds_dir, f'seed_{best_seed}_model.pkl')
        logger.info(f"  Loading best model from {best_model_path}")
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)

    if best_model is None:
        logger.error(f"{fold_label}: All seeds failed!")
        sys.exit(1)

    # Evaluate on test set
    test_ll = best_model.score(X_test, lengths=lengths_test)
    test_ll_ps = test_ll / n_test
    overfit_gap = float(best_ll - test_ll_ps)

    # Dual baselines
    n_states = int(best_model.n_components)
    n_active = max(
        1,
        infer_n_active_states(
            best_model,
            min_state_usage=selected_config['min_state_usage'],
        ),
    )
    baseline_total = np.log(1.0 / n_states)
    baseline_active = np.log(1.0 / n_active)

    logger.info(f"\n{fold_label} TEST LL/sample: {test_ll_ps:.6f}")
    logger.info(
        f"Baseline(total={n_states}) LL/sample: {baseline_total:.6f}, "
        f"Baseline(active={n_active}) LL/sample: {baseline_active:.6f}"
    )
    logger.info(
        f"Generalizes vs total: {test_ll_ps > baseline_total}, "
        f"vs active: {test_ll_ps > baseline_active}"
    )
    logger.info(f"Overfit gap (trainval - test): {overfit_gap:.6f}")

    # Decode test runs
    test_only_split = {'train': [], 'valid': [], 'test': split['test']}
    decoded = decode_all_runs(best_model, test_only_split, projected_dir, n_pcs)
    logger.info(f"Decoded {len(decoded)} test runs")

    # Back-project to parcel space using this fold's PCA
    state_means_parcel, state_covars_parcel = back_project_states(
        best_model, pca, n_pcs
    )

    # Save common artifacts
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f, protocol=4)
    with open(os.path.join(output_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded, f, protocol=4)
    np.save(os.path.join(output_dir, 'state_means_parcel.npy'), state_means_parcel)
    np.save(os.path.join(output_dir, 'state_covars_parcel.npy'), state_covars_parcel)

    return {
        'best_model': best_model,
        'best_seed': best_seed,
        'trainval_ll_per_sample': float(best_ll),
        'test_ll_per_sample': float(test_ll_ps),
        'overfit_gap': overfit_gap,
        'n_states': n_states,
        'n_active_states': n_active,
        'baseline_total': float(baseline_total),
        'baseline_active': float(baseline_active),
        'generalizes_vs_total': bool(test_ll_ps > baseline_total),
        'generalizes_vs_active': bool(test_ll_ps > baseline_active),
        'decoded_states': decoded,
        'seed_results': seed_results,
        'data_info': {
            'n_train_runs': len(split['train']),
            'n_valid_runs': len(split['valid']),
            'n_test_runs': len(split['test']),
            'n_trainval_samples': n_trainval,
            'n_test_samples': n_test,
        },
    }


def loso_fit(sub_id, parcellation, loso_season, n_final_seeds, force_refit=False,
             n_jobs=1, fixed_vt=None):
    """Refit the winning config for one Leave-One-Season-Out fold (Strategy B).

    Strategy B directly tests the central scientific claim: do brain states
    generalize to a completely held-out season? If test LL/TR exceeds the
    uniform-state baseline, the states learned from other seasons have
    non-trivial predictive structure for the held-out season.

    Each LOSO fold uses its own PCA (fitted by 03a_pca4combined_hmm.py on that
    fold's training data only), ensuring no data leakage from the held-out season
    into the PCA coordinate system.

    Args:
        sub_id:        Subject ID
        parcellation:  Parcellation name
        loso_season:   Integer (1-6; 1-4 for sub-04) - season to hold out
        n_final_seeds: Number of seeds for refit
        force_refit:   If True, redo even if results exist
        fixed_vt:      Optional fixed variance threshold (bypasses stage1_result.json)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"LOSO FIT - Season {loso_season} held out: {sub_id}")
    logger.info(f"{'=' * 60}")

    combined_base = get_combined_base(sub_id, parcellation)
    output_base = get_output_base(sub_id, parcellation)

    # Resolve variance threshold (from stage1_result.json or --fixed_vt)
    selected_vt = _resolve_vt(sub_id, parcellation, fixed_vt=fixed_vt)
    logger.info(f"Using variance threshold: vt={selected_vt}")

    # Load best config from primary Strategy A selection
    final_dir = get_final_dir(output_base, selected_vt)
    final_result_path = os.path.join(final_dir, 'final_results.json')
    if not os.path.exists(final_result_path):
        logger.error(
            f"Primary select results not found: {final_result_path}\n"
            "Run --mode select first."
        )
        sys.exit(1)
    with open(final_result_path, 'r') as f:
        primary_results = json.load(f)
    selected_config = primary_results['selected_config']
    logger.info(f"Using config: {primary_results['selected_config_name']}")

    # LOSO output directory
    loso_output_dir = os.path.join(output_base, 'loso', f'season_{loso_season}')
    loso_result_path = os.path.join(loso_output_dir, 'loso_results.json')
    if os.path.exists(loso_result_path) and not force_refit:
        logger.info("LOSO results exist. Use --force_refit to redo.")
        return
    os.makedirs(loso_output_dir, exist_ok=True)

    # Load this fold's own PCA - trained on seasons excluding loso_season.
    loso_pca_path = os.path.join(
        combined_base, 'loso', f'season_{loso_season}', 'pca_model.pkl'
    )
    if not os.path.exists(loso_pca_path):
        raise FileNotFoundError(
            f"LOSO PCA not found: {loso_pca_path}\n"
            "Run 03a_pca4combined_hmm.py first."
        )
    with open(loso_pca_path, 'rb') as f:
        loso_pca = pickle.load(f)

    # Load this fold's n_pcs_lookup (may differ from primary due to different data)
    loso_n_pcs_lookup = load_n_pcs_lookup(combined_base, loso_season=loso_season)
    vt_key = f"{selected_config['variance_threshold']:.2f}"
    n_pcs = loso_n_pcs_lookup[vt_key]
    logger.info(f"LOSO S{loso_season}: variance threshold {vt_key} -> {n_pcs} PCs")

    loso_split = load_split(combined_base, split_type='loso', loso_season=loso_season)
    loso_projected_dir = get_projected_dir(combined_base, loso_season=loso_season)

    # Delegate to shared refit helper
    fold_result = _refit_fold(
        selected_config=selected_config,
        config_name_str=primary_results.get('selected_config_name', ''),
        pca=loso_pca,
        n_pcs=n_pcs,
        split=loso_split,
        projected_dir=loso_projected_dir,
        output_dir=loso_output_dir,
        fold_label=f"LOSO S{loso_season}",
        n_seeds=n_final_seeds,
        n_jobs=n_jobs,
        force_refit=force_refit,
    )

    # Build LOSO-specific result JSON
    loso_result = {
        'sub_id': sub_id,
        'parcellation': parcellation,
        'loso_season': loso_season,
        'selected_config': selected_config,
        'selected_config_name': primary_results['selected_config_name'],
        'n_pcs': n_pcs,
        'loso_refit': {
            'n_seeds': n_final_seeds,
            'best_seed': fold_result['best_seed'],
            'trainval_ll_per_sample': fold_result['trainval_ll_per_sample'],
            'test_ll_per_sample': fold_result['test_ll_per_sample'],
            # Explicit dual-baseline schema.
            'baseline_ll_per_sample_total_states': fold_result['baseline_total'],
            'baseline_ll_per_sample_active_states': fold_result['baseline_active'],
            'generalizes_vs_baseline_total_states': fold_result['generalizes_vs_total'],
            'generalizes_vs_baseline_active_states': fold_result['generalizes_vs_active'],
            'n_total_states': fold_result['n_states'],
            'n_active_states': fold_result['n_active_states'],
            'overfit_gap': fold_result['overfit_gap'],
            'seed_results': fold_result['seed_results'],
        },
        'data_info': fold_result['data_info'],
        'schema_info': {
            'schema_version': '2.1',
            'baseline_primary': 'active_states',
            'legacy_keys_preserved': False,
        },
        'n_decoded_runs': len(fold_result['decoded_states']),
        'timestamp': datetime.now().isoformat(),
    }

    with open(loso_result_path, 'w') as f:
        json.dump(loso_result, f, indent=2)
    logger.info(f"LOSO results saved to {loso_output_dir}")


def split_half_fit(sub_id, parcellation, half, n_final_seeds, force_refit=False,
                   n_jobs=1, fixed_vt=None):
    """Fit the winning config on one split-half fold for reliability analysis.

    Each half uses its own PCA (fitted by 03a_pca4combined_hmm.py with
    --split_mode split_half). The resulting model is saved for comparison
    by 04rb_split_half_reliability.py.

    Args:
        sub_id:        Subject ID
        parcellation:  Parcellation name
        half:          'A' or 'B'
        n_final_seeds: Number of seeds for refit
        force_refit:   If True, redo even if results exist
        fixed_vt:      Optional fixed variance threshold
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SPLIT-HALF FIT - Half {half}: {sub_id}")
    logger.info(f"{'=' * 60}")

    combined_base = get_combined_base(sub_id, parcellation)
    output_base = get_output_base(sub_id, parcellation)

    # Resolve variance threshold
    selected_vt = _resolve_vt(sub_id, parcellation, fixed_vt=fixed_vt)
    logger.info(f"Using variance threshold: vt={selected_vt}")

    # Load best config from primary selection
    final_dir = get_final_dir(output_base, selected_vt)
    final_result_path = os.path.join(final_dir, 'final_results.json')
    if not os.path.exists(final_result_path):
        logger.error(
            f"Primary select results not found: {final_result_path}\n"
            "Run --mode select first."
        )
        sys.exit(1)
    with open(final_result_path, 'r') as f:
        primary_results = json.load(f)
    selected_config = primary_results['selected_config']
    logger.info(f"Using config: {primary_results['selected_config_name']}")

    # Output directory
    half_output_dir = os.path.join(output_base, 'split_half', half)
    half_result_path = os.path.join(half_output_dir, 'split_half_results.json')
    if os.path.exists(half_result_path) and not force_refit:
        logger.info("Split-half results exist. Use --force_refit to redo.")
        return
    os.makedirs(half_output_dir, exist_ok=True)

    # Load this half's PCA
    fold_spec = f'split_half_{half}'
    half_pca_path = os.path.join(
        combined_base, 'split_half', half, 'pca_model.pkl'
    )
    if not os.path.exists(half_pca_path):
        raise FileNotFoundError(
            f"Split-half PCA not found: {half_pca_path}\n"
            "Run 03a_pca4combined_hmm.py --split_mode split_half first."
        )
    with open(half_pca_path, 'rb') as f:
        half_pca = pickle.load(f)

    # Load this half's n_pcs_lookup
    half_n_pcs_lookup = load_n_pcs_lookup(combined_base, fold_spec=fold_spec)
    vt_key = f"{selected_config['variance_threshold']:.2f}"
    n_pcs = half_n_pcs_lookup[vt_key]
    logger.info(f"Split-half {half}: variance threshold {vt_key} -> {n_pcs} PCs")

    # Load split - note: 'test' is empty for split-half (no within-half test set).
    # We use train+valid for fitting, then decode ALL runs in this half.
    half_split = load_split(combined_base, fold_spec=fold_spec)
    half_projected_dir = get_projected_dir(combined_base, fold_spec=fold_spec)

    # For _refit_fold, we need a non-empty test set to evaluate on.
    # Use the validation set as the "test" for LL evaluation.
    # The real comparison is cross-half (done by 04rb).
    refit_split = {
        'train': half_split['train'],
        'valid': half_split['valid'],
        'test': half_split['valid'],  # evaluate on valid as proxy
    }

    # Delegate to shared refit helper
    fold_result = _refit_fold(
        selected_config=selected_config,
        config_name_str=primary_results.get('selected_config_name', ''),
        pca=half_pca,
        n_pcs=n_pcs,
        split=refit_split,
        projected_dir=half_projected_dir,
        output_dir=half_output_dir,
        fold_label=f"Half {half}",
        n_seeds=n_final_seeds,
        n_jobs=n_jobs,
        force_refit=force_refit,
        test_split_name='valid',  # validation runs reused as proxy test
    )

    # Also decode all runs in this half (train+valid) for recurrence analysis
    all_half_split = {
        'train': half_split['train'],
        'valid': half_split['valid'],
        'test': [],
    }
    decoded_all = decode_all_runs(
        fold_result['best_model'], all_half_split, half_projected_dir, n_pcs
    )
    logger.info(f"Decoded {len(decoded_all)} total runs in half {half}")

    # Overwrite decoded_states.pkl with ALL runs (not just valid-as-test)
    with open(os.path.join(half_output_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded_all, f, protocol=4)

    # Build result JSON
    half_result = {
        'sub_id': sub_id,
        'parcellation': parcellation,
        'half': half,
        'selected_config': selected_config,
        'selected_config_name': primary_results['selected_config_name'],
        'n_pcs': n_pcs,
        'refit': {
            'n_seeds': n_final_seeds,
            'best_seed': fold_result['best_seed'],
            'trainval_ll_per_sample': fold_result['trainval_ll_per_sample'],
            'valid_ll_per_sample': fold_result['test_ll_per_sample'],  # valid used as test
            'n_states': fold_result['n_states'],
            'n_active_states': fold_result['n_active_states'],
            'seed_results': fold_result['seed_results'],
        },
        'data_info': {
            'n_train_runs': len(half_split['train']),
            'n_valid_runs': len(half_split['valid']),
            'n_total_runs': len(half_split['train']) + len(half_split['valid']),
            'n_decoded_runs': len(decoded_all),
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(half_result_path, 'w') as f:
        json.dump(half_result, f, indent=2)
    logger.info(f"Split-half results saved to {half_output_dir}")


# =============================================================================
# Argument parsing
# =============================================================================

def _load_stage1_result(sub_id, parcellation):
    """Load Stage 1 result (selected vt) from disk.

    Returns:
        float: selected variance threshold, or None if not found.
    """
    output_base = get_output_base(sub_id, parcellation)
    stage1_result_path = os.path.join(output_base, 'stage1_result.json')
    if not os.path.exists(stage1_result_path):
        return None
    try:
        with open(stage1_result_path, 'r') as f:
            result = json.load(f)
        return result.get('selected_vt')
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Corrupt or invalid stage1_result.json: {e}")
        return None


def _resolve_vt(sub_id, parcellation, fixed_vt=None):
    """Resolve the selected variance threshold without scoring configs.

    If fixed_vt is provided, returns it directly.
    Otherwise loads from stage1_result.json.
    Raises RuntimeError if vt cannot be determined.
    """
    if fixed_vt is not None:
        return fixed_vt
    vt = _load_stage1_result(sub_id, parcellation)
    if vt is None:
        raise RuntimeError(
            f"Cannot determine vt for {sub_id}: no stage1_result.json and no --fixed_vt. "
            f"Run Stage 1 fit first, or pass --fixed_vt."
        )
    return vt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fit combined weak-limit HMM across all seasons with two-stage model selection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1 fit (5 vt configs, array 0-4):
  python script/04_combined_hdphmm.py --sub_id sub-01 --mode fit --stage 1 --task_index 0

  # Stage 2 fit (nc x gamma x cov configs):
  python script/04_combined_hdphmm.py --sub_id sub-01 --mode fit --stage 2 --task_index 0

  # Select mode (two-stage orchestration):
  python script/04_combined_hdphmm.py --sub_id sub-01 --mode select

  # LOSO fit mode (one fold per run):
  python script/04_combined_hdphmm.py --sub_id sub-01 --mode loso_fit --loso_season 1
        """
    )
    parser.add_argument('--sub_id', required=True,
                        help='Subject ID (e.g., "sub-01")')
    parser.add_argument('--parcellation', default='atlas-4S156Parcels',
                        help='Parcellation name (default: atlas-4S156Parcels)')
    parser.add_argument('--mode', required=True,
                        choices=['fit', 'select', 'select_seed', 'select_finalize',
                                 'loso_fit', 'split_half_fit'],
                        help='Operation mode')

    # Fit mode
    parser.add_argument('--stage', type=int, default=None, choices=[1, 2],
                        help='Model selection stage (fit mode): 1=vt sweep, 2=K/gamma/cov sweep')
    parser.add_argument('--task_index', type=int, default=None,
                        help='SLURM array task index (fit mode; 0-indexed)')
    parser.add_argument('--n_fit_seeds', type=int, default=5,
                        help='Seeds per config in fit mode (default: 5)')

    # Select + LOSO fit modes
    parser.add_argument('--n_final_seeds', type=int, default=10,
                        help='Seeds for final refit in select/loso_fit (default: 10)')
    parser.add_argument('--seed_index', type=int, default=None,
                        help='Seed index for select_seed array task')
    parser.add_argument('--force_refit', action='store_true',
                        help='Re-run select/loso_fit even if results already exist')

    # LOSO fit mode
    parser.add_argument('--loso_season', type=int, default=None,
                        help='Season to hold out in loso_fit mode (1-6; 1-4 for sub-04)')

    # Split-half fit mode
    parser.add_argument('--half', type=str, default=None, choices=['A', 'B'],
                        help='Which half to fit in split_half_fit mode')

    # Model selection metric override (mainly for backward compat / debugging)
    parser.add_argument(
        '--selection_metric',
        choices=['ll_per_sample', 'll_per_dim', 'll_per_sample_per_pc', 'bic'],
        default=None,
        help=(
            'Override selection metric. Default: ll_per_dim for Stage 1, '
            'bic for Stage 2. ll_per_sample is legacy behavior.'
        )
    )

    # Fixed variance threshold (bypasses Stage 1 selection)
    parser.add_argument(
        '--fixed_vt', type=float, default=None,
        choices=[0.80, 0.85, 0.90, 0.95, 0.99],
        help=(
            'Skip Stage 1 selection and use this fixed variance threshold. '
            'Bypasses ll_per_dim metric; use when vt is externally validated '
            '(e.g., from 03b scree analysis). Also allows Stage 2 fit mode '
            'to run without a stage1_result.json on disk.'
        )
    )

    # Manual config override (bypasses all auto-selection)
    parser.add_argument(
        '--select_config', type=str, default=None,
        help=(
            'Manually specify the config to use for select/select_seed/select_finalize, '
            'bypassing BIC or other auto-selection. Value is the config directory name '
            '(e.g., vt0.95_covdiag_nc50_g1). The vt is extracted from the config, '
            'so --fixed_vt is not needed when using this flag.'
        )
    )

    # Parallelism
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS,
                        help=f'Parallel jobs for E-step (default: {DEFAULT_N_JOBS})')

    args = parser.parse_args()

    if args.mode == 'fit':
        if args.stage is None:
            parser.error("--stage {1,2} is required in fit mode")

        if args.task_index is None:
            slurm_task = os.environ.get('SLURM_ARRAY_TASK_ID')
            if slurm_task is not None:
                args.task_index = int(slurm_task)
            else:
                parser.error(
                    "--task_index required in fit mode "
                    "(or set SLURM_ARRAY_TASK_ID)"
                )

        if args.stage == 1:
            if not (0 <= args.task_index < N_CONFIGS_STAGE1):
                parser.error(
                    f"--task_index must be 0-{N_CONFIGS_STAGE1 - 1} "
                    f"for Stage 1 ({N_CONFIGS_STAGE1} configs)"
                )
        # Stage 2 bounds are checked at runtime after loading selected_vt

    if args.mode == 'select_seed':
        if args.seed_index is None:
            slurm_task = os.environ.get('SLURM_ARRAY_TASK_ID')
            if slurm_task is not None:
                args.seed_index = int(slurm_task)
            else:
                parser.error(
                    "--seed_index required in select_seed mode "
                    "(or set SLURM_ARRAY_TASK_ID)"
                )

    if args.mode == 'loso_fit':
        if args.loso_season is None:
            slurm_task = os.environ.get('SLURM_ARRAY_TASK_ID')
            if slurm_task is not None:
                args.loso_season = int(slurm_task)
            else:
                parser.error(
                    "--loso_season required in loso_fit mode "
                    "(or set SLURM_ARRAY_TASK_ID)"
                )
        max_season = SUB04_MAX_SEASON if args.sub_id == 'sub-04' else 6
        if not (1 <= args.loso_season <= max_season):
            parser.error(
                f"--loso_season must be 1-{max_season} for {args.sub_id}"
            )

    if args.mode == 'split_half_fit':
        if args.half is None:
            slurm_task = os.environ.get('SLURM_ARRAY_TASK_ID')
            if slurm_task is not None:
                args.half = 'A' if int(slurm_task) == 0 else 'B'
            else:
                parser.error(
                    "--half required in split_half_fit mode "
                    "(or set SLURM_ARRAY_TASK_ID: 0=A, 1=B)"
                )

    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    parcellation = normalize_parcellation_name(args.parcellation)

    logger.info("=" * 60)
    logger.info("04_combined_hdphmm.py")
    logger.info("=" * 60)
    logger.info(f"Subject:       {args.sub_id}")
    logger.info(f"Parcellation:  {parcellation}")
    logger.info(f"Mode:          {args.mode}")
    if args.stage is not None:
        logger.info(f"Stage:         {args.stage}")
    logger.info(f"n_jobs:        {args.n_jobs}")

    # Validate 03a_pca4combined_hmm prerequisites
    combined_base = get_combined_base(args.sub_id, parcellation)
    if not os.path.exists(os.path.join(combined_base, 'n_pcs_lookup.json')):
        logger.error(f"03a_pca4combined_hmm output not found: {combined_base}")
        logger.error("Run script/03a_pca4combined_hmm.py first.")
        sys.exit(1)
    if not os.path.exists(os.path.join(combined_base, 'splits', 'primary.json')):
        logger.error(
            f"Primary split not found: {combined_base}/splits/primary.json"
        )
        logger.error("Run script/03a_pca4combined_hmm.py first.")
        sys.exit(1)

    if args.mode == 'fit':
        if args.stage == 1:
            config_grid = build_stage1_grid()
            config = config_grid[args.task_index]
            logger.info(f"Stage 1 fit: vt sweep")
            logger.info(f"Task index:    {args.task_index} / {N_CONFIGS_STAGE1 - 1}")
            logger.info(f"Config:        {config_name(config)}")
            logger.info(f"Seeds:         {args.n_fit_seeds}")
            logger.info("=" * 60)
            fit_single_config(
                config=config,
                sub_id=args.sub_id,
                parcellation=parcellation,
                n_fit_seeds=args.n_fit_seeds,
                n_jobs=args.n_jobs,
            )
        elif args.stage == 2:
            if args.fixed_vt is not None:
                selected_vt = args.fixed_vt
                logger.info(f"Using fixed variance threshold (bypassing Stage 1): vt={selected_vt}")
            else:
                selected_vt = _load_stage1_result(args.sub_id, parcellation)
                if selected_vt is None:
                    logger.error(
                        "Stage 1 result not found. Run --mode select --stage 1 first, "
                        "or run Stage 1 fit tasks then select. "
                        "Alternatively, pass --fixed_vt to skip Stage 1."
                    )
                    sys.exit(1)
            config_grid = build_stage2_grid(selected_vt)
            n_stage2 = len(config_grid)
            if not (0 <= args.task_index < n_stage2):
                logger.error(
                    f"--task_index must be 0-{n_stage2 - 1} "
                    f"for Stage 2 with vt={selected_vt} ({n_stage2} configs)"
                )
                sys.exit(1)
            config = config_grid[args.task_index]
            logger.info(f"Stage 2 fit: K/gamma/cov sweep (vt={selected_vt})")
            logger.info(f"Task index:    {args.task_index} / {n_stage2 - 1}")
            logger.info(f"Config:        {config_name(config)}")
            logger.info(f"Seeds:         {args.n_fit_seeds}")
            logger.info("=" * 60)
            fit_single_config(
                config=config,
                sub_id=args.sub_id,
                parcellation=parcellation,
                n_fit_seeds=args.n_fit_seeds,
                n_jobs=args.n_jobs,
            )

    elif args.mode == 'select':
        # Determine selection metric: default depends on whether --stage is given
        selection_metric = args.selection_metric
        if selection_metric is None:
            selection_metric = 'bic'  # two-stage default for final select
        logger.info(f"Final seeds:   {args.n_final_seeds}")
        logger.info(f"Force refit:   {args.force_refit}")
        logger.info(f"Selection:     {selection_metric}")
        if args.select_config:
            logger.info(f"Config override: {args.select_config}")
        elif args.fixed_vt is not None:
            logger.info(f"Fixed vt:      {args.fixed_vt} (bypassing Stage 1)")
        logger.info("=" * 60)
        select_and_refit(
            sub_id=args.sub_id,
            parcellation=parcellation,
            n_final_seeds=args.n_final_seeds,
            force_refit=args.force_refit,
            n_jobs=args.n_jobs,
            selection_metric=selection_metric,
            fixed_vt=args.fixed_vt,
            select_config=args.select_config,
        )

    elif args.mode == 'select_seed':
        selection_metric = args.selection_metric or 'bic'
        logger.info(f"Seed index:    {args.seed_index}")
        logger.info(f"Force refit:   {args.force_refit}")
        logger.info(f"Selection:     {selection_metric}")
        if args.select_config:
            logger.info(f"Config override: {args.select_config}")
        logger.info("=" * 60)
        select_seed(
            sub_id=args.sub_id,
            parcellation=parcellation,
            seed_index=args.seed_index,
            force_refit=args.force_refit,
            n_jobs=args.n_jobs,
            selection_metric=selection_metric,
            fixed_vt=args.fixed_vt,
            select_config=args.select_config,
        )

    elif args.mode == 'select_finalize':
        selection_metric = args.selection_metric or 'bic'
        logger.info(f"Final seeds:   {args.n_final_seeds}")
        logger.info(f"Force refit:   {args.force_refit}")
        logger.info(f"Selection:     {selection_metric}")
        if args.select_config:
            logger.info(f"Config override: {args.select_config}")
        logger.info("=" * 60)
        select_finalize(
            sub_id=args.sub_id,
            parcellation=parcellation,
            n_final_seeds=args.n_final_seeds,
            force_refit=args.force_refit,
            n_jobs=args.n_jobs,
            selection_metric=selection_metric,
            fixed_vt=args.fixed_vt,
            select_config=args.select_config,
        )

    elif args.mode == 'loso_fit':
        logger.info(f"LOSO season:   {args.loso_season}")
        logger.info(f"Final seeds:   {args.n_final_seeds}")
        logger.info(f"Force refit:   {args.force_refit}")
        if args.fixed_vt is not None:
            logger.info(f"Fixed vt:      {args.fixed_vt}")
        logger.info("=" * 60)
        loso_fit(
            sub_id=args.sub_id,
            parcellation=parcellation,
            loso_season=args.loso_season,
            n_final_seeds=args.n_final_seeds,
            force_refit=args.force_refit,
            n_jobs=args.n_jobs,
            fixed_vt=args.fixed_vt,
        )

    elif args.mode == 'split_half_fit':
        logger.info(f"Half:          {args.half}")
        logger.info(f"Final seeds:   {args.n_final_seeds}")
        logger.info(f"Force refit:   {args.force_refit}")
        if args.fixed_vt is not None:
            logger.info(f"Fixed vt:      {args.fixed_vt}")
        logger.info("=" * 60)
        split_half_fit(
            sub_id=args.sub_id,
            parcellation=parcellation,
            half=args.half,
            n_final_seeds=args.n_final_seeds,
            force_refit=args.force_refit,
            n_jobs=args.n_jobs,
            fixed_vt=args.fixed_vt,
        )

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
