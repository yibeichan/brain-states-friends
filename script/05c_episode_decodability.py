#!/usr/bin/env python3
"""
05c_episode_decodability.py - Season decodability from FO profiles (exploratory).

Tests whether season identity can be decoded from brain-state fractional
occupancy vectors.  This is an EXPLORATORY analysis:
- Above-chance accuracy does not prove content coding (session/longitudinal
  confounds cannot be separated from content effects in single-subject data).
- Chance-level accuracy does not prove context invariance, only that FO
  profiles do not linearly separate seasons.

FO is at run level (each scan run is an independent sample).  Multipart
episodes (e.g. s01e01a, s01e01b) are treated as separate runs.

Prerequisites:
    - 05a_recurrence_analysis.py completed for this subject
    - Outputs available at {SCRATCH_DIR}/output/05a_recurrence_analysis/{parcellation}/{sub_id}/

Analyses:
    1. Feature matrix X[r, k] = FO(state k, run r) at run level
    2. CLR transform applied to X before logistic regression (FO vectors sum to 1;
       raw FO is compositional data — CLR maps the simplex to Euclidean space)
    3. Season decoding via L2-regularized multinomial logistic regression (LOO-CV)
    4. Permutation test (shuffle season labels) for statistical significance
    5. Nuisance control: decode ordinal run number (session order)
    6. Per-state Kruskal-Wallis test across seasons with FDR correction
       (uses raw FO, not CLR — KW is non-parametric and scale-invariant)

CLR transform: CLR(x)[k] = log(x[k] / geom_mean(x)), after adding a small
pseudocount to handle zero-FO entries (inactive states or states absent from
a run). Pseudocount = 1e-4 (much smaller than typical non-zero FO, which
is on the order of 1/n_active_states ≈ 0.02).

Outputs:
    Saves to {SCRATCH_DIR}/output/05c_episode_decodability/{parcellation}/{sub_id}/
    - decodability_results.json (includes analysis_scope metadata)
    - per_state_kruskal_wallis.json
    - confusion_matrix.png / .pdf
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

def clr_transform(X, pseudocount=1e-4):
    """Centered log-ratio transform for compositional FO data.

    FO vectors sum to 1 (simplex constraint), making raw FO compositional data.
    Logistic regression assumes Euclidean geometry, which is invalid on the
    simplex. CLR maps each row to Euclidean space by dividing by the geometric
    mean, removing the sum constraint.

    CLR(x)[k] = log((x[k] + eps) / geom_mean(x + eps))

    A pseudocount (eps=1e-4) is added before log to handle zero-FO entries
    (inactive states or states absent from a given run). This value is
    ~5-50x smaller than typical non-zero FO (~1/n_active_states ≈ 0.02),
    so it has negligible impact on non-zero entries.

    Args:
        X: np.array (n_runs, n_states), raw fractional occupancy
        pseudocount: small constant added before log (default 1e-4)

    Returns:
        X_clr: np.array (n_runs, n_states), CLR-transformed
    """
    X_ps = X + pseudocount
    log_X = np.log(X_ps)
    log_gm = log_X.mean(axis=1, keepdims=True)  # log geometric mean per run
    return log_X - log_gm


# ── Imports from project utilities ───────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import benjamini_hochberg
from utils.plot_style import apply_publication_style
from utils.common import normalize_parcellation_name, _get_season, parse_episode_order_key

apply_publication_style()

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv('SCRATCH_DIR')
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Core analysis functions
# =============================================================================

def build_feature_matrix(fo_dict, n_states):
    """Build run x state feature matrix from fractional occupancy dict.

    Runs are sorted in chronological order (by season then episode number),
    not lexicographic order.

    Args:
        fo_dict: dict run_id -> np.array(n_states,)
                 (run-level FO from 05a)
        n_states: total number of HMM states

    Returns:
        X: np.array (n_runs, n_states)
        run_ids: list of run_id strings (chronological order)
    """
    # Sort chronologically by (season, episode_number, part)
    run_ids = sorted(fo_dict.keys(), key=parse_episode_order_key)
    X = np.zeros((len(run_ids), n_states))
    for i, eid in enumerate(run_ids):
        fo_vec = fo_dict[eid]
        if len(fo_vec) != n_states:
            raise ValueError(
                f"FO vector length {len(fo_vec)} for {eid} does not match "
                f"n_states={n_states}. Check upstream 05a outputs."
            )
        X[i, :] = fo_vec
    return X, run_ids


def loo_logistic_regression(X, y, C=0.1):
    """Leave-one-out cross-validated L2-regularized logistic regression.

    Args:
        X: feature matrix (n_samples, n_features)
        y: integer labels (n_samples,)
        C: inverse regularization strength

    Returns:
        accuracy: float, LOO-CV accuracy
        predictions: np.array of predicted labels per fold
        mean_abs_coefs: np.array (n_features,) mean absolute coefficient
            across folds and classes (decodability weight per feature)
    """
    loo = LeaveOneOut()
    predictions = np.zeros(len(y), dtype=int)
    coef_accumulator = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Need at least 2 classes in training fold
        if len(np.unique(y_train)) < 2:
            predictions[test_idx] = y_train[0]
            continue

        clf = LogisticRegression(
            C=C,
            penalty='l2',
            solver='lbfgs',
            max_iter=5000,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        predictions[test_idx] = clf.predict(X_test)

        # Average absolute coefficients across classes
        coef_accumulator += np.mean(np.abs(clf.coef_), axis=0)
        n_folds += 1

    accuracy = np.mean(predictions == y)
    mean_abs_coefs = coef_accumulator / max(n_folds, 1)
    return accuracy, predictions, mean_abs_coefs


def _single_permutation(X, y, C, seed):
    """Run one permutation: shuffle labels and compute LOO accuracy."""
    rng_local = np.random.default_rng(seed)
    y_perm = rng_local.permutation(y)
    acc, _, _ = loo_logistic_regression(X, y_perm, C=C)
    return acc


def permutation_test(X, y, C, n_permutations, observed_accuracy, rng,
                     n_jobs=1):
    """Permutation test: shuffle labels, recompute LOO accuracy.

    Args:
        X: feature matrix
        y: labels
        C: regularization parameter
        n_permutations: number of shuffles
        observed_accuracy: accuracy from the real labels
        rng: numpy random generator (used to generate per-permutation seeds)
        n_jobs: number of parallel workers (default: 1 = serial)

    Returns:
        p_value: fraction of permuted accuracies >= observed
        null_accuracies: np.array (n_permutations,)
    """
    # Generate independent seeds for each permutation (reproducible)
    seeds = rng.integers(0, 2**31, size=n_permutations)

    if n_jobs == 1:
        # Serial path (preserves logging)
        null_accuracies = np.zeros(n_permutations)
        for i in range(n_permutations):
            null_accuracies[i] = _single_permutation(X, y, C, seeds[i])
            if (i + 1) % 500 == 0:
                logger.info(f"  Permutation {i + 1}/{n_permutations}")
    else:
        logger.info(f"  Running {n_permutations} permutations with {n_jobs} workers")
        null_accuracies = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_single_permutation)(X, y, C, seeds[i])
                for i in range(n_permutations)
            )
        )

    # p-value: fraction of null >= observed (Phipson-Smyth correction, NaN-safe)
    from utils.stats import permutation_pvalue
    p_value = permutation_pvalue(observed_accuracy, null_accuracies, alternative='greater')
    return p_value, null_accuracies


def build_confusion_matrix(y_true, y_pred, classes):
    """Build confusion matrix manually.

    Args:
        y_true: true labels
        y_pred: predicted labels
        classes: sorted unique class labels

    Returns:
        cm: np.array (n_classes, n_classes), cm[i, j] = count of true=i, pred=j
    """
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in class_to_idx or p not in class_to_idx:
            logger.warning(f"Unknown class in (true={t}, pred={p}) — skipping cell")
            continue
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def per_state_kruskal_wallis(X, season_labels):
    """Per-state Kruskal-Wallis test across seasons.

    NOTE: FO values are compositional (sum to 1 per run), so per-state
    KW tests are not strictly independent.  Treat results as a descriptive
    screening tool; the CLR-based logistic regression (above) is the primary
    inferential test because it accounts for the simplex constraint.

    Args:
        X: feature matrix (n_runs, n_states)
        season_labels: np.array of season integers (n_runs,)

    Returns:
        results: list of dicts with keys: state, H, p_value, p_fdr
    """
    n_states = X.shape[1]
    unique_seasons = np.unique(season_labels)
    p_values = np.zeros(n_states)
    h_stats = np.zeros(n_states)

    for k in range(n_states):
        groups = [X[season_labels == s, k] for s in unique_seasons]
        # Filter out empty groups and groups with no variance
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            h_stats[k] = 0.0
            p_values[k] = 1.0
            continue
        try:
            h, p = stats.kruskal(*groups)
            if np.isnan(h) or np.isnan(p):
                # All values identical across groups → no effect
                h_stats[k] = 0.0
                p_values[k] = 1.0
            else:
                h_stats[k] = h
                p_values[k] = p
        except ValueError:
            # All values identical in all groups
            h_stats[k] = 0.0
            p_values[k] = 1.0

    q_values = benjamini_hochberg(p_values)

    results = []
    for k in range(n_states):
        results.append({
            'state': int(k),
            'H': float(h_stats[k]),
            'p_value': float(p_values[k]),
            'p_fdr': float(q_values[k]),
            'note': ('Descriptive screening: FO is compositional (sums to 1 '
                     'per run), so per-state tests are not independent. '
                     'Use CLR-based logistic regression for formal inference.'),
        })
    return results


# =============================================================================
# Plotting functions
# =============================================================================

def plot_confusion_matrix(cm, classes, out_dir):
    """Plot and save confusion matrix heatmap.

    Args:
        cm: confusion matrix (n_classes, n_classes)
        classes: class labels for axes
        out_dir: output directory
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Season Decodability: Confusion Matrix (LOO-CV)')
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([f'S{c}' for c in classes])
    ax.set_yticklabels([f'S{c}' for c in classes])
    ax.set_xlabel('Predicted Season')
    ax.set_ylabel('True Season')

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=11)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(out_dir, f'confusion_matrix.{ext}'), dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot.")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether season identity can be decoded from "
                    "brain-state fractional occupancy vectors."
    )
    parser.add_argument('--sub_id', type=str, required=True,
                        help="Subject ID (e.g., sub-01)")
    parser.add_argument('--parcellation', type=str, default='atlas-4S156Parcels',
                        help="Parcellation name (default: atlas-4S156Parcels)")
    parser.add_argument('--n_permutations', type=int, default=1000,
                        help="Number of permutations for significance test (default: 1000)")
    parser.add_argument('--vt', type=str, default=None,
                        help="Variance threshold subdirectory (e.g., 0.95). "
                             "Reads 05a from vt{VT}/ subdir. If omitted, reads flat (legacy).")
    parser.add_argument('--n_jobs', type=int, default=1,
                        help="Number of parallel workers for permutation tests (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    n_permutations = args.n_permutations

    # ── Input paths (from 05a, vt-aware) ────────────────────────────────
    recurrence_dir = os.path.join(
        SCRATCH_DIR, 'output', '05a_recurrence_analysis', parc, sub_id
    )
    if args.vt is not None:
        recurrence_dir = os.path.join(recurrence_dir, f'vt{args.vt}')
    fo_path = os.path.join(recurrence_dir, 'fractional_occupancy.pkl')
    summary_path = os.path.join(recurrence_dir, 'recurrence_summary.json')

    if not os.path.exists(fo_path):
        logger.error(f"Missing fractional occupancy: {fo_path}")
        logger.error("Run 05a_recurrence_analysis.py first.")
        sys.exit(1)
    if not os.path.exists(summary_path):
        logger.error(f"Missing recurrence summary: {summary_path}")
        sys.exit(1)

    # ── Output directory (vt-aware) ─────────────────────────────────────
    out_dir = os.path.join(
        SCRATCH_DIR, 'output', '05c_episode_decodability', parc, sub_id
    )
    if args.vt is not None:
        out_dir = os.path.join(out_dir, f'vt{args.vt}')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info(f"Loading fractional occupancy from {fo_path}")
    with open(fo_path, 'rb') as f:
        fo_dict = pickle.load(f)

    with open(summary_path, 'r') as f:
        recurrence_summary = json.load(f)

    n_states = recurrence_summary['n_states']
    logger.info(f"Loaded {len(fo_dict)} runs, {n_states} states")

    # ── 1. Build feature matrix and targets ──────────────────────────────
    # FO from 05a is run-level (no episode aggregation)
    X, run_ids = build_feature_matrix(fo_dict, n_states)
    season_labels = np.array([_get_season(eid) for eid in run_ids])

    # CLR-transform X for logistic regression.
    # Raw FO vectors sum to 1 (compositional data); this violates Euclidean
    # independence assumptions. CLR maps each row to Euclidean space.
    # Raw X is kept for Kruskal-Wallis (non-parametric, no geometry assumption).
    CLR_PSEUDOCOUNT = 1e-4
    X_clr = clr_transform(X, pseudocount=CLR_PSEUDOCOUNT)
    logger.info(f"CLR transform applied (pseudocount={CLR_PSEUDOCOUNT})")
    unique_seasons = np.sort(np.unique(season_labels))
    n_runs = len(run_ids)

    # ── Preflight guards ─────────────────────────────────────────────────
    if n_runs < 2:
        logger.error(f"Need at least 2 runs, got {n_runs}. Aborting.")
        sys.exit(1)
    if len(unique_seasons) < 2:
        logger.error(
            f"Need at least 2 seasons, got {len(unique_seasons)}: "
            f"{unique_seasons.tolist()}. Aborting."
        )
        sys.exit(1)
    # Check minimum class size for LOO-CV
    min_class_size = min(np.sum(season_labels == s) for s in unique_seasons)
    if min_class_size < 2:
        logger.error(
            f"Smallest season class has {min_class_size} runs. "
            f"Need at least 2 per class for LOO-CV. Aborting."
        )
        sys.exit(1)

    chance_level = 1.0 / len(unique_seasons)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(
        f"Seasons present: {unique_seasons.tolist()}, "
        f"runs per season: {[int(np.sum(season_labels == s)) for s in unique_seasons]}, "
        f"chance = {chance_level:.4f}"
    )

    # Ordinal run numbers (nuisance control)
    run_order = np.arange(n_runs)

    # ── 2. Season decoding ───────────────────────────────────────────────
    C = 0.1
    logger.info("Running LOO-CV logistic regression for season decoding (CLR features)...")
    season_acc, season_preds, _coefs = loo_logistic_regression(
        X_clr, season_labels, C=C
    )
    logger.info(f"Season decoding accuracy: {season_acc:.4f} (chance: {chance_level:.4f})")

    cm = build_confusion_matrix(season_labels, season_preds, unique_seasons)

    # ── 3. Permutation test for season decoding ──────────────────────────
    logger.info(f"Running permutation test ({n_permutations} shuffles)...")
    rng = np.random.default_rng(seed=42)
    n_jobs = args.n_jobs
    season_pvalue, null_accs = permutation_test(
        X_clr, season_labels, C=C,
        n_permutations=n_permutations,
        observed_accuracy=season_acc,
        rng=rng,
        n_jobs=n_jobs,
    )
    logger.info(f"Permutation p-value: {season_pvalue:.6f}")

    # ── 4. Nuisance control: decode session order ────────────────────────
    # Bin ordinal run numbers into quantile groups (same number of groups
    # as seasons) so the classifier has a comparable task
    n_order_bins = len(unique_seasons)
    order_bins = np.digitize(
        run_order,
        bins=np.linspace(0, n_runs, n_order_bins + 1)[1:-1]
    )
    logger.info("Running LOO-CV logistic regression for session-order control (CLR features)...")
    order_acc, _, _ = loo_logistic_regression(X_clr, order_bins, C=C)

    logger.info(f"Running permutation test for session-order control ({n_permutations} shuffles)...")
    order_pvalue, _ = permutation_test(
        X_clr, order_bins, C=C,
        n_permutations=n_permutations,
        observed_accuracy=order_acc,
        rng=rng,
        n_jobs=n_jobs,
    )
    logger.info(f"Session-order decoding accuracy: {order_acc:.4f}, p={order_pvalue:.6f}")
    logger.info(f"Order/season accuracy ratio: {order_acc / season_acc:.3f} "
                f"(>0.9 suggests longitudinal confounds dominate)")

    # ── 5. Per-state Kruskal-Wallis ──────────────────────────────────────
    logger.info("Running per-state Kruskal-Wallis tests...")
    kw_results = per_state_kruskal_wallis(X, season_labels)
    n_sig = sum(1 for r in kw_results if r['p_fdr'] < 0.05)
    logger.info(f"States with significant season effect (FDR < 0.05): {n_sig}/{n_states}")

    # ── Save outputs ─────────────────────────────────────────────────────

    # 1. Main decodability results
    decodability_results = {
        'analysis_scope': 'single_subject',
        'analysis_type': 'season_decodability_from_fo_profiles',
        'analysis_tier': 'exploratory',
        'note': (
            'Season decodability remains vulnerable to longitudinal/session '
            'confounds in single-subject data. Above-chance accuracy does not '
            'prove content coding; chance-level does not prove context invariance. '
            'FDR correction does not replace cross-subject replication.'
        ),
        'sub_id': sub_id,
        'parcellation': parc,
        'n_runs': n_runs,
        'n_runs': n_runs,  # backward compat alias (unit is runs)
        'n_states': n_states,
        'n_seasons': int(len(unique_seasons)),
        'seasons': unique_seasons.tolist(),
        'runs_per_season': [int(np.sum(season_labels == s)) for s in unique_seasons],
        'chance_level': float(chance_level),
        'regularization_C': C,
        'fo_transform': 'clr',
        'clr_pseudocount': CLR_PSEUDOCOUNT,
        'n_permutations': n_permutations,
        'season_decoding': {
            'accuracy': float(season_acc),
            'permutation_p_value': float(season_pvalue),
            'confusion_matrix': cm.tolist(),
            'null_accuracy_mean': float(np.mean(null_accs)),
            'null_accuracy_std': float(np.std(null_accs)),
        },
        'nuisance_control': {
            'session_order_accuracy': float(order_acc),
            'session_order_p_value': float(order_pvalue),
            'n_order_bins': n_order_bins,
            'order_season_accuracy_ratio': float(order_acc / season_acc) if season_acc > 0 else None,
        },
    }
    results_path = os.path.join(out_dir, 'decodability_results.json')
    with open(results_path, 'w') as f:
        json.dump(decodability_results, f, indent=2)
    logger.info(f"Saved decodability results to {results_path}")

    # 2. Per-state Kruskal-Wallis
    kw_path = os.path.join(out_dir, 'per_state_kruskal_wallis.json')
    with open(kw_path, 'w') as f:
        json.dump(kw_results, f, indent=2)
    logger.info(f"Saved per-state Kruskal-Wallis to {kw_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, unique_seasons, out_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Season Decodability Analysis Complete (exploratory)")
    logger.info("=" * 60)
    logger.info(f"Season accuracy:       {season_acc:.4f} (chance: {chance_level:.4f})")
    logger.info(f"Season p-value:        {season_pvalue:.6f}")
    logger.info(f"Session-order accuracy: {order_acc:.4f} (p={order_pvalue:.6f})")
    logger.info(f"States with sig. season effect (FDR<0.05): {n_sig}/{n_states}")
    logger.info(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()
