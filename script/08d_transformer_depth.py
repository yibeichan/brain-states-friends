#!/usr/bin/env python3
"""
08d_transformer_depth.py - within-stimulus layer-depth analyses for the
transformer-state correspondence sweep.

Analyses (per subject × stimulus × model):

    D1 main               - per-layer, per-lag LORO RidgeClassifier on
                            ``content_eligible`` states. Circular-shift null,
                            global BH-FDR across the full layers × lags grid.
                            Lag grid is [0..8] (~0–12 s) to cover both the
                            canonical HRF peak and subcortical / association
                            delays; lag=0 is kept as a synchrony /
                            autocorrelation diagnostic but excluded from the
                            peak-lag search (see :data:`PEAK_LAG_EXCLUDE`).
    D1 negative control   - same decoder on ``run_onset_anchored`` states. A
                            statistical gate at the D1-main peak cell
                            records ``neg_control_passed`` plus
                            ``delta_peak`` / ``p_main`` / ``p_neg`` in
                            ``D1_neg_control_gate.json`` for downstream
                            consumption.
    D1 confound baseline  - RidgeClassifier using 6 timing regressors
                            (run_onset, tr_since_onset_norm + quadratic /
                            cubic drift, episode_idx, season_idx) on the
                            ``content_eligible`` TR subset. Gives a
                            timing-only accuracy floor that also absorbs
                            within-run drift.
    D1-net                - same D1 machinery applied per group in a
                            ``NETWORK_ORDER × {+, −}`` stratification. A
                            state with strong DMN *deactivation* lands in
                            ``DMN_neg``, separate from DMN activation
                            states in ``DMN_pos``. Uses the best lag from
                            D1 main (with lag=0 excluded).
    D2                    - per-state, per-layer LORO Logistic+AUC on the
                            ``content_eligible`` state set (FO ≥ 1%). Null
                            via precomputed circular-shift sequences.
                            Selectivity flags the canonical threshold
                            (``max − median < 0.05``) PLUS sensitivity
                            entries at 0.03 / 0.05 / 0.10.

Cross-stimulus runs (movie10 / harrypotter / petitprince_*) reuse the
Friends-fit per-layer PCA (same pattern as 08e) instead of refitting on the
test stimulus, which would leak test-fold data into the decoder's feature
basis.

Outputs are written to
``{SCRATCH_DIR}/output/08d_transformer_depth/{parcellation}/{sub_id}/{stimulus}_{model}/``.

Prerequisites
-------------
* ``08c`` raw features under
  ``{SCRATCH_DIR}/output/08c_transformer_features/{stimulus}/{model}/layer_NN/{run}_raw.npy``
  (for cross-stimulus runs, the corresponding Friends features are ALSO
  required - they drive the shared PCA basis)
* ``04_combined_hdphmm/{parc}/{sub}/final/[vt*/]decoded_states.pkl``
  (and ``state_means_parcel.npy`` for D1-net)
* ``05e_temporal_trend_a4/{parc}/{sub}/[vt*/]state_flags.csv`` (falls back to
  05a sub-HRF with a warning)
* ``03a_pca4combined_hmm/{parc}/{sub}/splits/primary.json`` (for PCA training
  split; required for BOTH Friends and cross-stimulus invocations - the
  training split is always defined relative to Friends)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import (
    check_checkpoint, feature_key_for_cross_stim_run_id, load_training_split,
    normalize_parcellation_name, resolve_stage_file,
)
from utils.plot_style import NETWORK_ORDER
from utils.stats import benjamini_hochberg, permutation_pvalue
from utils.transformer_analysis import (
    D2_SELECTIVITY_THRESHOLD, batch_loro_ridge_classify,
    batch_loro_ridge_per_state_auc, build_layer_feature_matrix,
    build_loro_folds, build_run_boundaries, compute_effect_size,
    layer_selectivity, load_content_eligibility, load_or_fit_pca_cache,
    loro_logistic_auc_cv, loro_ridge_classifier_cv,
    precompute_eligible_null_state_sequences,
    precompute_null_state_sequences, stratify_states_by_dominant_network,
    stream_pca_features,
)
from utils.transformer_io import MODEL_REGISTRY, validate_stimulus_model

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise RuntimeError("SCRATCH_DIR must be set in the environment / .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("08d_transformer_depth")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAGS_TO_TEST = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # TRs - ~0–12s HRF window.
#   Lag 0 is retained as an autocorrelation / stimulus–state synchrony
#   diagnostic, but it is excluded from the *peak-lag* search used by
#   D1-net, D1-confound, and D2 (see PEAK_LAG_EXCLUDE and
#   _find_best_lag_layer) because pre-HRF accuracy is not a meaningful
#   "content encoding" proxy.
PEAK_LAG_EXCLUDE = {0}
N_PERMUTATIONS_DEFAULT = 1000
PCA_VARIANCE_THRESHOLD = 0.95
D2_MIN_FRACTIONAL_OCCUPANCY = 0.01        # raised from 0.005 (old 08d)
# D2_SELECTIVITY_THRESHOLD is imported from utils.transformer_analysis so
# that 08d, 08f, and 08g share the same source of truth (== 0.05).
D2_SELECTIVITY_THRESHOLDS_REPORT = (0.03, 0.05, 0.10)  # sensitivity levels
D1NET_MIN_STATES = 5
D1NET_MIN_TRS = 200

# ---------------------------------------------------------------------------
# Data loading helpers (08d-specific; generic helpers live in utils/*)
# ---------------------------------------------------------------------------


def _resolve_04_final_dir(sub_id, parcellation):
    """Return the 04 combined HMM `final` directory, vt-aware."""
    base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation, sub_id, "final",
    )
    decoded_path = resolve_stage_file(base, "decoded_states.pkl", "decoded states")
    return os.path.dirname(decoded_path)


def _load_decoded_states_friends(sub_id, parcellation):
    final_dir = _resolve_04_final_dir(sub_id, parcellation)
    with open(os.path.join(final_dir, "decoded_states.pkl"), "rb") as f:
        decoded = pickle.load(f)
    return decoded, final_dir


def _load_decoded_states_cross_stim(sub_id, parcellation, stimulus):
    stage_map = {
        "movie10": "m10_04_decoded",
        "harrypotter": "hp_04_decoded",
        "petitprince_fr": "pp_04_decoded",
        "petitprince_en": "pp_04_decoded",
    }
    if stimulus not in stage_map:
        raise ValueError(f"No cross-stim decoder for stimulus={stimulus}")
    base = os.path.join(SCRATCH_DIR, "output", stage_map[stimulus], parcellation, sub_id)
    ds_path = resolve_stage_file(
        base, "decoded_states.pkl", f"{stimulus} decoded states",
    )
    with open(ds_path, "rb") as f:
        return pickle.load(f)


def _load_recurrence(sub_id, parcellation):
    rec_base = os.path.join(
        SCRATCH_DIR, "output", "05a_recurrence_analysis", parcellation, sub_id,
    )
    rec_path = resolve_stage_file(
        rec_base, "recurrence_summary.json", "recurrence summary",
    )
    with open(rec_path) as f:
        return np.asarray(json.load(f)["recurrence_scores"])


def _episode_season_indices(run_ids):
    """Return ``(episode_idx, season_idx)`` arrays for each run.

    Assumes BIDS-style run IDs of the form ``task-sNNeMM[a-z]``. Falls back
    to ``season_idx=0`` and ``episode_idx=i`` (run order) on parse failure,
    logging a warning so silent drift in run-id format is caught early.
    """
    season_idx = []
    episode_idx = []
    parse_failures = []
    for i, rid in enumerate(run_ids):
        s = 0
        parsed = False
        try:
            token = rid.split("task-")[-1]  # e.g. "s01e02a"
            if token.startswith("s") and "e" in token:
                s = int(token[1:token.index("e")])
                parsed = True
        except (ValueError, IndexError):
            s = 0
        if not parsed:
            parse_failures.append(rid)
        season_idx.append(s)
        episode_idx.append(i)
    if parse_failures:
        logger.warning(
            "_episode_season_indices: could not parse season for %d/%d runs "
            "(e.g. %s). Falling back to season=0 and run-index order - "
            "the D1 confound baseline will under-represent actual episode "
            "structure for these runs.",
            len(parse_failures), len(run_ids), parse_failures[:3],
        )
    return np.asarray(episode_idx), np.asarray(season_idx)


# Feature loading, drift alignment, and per-layer PCA all live in
# :func:`utils.transformer_analysis.stream_pca_features`. 08d just calls it.


# ---------------------------------------------------------------------------
# Feature-matrix construction
# ---------------------------------------------------------------------------


CONFOUND_FEATURE_NAMES = [
    "run_onset_indicator",
    "tr_since_onset_norm",
    "tr_since_onset_norm_sq",
    "tr_since_onset_norm_cu",
    "episode_idx_norm",
    "season_idx_norm",
]


def _build_confound_design_matrix(run_ids, decoded_states):
    """Build the timing-only confound matrix for the D1 confound baseline.

    Columns (in order, see :data:`CONFOUND_FEATURE_NAMES`):

    * ``run_onset_indicator`` - 1.0 at the first TR of each run, else 0.
    * ``tr_since_onset_norm`` - linear within-run position, normalized to
      ``[0, 1]``.
    * ``tr_since_onset_norm_sq`` / ``_cu`` - quadratic and cubic polynomial
      drift terms. Added in the 2026-04 refactor to soak up scan-time /
      narrative-position drift that the linear term alone misses
      (previously, any state tracking "climax scenes happen late in a run"
      could beat the 4-feature baseline despite being a timing confound).
    * ``episode_idx_norm`` - run-index over the subject's episodes,
      normalized to ``[0, 1]``.
    * ``season_idx_norm`` - parsed from the BIDS ``task-sNNeMM`` token,
      normalized to ``[0, 1]``.
    """
    episode_idx_arr, season_idx_arr = _episode_season_indices(run_ids)
    rows = []
    for i, run_id in enumerate(run_ids):
        n_trs = len(decoded_states[run_id])
        run_onset = np.zeros(n_trs)
        run_onset[0] = 1.0
        if n_trs > 1:
            tr_since_onset = np.arange(n_trs) / (n_trs - 1)
        else:
            tr_since_onset = np.zeros(1)
        tr_sq = tr_since_onset ** 2
        tr_cu = tr_since_onset ** 3
        ep = np.full(n_trs, episode_idx_arr[i], dtype=float)
        se = np.full(n_trs, season_idx_arr[i], dtype=float)
        rows.append(np.column_stack([
            run_onset, tr_since_onset, tr_sq, tr_cu, ep, se,
        ]))
    X = np.vstack(rows)
    # Normalize episode and season indices to [0,1] for numerical stability.
    if episode_idx_arr.max() > 0:
        X[:, 4] = X[:, 4] / episode_idx_arr.max()
    if season_idx_arr.max() > 0:
        X[:, 5] = X[:, 5] / season_idx_arr.max()
    return X


# ---------------------------------------------------------------------------
# D1 main (+ neg control + confound baseline)
# ---------------------------------------------------------------------------


def _apply_global_fdr(results_per_lag):
    """Apply BH-FDR correction in-place across the full lag × layer grid."""
    flat = [
        (k, l)
        for k, d in results_per_lag.items()
        for l, e in d.items()
        if "p_perm" in e
    ]
    if flat:
        raw_p = np.array([results_per_lag[k][l]["p_perm"] for k, l in flat])
        fdr_p = benjamini_hochberg(raw_p)
        for (k, l), p_fdr in zip(flat, fdr_p):
            results_per_lag[k][l]["p_fdr"] = round(float(p_fdr), 4)


def _save_partial_atomic(data, path):
    """Atomically write *data* as JSON to *path* (POSIX rename)."""
    import tempfile
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        # Clean up temp file on any failure (including SIGTERM).
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pca_cache_dir(scratch_dir, parcellation, sub_id, model_key, stimulus,
                   output_subdir_suffix=""):
    """Return the PCA cache directory for (subject, parcellation, model, stimulus).

    Shared across all 9 same-stimulus lag jobs. The stimulus is included in
    the path even though the cache is currently only used for Friends fits
    (in-stim Friends D1 and cross-stim Friends-fit on shared basis) - this
    makes the directory structure robust against a future caller that
    accidentally passes a different stimulus.

    The cache lives under the same suffixed 08d output directory so that
    W-sweep variants (W=1, W=3, …) keep separate caches (different 08c
    features → different fitted PCAs). With the default empty suffix, the
    production cache path is unchanged.
    """
    return os.path.join(
        scratch_dir, "output",
        f"08d_transformer_depth{output_subdir_suffix}",
        parcellation, sub_id,
        "_pca_cache", model_key, stimulus,
    )


def _pca_split_hash_and_path(sub_id, parcellation, scratch_dir):
    """Return (split_hash, split_path_rel) for the 03a primary split file.

    Hash is MD5 of file contents (not mtime) so touching the file does not
    invalidate the cache. Changing the train run list does.
    """
    import hashlib
    split_path = os.path.join(
        scratch_dir, "output", "03a_pca4combined_hmm", parcellation, sub_id,
        "splits", "primary.json",
    )
    with open(split_path, "rb") as f:
        split_hash = hashlib.md5(f.read()).hexdigest()[:12]
    split_path_rel = os.path.relpath(split_path, scratch_dir)
    return split_hash, split_path_rel


def _d1_one_layer(
    layer_idx, features_by_layer, run_ids, decoded_states, lag,
    eligible_mask, y_full, null_y_masked, folds, label,
):
    """Process a single layer for D1 decoding.  Designed to be called from
    ``joblib.Parallel`` - all arguments are read-only numpy arrays or small
    Python objects so that the loky backend can serialise them efficiently.

    Returns ``(layer_idx, result_dict)`` or ``(layer_idx, None)`` on skip.
    """
    layer_runs = features_by_layer.get(layer_idx, {})
    if not layer_runs:
        return layer_idx, None
    try:
        X = build_layer_feature_matrix(
            layer_runs, run_ids, decoded_states, lag=lag,
        )
    except ValueError:
        return layer_idx, None

    X_elig = X[eligible_mask]
    y_elig = y_full[eligible_mask]
    if len(np.unique(y_elig)) < 2:
        return layer_idx, None

    batch_result = batch_loro_ridge_classify(
        X_elig, y_elig, null_y_masked, folds,
    )
    if batch_result is None:
        return layer_idx, None

    observed = batch_result["observed"]
    null_accs = batch_result["null_balanced_accuracies"]

    p_perm = permutation_pvalue(
        observed["balanced_accuracy"], null_accs, alternative="greater",
    )
    n_classes = len(np.unique(y_elig))
    eff = compute_effect_size(observed["balanced_accuracy"], n_classes)

    result = {
        "balanced_accuracy": round(observed["balanced_accuracy"], 4),
        "weighted_f1": round(observed["weighted_f1"], 4),
        "cohen_kappa": round(observed["cohen_kappa"], 4),
        "n_classes": n_classes,
        "chance_level": round(eff["chance_level"], 4),
        "normalized_effect_size": round(eff["normalized_effect_size"], 4),
        "p_perm": round(float(p_perm), 4),
        "null_mean": round(float(np.mean(null_accs)), 4) if null_accs else None,
        "null_std": round(float(np.std(null_accs)), 4) if null_accs else None,
        "n_permutations": len(null_accs),
    }
    return layer_idx, result


def _run_d1_decoder_set(
    label, decoded_states, features_by_layer, state_subset, run_ids,
    run_boundaries, all_states, n_perm, perm_seed_base,
    *, lags_subset=None, partials_dir=None, n_jobs=1,
):
    """Run D1-style decoding for one state subset across all lags and layers.

    Parameters
    ----------
    lags_subset : list[int] | None
        If given, only process these lag values (must be in LAGS_TO_TEST).
    partials_dir : str | None
        If given, enable per-lag checkpointing. Each lag's results are
        saved atomically to ``{partials_dir}/D1_{label}_lag{N}.json``
        and BH-FDR is **skipped** (deferred to a D1merge step).
    n_jobs : int
        Number of parallel workers for the layer loop (default 1 = serial).
        When > 1, layers within each lag are processed via ``joblib.Parallel``.
        Set ``OPENBLAS_NUM_THREADS=1`` in the environment to avoid BLAS thread
        contention.  Checkpointing granularity changes from per-layer to
        per-lag when ``n_jobs > 1``.

    Returns a nested dict ``{"lag_L": {layer: {...}}}``.  When
    ``partials_dir`` is None the dict includes BH-FDR corrected p-values;
    when set it contains only raw ``p_perm``.
    """
    state_subset_set = set(int(s) for s in state_subset)
    if not state_subset_set:
        logger.warning("%s: empty state subset - skipping", label)
        return {}

    eligible_mask = np.array(
        [int(s) in state_subset_set for s in all_states], dtype=bool,
    )
    if eligible_mask.sum() == 0:
        logger.warning("%s: 0 eligible TRs - skipping", label)
        return {}

    folds = build_loro_folds(run_boundaries, eligible_mask=eligible_mask)
    if len(folds) < 3:
        logger.warning("%s: < 3 usable LORO folds - skipping", label)
        return {}

    y_full = all_states.astype(np.int8)
    n_layers = max(features_by_layer.keys()) + 1

    lags_to_run = lags_subset if lags_subset is not None else LAGS_TO_TEST
    if partials_dir is not None:
        os.makedirs(partials_dir, exist_ok=True)

    results_per_lag: dict[str, dict[int, dict]] = {}

    for lag_i, lag in enumerate(lags_to_run):
        logger.info(
            "%s: lag %d/%d (lag=%d)", label, lag_i + 1, len(lags_to_run), lag,
        )

        # ── Per-layer checkpoint: load existing partial ────────────────
        partial_path = None
        completed_layers: set[int] = set()
        lag_results: dict[int, dict] = {}
        if partials_dir is not None:
            partial_path = os.path.join(
                partials_dir, f"D1_{label}_lag{lag}.json",
            )
            if os.path.exists(partial_path):
                with open(partial_path) as f:
                    partial_data = json.load(f)
                for k, v in partial_data.get("results", {}).items():
                    lag_results[int(k)] = v
                completed_layers = {int(k) for k in lag_results}
                logger.info(
                    "%s lag=%d: resuming from checkpoint (%d/%d layers done)",
                    label, lag, len(completed_layers), n_layers,
                )

        # Permutation null is generated directly in the eligible-subspace
        # so shifted labels can only come from the intended class set.
        # The prior pattern (precompute on full y_full, then mask by
        # original positions) leaked non-eligible classes into the null
        # training labels and depressed null_mean below true chance -
        # see 2026-04-10 null-leakage plan.
        null_y_masked = precompute_eligible_null_state_sequences(
            y_full, run_boundaries, eligible_mask, n_perm,
            perm_seed_base + lag * 100003,
        )

        # ── Recover best-layer tracking from checkpointed layers ─────
        best_layer_acc = -np.inf
        best_layer_idx = None
        best_layer_p = None
        best_layer_eff = None
        for layer_idx in completed_layers:
            acc = lag_results[layer_idx].get("balanced_accuracy", -np.inf)
            if acc is not None and acc > best_layer_acc:
                best_layer_acc = acc
                best_layer_idx = layer_idx
                best_layer_p = lag_results[layer_idx].get("p_perm")
                best_layer_eff = lag_results[layer_idx].get("normalized_effect_size")

        # ── Determine which layers still need to be computed ──────
        layers_to_run = [
            idx for idx in range(n_layers) if idx not in completed_layers
        ]

        if layers_to_run:
            _lag_t0 = time.monotonic()
            # Process layers in chunks of n_jobs for incremental checkpointing
            chunk_size = max(n_jobs, 1)
            total_computed = 0
            for chunk_start in range(0, len(layers_to_run), chunk_size):
                chunk = layers_to_run[chunk_start:chunk_start + chunk_size]
                chunk_results = Parallel(n_jobs=n_jobs)(
                    # Pass only the worker's own layer, not the full all-layer
                    # dict: shipping ~47G to each of n_jobs workers OOM-killed
                    # n_jobs>2 at startup (fixed 2026-05-29). Result-neutral -
                    # _d1_one_layer reads only features_by_layer.get(idx).
                    delayed(_d1_one_layer)(
                        idx, {idx: features_by_layer.get(idx, {})},
                        run_ids, decoded_states, lag,
                        eligible_mask, y_full, null_y_masked, folds, label,
                    )
                    for idx in chunk
                )

                for layer_idx, result in chunk_results:
                    if result is None:
                        continue
                    lag_results[layer_idx] = result
                    total_computed += 1
                    acc = result["balanced_accuracy"]
                    if acc > best_layer_acc:
                        best_layer_acc = acc
                        best_layer_idx = layer_idx
                        best_layer_p = result["p_perm"]
                        best_layer_eff = result["normalized_effect_size"]

                # ── Incremental checkpoint after each chunk ──────────
                if partial_path is not None:
                    _save_partial_atomic(
                        {
                            "lag": lag,
                            "label": label,
                            "completed_layers": sorted(lag_results.keys()),
                            "n_layers_total": n_layers,
                            "results": {str(k): v for k, v in lag_results.items()},
                        },
                        partial_path,
                    )
                _chunk_elapsed = time.monotonic() - _lag_t0
                logger.info(
                    "[%s] lag=%d: chunk %d/%d done (%d/%d layers, %.1fs elapsed, n_jobs=%d)",
                    label, lag, chunk_start // chunk_size + 1,
                    (len(layers_to_run) + chunk_size - 1) // chunk_size,
                    len(lag_results), n_layers, _chunk_elapsed, n_jobs,
                )

            _lag_elapsed = time.monotonic() - _lag_t0
            logger.info(
                "[%s] lag=%d: all %d layers computed in %.1fs (n_jobs=%d)",
                label, lag, total_computed, _lag_elapsed, n_jobs,
            )

            # ── Final checkpoint for this lag ──────────────────────
            if partial_path is not None:
                _save_partial_atomic(
                    {
                        "lag": lag,
                        "label": label,
                        "completed_layers": sorted(lag_results.keys()),
                        "n_layers_total": n_layers,
                        "results": {str(k): v for k, v in lag_results.items()},
                    },
                    partial_path,
                )

        if best_layer_idx is not None:
            logger.info(
                "[%s] lag=%d: best layer=%d bal_acc=%.4f eff=%.3f p=%.4f",
                label, lag, best_layer_idx, best_layer_acc,
                best_layer_eff, best_layer_p,
            )
        results_per_lag[f"lag_{lag}"] = lag_results

    # Global BH-FDR across the full layers × lags grid.
    # Skipped when using partials (deferred to D1merge).
    if partials_dir is None:
        _apply_global_fdr(results_per_lag)
    return results_per_lag


_GATE_DEPRECATED = {
    "neg_control_passed": None,
    "status": "deprecated",
    "reason": (
        "Run-onset negative control dropped from the manuscript 2026-05-31: "
        "run-onset-anchored states are feature-distinctive (stereotyped onset "
        "content), not content-free, so they are a poor content-free control "
        "and decode at least as well as content-eligible states under a "
        "chance-corrected effect size. R4b's depth claim rests on the "
        "within-run circular-shift permutation null + timing-regressor floor "
        "instead. Gate no longer computed or consumed by any script. See "
        "findings_08d_transformer_depth.md."
    ),
}


def _write_deprecated_gate(out_dir):
    """Write the deprecation stub in place of the old neg-control gate.

    Keeps the ``D1_neg_control_gate.json`` filename present (so ``--force``
    output-existence checks and re-runs still find a file) while recording why
    no verdict is produced. See ``_GATE_DEPRECATED``.
    """
    with open(os.path.join(out_dir, "D1_neg_control_gate.json"), "w") as f:
        json.dump(_GATE_DEPRECATED, f, indent=2)


def _compute_neg_control_gate(d1_main, d1_neg):
    """DEPRECATED (2026-05-31) - no longer called; retained for reference only.

    The run-onset negative control was dropped from the manuscript (see
    ``_GATE_DEPRECATED``). This function's raw-balanced-accuracy comparison was
    also confounded by the differing class counts of the content-eligible
    (~16-class) and run-onset (~7-class) decoders. D1merge now writes
    ``_GATE_DEPRECATED`` via :func:`_write_deprecated_gate` instead of calling
    this.

    Decide whether D1 main beats the design-driven negative control.

    At the D1 main peak ``(lag, layer)`` (lag=0 excluded, matching the
    downstream peak search), look up the matching layer in the neg-control
    results at the *same lag*. The test passes iff D1 main beats the neg
    control by more than half the random-null standard deviation of
    D1 main at that cell - a cheap proxy for a permutation-based
    ``Δ > 0`` test that avoids holding the full null distributions in
    memory. We also record ``p_main`` (content-eligible p_perm) and
    ``p_neg`` (design-driven p_perm) at the peak so downstream tools can
    apply a stricter joint criterion if needed.

    Returns a JSON-safe dict that is written to
    ``D1_neg_control_gate.json`` for 08e/08f/08g consumption.
    """
    if not d1_main:
        return {
            "neg_control_passed": None,
            "skipped_reason": "d1_main empty",
        }

    best_lag_key, best_layer, main_acc = _find_best_lag_layer(
        d1_main, exclude_lags=PEAK_LAG_EXCLUDE,
    )
    if best_lag_key is None:
        return {
            "neg_control_passed": None,
            "skipped_reason": "d1_main has no decodable (lag, layer) cell",
        }

    main_entry = d1_main.get(best_lag_key, {}).get(best_layer, {})
    null_std = main_entry.get("null_std")
    p_main = main_entry.get("p_perm")

    if not d1_neg:
        return {
            "best_lag": int(best_lag_key.split("_")[1]),
            "best_layer": int(best_layer),
            "main_peak_acc": float(main_acc),
            "p_main": p_main,
            "neg_control_passed": None,
            "skipped_reason": "no run_onset_anchored states available",
        }

    neg_lag_data = d1_neg.get(best_lag_key, {})
    neg_entry = neg_lag_data.get(best_layer, {})
    if not neg_entry:
        # Neg control has no coverage of this (lag, layer). Fall back to
        # the maximum neg-control accuracy at the same lag, so we are
        # still answering "does main beat neg at the chosen lag?".
        if neg_lag_data:
            fallback_layer, fallback_entry = max(
                neg_lag_data.items(),
                key=lambda kv: kv[1].get("balanced_accuracy", -np.inf),
            )
            neg_entry = fallback_entry
            neg_layer_used = int(fallback_layer)
        else:
            return {
                "best_lag": int(best_lag_key.split("_")[1]),
                "best_layer": int(best_layer),
                "main_peak_acc": float(main_acc),
                "p_main": p_main,
                "neg_control_passed": None,
                "skipped_reason": "no neg control results at best lag",
            }
    else:
        neg_layer_used = int(best_layer)

    neg_acc = neg_entry.get("balanced_accuracy")
    p_neg = neg_entry.get("p_perm")
    if neg_acc is None:
        return {
            "best_lag": int(best_lag_key.split("_")[1]),
            "best_layer": int(best_layer),
            "main_peak_acc": float(main_acc),
            "p_main": p_main,
            "neg_control_passed": None,
            "skipped_reason": "neg control cell has no accuracy",
        }

    delta = float(main_acc) - float(neg_acc)
    # Half-null-std margin: passes if main beats neg by at least
    # 0.5 × (D1 main null std at the peak cell). If null_std is missing
    # or zero, fall back to a positive-delta criterion.
    margin = 0.5 * float(null_std) if null_std not in (None, 0) else 0.0
    passed = bool(delta > margin)

    return {
        "best_lag": int(best_lag_key.split("_")[1]),
        "best_layer": int(best_layer),
        "neg_layer_used": neg_layer_used,
        "main_peak_acc": float(main_acc),
        "neg_peak_acc": float(neg_acc),
        "delta_peak": round(delta, 4),
        "null_std_main": (float(null_std) if null_std is not None else None),
        "margin_used": round(float(margin), 4),
        "p_main": p_main,
        "p_neg": p_neg,
        "neg_control_passed": passed,
    }


def _find_best_lag_layer(results_per_lag, exclude_lags=None):
    """Return ``(best_lag_key, best_layer_idx, best_acc)`` over a D1 result dict.

    Parameters
    ----------
    results_per_lag : dict
        ``{lag_key: {layer_idx: entry_dict}}`` as produced by
        :func:`_run_d1_decoder_set`.
    exclude_lags : iterable of int, optional
        Integer lag values to exclude from the peak search (e.g. ``{0}`` to
        skip the synchronous / autocorrelation diagnostic). Lag values are
        extracted from the ``lag_N`` key suffix. ``None`` considers every
        lag.
    """
    exclude = set(int(l) for l in exclude_lags) if exclude_lags else set()
    best = (None, None, -np.inf)
    for lag_key, lag_data in results_per_lag.items():
        try:
            lag_val = int(lag_key.split("_")[1])
        except (IndexError, ValueError):
            lag_val = None
        if lag_val is not None and lag_val in exclude:
            continue
        for layer, entry in lag_data.items():
            acc = entry.get("balanced_accuracy", -np.inf)
            if acc is not None and acc > best[2]:
                best = (lag_key, int(layer), float(acc))
    return best


def _plot_depth_profile(results_by_label, out_path, title):
    """Multi-line depth-profile plot (one line per label at its best lag)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, results in results_by_label.items():
        if not results:
            continue
        best_lag, _, _ = _find_best_lag_layer(results)
        if best_lag is None:
            continue
        lag_data = results[best_lag]
        layers = sorted(int(k) for k in lag_data.keys())
        accs = [lag_data[k]["balanced_accuracy"] for k in layers]
        chance = lag_data[layers[0]]["chance_level"] if layers else None
        ax.plot(layers, accs, marker="o", markersize=3, label=f"{label} ({best_lag})")
        if chance is not None:
            ax.axhline(chance, linestyle=":", color="gray", alpha=0.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _d1net_group_order():
    """Return the canonical row order for D1-net groups (network × polarity)."""
    order = []
    for net in NETWORK_ORDER:
        for polarity in ("+", "-"):
            order.append(f"{net}_{'pos' if polarity == '+' else 'neg'}")
    return order


def _plot_d1net_heatmap(d1net_results, out_path):
    """Group × layer heatmap of balanced accuracy.

    Rows are ``{network}_{pos|neg}`` groups that actually passed the
    ``D1NET_MIN_STATES`` / ``D1NET_MIN_TRS`` gates. Polarity is preserved so
    functionally opposite states (e.g. DMN activation vs. deactivation)
    appear on separate rows instead of being averaged together.
    """
    row_order = _d1net_group_order()
    rows = [
        g for g in row_order
        if g in d1net_results and d1net_results[g].get("per_layer")
    ]
    if not rows:
        return
    layers = sorted(
        {int(l) for g in rows for l in d1net_results[g]["per_layer"]}
    )
    if not layers:
        return
    mat = np.full((len(rows), len(layers)), np.nan)
    for i, g in enumerate(rows):
        for j, lyr in enumerate(layers):
            entry = d1net_results[g]["per_layer"].get(lyr)
            if entry is not None:
                mat[i, j] = entry["balanced_accuracy"]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(layers)),
                                     max(3, 0.35 * len(rows))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer index")
    ax.set_title("D1-net: balanced accuracy by (network, polarity)")
    fig.colorbar(im, ax=ax, label="bal_acc")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_d2_heatmap(d2_results, n_layers, out_path):
    """State × layer AUC heatmap for D2, sorted by peak layer."""
    states = sorted(d2_results.keys(),
                    key=lambda s: d2_results[s]["selectivity"]["peak_layer"] or 0)
    if not states:
        return
    mat = np.full((len(states), n_layers), np.nan)
    for i, s in enumerate(states):
        for lyr_str, auc in d2_results[s]["layer_auc"].items():
            lyr = int(lyr_str)
            if 0 <= lyr < n_layers and auc is not None:
                mat[i, lyr] = auc

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * n_layers),
                                     max(3, 0.25 * len(states))))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.4, vmax=1.0)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("State (sorted by peak layer)")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_title("D2: per-state layer AUC")
    fig.colorbar(im, ax=ax, label="AUC")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# D2 - per-state layer AUC
# ---------------------------------------------------------------------------


D2_SCHEMA_VERSION = 3  # Bumped 2026-04-30: vectorised Ridge AUC + n_perm=1000 + per-layer checkpointing


def _d2_one_state(
    state_id, all_states, layer_matrices, null_seqs, folds, n_perm,
):
    """Process a single state for D2 (per-state layer AUC).  Designed to be
    called from ``joblib.Parallel``.

    Returns ``(state_id, result_dict)`` or ``(state_id, None)`` on skip.
    """
    y = (all_states == state_id).astype(int)
    fo = float(y.mean())
    if fo < D2_MIN_FRACTIONAL_OCCUPANCY:
        return state_id, None

    layer_auc: dict[int, float] = {}
    layer_p: dict[int, float] = {}
    n_pos_obs = int(y.sum())
    n_pos_per_fold = [int(y[test_idx].sum()) for _, test_idx in folds]

    for layer_idx, X in layer_matrices.items():
        observed = loro_logistic_auc_cv(X, y, folds)
        if observed is None:
            continue

        null_vals = []
        for i in range(n_perm):
            y_perm = (null_seqs[i] == state_id).astype(int)
            if y_perm.sum() < 2:
                continue
            null_res = loro_logistic_auc_cv(X, y_perm, folds)
            if null_res is not None:
                null_vals.append(null_res["roc_auc"])

        p = permutation_pvalue(
            observed["roc_auc"], null_vals, alternative="greater",
        )
        layer_auc[layer_idx] = round(float(observed["roc_auc"]), 4)
        layer_p[layer_idx] = round(float(p), 4)

    if not layer_auc:
        return state_id, None

    # BH-FDR per state across layers.
    layer_keys = sorted(layer_auc.keys())
    raw_p = np.array([layer_p[k] for k in layer_keys])
    p_fdr = benjamini_hochberg(raw_p)
    layer_p_fdr = {k: round(float(f), 4) for k, f in zip(layer_keys, p_fdr)}

    selectivity = layer_selectivity(layer_auc)
    m_minus_med = selectivity["max_minus_median"]
    m_minus_med_finite = (
        m_minus_med is not None and np.isfinite(m_minus_med)
    )
    non_selective = (
        m_minus_med_finite
        and m_minus_med < D2_SELECTIVITY_THRESHOLD
    )
    non_selective_at = {
        f"{t:g}": bool(m_minus_med_finite and m_minus_med < t)
        for t in D2_SELECTIVITY_THRESHOLDS_REPORT
    }

    result = {
        "fractional_occupancy": round(fo, 4),
        "n_positive": n_pos_obs,
        "n_positive_per_fold_min": int(min(n_pos_per_fold)) if n_pos_per_fold else 0,
        "n_positive_per_fold_median": int(np.median(n_pos_per_fold)) if n_pos_per_fold else 0,
        "layer_auc": {str(k): layer_auc[k] for k in layer_keys},
        "layer_p": {str(k): layer_p[k] for k in layer_keys},
        "layer_p_fdr": {str(k): layer_p_fdr[k] for k in layer_keys},
        "selectivity": {
            k: (int(v) if k == "peak_layer" and v is not None else
                (round(float(v), 4) if v is not None and np.isfinite(v) else None))
            for k, v in selectivity.items()
        },
        "non_selective": bool(non_selective),
        "non_selective_at": non_selective_at,
        "selectivity_threshold_primary": D2_SELECTIVITY_THRESHOLD,
    }
    return state_id, result


def _build_d2_folds(
    run_ids, run_boundaries, total_trs, n_splits=10, seed=42,
    eligible_mask=None,
):
    """Build StratifiedGroupKFold folds for D2.

    Replaces 292-fold LORO with 10-fold-by-run, stratified by season. Each run
    lives entirely in one fold (groups=run-idx), and folds are balanced across
    the 6 Friends seasons (y=season-id). Empirically validated 2026-04-28 to
    yield ~700 positives/fold for FO≈5% states vs ~22 in LORO - lower per-fold
    AUC variance and methodologically superior for binary AUC (Stats C2,
    Neuro A1 of 2026-04-28 D2 review).

    Parameters
    ----------
    eligible_mask : np.ndarray of bool, optional
        Length-``total_trs`` eligibility mask. When provided, returned fold
        indices address into ``X[eligible_mask]``/``y[eligible_mask]`` directly,
        matching the convention of :func:`build_loro_folds`. The stratification
        and groups are still computed on the full index space (one run lives
        entirely in one fold) before remapping.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.empty(total_trs, dtype=np.int32)
    season_per_run = np.empty(len(run_boundaries), dtype=np.int32)
    for run_idx, ((start, end), run_id) in enumerate(zip(run_boundaries, run_ids)):
        groups[start:end] = run_idx
        # Parse "task-sXXeYY[a|b]" → season int. Default to 0 if unparseable.
        season = 0
        try:
            tag = run_id.split("task-s")[1][:2]
            season = int(tag)
        except (IndexError, ValueError):
            pass
        season_per_run[run_idx] = season
    y_strat = season_per_run[groups]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw_folds = [
        (np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64))
        for train_idx, test_idx in splitter.split(
            np.zeros(total_trs), y_strat, groups=groups,
        )
    ]
    if eligible_mask is None:
        return raw_folds

    # Remap full-space indices into eligible-only-space (matches build_loro_folds)
    full_to_eligible = np.full(total_trs, -1, dtype=np.int64)
    eligible_pos = np.flatnonzero(eligible_mask)
    full_to_eligible[eligible_pos] = np.arange(len(eligible_pos))
    remapped = []
    for train_idx, test_idx in raw_folds:
        train_remap = full_to_eligible[train_idx[eligible_mask[train_idx]]]
        test_remap = full_to_eligible[test_idx[eligible_mask[test_idx]]]
        if len(train_remap) > 0 and len(test_remap) > 0:
            remapped.append((train_remap, test_remap))
    return remapped


def _hash_eligible_ids(eligible_ids):
    import hashlib
    return hashlib.md5(",".join(str(s) for s in sorted(eligible_ids)).encode()).hexdigest()[:12]


def _run_d2(
    decoded_states, features_by_layer, content_eligible, run_ids,
    run_boundaries, all_states, best_lag, n_perm, perm_seed_base, out_dir,
    *, n_jobs=1, force=False,
):
    """Vectorised D2: per-state × layer AUC via batched Ridge regression.

    Replaces the legacy LogReg-per-fold path (~120-160 h/cell) with a Cholesky-
    once-per-fold Ridge solve (~6-8 h/cell). Each layer is one
    :func:`batch_loro_ridge_per_state_auc` call returning per-state observed +
    null AUCs. Per-layer atomic checkpoints to disk; resume reads `.partial`
    and skips completed layers.

    Design: ``the design notes``
    Schema: v3 (Ridge engine + n_perm=1000 + selectivity_score continuous).

    The legacy ``_d2_one_state`` (LogReg per state × layer) is retained
    elsewhere in this file for the validation harness only.
    """
    if n_jobs > 1:
        logger.warning(
            "D2 vectorised path ignores n_jobs=%d (single big op per layer; "
            "set OPENBLAS_NUM_THREADS via SLURM --cpus-per-task instead)",
            n_jobs,
        )

    logger.info("=" * 40)
    logger.info(
        "Analysis D2 vectorised (best lag=%d, n_perm=%d, engine=ridge_batch_cholesky, schema=%d)",
        best_lag, n_perm, D2_SCHEMA_VERSION,
    )
    logger.info("=" * 40)

    n_layers = max(features_by_layer.keys()) + 1
    total_trs = len(all_states)

    # Pre-build per-layer lagged X matrices ONCE (single lag).
    layer_matrices: dict[int, np.ndarray] = {}
    for layer_idx in range(n_layers):
        runs_data = features_by_layer.get(layer_idx, {})
        if not runs_data:
            continue
        try:
            layer_matrices[layer_idx] = build_layer_feature_matrix(
                runs_data, run_ids, decoded_states, lag=best_lag,
            )
        except ValueError:
            continue
    available_layers = sorted(layer_matrices.keys())

    # Eligible-subspace null (Stats V2 of 04-30 D2 review).
    eligible_ids = sorted(int(s) for s in content_eligible)
    if len(eligible_ids) < 2:
        logger.warning("D2: only %d eligible states; skipping", len(eligible_ids))
        return {}
    eligible_hash = _hash_eligible_ids(eligible_ids)
    eligible_mask = np.isin(all_states.astype(int), eligible_ids)
    if eligible_mask.sum() < 100:
        logger.warning("D2: <100 eligible TRs (n=%d); skipping", int(eligible_mask.sum()))
        return {}

    # Build folds in eligible-only index space (so we can pass X[eligible_mask]
    # and folds together - matches build_loro_folds convention).
    folds = _build_d2_folds(
        run_ids, run_boundaries, total_trs, eligible_mask=eligible_mask,
    )
    logger.info(
        "D2 folds: %d (StratifiedGroupKFold, runs-as-groups, season-stratified, eligible-masked)",
        len(folds),
    )

    null_seqs_full = precompute_eligible_null_state_sequences(
        all_states.astype(np.int8), run_boundaries, eligible_mask, n_perm,
        perm_seed_base,
    )

    # Per-state per-fold positive counts (Coding C6).
    eligible_only_states = all_states[eligible_mask].astype(np.int8)
    n_pos_per_fold_per_state: dict[int, list[int]] = {}
    for sid in eligible_ids:
        n_pos_per_fold_per_state[sid] = [
            int((eligible_only_states[test_idx] == sid).sum())
            for _, test_idx in folds
        ]

    out_path = os.path.join(out_dir, "D2_state_layer_auc.json")
    partial_path = out_path + ".partial"

    # Resume logic. Schema-version mismatch (e.g. v2 from 04-28) force-discards
    # the partial - confirmed by 04-30 Coding A3 that this is the right behaviour.
    completed_layers: dict[int, dict] = {}  # layer_idx -> per-state AUC + null AUC
    if force and os.path.exists(partial_path):
        os.remove(partial_path)
        logger.info("D2 --force: removed stale .partial")
    elif os.path.exists(partial_path):
        try:
            with open(partial_path) as f:
                prior = json.load(f)
            if (prior.get("schema_version") == D2_SCHEMA_VERSION
                    and prior.get("eligible_hash") == eligible_hash
                    and int(prior.get("best_lag", -1)) == int(best_lag)
                    and int(prior.get("n_perm", -1)) == int(n_perm)
                    and prior.get("engine") == "ridge_batch_cholesky"):
                # Per-layer schema: layers_completed: {layer_idx_str: {observed_auc, null_auc}}
                for layer_str, layer_payload in prior.get("layers_completed", {}).items():
                    layer_idx = int(layer_str)
                    completed_layers[layer_idx] = {
                        "observed_auc": np.asarray(
                            layer_payload["observed_auc"], dtype=np.float64,
                        ),
                        "null_auc": np.asarray(
                            layer_payload["null_auc"], dtype=np.float64,
                        ),
                        "n_valid_folds_observed": int(
                            layer_payload.get("n_valid_folds_observed", 0),
                        ),
                    }
                logger.info(
                    "D2 resume: loaded %d completed layers from .partial",
                    len(completed_layers),
                )
            else:
                logger.warning(
                    "D2 .partial mismatch (schema=%s eligible_hash=%s best_lag=%s n_perm=%s engine=%s) - discarding",
                    prior.get("schema_version"), prior.get("eligible_hash"),
                    prior.get("best_lag"), prior.get("n_perm"), prior.get("engine"),
                )
                os.remove(partial_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("D2 .partial unreadable (%s) - discarding", exc)
            try:
                os.remove(partial_path)
            except OSError:
                pass

    def _serialise_layer(layer_payload):
        """Convert per-layer numpy AUC arrays to JSON-friendly form."""
        return {
            "observed_auc": layer_payload["observed_auc"].tolist(),
            "null_auc": layer_payload["null_auc"].tolist(),
            "n_valid_folds_observed": int(layer_payload["n_valid_folds_observed"]),
        }

    def _write_partial_layers(layers_dict, complete=False, final_payload=None):
        payload = {
            "schema_version": D2_SCHEMA_VERSION,
            "engine": "ridge_batch_cholesky",
            "best_lag": int(best_lag),
            "n_perm": int(n_perm),
            "n_folds": len(folds),
            "fold_scheme": "stratified_group_kfold_10_season_stratified",
            "eligible_hash": eligible_hash,
            "eligible_states": eligible_ids,
            "selectivity_thresholds_reported": list(D2_SELECTIVITY_THRESHOLDS_REPORT),
            "selectivity_threshold_primary": D2_SELECTIVITY_THRESHOLD,
            "layers_completed": {
                str(layer_idx): _serialise_layer(layer_payload)
                for layer_idx, layer_payload in layers_dict.items()
            },
            "complete": bool(complete),
        }
        if complete and final_payload is not None:
            payload.update(final_payload)
        target = out_path if complete else partial_path
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)
        if complete and os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass

    # ── Per-layer compute loop ─────────────────────────────────────────────
    _d2_t0 = time.monotonic()
    layers_to_run = [L for L in available_layers if L not in completed_layers]
    if not layers_to_run:
        logger.info(
            "D2 resume: all %d layers already done; finalising", len(completed_layers),
        )

    X_eligible_cache: dict[int, np.ndarray] = {}  # masked X per layer
    for i, layer_idx in enumerate(layers_to_run):
        X_full = layer_matrices[layer_idx]
        X_eligible_cache[layer_idx] = X_full[eligible_mask]
        result = batch_loro_ridge_per_state_auc(
            X_eligible_cache[layer_idx],
            eligible_only_states,
            null_seqs_full,
            folds,
            eligible_ids,
            alpha=1.0,
        )
        if result is None:
            logger.warning(
                "D2: layer %d returned None (degenerate); marking with NaN AUCs",
                layer_idx,
            )
            n_states = len(eligible_ids)
            completed_layers[layer_idx] = {
                "observed_auc": np.full(n_states, np.nan, dtype=np.float64),
                "null_auc": np.full((n_perm, n_states), np.nan, dtype=np.float64),
                "n_valid_folds_observed": 0,
            }
        else:
            completed_layers[layer_idx] = {
                "observed_auc": result["observed_auc"],
                "null_auc": result["null_auc"],
                "n_valid_folds_observed": result["n_valid_folds_observed"],
            }
        # Free per-layer cached X (we don't need it again)
        del X_eligible_cache[layer_idx]

        _layer_elapsed = time.monotonic() - _d2_t0
        logger.info(
            "D2: layer %d done (%d/%d, %.1fs elapsed, partial saved)",
            layer_idx, len(completed_layers), len(available_layers), _layer_elapsed,
        )
        # Atomic per-layer checkpoint
        _write_partial_layers(completed_layers, complete=False)

    # ── Aggregate per-state results across layers ──────────────────────────
    n_states = len(eligible_ids)
    results: dict[int, dict] = {}
    for s_idx, state_id in enumerate(eligible_ids):
        layer_auc: dict[int, float] = {}
        layer_p: dict[int, float] = {}
        n_valid_per_layer: dict[int, int] = {}

        for layer_idx, layer_payload in sorted(completed_layers.items()):
            obs_auc = layer_payload["observed_auc"][s_idx]
            null_auc_state = layer_payload["null_auc"][:, s_idx]
            null_auc_state = null_auc_state[~np.isnan(null_auc_state)]
            if not np.isfinite(obs_auc) or len(null_auc_state) < 10:
                continue
            p = permutation_pvalue(obs_auc, null_auc_state.tolist(), alternative="greater")
            layer_auc[int(layer_idx)] = round(float(obs_auc), 4)
            layer_p[int(layer_idx)] = round(float(p), 4)
            n_valid_per_layer[int(layer_idx)] = int(len(null_auc_state))

        if not layer_auc:
            continue

        # BH-FDR per state across layers
        layer_keys = sorted(layer_auc.keys())
        raw_p = np.array([layer_p[k] for k in layer_keys])
        p_fdr = benjamini_hochberg(raw_p)
        layer_p_fdr = {k: round(float(f), 4) for k, f in zip(layer_keys, p_fdr)}

        selectivity = layer_selectivity(layer_auc)
        m_minus_med = selectivity["max_minus_median"]
        m_minus_med_finite = (
            m_minus_med is not None and np.isfinite(m_minus_med)
        )
        non_selective = (
            m_minus_med_finite and m_minus_med < D2_SELECTIVITY_THRESHOLD
        )
        non_selective_at = {
            f"{t:g}": bool(m_minus_med_finite and m_minus_med < t)
            for t in D2_SELECTIVITY_THRESHOLDS_REPORT
        }

        # Per-state per-fold positive counts (Coding C6)
        n_pos_per_fold_state = n_pos_per_fold_per_state[state_id]
        fo = float((eligible_only_states == state_id).mean())
        n_pos_obs = int((eligible_only_states == state_id).sum())

        results[int(state_id)] = {
            "fractional_occupancy": round(fo, 4),
            "n_positive": n_pos_obs,
            "n_positive_per_fold_min": int(min(n_pos_per_fold_state))
            if n_pos_per_fold_state else 0,
            "n_positive_per_fold_median": int(np.median(n_pos_per_fold_state))
            if n_pos_per_fold_state else 0,
            "layer_auc": {str(k): layer_auc[k] for k in layer_keys},
            "layer_p": {str(k): layer_p[k] for k in layer_keys},
            "layer_p_fdr": {str(k): layer_p_fdr[k] for k in layer_keys},
            "selectivity": {
                k: (int(v) if k == "peak_layer" and v is not None else
                    (round(float(v), 4) if v is not None and np.isfinite(v) else None))
                for k, v in selectivity.items()
            },
            # Continuous selectivity score (Neuro N3) - for 08g/08f to use
            # instead of (or alongside) the binary non_selective flag.
            "selectivity_score": (
                round(float(m_minus_med), 4)
                if m_minus_med_finite else None
            ),
            "non_selective": bool(non_selective),
            "non_selective_at": non_selective_at,
            "selectivity_threshold_primary": D2_SELECTIVITY_THRESHOLD,
        }

    _d2_elapsed = time.monotonic() - _d2_t0
    logger.info(
        "D2: all %d/%d layers computed in %.1fs",
        len(completed_layers), len(available_layers), _d2_elapsed,
    )

    final_payload = {"states": results}
    _write_partial_layers(completed_layers, complete=True, final_payload=final_payload)

    n_non_sel = sum(1 for r in results.values() if r["non_selective"])
    logger.info(
        "Saved D2: %d states (%d selective, %d non-selective @ %.2f) -> %s",
        len(results), len(results) - n_non_sel, n_non_sel,
        D2_SELECTIVITY_THRESHOLD, out_path,
    )

    _plot_d2_heatmap(
        results, max(available_layers) + 1 if available_layers else 0,
        os.path.join(out_dir, "D2_state_layer_auc.png"),
    )
    return results


# ---------------------------------------------------------------------------
# D1-net
# ---------------------------------------------------------------------------


def _run_d1_net(
    decoded_states, features_by_layer, eligibility, run_ids,
    run_boundaries, all_states, best_lag, n_perm, perm_seed_base,
    parcellation, final_dir, out_dir, *, n_jobs=1,
):
    logger.info("=" * 40)
    logger.info("Analysis D1-net (best lag = %d)", best_lag)
    logger.info("=" * 40)

    state_means_path = os.path.join(final_dir, "state_means_parcel.npy")
    if not os.path.exists(state_means_path):
        logger.warning("state_means_parcel.npy missing at %s - skipping D1-net",
                       state_means_path)
        return {}

    state_means = np.load(state_means_path)
    # Polarity-aware stratification: each content-eligible state is tagged
    # with (network, "+"|"-") so that, e.g., DMN activation and DMN
    # deactivation states end up in separate groups instead of being pooled
    # together and averaged into a single DMN depth profile.
    try:
        groups = stratify_states_by_dominant_network(
            state_means, eligibility["content_eligible"], parcellation,
            include_sign=True,
        )
    except RuntimeError as exc:
        logger.warning("%s - skipping D1-net", exc)
        return {}

    results: dict[str, dict] = {}
    y_full = all_states.astype(np.int8)
    n_layers = max(features_by_layer.keys()) + 1

    # Iterate 13 networks × 2 polarities = up to 26 groups, in a canonical
    # order (NETWORK_ORDER × [+, -]). Groups are keyed as "{network}_pos" /
    # "{network}_neg" in the JSON output.
    for net_name in NETWORK_ORDER:
        for polarity in ("+", "-"):
            group_key = f"{net_name}_{'pos' if polarity == '+' else 'neg'}"
            state_ids = [int(s) for s in groups.get((net_name, polarity), [])]
            if len(state_ids) < D1NET_MIN_STATES:
                logger.info(
                    "  D1-net[%s]: only %d states - skipping",
                    group_key, len(state_ids),
                )
                results[group_key] = {
                    "network": net_name,
                    "polarity": polarity,
                    "n_states": len(state_ids),
                    "skipped": "too_few_states",
                }
                continue
            mask = np.array(
                [int(s) in set(state_ids) for s in all_states], dtype=bool,
            )
            n_trs_mask = int(mask.sum())
            if n_trs_mask < D1NET_MIN_TRS:
                logger.info(
                    "  D1-net[%s]: only %d TRs - skipping",
                    group_key, n_trs_mask,
                )
                results[group_key] = {
                    "network": net_name,
                    "polarity": polarity,
                    "n_states": len(state_ids),
                    "n_trs": n_trs_mask,
                    "skipped": "too_few_trs",
                }
                continue

            folds = build_loro_folds(run_boundaries, eligible_mask=mask)
            if len(folds) < 3:
                results[group_key] = {
                    "network": net_name,
                    "polarity": polarity,
                    "n_states": len(state_ids),
                    "n_trs": n_trs_mask,
                    "skipped": "too_few_folds",
                }
                continue

            # Deterministic, environment-independent seed derivation.
            # Python's built-in hash() is randomised per process unless
            # PYTHONHASHSEED is set; md5 gives us a stable integer keyed on
            # the group label so reruns reproduce byte-for-byte.
            group_digest = int(
                hashlib.md5(group_key.encode("utf-8")).hexdigest()[:8], 16,
            )
            net_seed = perm_seed_base + group_digest % 1_000_000

            # Eligible-subspace null (see 2026-04-10 null-leakage plan):
            # shifts within the per-network eligible TR subset so shifted
            # labels can only come from that network's state subset.
            null_y_masked = precompute_eligible_null_state_sequences(
                y_full, run_boundaries, mask, n_perm, net_seed,
            )

            per_layer: dict[int, dict] = {}
            layers_all = list(range(n_layers))
            chunk_size_net = max(n_jobs, 1)
            for chunk_start in range(0, len(layers_all), chunk_size_net):
                chunk = layers_all[chunk_start:chunk_start + chunk_size_net]
                chunk_results = Parallel(n_jobs=n_jobs)(
                    # Per-layer slice only (see D1_main note above) - avoids
                    # shipping the full all-layer dict to every worker.
                    delayed(_d1_one_layer)(
                        idx, {idx: features_by_layer.get(idx, {})},
                        run_ids, decoded_states, best_lag,
                        mask, y_full, null_y_masked, folds, group_key,
                    )
                    for idx in chunk
                )
                for layer_idx, result in chunk_results:
                    if result is None:
                        continue
                    per_layer[layer_idx] = {
                        "balanced_accuracy": result["balanced_accuracy"],
                        "chance_level": result["chance_level"],
                        "normalized_effect_size": result["normalized_effect_size"],
                        "p_perm": result["p_perm"],
                        "n_permutations": result["n_permutations"],
                    }
                logger.info(
                    "D1-net [%s]: chunk %d done (%d/%d layers)",
                    group_key, chunk_start // chunk_size_net + 1,
                    len(per_layer), n_layers,
                )

            if not per_layer:
                results[group_key] = {
                    "network": net_name,
                    "polarity": polarity,
                    "n_states": len(state_ids),
                    "n_trs": n_trs_mask,
                    "skipped": "no_valid_layers",
                }
                continue

            # Per-group BH-FDR across layers.
            layer_keys = sorted(per_layer.keys())
            raw_p = np.array([per_layer[k]["p_perm"] for k in layer_keys])
            p_fdr = benjamini_hochberg(raw_p)
            for k, f in zip(layer_keys, p_fdr):
                per_layer[k]["p_fdr"] = round(float(f), 4)

            results[group_key] = {
                "network": net_name,
                "polarity": polarity,
                "n_states": len(state_ids),
                "n_trs": n_trs_mask,
                "state_ids": state_ids,
                "per_layer": per_layer,
            }
            logger.info(
                "  D1-net[%s]: %d states, %d TRs, peak_acc=%.4f",
                group_key, len(state_ids), n_trs_mask,
                max(entry["balanced_accuracy"] for entry in per_layer.values()),
            )

    out_path = os.path.join(out_dir, "D1_net.json")
    with open(out_path, "w") as f:
        json.dump(
            {"best_lag": int(best_lag), "groups": results},
            f, indent=2,
        )
    _plot_d1net_heatmap(results, os.path.join(out_dir, "D1_net.png"))
    logger.info("Saved D1-net -> %s", out_path)
    return results


# ---------------------------------------------------------------------------
# D1 confound baseline
# ---------------------------------------------------------------------------


def _run_d1_confound_baseline(
    decoded_states, content_eligible, run_ids, run_boundaries, all_states,
    best_lag, n_perm, perm_seed_base, out_dir,
):
    logger.info("=" * 40)
    logger.info("Analysis D1 confound baseline (best lag = %d)", best_lag)
    logger.info("=" * 40)

    if not content_eligible:
        return {}

    mask = np.array(
        [int(s) in set(int(x) for x in content_eligible) for s in all_states],
        dtype=bool,
    )
    folds = build_loro_folds(run_boundaries, eligible_mask=mask)
    if len(folds) < 3:
        return {}

    X = _build_confound_design_matrix(run_ids, decoded_states)
    # Apply the same lag-shift convention as the transformer features.
    if best_lag > 0:
        X = np.vstack([np.zeros((best_lag, X.shape[1])), X[:-best_lag]])

    X_elig = X[mask]
    y_elig = all_states[mask].astype(np.int8)

    # Eligible-subspace null (see 2026-04-10 null-leakage plan).
    null_y_masked = precompute_eligible_null_state_sequences(
        all_states.astype(np.int8), run_boundaries, mask, n_perm,
        perm_seed_base + 9_000_000,
    )

    batch_result = batch_loro_ridge_classify(
        X_elig, y_elig, null_y_masked, folds,
    )
    if batch_result is None:
        return {}

    observed = batch_result["observed"]
    null_accs = batch_result["null_balanced_accuracies"]

    p = permutation_pvalue(
        observed["balanced_accuracy"], null_accs, alternative="greater",
    )
    eff = compute_effect_size(
        observed["balanced_accuracy"], len(np.unique(y_elig)),
    )
    result = {
        "best_lag": int(best_lag),
        "balanced_accuracy": round(observed["balanced_accuracy"], 4),
        "chance_level": round(eff["chance_level"], 4),
        "normalized_effect_size": round(eff["normalized_effect_size"], 4),
        "p_perm": round(float(p), 4),
        "null_mean": round(float(np.mean(null_accs)), 4) if null_accs else None,
        "n_permutations": len(null_accs),
        "n_features": int(X.shape[1]),
        "feature_names": list(CONFOUND_FEATURE_NAMES),
    }
    out_path = os.path.join(out_dir, "D1_confound_baseline.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved D1 confound baseline -> %s", out_path)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Within-stimulus transformer depth analyses (08d refactor).",
    )
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--stimulus", required=True,
                   choices=["friends", "movie10", "harrypotter",
                            "petitprince_fr", "petitprince_en"])
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--vt", default=None, help="VT suffix for 05e_a4 lookup")
    p.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS_DEFAULT)
    p.add_argument(
        "--analyses", nargs="+",
        default=["D1", "D1net", "D1confound", "D2"],
        choices=["D1", "D1net", "D1confound", "D2", "D1merge"],
    )
    p.add_argument(
        "--lags", nargs="+", type=int, default=None,
        help="Subset of lags to process for D1 (default: all from LAGS_TO_TEST). "
             "Used for per-lag SLURM splitting. Incompatible with D1net/D1confound/D2.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run analyses even if checkpoint outputs already exist.",
    )
    p.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel workers for layer (D1) and state (D2) loops. "
             "Default 1 (serial). Set OPENBLAS_NUM_THREADS=1 in the environment "
             "when using n_jobs > 1 to avoid BLAS thread contention.",
    )
    p.add_argument(
        "--features_subdir_suffix", type=str, default="",
        help="Suffix for the 08c features directory "
             "(e.g. '_sweep_w3' reads from 08c_transformer_features_sweep_w3/). "
             "Default empty = production path. Used by the LLaMA W-sweep "
             "(2026-05-01_08c_llama_local_window_design.md §11.6.1).",
    )
    p.add_argument(
        "--output_subdir_suffix", type=str, default="",
        help="Suffix for the 08d output directory "
             "(e.g. '_sweep_w3' writes to 08d_transformer_depth_sweep_w3/). "
             "Default empty = production path. Pair with "
             "--features_subdir_suffix when running W-sweep variants so "
             "results don't clobber each other.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    stimulus = args.stimulus
    model_key = args.model

    validate_stimulus_model(stimulus, model_key)

    force = args.force

    # -- Checkpoint file definitions per analysis --
    CHECKPOINT_FILES = {
        "D1": ["D1_depth_profile.json", "D1_neg_control_run_onset_anchored.json",
               "D1_neg_control_gate.json"],
        "D1merge": ["D1_depth_profile.json", "D1_neg_control_run_onset_anchored.json",
                    "D1_neg_control_gate.json"],
        "D1net": ["D1_net.json"],
        "D1confound": ["D1_confound_baseline.json"],
        "D2": ["D2_state_layer_auc.json"],
    }

    # Warn if parallel mode requested without BLAS thread control.
    if args.n_jobs > 1:
        for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            if os.environ.get(var) not in (None, "1"):
                break
        else:
            # None of the threading env vars are set to 1 - set them.
            for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            logger.info(
                "n_jobs=%d: set OPENBLAS/MKL/OMP_NUM_THREADS=1 to avoid "
                "BLAS thread contention with joblib workers",
                args.n_jobs,
            )

    logger.info("=" * 60)
    logger.info("08d - Transformer Depth (refactored)")
    logger.info("Sub=%s Stim=%s Model=%s n_perm=%d n_jobs=%d",
                sub_id, stimulus, model_key, args.n_permutations, args.n_jobs)
    logger.info("=" * 60)

    out_dir = os.path.join(
        SCRATCH_DIR, "output",
        f"08d_transformer_depth{args.output_subdir_suffix}",
        parc, sub_id, f"{stimulus}_{model_key}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Checkpoint filtering ──────────────────────────────────────────
    to_run = []
    for a in args.analyses:
        if check_checkpoint(out_dir, CHECKPOINT_FILES[a], a, force=force):
            continue
        to_run.append(a)

    if not to_run:
        logger.info("All requested analyses already checkpointed. Nothing to do.")
        return

    logger.info("Will run: %s (force=%s)", " ".join(to_run), force)

    # Per-lag mode is incompatible with downstream analyses (they need the
    # full D1 grid to determine best_lag).
    if args.lags is not None:
        bad = [a for a in to_run if a in ("D1net", "D1confound", "D2")]
        if bad:
            raise SystemExit(
                f"--lags is incompatible with downstream analyses "
                f"{bad}. Run D1merge first, then D1net/D1confound/D2 "
                f"in a separate invocation."
            )

    # D1 is a prerequisite for D1net / D1confound / D2 because the peak
    # (lag, layer) from D1 main is what those analyses condition on.
    downstream_requested = any(a in to_run for a in ("D1net", "D1confound", "D2"))
    d1_checkpoint_path = os.path.join(out_dir, "D1_depth_profile.json")
    if downstream_requested and "D1" not in to_run:
        if not os.path.exists(d1_checkpoint_path):
            raise SystemExit(
                "08d: D1net/D1confound/D2 require D1 output, but "
                "D1_depth_profile.json is missing and D1 is not in "
                "analyses to run. Either add 'D1' to --analyses or "
                "ensure D1 has been run previously."
            )
        logger.info(
            "D1 checkpointed - will load best (lag, layer) from %s",
            d1_checkpoint_path,
        )

    # ── States ────────────────────────────────────────────────────────
    if stimulus == "friends":
        decoded_states, final_dir = _load_decoded_states_friends(sub_id, parc)
    else:
        decoded_states = _load_decoded_states_cross_stim(sub_id, parc, stimulus)
        # For D1-net we still need Friends state_means_parcel (shared state space).
        _, final_dir = _load_decoded_states_friends(sub_id, parc)

    # Initial run inventory (pre-alignment).
    candidate_run_ids = sorted(decoded_states.keys())
    initial_n_trs_per_run = {r: len(decoded_states[r]) for r in candidate_run_ids}

    # ── Eligibility ───────────────────────────────────────────────────
    eligibility = load_content_eligibility(sub_id, parc, SCRATCH_DIR, vt=args.vt)
    logger.info(
        "Eligibility source: %s | content_eligible=%d run_onset_anchored=%d",
        eligibility["eligibility_source"],
        len(eligibility["content_eligible"]),
        len(eligibility["run_onset_anchored"]),
    )
    if not eligibility["content_eligible"]:
        logger.error("No content_eligible states - exiting")
        sys.exit(1)

    # D1merge only needs eligibility metadata - skip heavy feature loading.
    d1merge_only = to_run == ["D1merge"]
    if d1merge_only:
        # Jump directly to the D1merge block (and downstream best_lag).
        # Declare variables that the D1merge block and downstream code
        # expect to exist, even though they won't be used in merge-only.
        d1_main = None
        per_lag_mode = False
        partials_dir = None

        # ── D1merge fast path (no feature loading) ────────────────────
        merge_partials_dir = os.path.join(out_dir, "partials")
        all_lags_main: dict[str, dict[int, dict]] = {}
        all_lags_neg: dict[str, dict[int, dict]] = {}
        missing_main = []

        # Honor --lags subset (single-lag W-sweep mode). When --lags is
        # unset, merge over the canonical 9-lag grid (LAGS_TO_TEST).
        merge_lags = args.lags if args.lags is not None else LAGS_TO_TEST
        for lag in merge_lags:
            main_path = os.path.join(
                merge_partials_dir, f"D1_D1_main_lag{lag}.json",
            )
            if not os.path.exists(main_path):
                missing_main.append(lag)
                continue
            with open(main_path) as f:
                pdata = json.load(f)
            n_layers_total = pdata.get("n_layers_total", 0)
            results = pdata.get("results", {})
            if len(results) < n_layers_total:
                missing_main.append(lag)
                logger.warning(
                    "D1merge: lag=%d partial incomplete (%d/%d layers)",
                    lag, len(results), n_layers_total,
                )
                continue
            all_lags_main[f"lag_{lag}"] = {int(k): v for k, v in results.items()}

            # Neg-control partial (exploratory only - the run-onset negative
            # control was dropped from the manuscript 2026-05-31). It does NOT
            # gate the merge: include only COMPLETE neg lags so the neg BH-FDR
            # pool (m) is consistent; skip incomplete/missing neg with a
            # warning. Never abort on neg.
            neg_path = os.path.join(
                merge_partials_dir, f"D1_D1_neg_control_lag{lag}.json",
            )
            if os.path.exists(neg_path):
                with open(neg_path) as f:
                    ndata = json.load(f)
                neg_results = ndata.get("results", {})
                neg_total = ndata.get("n_layers_total", 0)
                if (not neg_total) or len(neg_results) < neg_total:
                    logger.warning(
                        "D1merge: neg lag=%d incomplete or missing layer count "
                        "(%d/%s layers) - excluding from exploratory neg output "
                        "(non-blocking)",
                        lag, len(neg_results), neg_total or "?",
                    )
                else:
                    all_lags_neg[f"lag_{lag}"] = {
                        int(k): v for k, v in neg_results.items()
                    }

        if missing_main:
            raise SystemExit(
                f"D1merge: missing or incomplete main partials for lags "
                f"{missing_main}. Re-run those lags first."
            )

        logger.info(
            "D1merge: assembled %d lags × %d layers (main) + %d complete lags (neg)",
            len(all_lags_main),
            max(len(v) for v in all_lags_main.values()) if all_lags_main else 0,
            len(all_lags_neg),
        )

        _apply_global_fdr(all_lags_main)
        if all_lags_neg:
            _apply_global_fdr(all_lags_neg)

        with open(os.path.join(out_dir, "D1_depth_profile.json"), "w") as f:
            json.dump({
                "eligibility_source": eligibility["eligibility_source"],
                "n_eligible_states": len(eligibility["content_eligible"]),
                "lags": list(merge_lags),
                "results": all_lags_main,
            }, f, indent=2)

        with open(os.path.join(out_dir, "D1_neg_control_run_onset_anchored.json"), "w") as f:
            json.dump({
                "n_run_onset_anchored_states": len(eligibility["run_onset_anchored"]),
                "results": all_lags_neg,
            }, f, indent=2)

        _plot_depth_profile(
            {"content_eligible": all_lags_main, "run_onset_anchored": all_lags_neg},
            os.path.join(out_dir, "D1_depth_profile.png"),
            f"D1 - {stimulus}/{model_key}",
        )

        # Neg-control gate dropped 2026-05-31 (see _GATE_DEPRECATED).
        _write_deprecated_gate(out_dir)

        logger.info("=" * 60)
        logger.info("08d_transformer_depth complete (D1merge only)")
        logger.info("Output: %s", out_dir)
        logger.info("=" * 60)
        return

    # ── Features + PCA (streamed per layer, drift-aware) ──────────────
    # `stream_pca_features` loads one layer at a time, fits PCA on the
    # training split, projects all runs, and discards raw layer data
    # before moving to the next layer - peak memory ≈ one raw layer +
    # accumulated PCA'd output (~2.4 GB for Friends × llama vs. ~46 GB
    # for the old load-everything approach).
    #
    # For cross-stimulus runs (movie10 / harrypotter / petitprince_*) we
    # MUST reuse the Friends-fit PCA instead of fitting on the test
    # stimulus, otherwise the LORO decoder sees a feature basis that was
    # shaped by its own test-fold data. We do this by running
    # stream_pca_features once on Friends (fit) and once on the test
    # stimulus (project-only), mirroring the pattern in 08e.
    splits = load_training_split(sub_id, parc, SCRATCH_DIR)
    train_run_ids = splits["train"]

    if stimulus == "friends":
        logger.info(
            "Loading 08c features and fitting per-layer PCA "
            "(variance threshold = %.0f%%)",
            PCA_VARIANCE_THRESHOLD * 100,
        )
        cache_dir = _pca_cache_dir(
            SCRATCH_DIR, parc, sub_id, model_key, stimulus,
            output_subdir_suffix=args.output_subdir_suffix,
        )
        split_hash, split_path_rel = _pca_split_hash_and_path(
            sub_id, parc, SCRATCH_DIR,
        )
        (
            features_by_layer,
            pca_info,
            _pca_models,
            effective_n_trs,
            dropped_runs,
        ) = load_or_fit_pca_cache(
            stimulus, model_key, candidate_run_ids, initial_n_trs_per_run,
            SCRATCH_DIR,
            cache_dir=cache_dir,
            split_hash=split_hash,
            sub_id=sub_id,
            parcellation=parc,
            split_path_rel=split_path_rel,
            train_run_ids=train_run_ids,
            variance_threshold=PCA_VARIANCE_THRESHOLD,
            extraction_subdir_suffix=args.features_subdir_suffix,
        )
    else:
        # Fit on Friends first (discard the projected Friends features -
        # we only need the PCA basis), then project the test stimulus
        # through the Friends-fit PCA. Uses the shared PCA cache so
        # cross-stim runs skip the ~90 min LLaMA refit entirely.
        friends_decoded, _ = _load_decoded_states_friends(sub_id, parc)
        friends_run_ids = sorted(friends_decoded.keys())
        friends_n_trs_init = {
            r: len(friends_decoded[r]) for r in friends_run_ids
        }
        logger.info(
            "Cross-stim: fitting PCA on Friends features (variance "
            "threshold = %.0f%%) to keep a shared basis with in-stim "
            "results and avoid LORO data leakage.",
            PCA_VARIANCE_THRESHOLD * 100,
        )
        cache_dir = _pca_cache_dir(
            SCRATCH_DIR, parc, sub_id, model_key, "friends",
            output_subdir_suffix=args.output_subdir_suffix,
        )
        split_hash, split_path_rel = _pca_split_hash_and_path(
            sub_id, parc, SCRATCH_DIR,
        )
        (
            _friends_features,
            pca_info,
            pca_models,
            _friends_eff,
            _friends_dropped,
        ) = load_or_fit_pca_cache(
            "friends", model_key, friends_run_ids, friends_n_trs_init,
            SCRATCH_DIR,
            cache_dir=cache_dir,
            split_hash=split_hash,
            sub_id=sub_id,
            parcellation=parc,
            split_path_rel=split_path_rel,
            train_run_ids=train_run_ids,
            variance_threshold=PCA_VARIANCE_THRESHOLD,
            extraction_subdir_suffix=args.features_subdir_suffix,
        )
        # Drop the Friends projections immediately - 08d cross-stim only
        # decodes on the test stimulus, so carrying them would waste RAM.
        del _friends_features, friends_decoded

        logger.info(
            "Cross-stim: projecting %s features through the Friends PCA",
            stimulus,
        )
        # movie10 has two viewings of figures/life per subject that share
        # one 08c feature file; strip the _run-N suffix for feature lookup.
        cross_stim_feature_key_fn = (
            (lambda r: feature_key_for_cross_stim_run_id(r, stimulus))
            if stimulus == "movie10" else None
        )
        (
            features_by_layer,
            _,
            _,
            effective_n_trs,
            dropped_runs,
        ) = stream_pca_features(
            stimulus, model_key, candidate_run_ids, initial_n_trs_per_run,
            SCRATCH_DIR,
            pca_models=pca_models, pca_info=pca_info,
            feature_key_fn=cross_stim_feature_key_fn,
            extraction_subdir_suffix=args.features_subdir_suffix,
        )

    if not effective_n_trs:
        logger.error("No features loaded - exiting")
        sys.exit(1)

    # Truncate decoded_states to the effective per-run TR count, then
    # rebuild run_boundaries / all_states from the aligned state dict.
    for rid in list(decoded_states.keys()):
        if rid not in effective_n_trs:
            decoded_states.pop(rid)
            continue
        eff = effective_n_trs[rid]
        if len(decoded_states[rid]) != eff:
            decoded_states[rid] = np.asarray(decoded_states[rid])[:eff]

    run_ids = sorted(decoded_states.keys())
    run_boundaries = build_run_boundaries(run_ids, decoded_states)
    all_states = np.concatenate([decoded_states[r] for r in run_ids])
    n_trs_per_run = {r: len(decoded_states[r]) for r in run_ids}
    logger.info(
        "Post-alignment: %d runs, %d total TRs (%d runs dropped)",
        len(run_ids), len(all_states), len(dropped_runs),
    )

    with open(os.path.join(out_dir, "pca_info.json"), "w") as f:
        json.dump({str(k): v for k, v in pca_info.items()}, f, indent=2)

    # ── Run analyses ──────────────────────────────────────────────────
    d1_main = None
    per_lag_mode = args.lags is not None and "D1" in to_run
    partials_dir = os.path.join(out_dir, "partials") if per_lag_mode else None

    if "D1" in to_run:
        d1_main = _run_d1_decoder_set(
            label="D1_main",
            decoded_states=decoded_states,
            features_by_layer=features_by_layer,
            state_subset=eligibility["content_eligible"],
            run_ids=run_ids,
            run_boundaries=run_boundaries,
            all_states=all_states,
            n_perm=args.n_permutations,
            perm_seed_base=10_000,
            lags_subset=args.lags,
            partials_dir=partials_dir,
            n_jobs=args.n_jobs,
        )

        # Negative control: design-driven
        d1_neg = {}
        if eligibility["run_onset_anchored"]:
            d1_neg = _run_d1_decoder_set(
                label="D1_neg_control",
                decoded_states=decoded_states,
                features_by_layer=features_by_layer,
                state_subset=eligibility["run_onset_anchored"],
                run_ids=run_ids,
                run_boundaries=run_boundaries,
                all_states=all_states,
                n_perm=args.n_permutations,
                perm_seed_base=20_000,
                lags_subset=args.lags,
                partials_dir=partials_dir,
                n_jobs=args.n_jobs,
            )
        else:
            logger.warning("No run_onset_anchored states available - skipping D1 neg control")

        if per_lag_mode:
            # Per-lag mode: partials saved by _run_d1_decoder_set.
            # Canonical files, plot, and gate are deferred to D1merge.
            logger.info(
                "D1 per-lag partials saved to %s - run D1merge after "
                "all lags complete to produce canonical outputs.",
                partials_dir,
            )
        else:
            # Monolithic mode: write canonical files immediately.
            # Honor --lags subset; default to canonical 9-lag grid.
            monolithic_lags = (
                args.lags if args.lags is not None else LAGS_TO_TEST
            )
            with open(os.path.join(out_dir, "D1_depth_profile.json"), "w") as f:
                json.dump({
                    "eligibility_source": eligibility["eligibility_source"],
                    "n_eligible_states": len(eligibility["content_eligible"]),
                    "lags": list(monolithic_lags),
                    "results": d1_main,
                }, f, indent=2)

            with open(os.path.join(out_dir, "D1_neg_control_run_onset_anchored.json"), "w") as f:
                json.dump({
                    "n_run_onset_anchored_states": len(eligibility["run_onset_anchored"]),
                    "results": d1_neg,
                }, f, indent=2)

            _plot_depth_profile(
                {"content_eligible": d1_main, "run_onset_anchored": d1_neg},
                os.path.join(out_dir, "D1_depth_profile.png"),
                f"D1 - {stimulus}/{model_key}",
            )

            # Neg-control gate dropped 2026-05-31 (see _GATE_DEPRECATED).
            _write_deprecated_gate(out_dir)

    # ── D1merge: assemble per-lag partials → canonical outputs ───────
    if "D1merge" in to_run:
        merge_partials_dir = os.path.join(out_dir, "partials")
        all_lags_main: dict[str, dict[int, dict]] = {}
        all_lags_neg: dict[str, dict[int, dict]] = {}
        missing_main = []

        # Honor --lags subset (single-lag W-sweep mode); see fast-path
        # comment above for rationale.
        merge_lags = args.lags if args.lags is not None else LAGS_TO_TEST
        for lag in merge_lags:
            main_path = os.path.join(
                merge_partials_dir, f"D1_D1_main_lag{lag}.json",
            )
            if not os.path.exists(main_path):
                missing_main.append(lag)
                continue
            with open(main_path) as f:
                pdata = json.load(f)
            n_layers_total = pdata.get("n_layers_total", 0)
            results = pdata.get("results", {})
            if len(results) < n_layers_total:
                missing_main.append(lag)
                logger.warning(
                    "D1merge: lag=%d partial incomplete (%d/%d layers)",
                    lag, len(results), n_layers_total,
                )
                continue
            all_lags_main[f"lag_{lag}"] = {int(k): v for k, v in results.items()}

            # Neg-control partial (exploratory only; dropped from the
            # manuscript 2026-05-31). Non-blocking: include only COMPLETE neg
            # lags so the neg BH-FDR pool is consistent; skip incomplete/missing
            # with a warning. Never abort on neg.
            neg_path = os.path.join(
                merge_partials_dir, f"D1_D1_neg_control_lag{lag}.json",
            )
            if os.path.exists(neg_path):
                with open(neg_path) as f:
                    ndata = json.load(f)
                neg_results = ndata.get("results", {})
                neg_total = ndata.get("n_layers_total", 0)
                if (not neg_total) or len(neg_results) < neg_total:
                    logger.warning(
                        "D1merge: neg lag=%d incomplete or missing layer count "
                        "(%d/%s layers) - excluding from exploratory neg output "
                        "(non-blocking)",
                        lag, len(neg_results), neg_total or "?",
                    )
                else:
                    all_lags_neg[f"lag_{lag}"] = {
                        int(k): v for k, v in neg_results.items()
                    }

        if missing_main:
            raise SystemExit(
                f"D1merge: missing or incomplete main partials for lags "
                f"{missing_main}. Re-run those lags first."
            )

        logger.info(
            "D1merge: assembled %d lags × %d layers (main) + %d complete lags (neg)",
            len(all_lags_main),
            max(len(v) for v in all_lags_main.values()) if all_lags_main else 0,
            len(all_lags_neg),
        )

        # Apply global BH-FDR across the merged grid (subset-aware when
        # --lags is set; single-lag FDR pool is n_layers, multi-lag is
        # n_lags × n_layers).
        _apply_global_fdr(all_lags_main)
        if all_lags_neg:
            _apply_global_fdr(all_lags_neg)

        # Write canonical checkpoint files.
        with open(os.path.join(out_dir, "D1_depth_profile.json"), "w") as f:
            json.dump({
                "eligibility_source": eligibility["eligibility_source"],
                "n_eligible_states": len(eligibility["content_eligible"]),
                "lags": list(merge_lags),
                "results": all_lags_main,
            }, f, indent=2)

        with open(os.path.join(out_dir, "D1_neg_control_run_onset_anchored.json"), "w") as f:
            json.dump({
                "n_run_onset_anchored_states": len(eligibility["run_onset_anchored"]),
                "results": all_lags_neg,
            }, f, indent=2)

        _plot_depth_profile(
            {"content_eligible": all_lags_main, "run_onset_anchored": all_lags_neg},
            os.path.join(out_dir, "D1_depth_profile.png"),
            f"D1 - {stimulus}/{model_key}",
        )

        # Neg-control gate dropped 2026-05-31 (see _GATE_DEPRECATED).
        _write_deprecated_gate(out_dir)

        # Make merged results available for downstream best_lag extraction.
        d1_main = all_lags_main

    # Determine best (lag, layer) from D1 main for D1-net / D2 / D1-confound.
    # We exclude lag=0 from the peak search because at TR=1.49s the HRF
    # has not risen yet; lag=0 is kept in the main D1 grid as a
    # synchrony / autocorrelation diagnostic only.
    best_lag_val = 0
    if d1_main:
        # D1 was just computed in this invocation.
        best_lag_key, best_layer_val, best_acc = _find_best_lag_layer(
            d1_main, exclude_lags=PEAK_LAG_EXCLUDE,
        )
        if best_lag_key is not None:
            best_lag_val = int(best_lag_key.split("_")[1])
            logger.info(
                "Best D1 main (lag=0 excluded): %s layer=%d bal_acc=%.4f",
                best_lag_key, best_layer_val, best_acc,
            )
    elif os.path.exists(d1_checkpoint_path):
        # D1 was checkpointed - load best (lag, layer) from saved JSON.
        with open(d1_checkpoint_path) as f:
            d1_saved = json.load(f)
        d1_main_saved = d1_saved.get("results", {})
        best_lag_key, best_layer_val, best_acc = _find_best_lag_layer(
            d1_main_saved, exclude_lags=PEAK_LAG_EXCLUDE,
        )
        if best_lag_key is not None:
            best_lag_val = int(best_lag_key.split("_")[1])
            logger.info(
                "Best D1 main (from checkpoint, lag=0 excluded): "
                "%s layer=%d bal_acc=%.4f",
                best_lag_key, best_layer_val, best_acc,
            )

    if "D1net" in to_run:
        _run_d1_net(
            decoded_states=decoded_states,
            features_by_layer=features_by_layer,
            eligibility=eligibility,
            run_ids=run_ids,
            run_boundaries=run_boundaries,
            all_states=all_states,
            best_lag=best_lag_val,
            n_perm=args.n_permutations,
            perm_seed_base=30_000,
            parcellation=parc,
            final_dir=final_dir,
            out_dir=out_dir,
            n_jobs=args.n_jobs,
        )

    if "D1confound" in to_run:
        _run_d1_confound_baseline(
            decoded_states=decoded_states,
            content_eligible=eligibility["content_eligible"],
            run_ids=run_ids,
            run_boundaries=run_boundaries,
            all_states=all_states,
            best_lag=best_lag_val,
            n_perm=args.n_permutations,
            perm_seed_base=40_000,
            out_dir=out_dir,
        )

    if "D2" in to_run:
        # D2 redesign 2026-04-28: default n_perm=500 (Stats C1 of D2 review)
        # to keep BH-FDR rank-1 reachable across L≈28 layers per state.
        # Override if user explicitly passed a different --n_permutations.
        d2_n_perm = args.n_permutations if args.n_permutations != 1000 else 500
        if d2_n_perm != args.n_permutations:
            logger.info(
                "D2: applying redesign default n_perm=500 (was %d)",
                args.n_permutations,
            )
        _run_d2(
            decoded_states=decoded_states,
            features_by_layer=features_by_layer,
            content_eligible=eligibility["content_eligible"],
            run_ids=run_ids,
            run_boundaries=run_boundaries,
            all_states=all_states,
            best_lag=best_lag_val,
            n_perm=d2_n_perm,
            perm_seed_base=50_000,
            out_dir=out_dir,
            n_jobs=args.n_jobs,
            force=force,
        )

    logger.info("=" * 60)
    logger.info("08d_transformer_depth complete")
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
