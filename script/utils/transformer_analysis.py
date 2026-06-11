#!/usr/bin/env python
"""
transformer_analysis.py — shared analysis helpers for the 08d/08e/08f/08g
transformer-state correspondence scripts and the refactored 08b.

This module consolidates logic that used to live duplicated in 08b and the old
monolithic 08d:

* State-category loading (content-eligibility standard, §6 of the 2026-04-09
  design doc)
* Run-structure helpers for leave-one-run-out cross-validation and
  within-run circular-shift null models
* PCA (fit on Friends training split, project on arbitrary stimuli)
* LORO decoders: multi-class RidgeClassifier for D1 / D3a, per-state binary
  LogisticRegression+AUC for D2
* Chance-level and effect-size helpers, per-state layer-selectivity metrics,
  and state stratification by dominant network for D1-net

Nothing in this module performs I/O beyond what the helper signatures advertise
(reads state_flags.csv via ``state_flags_io``; reads parcel network mapping
via ``plot_style``). All heavier data loading stays in the calling scripts.
"""

from __future__ import annotations

import logging
import json
import os
from typing import Callable, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared thresholds for the 08-series transformer correspondence pipeline
# ---------------------------------------------------------------------------
#
# These constants were previously defined in 08d / 08e module scope and are
# now hoisted here so 08d, 08e, 08f, and 08g all import the same source of
# truth. Don't duplicate the numeric values in any downstream script — import
# from here instead.

#: Minimum per-stimulus fractional occupancy (fraction of TRs) required for
#: a state to enter a cross-stimulus intersection-based analysis (08e D3a,
#: 08f D3c, 08f D4-lang). States below this threshold in *either* stimulus
#: are dropped to avoid unstable AUC / rank estimates on sparse occupancy.
INTERSECTION_MIN_FO = 0.01

#: Primary selectivity threshold for D2 per-state layer AUC profiles: a
#: state is considered layer-*selective* if ``max(layer_auc) -
#: median(layer_auc) >= D2_SELECTIVITY_THRESHOLD``. 08d uses this to set
#: the ``non_selective`` flag in ``D2_state_layer_auc.json``; 08f re-checks
#: the same threshold when deciding which states feed D3c / D4-lang.
D2_SELECTIVITY_THRESHOLD = 0.05

#: Minimum number of states required for an aggregate convergence statistic
#: to be reported (08f D3c, 08g D5 / cross-modality / recurrence × depth).
#: Aggregates with fewer than this many states return ``insufficient_states``
#: and are still written to disk for downstream auditing.
MIN_CONVERGENCE_STATES = 5

# ---------------------------------------------------------------------------
# 08-series RNG seed slot reservation
# ---------------------------------------------------------------------------
#
# Each 08-series analysis owns a disjoint integer block so that null /
# bootstrap sequences cannot accidentally collide when multiple analyses run
# in the same process. Each analysis is free to use sub-offsets ``+1, +2, …``
# from its base seed for distinct sub-statistics. The full slot table is
# documented in ``project_08c_design.md``.
#
#   D1            10_000   (08d main; non-base sub-offsets land in the 10k range)
#   D1-neg        20_000   (08d design-driven negative control)
#   D1-net        30_000   (08d network-stratified, base + md5(group_key))
#   D1-confound   40_000   (08d confound-baseline)
#   D2            50_000   (08d per-state layer AUC)
#   D3a           60_000   (08e cross-stimulus aggregate)
#   D3c           70_000   (08f cross-stim per-state, +1..+4 sub-offsets)
#   D5            80_000   (08g convergence with 08b)
#   X-modality    90_000   (08g cross-modality dissociation)
#   Rec×depth    100_000   (08g recurrence × depth interaction)
BOOTSTRAP_SEED_D5 = 80_000
BOOTSTRAP_SEED_CROSS_MODALITY = 90_000
BOOTSTRAP_SEED_RECURRENCE_DEPTH = 100_000


# ---------------------------------------------------------------------------
# State eligibility (project-wide content-eligibility convention)
# ---------------------------------------------------------------------------


def load_content_eligibility(sub_id, parcellation, scratch_dir, vt=None):
    """Return per-category state ID lists from 05e_a4 ``state_flags.csv``.

    This is the **authoritative** content-eligibility loader for the 08 series
    and for 08b. It wraps :func:`utils.state_flags_io.load_state_flags` and
    falls back to the 05a sub-HRF filter with a prominent warning if
    ``state_flags.csv`` is missing (e.g. when 05e_a4 has not been run yet).

    Parameters
    ----------
    sub_id : str
        Subject ID (``'sub-01'``).
    parcellation : str
        Parcellation directory name (e.g. ``'atlas-4S156Parcels'``). Should
        already be normalized via
        :func:`utils.common.normalize_parcellation_name`.
    scratch_dir : str
        Scratch directory root (``os.getenv('SCRATCH_DIR')``).
    vt : str, optional
        VT subdirectory suffix (e.g. ``'0.95'``). Passed through to
        ``load_state_flags``.

    Returns
    -------
    dict
        Keys:

        - ``content_eligible``: list[int] — states with
          ``summary_category == 'eligible_for_content_analysis'``
        - ``run_onset_anchored``: list[int] — states in ``run_onset_anchored`` (used as
          a negative control in D1)
        - ``season_temporal``: list[int] — ``season_temporal`` states
          (excluded from content analyses; never used as a control — they are
          confounded with episode-order drift)
        - ``basic_sub_hrf``: list[int] — the looser sub-HRF filter from 05a;
          populated only when 05e_a4's CSV is missing and used as a fallback
          for ``content_eligible``. Consumers should check
          ``eligibility_source`` to tell whether the main set was derived from
          05e_a4 or this fallback.
        - ``eligibility_source``: str — either ``'05e_a4'`` or
          ``'sub_hrf_fallback'``.
    """
    from utils.state_flags_io import load_state_flags

    state_flags_df = load_state_flags(
        sub_id=sub_id,
        parcellation=parcellation,
        scratch_dir=scratch_dir,
        vt=vt,
    )

    if state_flags_df is not None and "summary_category" in state_flags_df.columns:
        def _ids_with(category):
            return sorted(
                int(s) for s in
                state_flags_df.loc[
                    state_flags_df["summary_category"] == category, "state"
                ].tolist()
            )

        result = {
            "content_eligible": _ids_with("eligible_for_content_analysis"),
            "run_onset_anchored": _ids_with("run_onset_anchored"),
            "season_temporal": _ids_with("season_temporal"),
            "basic_sub_hrf": [],  # not used when 05e_a4 is available
            "eligibility_source": "05e_a4",
        }
        logger.info(
            "load_content_eligibility[%s/%s%s]: %d content_eligible, "
            "%d run_onset_anchored, %d season_temporal (source=05e_a4)",
            sub_id, parcellation,
            f"/vt{vt}" if vt is not None else "",
            len(result["content_eligible"]),
            len(result["run_onset_anchored"]),
            len(result["season_temporal"]),
        )
        return result

    # ── Fallback: 05a sub-HRF filter ────────────────────────────────────
    # Logged at ERROR (not WARNING) so the fallback is unmissable in
    # SLURM logs — content-eligibility claims downstream of this fallback
    # are unreliable and the user must rerun 05e_a4 before interpretation.
    # The output JSON still records ``eligibility_source = sub_hrf_fallback``
    # for downstream auditing, but the log line is the primary signal.
    logger.error(
        "state_flags.csv missing for %s / %s%s — falling back to 05a "
        "sub-HRF filter. Content-eligibility claims from downstream analyses "
        "will be UNRELIABLE: rerun 05e_a4 before interpreting results.",
        sub_id, parcellation, f"/vt{vt}" if vt is not None else "",
    )

    import json
    import os
    from utils.common import resolve_stage_file
    from utils.state_blocks import load_eligible_states

    rec_base = os.path.join(
        scratch_dir, "output", "05a_recurrence_analysis", parcellation, sub_id,
    )
    rec_path = resolve_stage_file(
        rec_base, "recurrence_summary.json", "recurrence summary",
    )
    try:
        eligible_ids, _, _ = load_eligible_states(os.path.dirname(rec_path))
        fallback = sorted(int(s) for s in eligible_ids)
    except FileNotFoundError:
        with open(rec_path) as f:
            rec_summary = json.load(f)
        recurrence_scores = np.asarray(rec_summary["recurrence_scores"])
        fallback = [int(s) for s in np.where(recurrence_scores > 0)[0]]

    return {
        "content_eligible": fallback,
        "run_onset_anchored": [],
        "season_temporal": [],
        "basic_sub_hrf": fallback,
        "eligibility_source": "sub_hrf_fallback",
    }



def load_recurrence_scores(sub_id, parcellation, scratch_dir, vt=None):
    """Load per-state recurrence scores from 05a ``recurrence_summary.json``.

    Hoisted from 08f and 08g (which previously each defined a private
    ``_load_recurrence``) so the two scripts share one source of truth.
    Returns a 1-D ``np.ndarray`` of shape ``(n_states,)``.

    Parameters
    ----------
    sub_id : str
        Subject ID (``'sub-01'``).
    parcellation : str
        Parcellation directory name (e.g. ``'atlas-4S156Parcels'``). Should
        already be normalized via :func:`utils.common.normalize_parcellation_name`.
    scratch_dir : str
        Path to ``$SCRATCH_DIR`` (the caller's ``os.getenv('SCRATCH_DIR')``).
    vt : str | None
        Optional vt subdirectory tag (e.g. ``'0.95'``). Currently unused for
        recurrence (05a outputs do not branch by vt) but kept for symmetry
        with :func:`load_content_eligibility`.
    """
    # Local imports keep the helper free of cross-module top-level coupling
    # — `utils.common.resolve_stage_file` is the same lookup used by 08f
    # and matches the legacy ``_load_recurrence`` behaviour.
    import json
    import os

    from utils.common import resolve_stage_file

    rec_base = os.path.join(
        scratch_dir, "output", "05a_recurrence_analysis", parcellation, sub_id,
    )
    rec_path = resolve_stage_file(
        rec_base, "recurrence_summary.json", "recurrence summary",
    )
    with open(rec_path) as f:
        return np.asarray(json.load(f)["recurrence_scores"])


# ---------------------------------------------------------------------------
# Run-structure helpers
# ---------------------------------------------------------------------------


def build_run_boundaries(run_ids, decoded_states):
    """Compute contiguous ``(start, end)`` index ranges for each run.

    Parameters
    ----------
    run_ids : sequence of str
        Run IDs in the order they will be concatenated.
    decoded_states : dict
        ``{run_id: np.ndarray}`` of per-run state sequences. Only the length
        of each entry is used.

    Returns
    -------
    list of (int, int)
        One ``(start, end)`` pair per run. ``end`` is exclusive, matching
        NumPy slice conventions.
    """
    boundaries = []
    offset = 0
    for run_id in run_ids:
        n = len(decoded_states[run_id])
        boundaries.append((offset, offset + n))
        offset += n
    return boundaries


def circular_shift_states_by_run(states, run_boundaries, seed, min_shift=1):
    """Circularly shift state labels within each run with a deterministic seed.

    This is the canonical implementation; previously duplicated in 08b and the
    old 08d. Used as the null model for every LORO permutation test in the 08
    series.

    Parameters
    ----------
    states : array-like
        Concatenated state sequence (length = sum of per-run lengths).
    run_boundaries : sequence of (int, int)
        Output of :func:`build_run_boundaries`.
    seed : int
        Random seed for ``numpy.random.default_rng``.
    min_shift : int, default 1
        Minimum shift in TRs. Runs with length ``<= min_shift`` are left
        unshifted.

    Returns
    -------
    np.ndarray
        Shifted state sequence with the same shape as ``states``.
    """
    rng = np.random.default_rng(seed)
    shifted = np.asarray(states).copy()
    for start, end in run_boundaries:
        n = end - start
        if n > min_shift:
            shift = int(rng.integers(min_shift, n))
            shifted[start:end] = np.roll(shifted[start:end], shift)
    return shifted


def precompute_null_state_sequences(
    states, run_boundaries, n_perm, rng_seed,
):
    """Build an ``(n_perm, n_trs)`` matrix of circularly shifted sequences.

    Hoisting this computation outside the per-layer loop is one of the
    efficiency fixes relative to the old 08d: permuted state sequences are
    identical across transformer layers for a given ``(seed, lag)`` pair, so
    recomputing them per layer is pure waste.

    Parameters
    ----------
    states : array-like
        Concatenated state sequence.
    run_boundaries : sequence of (int, int)
    n_perm : int
    rng_seed : int
        Base seed; the ``i``-th permutation uses ``rng_seed + i``.

    Returns
    -------
    np.ndarray
        Shape ``(n_perm, n_trs)``.
    """
    n_trs = len(states)
    out = np.empty((n_perm, n_trs), dtype=np.asarray(states).dtype)
    for i in range(n_perm):
        out[i] = circular_shift_states_by_run(
            states, run_boundaries, rng_seed + i,
        )
    return out



def precompute_eligible_null_state_sequences(
    all_states,
    run_boundaries,
    eligible_mask,
    n_perm,
    rng_seed,
):
    """Build an ``(n_perm, n_eligible)`` matrix of circular-shift permutations
    performed **entirely within the eligible label subspace**.

    Unlike :func:`precompute_null_state_sequences`, this helper guarantees
    that every shifted label comes from the eligible class set. It extracts
    the eligible TR subsequence first, derives per-fMRI-run eligible-span
    boundaries, and only then applies :func:`circular_shift_states_by_run`
    within those eligible spans. The returned sequences are already
    restricted to eligible TRs and ready to be passed to a classifier fit
    on ``X_elig`` without any further masking.

    Background — the label-space leakage bug this fixes
    ----------------------------------------------------
    The prior pattern (used throughout 08d and 08e for multi-class decoders)
    was to precompute null sequences from the **full** multi-class
    ``all_states`` vector, then slice with a position-based ``eligible_mask``
    built from the **original** sequence. That's subtly wrong: after a
    circular shift, a position that was originally eligible may now hold a
    non-eligible class that rolled in from elsewhere in the same fMRI run.
    The downstream classifier is then fit on a mixed label space (eligible
    + leaked non-eligible classes), its predictions land in that mixed
    space, and ``sklearn.metrics.balanced_accuracy_score`` with the default
    ``labels=None`` averages per-class TPR over ``np.union1d(y_true, y_pred)``
    — a denominator larger than the intended ``n_classes``, which depresses
    the null mean below the true ``1/n_classes`` chance level. For an 08e
    smoke run this manifested as ``null_mean ≈ 0.012`` when chance should
    have been ``1/29 ≈ 0.0345``.

    Fixing this at the null-generation layer (instead of patching every
    call site) ensures all current and future callers are safe.

    Parameters
    ----------
    all_states : array-like, shape (n_total_trs,)
        Full TR-level state sequence (concatenated across fMRI runs in the
        same order as ``run_boundaries``).
    run_boundaries : sequence of (int, int)
        Contiguous ``(start, end)`` index ranges over ``all_states`` for
        each fMRI run (``end`` exclusive). Output of
        :func:`build_run_boundaries`.
    eligible_mask : array-like of bool, shape (n_total_trs,)
        Boolean mask over ``all_states`` marking TRs whose *original* label
        is in the eligible subset (e.g. ``content_eligible`` states, or the
        FO-intersection for 08e D3a).
    n_perm : int
        Number of permutations.
    rng_seed : int
        Base seed; the ``i``-th permutation uses ``rng_seed + i``.

    Returns
    -------
    np.ndarray
        Shape ``(n_perm, eligible_mask.sum())``. Every value is guaranteed
        to be a label drawn from the eligible class set, and the marginal
        class distribution of each row exactly matches
        ``all_states[eligible_mask]`` (shift is label-preserving within
        each eligible sub-span). The returned rows can be fed directly
        into a classifier's ``fit`` paired with ``X[eligible_mask]``.

    Notes
    -----
    * D2 per-state layer selectivity (08d ``_run_d2``) uses a binary
      indicator null ``(null_seqs[i] == state_id).astype(int)`` which is
      insensitive to which non-target class rolled into each position, so
      it is **not** affected by the label-space leakage and continues to
      use :func:`precompute_null_state_sequences` directly.
    * The eligible sub-span of an fMRI run may be shorter than the
      original run (non-eligible TRs are dropped), so the shift range is
      smaller. Empty sub-spans are skipped. Within-run temporal structure
      of the eligible subset is preserved.
    """
    mask = np.asarray(eligible_mask, dtype=bool)
    states = np.asarray(all_states)
    if mask.shape[0] != states.shape[0]:
        raise ValueError(
            f"eligible_mask length ({mask.shape[0]}) does not match "
            f"all_states length ({states.shape[0]})"
        )

    y_elig = states[mask]
    n_elig = int(y_elig.shape[0])
    if n_elig == 0:
        return np.empty((n_perm, 0), dtype=y_elig.dtype)

    # Per-fMRI-run eligible-span boundaries: for each original run
    # (start, end), the eligible subspan is (cum_elig[start], cum_elig[end]).
    # cum_elig has length n_total_trs + 1 so end-indexing is well-defined
    # (matches the Python half-open slice convention).
    cum_elig = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    elig_boundaries = []
    for start, end in run_boundaries:
        es = int(cum_elig[start])
        ee = int(cum_elig[end])
        if ee > es:
            elig_boundaries.append((es, ee))

    out = np.empty((n_perm, n_elig), dtype=y_elig.dtype)
    for i in range(n_perm):
        out[i] = circular_shift_states_by_run(
            y_elig, elig_boundaries, rng_seed + i,
        )
    return out


# ---------------------------------------------------------------------------
# Partial-effect residualization helpers (C4 negative control)
# ---------------------------------------------------------------------------
#
# These helpers implement the epoch-level partial-effect control described in
# the design notes §3.1 (C4) and §4.1.
# They are used by 08b to residualize state-block-level features and state
# indicators against an epoch-center run-position polynomial, and by 08d to
# residualize transformer features against the same polynomial at TR level.
#
# Immutability guarantee (per 2026-04-23 08b per-state redesign §8.1b V2):
# ``build_epoch_run_position_design`` and ``partial_effect_residualize`` NEVER
# mutate their inputs — they build new arrays (``np.full_like``,
# ``np.column_stack``, ``y2[finite] - D[finite] @ beta``) and return fresh
# results. This is what makes ``prefer="threads"`` safe for the per-state /
# per-feature parallel loops in 08b A1/A3. See
# ``test_residualize_does_not_mutate_input`` in ``tests/test_partial_effect_null.py``
# for the regression check.


def build_epoch_run_position_design(block_records, degree=3, intercept=True):
    """Build an (n_epochs, degree+1) design matrix of epoch-center run positions.

    For each epoch, compute the normalized center position within its run:
    ``t_bar = ((start_tr + end_tr - 1) / 2) / max(episode_length_tr - 1, 1)``,
    clipped to [0, 1], then evaluate ``[1, t_bar, t_bar^2, ..., t_bar^degree]``.

    Parameters
    ----------
    block_records : sequence of dict
        Output of :func:`utils.state_blocks.extract_state_block_records` —
        each record must carry ``start_tr``, ``end_tr``, ``episode_length_tr``.
    degree : int, default 3
        Polynomial degree. Cubic matches ``_build_confound_design_matrix`` in
        08d for consistency.
    intercept : bool, default True
        Prepend an all-ones column so the design also absorbs the mean.

    Returns
    -------
    np.ndarray
        Shape ``(len(block_records), degree + int(intercept))``.
    """
    n = len(block_records)
    if n == 0:
        return np.empty((0, degree + int(intercept)))

    t_bar = np.empty(n, dtype=np.float64)
    for i, rec in enumerate(block_records):
        denom = max(int(rec["episode_length_tr"]) - 1, 1)
        center = (int(rec["start_tr"]) + int(rec["end_tr"]) - 1) / 2.0
        t_bar[i] = np.clip(center / denom, 0.0, 1.0)

    cols = []
    if intercept:
        cols.append(np.ones(n))
    for d in range(1, degree + 1):
        cols.append(t_bar ** d)
    return np.column_stack(cols)


def partial_effect_residualize(y, D):
    """Residualize ``y`` against design matrix ``D`` via OLS.

    Returns ``y - D @ pinv(D) @ y``. Uses ``np.linalg.lstsq`` for numerical
    robustness (handles rank-deficient D cleanly).

    Parameters
    ----------
    y : np.ndarray, shape (n,) or (n, k)
        Response(s) to residualize.
    D : np.ndarray, shape (n, p)
        Design matrix (typically epoch-center polynomial from
        :func:`build_epoch_run_position_design`).

    Returns
    -------
    np.ndarray
        Residuals with the same shape as ``y``. If any row of ``y`` contains
        NaN, that row's residual is NaN (OLS is fit on finite rows only).
    """
    y = np.asarray(y, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if y.shape[0] != D.shape[0]:
        raise ValueError(
            f"y and D must share leading dim (got {y.shape[0]} vs {D.shape[0]})"
        )
    if D.shape[0] == 0:
        return y.copy()

    # Mask finite rows across both y (any column) and D (any column)
    y2 = y if y.ndim == 2 else y.reshape(-1, 1)
    finite = np.isfinite(y2).all(axis=1) & np.isfinite(D).all(axis=1)
    resid = np.full_like(y2, np.nan, dtype=np.float64)

    if finite.sum() <= D.shape[1]:
        # Underdetermined (<) or exactly determined (==): OLS residuals are
        # either non-unique or identically zero on the finite rows — either
        # case would produce degenerate downstream KW input, so fall back to
        # the raw values and let the caller's finite-check drop NaNs.
        return y.copy()

    beta, *_ = np.linalg.lstsq(D[finite], y2[finite], rcond=None)
    resid[finite] = y2[finite] - D[finite] @ beta

    return resid.ravel() if y.ndim == 1 else resid


def mask_a_run_opening(data_by_run, mask_tr=33):
    """Drop the first ``mask_tr`` TRs of each a-run from a per-run dict.

    a-runs are identified by ``run_id.endswith('a')`` (BIDS task field
    convention: ``task-s01e01a`` vs ``...b``). b-runs are returned
    unchanged. Use this to apply the C3 theme-song sensitivity mask to
    both ``decoded_states`` and ``content_features`` before downstream
    epoch extraction.

    Parameters
    ----------
    data_by_run : dict[str, np.ndarray]
        Per-run arrays indexed by run_id. First axis must be TR.
    mask_tr : int, default 33
        Number of leading TRs to drop from each a-run. Default matches the
        empirical theme-song + opening-credits window from
        ``findings_05e_temporal_trend_a2.md``.

    Returns
    -------
    dict[str, np.ndarray]
        New dict with a-run arrays truncated; b-run arrays shared by reference.
    """
    out = {}
    for run_id, arr in data_by_run.items():
        if run_id.endswith("a") and arr is not None and len(arr) > mask_tr:
            out[run_id] = arr[mask_tr:]
        else:
            out[run_id] = arr
    return out


# ---------------------------------------------------------------------------
# Streaming per-layer PCA (drift-aware feature loader + PCA fit/project)
# ---------------------------------------------------------------------------


def stream_pca_features(
    stimulus,
    model_key,
    run_ids,
    n_trs_per_run,
    scratch_dir,
    *,
    train_run_ids=None,
    variance_threshold=0.95,
    pca_models=None,
    pca_info=None,
    max_tr_drift=3,
    extraction_subdir_suffix="",
    feature_key_fn=None,
):
    """Per-layer streamed loader + PCA fit/project, bounding peak memory.

    Combines drift-aware feature loading, per-layer PCA fitting (on a
    training-run subset, no leakage), and projection into a single helper
    that processes **one layer at a time** and discards raw features after
    projection. Peak memory is bounded by ``max(raw_layer, accumulated_pca)``
    rather than ``n_layers × raw_layer`` as with the old
    load-then-fit-then-project approach.

    Why streaming: Friends × llama-3.2-3b raw features sum to ~46 GB (292
    runs × 28 layers × 3072 dims × 460 TRs × float32). Streaming drops peak
    usage to ~2.4 GB.

    Two-pass structure:

    * **Pass 1** (cheap header probes): for each run, ``np.load`` with
      ``mmap_mode='r'`` reads only the ``.npy`` header to get ``shape[0]``
      without materializing the data. The minimum length across all
      available layers gives the per-run feature length. Combined with the
      per-run state length this yields ``effective_n_trs[run] =
      min(state_len, feat_len)``. Runs exceeding ``max_tr_drift`` TRs of
      state/feature mismatch are dropped (with a warning).

    * **Pass 2** (per-layer): for each layer, materialize the full raw
      features across all kept runs (z-scored within run, truncated to the
      effective length), then either fit a new ``sklearn.decomposition.PCA``
      on the training-run subset (if ``pca_models is None``) or reuse a
      pre-fit PCA passed via ``pca_models`` / ``pca_info``. The projected
      ``(n_trs, K)`` arrays are stored in the output dict as ``float32``;
      the raw layer dict goes out of scope at the end of each iteration.

    Parameters
    ----------
    stimulus : str
        Stimulus key (e.g. ``'friends'``). Used to locate 08c feature files
        under ``{scratch_dir}/output/08c_transformer_features/{stimulus}/``.
    model_key : str
        Model key (e.g. ``'llama-3.2-3b'``). Must be present in
        :data:`utils.transformer_io.MODEL_REGISTRY`.
    run_ids : sequence of str
        Candidate run IDs to load (already deduplicated / sorted by the
        caller).
    n_trs_per_run : dict[str, int]
        Per-run HMM state length (from ``decoded_states[run]``). Used
        together with the observed feature length to compute the effective
        per-run TR count.
    scratch_dir : str
        Scratch directory root (``os.getenv('SCRATCH_DIR')``).
    train_run_ids : set of str, optional
        Training run IDs used for PCA fitting. Required when
        ``pca_models is None``. When ``None`` and ``pca_models is None``,
        all runs are used for fitting with a warning (this is the legacy
        behaviour for non-Friends stimuli that don't have their own 03a
        training split).
    variance_threshold : float, default 0.95
        Cumulative explained variance retained per layer. Passed directly
        as sklearn's fractional ``n_components``, so the effective K is
        always the minimum number of components needed to reach this
        threshold (no hidden upper bound).
    pca_models : dict, optional
        Pre-fit per-layer PCAs (output of a prior call). When supplied, the
        fitting step is skipped and features are only projected. Used by
        08e to project a test stimulus through the Friends-fit PCA.
    pca_info : dict, optional
        Per-layer PCA metadata accompanying ``pca_models`` (same shape as
        the return value). Required alongside ``pca_models`` because the
        per-layer ``K`` is needed for projection.
    max_tr_drift : int, default 3
        Runs with ``|state_len - feat_len| > max_tr_drift`` are dropped.
    feature_key_fn : callable, optional
        ``(run_id: str) -> feature_stem: str`` translating a
        decoded_states key into the 08c feature-file stem. Default is
        identity. Callers that have more decoded_states entries than
        08c feature files (movie10 has two viewings of ``figures``/
        ``life`` sharing one feature file) pass
        :func:`utils.common.feature_key_for_cross_stim_run_id` here.

    Returns
    -------
    features_by_layer : dict
        ``{layer_idx: {run_id: np.ndarray (effective_n_trs, K) float32}}``.
        Empty inner dict for layers with no feature data.
    out_pca_info : dict
        ``{layer_idx: {'K': int, 'explained_variance': float,
        'n_train_samples': int}}``.
    out_pca_models : dict
        ``{layer_idx: sklearn.decomposition.PCA | None}``.
    effective_n_trs : dict
        ``{run_id: int}`` for runs that survived drift alignment. Callers
        MUST use this to truncate ``decoded_states[run_id]`` to the same
        length before building run boundaries, concatenating state vectors,
        or computing fractional occupancy.
    dropped_runs : dict
        ``{run_id: reason_string}`` for runs removed from the result.

    Raises
    ------
    ValueError
        If both ``pca_models`` is ``None`` and ``train_run_ids`` is ``None``
        (or empty) — callers must be explicit about PCA fit vs. project.
    """
    import os

    from sklearn.decomposition import PCA

    from utils.transformer_io import MODEL_REGISTRY, zscore_within_run

    fit_pca = pca_models is None
    if fit_pca and not train_run_ids:
        raise ValueError(
            "stream_pca_features: either `pca_models` (for project-only) "
            "or a non-empty `train_run_ids` (for fitting) is required."
        )
    if not fit_pca and pca_info is None:
        raise ValueError(
            "stream_pca_features: `pca_info` is required alongside "
            "`pca_models` to know the per-layer component count K."
        )

    feat_dir = os.path.join(
        scratch_dir, "output",
        f"08c_transformer_features{extraction_subdir_suffix}",
        stimulus, model_key,
    )
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key={model_key!r}")
    n_layers = MODEL_REGISTRY[model_key]["n_layers"]

    # ------------------------------------------------------------------
    # Pass 1: drift alignment via mmap header probes.
    #
    # All layers for a given run come from the same forward pass, so they
    # share the same n_trs. We still probe every existing layer to guard
    # against corrupted / partial writes — np.load(mmap_mode='r') only
    # reads the header (~microseconds per file), so 8k probes for
    # Friends × llama cost ≈ a few seconds total.
    # ------------------------------------------------------------------
    # feature_key_fn lets callers disambiguate decoded_states entries at
    # finer granularity than the 08c feature files (e.g. movie10 has two
    # viewings of ``figures01`` → two decoded_states keys mapping to the
    # single 08c file ``figures01_raw.npy``). Default is identity.
    _fkey = feature_key_fn if feature_key_fn is not None else (lambda r: r)

    effective_n_trs = {}
    dropped_runs = {}
    delta_counts = {}

    for run_id in run_ids:
        state_n = n_trs_per_run[run_id]
        feat_stem = _fkey(run_id)
        per_layer_lens = []
        for layer_idx in range(n_layers):
            fpath = os.path.join(
                feat_dir, f"layer_{layer_idx:02d}", f"{feat_stem}_raw.npy",
            )
            if not os.path.exists(fpath):
                continue
            arr = np.load(fpath, mmap_mode="r")
            per_layer_lens.append(int(arr.shape[0]))
            del arr

        if not per_layer_lens:
            dropped_runs[run_id] = "no feature files found"
            continue

        feat_n = min(per_layer_lens)
        eff = min(state_n, feat_n)
        delta = state_n - feat_n  # positive => features shorter than states
        delta_counts[delta] = delta_counts.get(delta, 0) + 1

        if abs(delta) > max_tr_drift:
            dropped_runs[run_id] = (
                f"|states ({state_n}) - features ({feat_n})| = {abs(delta)} "
                f"TRs exceeds cap ({max_tr_drift})"
            )
            continue

        effective_n_trs[run_id] = eff

    if delta_counts:
        summary = ", ".join(
            f"Δ={d}: {c}" for d, c in sorted(delta_counts.items())
        )
        logger.info(
            "Feature/state TR drift for %s/%s: %s "
            "(positive Δ = features shorter than states)",
            stimulus, model_key, summary,
        )
    if dropped_runs:
        logger.warning(
            "Dropped %d/%d runs for %s/%s due to drift > %d TRs or missing files",
            len(dropped_runs), len(run_ids), stimulus, model_key, max_tr_drift,
        )
        for rid, reason in list(dropped_runs.items())[:10]:
            logger.warning("  %s: %s", rid, reason)

    kept_run_ids = [r for r in run_ids if r in effective_n_trs]
    if not kept_run_ids:
        return {}, {}, {}, effective_n_trs, dropped_runs

    # ------------------------------------------------------------------
    # Pass 2: per-layer load → z-score → truncate → PCA → project.
    #
    # The raw layer dict is rebuilt from scratch each iteration and goes
    # out of scope at the end, bounding peak memory to
    # (one raw layer) + (PCA'd output accumulated so far).
    # ------------------------------------------------------------------
    features_by_layer = {}
    out_pca_info = dict(pca_info) if pca_info else {}
    out_pca_models = dict(pca_models) if pca_models else {}

    train_set = set(train_run_ids) if train_run_ids else None

    for layer_idx in range(n_layers):
        raw_per_run = {}
        for run_id in kept_run_ids:
            feat_stem = _fkey(run_id)
            fpath = os.path.join(
                feat_dir, f"layer_{layer_idx:02d}", f"{feat_stem}_raw.npy",
            )
            if not os.path.exists(fpath):
                continue
            arr = np.load(fpath)
            eff = effective_n_trs[run_id]
            if arr.shape[0] < eff:
                # Should not happen: pass 1 should have clamped eff to this
                # layer's length already. Guard defensively and skip.
                logger.warning(
                    "Layer %d, run %s: feature length (%d) < effective (%d) "
                    "— skipping this (run, layer)",
                    layer_idx, run_id, arr.shape[0], eff,
                )
                continue
            raw_per_run[run_id] = zscore_within_run(arr[:eff])

        if not raw_per_run:
            features_by_layer[layer_idx] = {}
            if fit_pca:
                out_pca_info[layer_idx] = {
                    "K": 0, "explained_variance": 0.0, "n_train_samples": 0,
                }
                out_pca_models[layer_idx] = None
            continue

        # -------------------- Fit or reuse PCA --------------------
        if fit_pca:
            if train_set is not None:
                train_data = [
                    data for rid, data in raw_per_run.items() if rid in train_set
                ]
            else:
                train_data = list(raw_per_run.values())

            if not train_data:
                logger.warning(
                    "Layer %d: no training runs found for PCA — using all runs",
                    layer_idx,
                )
                train_data = list(raw_per_run.values())

            train_stacked = np.vstack(train_data)
            # Delegate K selection to sklearn. A float 0 < c < 1 in
            # n_components tells PCA to fit full SVD and auto-truncate at
            # the smallest K whose cumulative explained variance ≥ c. This
            # replaces the old hardcoded max_k=200 which silently violated
            # the threshold for high-dim models (LLaMA 3072-dim layers hit
            # K=200 at only ~78-87% variance under the 0.90 target).
            # ``svd_solver='full'`` is required because the fractional
            # n_components API is only supported with full SVD; for LLaMA
            # this adds ~15 min to the fit phase relative to the previous
            # randomized-SVD + cap approach but guarantees correctness.
            pca = PCA(n_components=variance_threshold, svd_solver="full")
            pca.fit(train_stacked)
            k = int(pca.n_components_)
            explained_var = float(np.sum(pca.explained_variance_ratio_))

            out_pca_models[layer_idx] = pca
            out_pca_info[layer_idx] = {
                "K": k,
                "explained_variance": explained_var,
                "n_train_samples": int(train_stacked.shape[0]),
            }
            logger.info(
                "Layer %d: PCA %d -> %d components (%.1f%% variance, "
                "fitted on %d train samples)",
                layer_idx, train_stacked.shape[1], k,
                explained_var * 100,
                train_stacked.shape[0],
            )
            # Free fit workspace before projecting.
            del train_stacked, train_data
        else:
            pca = out_pca_models.get(layer_idx)
            k = out_pca_info.get(layer_idx, {}).get("K", 0)
            if pca is None or k == 0:
                features_by_layer[layer_idx] = {}
                # Drop raw for this layer before moving on.
                del raw_per_run
                continue

        # -------------------- Project all runs --------------------
        projected = {}
        for run_id, data in raw_per_run.items():
            projected[run_id] = pca.transform(data)[:, :k].astype(np.float32)
        features_by_layer[layer_idx] = projected

        # Explicit drop (raw_per_run also goes out of scope at next iter,
        # but being explicit keeps peak memory predictable under Python's
        # reference counting).
        del raw_per_run

    return features_by_layer, out_pca_info, out_pca_models, effective_n_trs, dropped_runs


# ---------------------------------------------------------------------------
# PCA cache (file-lock, cross-job) — skip redundant refits across lag jobs
# ---------------------------------------------------------------------------
#
# Each 08d per-lag SLURM job reruns the PCA fit even though the inputs and
# outputs are identical across lags (the lag shift happens after PCA, in
# ``build_layer_feature_matrix``). For LLaMA this costs ~90 min per job;
# across 9 lag jobs per (subject, model), that is ~13.5h of pure waste per
# subject. The cache below lets the first job compute once, and every
# subsequent job (same-partition successor, preemptable resubmission,
# gap-fill, or cross-stimulus) load from disk in ~30 s.
#
# See ``the design notes`` and the
# plan file for full design rationale.

_PCA_CACHE_VERSION = 1


def _pca_cache_atomic_write_json(path: str, data: dict) -> None:
    """Atomically write *data* as JSON to *path* via temp file + rename."""
    import json
    import os
    import tempfile

    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp", prefix=".meta.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pca_cache_atomic_write_npy(path: str, array: np.ndarray) -> None:
    """Atomically write *array* as .npy to *path* via temp file + rename.

    ``np.save`` auto-appends ``.npy`` when the path does not already end in
    ``.npy``, so we write to a temp path that DOES end in ``.npy`` to avoid
    ambiguity, then rename.
    """
    import os
    import tempfile

    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".npy.tmp", prefix=".npy.")
    os.close(fd)  # np.save opens the path itself
    try:
        # Use write_to_file semantics: open in wb and use np.save with a
        # file object so nothing is appended to the name.
        with open(tmp, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pca_cache_try_load(
    cache_dir: str,
    stimulus: str,
    model_key: str,
    split_hash: str,
    variance_threshold: float,
):
    """Return (features_by_layer, pca_info, pca_models, effective_n_trs,
    dropped_runs) if cache is VALID and complete, else None.

    Validity is determined purely from on-disk state: meta.json presence +
    schema match + every (layer, run) .npy file present + pca_models.joblib
    present. The caller does NOT hold the lock while this runs — it is a
    fast-path check before acquiring the exclusive lock, then re-run under
    the lock to avoid duplicate refits.
    """
    import json
    import os

    import joblib

    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if meta.get("version") != _PCA_CACHE_VERSION:
        return None
    if meta.get("stimulus") != stimulus:
        return None
    if meta.get("model_key") != model_key:
        return None
    if meta.get("split_hash") != split_hash:
        return None
    if abs(float(meta.get("variance_threshold", -1)) - float(variance_threshold)) > 1e-9:
        return None

    models_path = os.path.join(cache_dir, "pca_models.joblib")
    if not os.path.exists(models_path):
        return None

    pca_info_raw = meta.get("pca_info", {})
    effective_n_trs_raw = meta.get("effective_n_trs", {})
    dropped_runs = dict(meta.get("dropped_runs", {}))

    # Rebuild typed dicts. pca_info keys come back as strings from JSON.
    pca_info = {}
    for layer_key, layer_meta in pca_info_raw.items():
        try:
            layer_idx = int(layer_key)
        except (TypeError, ValueError):
            return None
        pca_info[layer_idx] = layer_meta

    effective_n_trs = {str(k): int(v) for k, v in effective_n_trs_raw.items()}

    # Verify every per-(layer, run) .npy file exists before loading anything
    features_root = os.path.join(cache_dir, "features")
    for layer_idx, layer_meta in pca_info.items():
        if not layer_meta or int(layer_meta.get("K", 0)) == 0:
            continue  # empty layer — no per-run files expected
        layer_dir = os.path.join(features_root, f"layer_{layer_idx:02d}")
        if not os.path.isdir(layer_dir):
            return None
        for run_id in effective_n_trs:
            fpath = os.path.join(layer_dir, f"{run_id}.npy")
            if not os.path.exists(fpath):
                return None

    # All-or-nothing load. Any failure (corrupt file, partial write that
    # slipped past the write-order guard, joblib version mismatch) is
    # treated as a cache miss so the caller falls back to refit instead of
    # crashing.
    try:
        pca_models = joblib.load(models_path)
    except Exception:
        logger.warning(
            "PCA cache: failed to load %s — treating as miss",
            models_path, exc_info=True,
        )
        return None

    # Use mmap_mode='r' so arrays are backed by page cache (not RSS) and
    # can be shared across loky workers via COW fork without duplicating
    # the ~11 GB LLaMA payload. Without this, warm reads OOM the 48 GB
    # SLURM allocation when joblib dispatches parallel layer workers.
    features_by_layer = {}
    try:
        for layer_idx, layer_meta in pca_info.items():
            layer_runs = {}
            if layer_meta and int(layer_meta.get("K", 0)) > 0:
                layer_dir = os.path.join(features_root, f"layer_{layer_idx:02d}")
                for run_id in effective_n_trs:
                    fpath = os.path.join(layer_dir, f"{run_id}.npy")
                    layer_runs[run_id] = np.load(fpath, mmap_mode="r")
            features_by_layer[layer_idx] = layer_runs
    except Exception:
        logger.warning(
            "PCA cache: failed to load one or more .npy files under %s "
            "— treating as miss", features_root, exc_info=True,
        )
        return None

    return features_by_layer, pca_info, pca_models, effective_n_trs, dropped_runs


def _pca_cache_save(
    cache_dir: str,
    result: tuple,
    *,
    stimulus: str,
    model_key: str,
    sub_id: str,
    parcellation: str,
    split_path_rel: str,
    split_hash: str,
    variance_threshold: float,
    n_layers: int,
):
    """Persist the 5-tuple returned by ``stream_pca_features`` to disk.

    Order: per-run .npy files → pca_models.joblib → meta.json (last).
    meta.json absence signals an incomplete cache so a dead writer does not
    leave stale state.
    """
    import datetime
    import os

    import joblib

    features_by_layer, pca_info, pca_models, effective_n_trs, dropped_runs = result
    features_root = os.path.join(cache_dir, "features")
    os.makedirs(features_root, exist_ok=True)

    # Per-run .npy files
    for layer_idx, layer_runs in features_by_layer.items():
        if not layer_runs:
            continue
        layer_dir = os.path.join(features_root, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        for run_id, arr in layer_runs.items():
            fpath = os.path.join(layer_dir, f"{run_id}.npy")
            _pca_cache_atomic_write_npy(fpath, np.asarray(arr, dtype=np.float32))

    # PCA models (sklearn objects via joblib). Use mkstemp for the temp
    # path so concurrent writers (should the lock ever fail) do not clobber
    # each other's in-progress writes — matches the pattern used by the
    # .npy and meta.json writers.
    import tempfile as _tempfile
    models_path = os.path.join(cache_dir, "pca_models.joblib")
    fd, models_tmp = _tempfile.mkstemp(
        dir=cache_dir, suffix=".joblib.tmp", prefix=".pca_models.",
    )
    os.close(fd)
    try:
        joblib.dump(pca_models, models_tmp, compress=3)
        os.replace(models_tmp, models_path)
    except BaseException:
        try:
            os.unlink(models_tmp)
        except OSError:
            pass
        raise

    # meta.json LAST — its presence marks the cache complete
    meta = {
        "version": _PCA_CACHE_VERSION,
        "stimulus": stimulus,
        "model_key": model_key,
        "sub_id": sub_id,
        "parcellation": parcellation,
        "variance_threshold": float(variance_threshold),
        "split_path": split_path_rel,
        "split_hash": split_hash,
        "n_layers": int(n_layers),
        "pca_info": {str(k): v for k, v in pca_info.items()},
        "effective_n_trs": {str(k): int(v) for k, v in effective_n_trs.items()},
        "dropped_runs": dict(dropped_runs),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "created_by_jobid": os.environ.get("SLURM_JOB_ID", ""),
    }
    _pca_cache_atomic_write_json(os.path.join(cache_dir, "meta.json"), meta)


def load_or_fit_pca_cache(
    stimulus: str,
    model_key: str,
    run_ids: Sequence[str],
    n_trs_per_run: dict,
    scratch_dir: str,
    *,
    cache_dir: str,
    split_hash: str,
    sub_id: str,
    parcellation: str,
    split_path_rel: str,
    train_run_ids: Iterable[str] | None = None,
    variance_threshold: float = 0.95,
    max_tr_drift: int = 3,
    feature_key_fn: Callable[[str], str] | None = None,
    extraction_subdir_suffix: str = "",
):
    """Load cached PCA outputs or compute + cache them.

    Returns the same 5-tuple as :func:`stream_pca_features`:
    ``(features_by_layer, pca_info, pca_models, effective_n_trs, dropped_runs)``.

    Uses ``fcntl.flock`` on ``{cache_dir}/.lock`` for cross-process
    coordination. Cache is keyed on (stimulus, model_key, split_hash,
    variance_threshold) — if any of these change, the cache misses and
    ``stream_pca_features`` is called to refill.

    Blocking semantics: if process A holds the lock (actively fitting),
    process B blocks on ``flock`` until A releases. When B acquires the
    lock it re-checks the cache and typically finds a fresh hit written
    by A — so B pays only the cache-load cost (~30 s) instead of a full
    refit (~90 min for LLaMA).
    """
    import fcntl
    import os

    os.makedirs(cache_dir, exist_ok=True)
    lock_path = os.path.join(cache_dir, ".lock")

    # Fast path: check cache without the lock
    hit = _pca_cache_try_load(
        cache_dir, stimulus, model_key, split_hash, variance_threshold,
    )
    if hit is not None:
        logger.info(
            "PCA cache HIT (fast path) for %s/%s at %s",
            stimulus, model_key, cache_dir,
        )
        return hit

    logger.info(
        "PCA cache MISS for %s/%s — acquiring lock at %s",
        stimulus, model_key, lock_path,
    )
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # blocks

        # Re-check under lock — another process may have populated it
        hit = _pca_cache_try_load(
            cache_dir, stimulus, model_key, split_hash, variance_threshold,
        )
        if hit is not None:
            logger.info(
                "PCA cache HIT (after lock wait) for %s/%s at %s",
                stimulus, model_key, cache_dir,
            )
            return hit

        logger.info(
            "PCA cache: fitting under lock for %s/%s "
            "(this process is the writer)",
            stimulus, model_key,
        )
        result = stream_pca_features(
            stimulus, model_key, run_ids, n_trs_per_run, scratch_dir,
            train_run_ids=train_run_ids,
            variance_threshold=variance_threshold,
            feature_key_fn=feature_key_fn,
            max_tr_drift=max_tr_drift,
            extraction_subdir_suffix=extraction_subdir_suffix,
        )
        _, pca_info, _, _, _ = result
        n_layers = len(pca_info)
        _pca_cache_save(
            cache_dir, result,
            stimulus=stimulus,
            model_key=model_key,
            sub_id=sub_id,
            parcellation=parcellation,
            split_path_rel=split_path_rel,
            split_hash=split_hash,
            variance_threshold=variance_threshold,
            n_layers=n_layers,
        )
        logger.info(
            "PCA cache SAVED for %s/%s at %s",
            stimulus, model_key, cache_dir,
        )
        return result
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


# ---------------------------------------------------------------------------
# Per-layer feature-matrix stacking (shared by 08d D1 and 08e D3a)
# ---------------------------------------------------------------------------


def build_layer_feature_matrix(
    layer_runs: dict,
    run_ids: Sequence[str],
    decoded_states: dict,
    lag: int = 0,
) -> np.ndarray:
    """Stack per-run feature arrays into one ``(total_n_trs, K)`` matrix.

    Used by both [08d D1](../08d_transformer_depth.py) (within-stimulus, per-lag
    decoding) and [08e D3a](../08e_transformer_cross_stim_aggregate.py)
    (cross-stimulus aggregate transfer). Previously duplicated as a private
    helper in each script — hoisted here so both can share the same safeguards
    and lag convention.

    Parameters
    ----------
    layer_runs : dict[str, np.ndarray]
        ``{run_id: (n_trs_run, K)}`` feature matrix per run for a single
        transformer layer, typically an element of
        ``features_by_layer[layer_idx]`` returned by
        :func:`stream_pca_features`.
    run_ids : Sequence[str]
        Concatenation order. Runs must also be keys in ``decoded_states``;
        runs missing from ``layer_runs`` are zero-padded to
        ``len(decoded_states[run_id])``.
    decoded_states : dict[str, np.ndarray]
        ``{run_id: (n_trs_run,)}`` HMM state sequences, already truncated to
        the effective per-run TR count by the caller (see
        :func:`stream_pca_features`'s ``effective_n_trs`` output).
    lag : int, default 0
        Lag in TRs applied via ``new_feats[t] = old_feats[t - lag]`` with
        zero-padding at the start, matching the 08b content-decoding
        convention. 08e currently calls with ``lag=0``; 08d sweeps
        ``lag ∈ LAGS_TO_TEST``.

    Returns
    -------
    np.ndarray
        ``(sum(len(decoded_states[r]) for r in run_ids), K)``.

    Raises
    ------
    ValueError
        If ``layer_runs`` is empty or if no run in ``run_ids`` has feature
        data (callers should ``try/except ValueError`` and skip the layer).

    Notes
    -----
    * Runs **in** ``run_ids`` but **not in** ``layer_runs`` get
      ``np.zeros((n_trs, K))`` — accepted project-wide behavior for
      tolerating partial layer coverage. Callers that want strict matching
      should pre-filter ``run_ids`` to the layer's keys.
    * For runs present in both dicts, the feature length must match
      ``len(decoded_states[run_id])`` after the lag shift; a mismatch
      raises ``AssertionError`` to surface drift-alignment regressions
      (previously caused silent corruption in 08e).
    """
    if not layer_runs:
        raise ValueError("layer_runs is empty")

    # Infer K from the first run that has feature data. If none do, the
    # layer has zero coverage — raise rather than building a degenerate
    # all-zeros matrix.
    first_runs = [r for r in run_ids if r in layer_runs]
    if not first_runs:
        raise ValueError("No runs have feature data for this layer")
    dim = layer_runs[first_runs[0]].shape[1]

    parts = []
    for run_id in run_ids:
        n_trs = len(decoded_states[run_id])
        if run_id not in layer_runs:
            parts.append(np.zeros((n_trs, dim)))
            continue
        feats = layer_runs[run_id]
        if lag > 0:
            n_f = feats.shape[0]
            if lag >= n_f:
                feats = np.zeros_like(feats)
            else:
                feats = np.vstack([
                    np.zeros((lag, dim)),
                    feats[:n_f - lag],
                ])
        # Drift-alignment invariant: after the caller's effective_n_trs
        # truncation, feats and decoded_states[run_id] must match in length.
        # A mismatch here would otherwise surface as a cryptic IndexError
        # under ``X[eligible_mask]`` downstream.
        assert feats.shape[0] == n_trs, (
            f"Feature/state length mismatch for run {run_id} at lag={lag}: "
            f"feats={feats.shape[0]} states={n_trs}"
        )
        parts.append(feats)
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# LORO decoders
# ---------------------------------------------------------------------------


def loro_ridge_classifier_cv(X, y, folds):
    """Leave-one-run-out RidgeClassifier with balanced class weights.

    This is the multi-class decoder used for D1 (within-stimulus depth
    profile) and D3a (cross-stimulus transfer, via ``fit`` on Friends and
    ``score`` on the test stimulus — D3a uses its own helper, not this one).

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    folds : list of (train_idx, test_idx) tuples

    Returns
    -------
    dict or None
        ``{"balanced_accuracy": float, "weighted_f1": float,
        "cohen_kappa": float}``, or ``None`` if pooled predictions contain
        fewer than 10 samples (e.g. every fold crashed).
    """
    from sklearn.exceptions import ConvergenceWarning, NotFittedError
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import (
        balanced_accuracy_score, cohen_kappa_score, f1_score,
    )

    # These are the exceptions we expect inside a fold: singular-matrix
    # errors (numpy), degenerate-class-set errors (sklearn), convergence
    # failures, and mismatched-shape ValueErrors. Anything else is a real
    # bug and should propagate.
    _expected = (
        ValueError, np.linalg.LinAlgError, NotFittedError, ConvergenceWarning,
    )

    all_true = []
    all_pred = []
    for train_idx, test_idx in folds:
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        try:
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            all_true.extend(y[test_idx])
            all_pred.extend(pred)
        except _expected as exc:
            logger.debug("Ridge fold failed: %s", exc)
            continue

    if len(all_true) < 10:
        return None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "weighted_f1": float(
            f1_score(all_true, all_pred, average="weighted", zero_division=0),
        ),
        "cohen_kappa": float(cohen_kappa_score(all_true, all_pred)),
    }


def _ridge_label_binarize(y, classes):
    """One-hot encode labels matching sklearn RidgeClassifier's LabelBinarizer.

    sklearn's ``RidgeClassifier`` uses ``LabelBinarizer(pos_label=1, neg_label=-1)``,
    which produces ``{-1, +1}`` encoding for **all** cases:

    - Multi-class (≥3 classes): returns (n, C) matrix of {-1, +1}.
    - Binary (2 classes): returns (n, 1) column of {-1, +1}.

    This must match sklearn exactly so that ``batch_loro_ridge_classify``
    produces the same predictions as sequential ``loro_ridge_classifier_cv``.
    """
    n = len(y)
    C = len(classes)
    if C == 2:
        # Binary: single column, positive class = classes[1]
        out = -np.ones((n, 1), dtype=np.float64)
        out[y == classes[1], 0] = 1.0
        return out
    # Multi-class: (n, C) with {-1, +1} — vectorized via class_lookup
    class_lookup = np.empty(int(classes.max()) + 1, dtype=np.intp)
    class_lookup[classes] = np.arange(C)
    out = -np.ones((n, C), dtype=np.float64)
    out[np.arange(n), class_lookup[y.astype(np.intp)]] = 1.0
    return out


def _batch_binarize(y_batch, classes, class_lookup):
    """Binarize B label vectors at once into a (n, B * n_ohe) matrix.

    Parameters
    ----------
    y_batch : (B, n) integer array of labels
    classes : sorted 1-D array of class labels
    class_lookup : array mapping label -> class index (pre-built)

    Returns
    -------
    Y : (n, B * n_ohe) float64 array with {-1, +1} encoding
    """
    B, n = y_batch.shape
    C = len(classes)
    if C == 2:
        # Binary: (n, B) where +1 if label == classes[1]
        Y = -np.ones((n, B), dtype=np.float64)
        Y[y_batch.T == classes[1]] = 1.0
        return Y
    # Multi-class: (n, B * C) with {-1, +1}
    n_ohe = C
    Y = -np.ones((n, B * n_ohe), dtype=np.float64)
    idx = class_lookup[y_batch.astype(np.intp)]  # (B, n)
    row = np.arange(n)
    for j in range(B):
        Y[row, j * n_ohe + idx[j]] = 1.0
    return Y


def _cholesky_solve_targets(
    X, y_all, classes, folds, *, alpha=1.0, batch_size=500,
    sample_weight_source=None,
):
    """Generator: per-fold, per-batch Cholesky-solved continuous Ridge scores.

    Extracts the inner solve loop shared by ``batch_loro_ridge_classify`` (D1
    main + D1 confound consumers, decode-to-prediction) and
    ``batch_loro_ridge_per_state_auc`` (D2 consumer, retain continuous scores
    for per-state AUC).

    Implementation references (preserved from prior 04-09 + 04-10 versions):
    - Opt A vectorized binarization (commit b9d4856)
    - Opt D adaptive batch cap (4 GB on binarized Y)
    - Sample weights from ``sample_weight_source`` (typically ``y_all[0]`` =
      observed labels), held fixed across all targets in a fold. See "Approx-
      imation note" in ``batch_loro_ridge_classify``.

    Yields
    ------
    (b_start, b_end, test_idx, scores_3d) per (fold × batch):
        scores_3d : (n_test, B, n_ohe_cols) float64
            Continuous Ridge scores for the B targets in this batch on the
            fold's test set. Caller decodes to argmax / threshold (D1) or
            keeps as-is for AUC (D2).
        Yields nothing for folds where any class has zero training samples
        under ``sample_weight_source`` (degenerate-fold skip).

    Notes
    -----
    The generator does NOT track a "valid" mask — that's the consumer's job.
    For degenerate folds the test_idx samples are simply never yielded,
    leaving the consumer's pre-allocated buffer in its initial sentinel state.
    """
    from scipy.linalg import cho_factor, cho_solve

    classes = np.asarray(classes)
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    n_ohe_cols = 1 if is_binary else n_classes

    n_targets, n_samples = y_all.shape
    if sample_weight_source is None:
        sample_weight_source = y_all[0]

    # Class-to-index lookup for vectorized binarization (Opt A)
    class_lookup = np.empty(int(classes.max()) + 1, dtype=np.intp)
    class_lookup[classes] = np.arange(n_classes)

    # Adaptive batch_size: cap so binarized Y stays under 4 GB (Opt D)
    _max_batch_bytes = 4 * 1024**3
    _adaptive_B = max(
        100, int(_max_batch_bytes / (n_ohe_cols * n_samples * 8)),
    )
    batch_size = min(batch_size, _adaptive_B, n_targets)

    for train_idx, test_idx in folds:
        n_train = len(train_idx)
        n_test = len(test_idx)
        X_train = X[train_idx]
        X_test = X[test_idx]

        # Balanced class weights from sample_weight_source training labels
        # (fixed across all targets — see batch_loro_ridge_classify docstring).
        class_counts = np.array(
            [np.sum(sample_weight_source[train_idx] == c) for c in classes],
            dtype=np.float64,
        )
        if np.any(class_counts == 0):
            continue
        weights_per_class = n_train / (n_classes * class_counts)
        sample_weights = np.empty(n_train, dtype=np.float64)
        for ci, c in enumerate(classes):
            sample_weights[sample_weight_source[train_idx] == c] = (
                weights_per_class[ci]
            )
        sqrt_w = np.sqrt(sample_weights)

        # Centre X using weighted mean (matches sklearn fit_intercept=True).
        X_offset = np.average(X_train, axis=0, weights=sample_weights)
        X_c = X_train - X_offset
        X_test_c = X_test - X_offset

        # Weighted centred design matrix + Cholesky factorisation (ONCE)
        X_w = X_c * sqrt_w[:, None]
        A = X_w.T @ X_w
        A[np.diag_indices_from(A)] += alpha
        try:
            L = cho_factor(A)
        except np.linalg.LinAlgError:
            logger.debug("Cholesky failed for fold — skipping")
            continue

        for b_start in range(0, n_targets, batch_size):
            b_end = min(b_start + batch_size, n_targets)
            B = b_end - b_start

            y_batch_train = y_all[b_start:b_end, train_idx]  # (B, n_train)
            Y_batch = _batch_binarize(y_batch_train, classes, class_lookup)

            # Weighted per-target offsets for centering (matches sklearn)
            Y_3d = Y_batch.reshape(n_train, B, n_ohe_cols)
            y_offsets = np.einsum("i,ijk->jk", sample_weights, Y_3d)
            y_offsets /= sample_weights.sum()
            Y_batch -= y_offsets.reshape(1, B * n_ohe_cols)
            Y_batch *= sqrt_w[:, None]

            # Single large GEMM instead of B small ones
            rhs = X_w.T @ Y_batch  # (K, B * n_ohe_cols)
            coef = cho_solve(L, rhs)  # (K, B * n_ohe_cols)

            scores = X_test_c @ coef  # (n_test, B * n_ohe_cols)
            scores += y_offsets.reshape(1, B * n_ohe_cols)

            scores_3d = scores.reshape(n_test, B, n_ohe_cols)
            yield b_start, b_end, test_idx, scores_3d


def batch_loro_ridge_classify(
    X, y_observed, y_null, folds, *, alpha=1.0, batch_size=500,
):
    """Batch LORO RidgeClassifier: observed + all null targets in one pass.

    Instead of fitting 1001 separate sklearn RidgeClassifier models per fold,
    this function precomputes the Cholesky factorization of the weighted Gram
    matrix **once** and solves all targets via matrix multiply.

    **Approximation note:** Sample weights (``class_weight="balanced"``) are
    computed from the *observed* training labels and held fixed for all
    permutation targets. Circular-shift permutations preserve per-run class
    counts, so the per-class weight *values* are identical; however, the
    sample-to-weight *assignment* changes (different samples belong to each
    class after the shift). This means the weighted Gram matrix is approximate
    for null targets.  Empirical testing shows ≤2 % error on individual null
    balanced accuracies. The **observed** metrics are exact (same labels →
    same weights). For well-separated results the p-value impact is negligible;
    borderline cases (p ≈ 0.04–0.06) should be interpreted with this caveat.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (already masked to eligible TRs).
    y_observed : np.ndarray, shape (n_samples,)
        Observed state labels.
    y_null : np.ndarray, shape (n_perm, n_samples)
        Null (circular-shifted) state labels for each permutation.
    folds : list of (train_idx, test_idx)
        LORO fold indices into X/y arrays.
    alpha : float
        Ridge regularisation parameter (default 1.0, matching sklearn).
    batch_size : int
        Number of null permutations to batch-solve per Cholesky pass.
        Controls peak memory.  Default 100 ≈ 25 MB for typical data shapes.

    Returns
    -------
    dict or None
        ``{"observed": {"balanced_accuracy", "weighted_f1", "cohen_kappa"},
          "null_balanced_accuracies": list[float]}``
        Returns ``None`` if fewer than 10 test samples are pooled.
    """
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

    classes = np.sort(np.unique(y_observed))
    n_classes = len(classes)
    if n_classes < 2:
        return None
    is_binary = (n_classes == 2)

    # HMM state IDs are non-negative integers in [0, nc-1]. A negative value
    # in `classes` almost always means silent integer overflow at the call
    # site (e.g. casting state_id=200 to int8 wraps to -56), and would later
    # crash _cholesky_solve_targets at `class_lookup[classes]` with a cryptic
    # IndexError. Fail loud here instead.
    if int(classes.min()) < 0:
        raise ValueError(
            f"y_observed contains negative class id={int(classes.min())} "
            f"(dtype={y_observed.dtype}). HMM state IDs must be non-negative; "
            f"this often indicates silent integer overflow at the call site "
            f"(e.g. cast to int8 of a state_id >= 128)."
        )

    n_perm = y_null.shape[0]
    n_targets = 1 + n_perm
    n_samples = len(y_observed)

    # Stack all label vectors: row 0 = observed, rows 1..n_perm = null
    y_all = np.empty((n_targets, n_samples), dtype=y_observed.dtype)
    y_all[0] = y_observed
    y_all[1:] = y_null

    # Pre-allocate pooled predictions (-1 = unfilled)
    preds = np.full((n_targets, n_samples), -1, dtype=y_observed.dtype)

    # Stream Cholesky-solved continuous scores; decode to per-target predictions.
    for b_start, b_end, test_idx, scores_3d in _cholesky_solve_targets(
        X, y_all, classes, folds, alpha=alpha, batch_size=batch_size,
        sample_weight_source=y_observed,
    ):
        if is_binary:
            # scores_3d: (n_test, B, 1)
            batch_pred_idx = (scores_3d[:, :, 0] > 0).astype(int)
            preds[b_start:b_end, test_idx] = classes[batch_pred_idx.T]
        else:
            batch_pred_idx = scores_3d.argmax(axis=2)  # (n_test, B)
            preds[b_start:b_end, test_idx] = classes[batch_pred_idx.T]

    # ── Compute metrics ──────────────────────────────────────────────────
    valid_obs = preds[0] != -1
    if valid_obs.sum() < 10:
        return None

    observed_metrics = {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_observed[valid_obs], preds[0, valid_obs]),
        ),
        "weighted_f1": float(
            f1_score(
                y_observed[valid_obs], preds[0, valid_obs],
                average="weighted", zero_division=0,
            ),
        ),
        "cohen_kappa": float(
            cohen_kappa_score(y_observed[valid_obs], preds[0, valid_obs]),
        ),
    }

    # Vectorized null balanced_accuracy (Opt C): replace 1000 sklearn calls
    # with numpy broadcasting. Valid mask is shared across all targets because
    # LORO folds fill predictions identically for all targets in each batch.
    null_preds = preds[1:]  # (n_perm, n_samples)
    valid_shared = np.all(null_preds != -1, axis=0) if n_perm > 0 else valid_obs
    if valid_shared.sum() >= 10 and np.array_equal(valid_shared, valid_obs):
        # Fast path: shared valid mask — fully vectorized
        yt = y_null[:, valid_obs]       # (n_perm, n_valid)
        yp = null_preds[:, valid_obs]   # (n_perm, n_valid)
        # Per-class recall for all perms at once
        recall_sum = np.zeros(n_perm, dtype=np.float64)
        n_classes_present = 0
        for c in classes:
            mask_c = (yt == c)               # (n_perm, n_valid)
            n_c = mask_c.sum(axis=1)         # (n_perm,)
            has_c = n_c > 0
            correct_c = ((yt == c) & (yp == c)).sum(axis=1)  # (n_perm,)
            # recall = correct / n_c where n_c > 0, else 0
            recall = np.where(has_c, correct_c / np.maximum(n_c, 1), 0.0)
            recall_sum += recall
            n_classes_present += 1
        null_accs = (recall_sum / n_classes_present).tolist()
    else:
        # Fallback: per-target sklearn calls (valid masks differ)
        null_accs = []
        for i in range(n_perm):
            valid_i = null_preds[i] != -1
            if valid_i.sum() < 10:
                continue
            null_accs.append(float(
                balanced_accuracy_score(y_null[i, valid_i], null_preds[i, valid_i]),
            ))

    return {"observed": observed_metrics, "null_balanced_accuracies": null_accs}


def batch_loro_ridge_per_state_auc(
    X, y_observed, y_null, folds, eligible_states,
    *, alpha=1.0, batch_size=500,
):
    """Vectorised D2: per-state binary AUC from multi-class Ridge scores.

    Computes per-state, per-(observed|null) ROC AUC by retaining the continuous
    Ridge regression scores from a Cholesky-once-per-fold solve. This replaces
    the legacy 08d D2 path (per-state binary :func:`loro_logistic_auc_cv` ×
    n_layers × n_perm), which was empirically infeasible at production scale
    (~1000 h/cell with 1000 perms × 292-fold LORO LogReg, ~120-160 h/cell with
    500 perms × 10-fold LogReg). This function brings per-cell cost down to
    ~6-8 h.

    **Algorithm.** Cast D2 as multi-class Ridge regression on the
    ``eligible_states`` label set (same engine as D1 main + D1 confound).
    Per fold: Cholesky-factorise the weighted Gram once, batch-solve all
    ``n_perm + 1`` targets in one large GEMM, retain continuous per-class
    scores. After all folds: per state ``s``, compute
    ``roc_auc_score(y == s, pooled_scores[target, :, s])``.

    **Statistical equivalence to LogReg-balanced AUC.** Ridge with sample-
    weighted balancing yields the same Bayes-optimal linear discriminant
    direction as L2-regularised logistic regression under shared-covariance
    Gaussian-like covariates (Hastie/Tibshirani/Friedman 2009 §4.3-4.4).
    AUC is invariant to monotone transforms of the score, so per-state AUC
    is rank-stable across the LogReg/Ridge swap. Validated empirically on
    sub-04 + sub-01 × dinov2 × lag=3 — see
    ``the design notes`` §6.

    **Cross-fold score scale.** Each fold's Cholesky produces different
    ``coef``; pooled scores from different folds are not on the same absolute
    scale. AUC pools across folds, treating the union of fold-scores as a
    single ranking. This inherits the same convention as
    :func:`loro_logistic_auc_cv` (which pools per-fold ``predict_proba`` the
    same way) and is empirically fine — but is documented here because the
    rank-based AUC argument depends on it.

    **Approximation note (inherited).** Sample weights are computed from
    observed labels and held fixed across permutations (see
    :func:`batch_loro_ridge_classify` docstring). For per-state binary AUC,
    the validation step measures null-AUC distribution shift between the
    fixed-weight path and a per-permutation reweight path; if KS distance
    > 0.05, fall back to per-permutation reweight (cheap — only affects the
    sample_weight vector inside :func:`_cholesky_solve_targets`).

    **Eligible-subspace null required.** ``y_null`` must be generated by
    :func:`precompute_eligible_null_state_sequences` (the 2026-04-10 leakage
    fix) so each null permutation contains only ``eligible_states``. Full-
    state nulls bias the per-state marginal P(y == s) and decalibrate the
    AUC null distribution. See ``feedback_subset_null_subspace`` memory.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Lagged feature matrix (typically PCA-projected layer activations).
    y_observed : np.ndarray, shape (n_samples,)
        Observed state labels (multi-class; only ``eligible_states`` participate
        in the regression — non-eligible labels are not part of the OHE).
        The caller is expected to have masked X/y to eligible TRs already.
    y_null : np.ndarray, shape (n_perm, n_samples)
        Eligible-subspace null state labels.
    folds : list of (train_idx, test_idx)
        Typically StratifiedGroupKFold(n_splits=10) over runs (see
        ``08d_transformer_depth.py:_build_d2_folds``).
    eligible_states : sequence of int
        Content-eligible state IDs. Must be a subset of unique values in
        ``y_observed`` and each row of ``y_null``.
    alpha : float
        Ridge regularisation. Validated insensitive to α ∈ {0.1, 1.0, 10}
        at production p/n (Neuro A2 of 04-30 D2 vectorisation review).
    batch_size : int
        Targets batched per Cholesky pass. Adaptive cap of 4 GB on the
        intermediate ``Y_batch`` matrix is applied automatically.

    Returns
    -------
    dict or None
        ``{"states": [...], "observed_auc": np.ndarray (n_states,),
           "null_auc": np.ndarray (n_perm, n_states),
           "n_valid_folds_observed": int,
           "n_valid_per_state_obs": np.ndarray (n_states,)}``
        Returns ``None`` if no valid fold yielded predictions.
    """
    from sklearn.metrics import roc_auc_score

    classes = np.asarray(sorted(int(s) for s in eligible_states))
    n_classes = len(classes)
    if n_classes < 2:
        return None

    # See the matching guard in batch_loro_ridge_classify. Silent int8
    # truncation of a state_id >= 128 produces a negative wraparound value
    # in y_observed, which would silently mismatch every (y_observed == s)
    # indicator and produce nonsense AUCs without crashing.
    if np.issubdtype(y_observed.dtype, np.integer) and int(y_observed.min()) < 0:
        raise ValueError(
            f"y_observed contains negative state id={int(y_observed.min())} "
            f"(dtype={y_observed.dtype}). HMM state IDs must be non-negative; "
            f"this often indicates silent integer overflow at the call site."
        )

    n_perm = y_null.shape[0]
    n_targets = 1 + n_perm
    n_samples = len(y_observed)

    y_all = np.empty((n_targets, n_samples), dtype=y_observed.dtype)
    y_all[0] = y_observed
    y_all[1:] = y_null

    # Continuous score buffer. NaN = unfilled (degenerate fold or sample not
    # in any test fold). float32 for memory; AUC is rank-based so precision
    # is more than enough.
    pooled_scores = np.full(
        (n_targets, n_samples, n_classes), np.nan, dtype=np.float32,
    )

    n_valid_folds = 0
    folds_seen_test_idx: list[np.ndarray] = []

    last_b_start = -1
    for b_start, b_end, test_idx, scores_3d in _cholesky_solve_targets(
        X, y_all, classes, folds, alpha=alpha, batch_size=batch_size,
        sample_weight_source=y_observed,
    ):
        # scores_3d is (n_test, B, n_classes); transpose to (B, n_test, n_classes)
        # for assignment into pooled_scores[b_start:b_end, test_idx, :].
        pooled_scores[b_start:b_end, test_idx, :] = (
            scores_3d.transpose(1, 0, 2).astype(np.float32)
        )
        # Track unique folds (b_start resets to 0 on each new fold)
        if b_start == 0 and (
            not folds_seen_test_idx
            or not np.array_equal(folds_seen_test_idx[-1], test_idx)
        ):
            folds_seen_test_idx.append(test_idx)
            n_valid_folds += 1
        last_b_start = b_start

    if n_valid_folds == 0:
        return None

    # Per-state AUC computation.
    observed_auc = np.full(n_classes, np.nan, dtype=np.float64)
    null_auc = np.full((n_perm, n_classes), np.nan, dtype=np.float64)
    n_valid_per_state_obs = np.zeros(n_classes, dtype=np.int64)

    for s_idx, state_id in enumerate(classes):
        # Observed AUC
        scores_obs = pooled_scores[0, :, s_idx]
        valid_obs = ~np.isnan(scores_obs)
        n_valid_per_state_obs[s_idx] = int(valid_obs.sum())
        if valid_obs.sum() < 10:
            continue
        y_ind_obs = (y_observed == state_id).astype(np.int8)
        if y_ind_obs[valid_obs].sum() < 2 or y_ind_obs[valid_obs].sum() == valid_obs.sum():
            continue
        try:
            observed_auc[s_idx] = roc_auc_score(
                y_ind_obs[valid_obs], scores_obs[valid_obs],
            )
        except ValueError:
            continue

        # Null AUC distribution (vectorise the indicator construction; loop
        # over permutations because roc_auc_score isn't broadcasted).
        null_indicators = (y_null == state_id).astype(np.int8)  # (n_perm, n)
        for i in range(n_perm):
            scores_null = pooled_scores[1 + i, :, s_idx]
            v = ~np.isnan(scores_null)
            if v.sum() < 10:
                continue
            yi = null_indicators[i, v]
            if yi.sum() < 2 or yi.sum() == v.sum():
                continue
            try:
                null_auc[i, s_idx] = roc_auc_score(yi, scores_null[v])
            except ValueError:
                continue

    return {
        "states": classes.tolist(),
        "observed_auc": observed_auc,
        "null_auc": null_auc,
        "n_valid_folds_observed": n_valid_folds,
        "n_valid_per_state_obs": n_valid_per_state_obs,
    }


def loro_logistic_auc_cv(X, y_binary, folds):
    """Leave-one-run-out logistic regression pooled-probabilities AUC.

    Used as the D2 metric: per-state layer selectivity. Replaces the old
    binary-Ridge-R² metric, which was unstable on binary targets.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
    y_binary : np.ndarray (n_samples,)
        0/1 state-indicator vector.
    folds : list of (train_idx, test_idx) tuples

    Returns
    -------
    dict or None
        ``{"roc_auc": float, "n_positive": int}``, or ``None`` if the pooled
        fold predictions contain fewer than 10 samples or only one class is
        represented.
    """
    from sklearn.exceptions import ConvergenceWarning, NotFittedError
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    _expected = (
        ValueError, np.linalg.LinAlgError, NotFittedError, ConvergenceWarning,
    )

    y_pred = np.full(len(y_binary), np.nan, dtype=float)
    for train_idx, test_idx in folds:
        if len(np.unique(y_binary[train_idx])) < 2:
            # Class-balance failure for this fold — leave as NaN
            continue
        clf = LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=200, C=1.0,
        )
        try:
            clf.fit(X[train_idx], y_binary[train_idx])
            y_pred[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        except _expected as exc:
            logger.debug("Logistic fold failed: %s", exc)
            continue

    valid = np.isfinite(y_pred)
    if valid.sum() < 10:
        return None
    if len(np.unique(y_binary[valid])) < 2:
        return None

    return {
        "roc_auc": float(roc_auc_score(y_binary[valid], y_pred[valid])),
        "n_positive": int(y_binary[valid].sum()),
    }


# ---------------------------------------------------------------------------
# Permutation-null runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Metrics and selectivity helpers
# ---------------------------------------------------------------------------


def compute_effect_size(observed, n_classes):
    """Return ``{chance_level, normalized_effect_size}`` for balanced accuracy.

    ``normalized_effect_size = (observed - chance) / (1 - chance)`` is the
    standard "distance above chance divided by distance remaining to ceiling"
    normalization. Reported alongside raw accuracy to avoid the current
    08d issue where accuracies were presented without a chance floor.
    """
    if n_classes < 2:
        return {"chance_level": float("nan"),
                "normalized_effect_size": float("nan")}
    chance = 1.0 / n_classes
    denom = 1.0 - chance
    if denom <= 0:
        return {"chance_level": chance,
                "normalized_effect_size": float("nan")}
    return {
        "chance_level": float(chance),
        "normalized_effect_size": float((observed - chance) / denom),
    }


def layer_selectivity(layer_metric_dict):
    """Summarize a per-layer metric profile for a single state.

    Parameters
    ----------
    layer_metric_dict : dict
        ``{layer_idx: float}``. NaN and missing layers are ignored.

    Returns
    -------
    dict
        ``{peak_layer, peak_value, max_minus_median, selectivity_entropy}``.
        The entropy is computed over the distribution
        ``p_ℓ = metric_ℓ / Σ metric_ℓ`` (higher = flatter profile).
        All fields are NaN if the profile has fewer than two finite values.
    """
    if not layer_metric_dict:
        return {
            "peak_layer": None, "peak_value": float("nan"),
            "max_minus_median": float("nan"),
            "selectivity_entropy": float("nan"),
        }

    items = [(int(k), float(v)) for k, v in layer_metric_dict.items()
             if v is not None and np.isfinite(v)]
    if len(items) < 2:
        return {
            "peak_layer": None, "peak_value": float("nan"),
            "max_minus_median": float("nan"),
            "selectivity_entropy": float("nan"),
        }

    items.sort(key=lambda kv: kv[0])
    layers = np.array([k for k, _ in items])
    values = np.array([v for _, v in items])

    peak_idx = int(np.argmax(values))
    peak_layer = int(layers[peak_idx])
    peak_value = float(values[peak_idx])
    max_minus_median = float(peak_value - np.median(values))

    # Selectivity entropy: shift to non-negative before normalizing. If the
    # values are all below zero (can happen with degenerate metrics) we
    # report NaN rather than silently flipping the sign.
    shifted = values - np.min(values)
    if shifted.sum() <= 0:
        entropy = float("nan")
    else:
        p = shifted / shifted.sum()
        p_pos = p[p > 0]
        entropy = float(-np.sum(p_pos * np.log(p_pos)))

    return {
        "peak_layer": peak_layer,
        "peak_value": peak_value,
        "max_minus_median": max_minus_median,
        "selectivity_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Network stratification (D1-net)
# ---------------------------------------------------------------------------


def stratify_states_by_dominant_network(
    state_means, active_states, parcellation, *, include_sign=False,
):
    """Group states by their dominant functional/anatomical network.

    Used by D1-net to stratify ``content_eligible`` states. When
    ``include_sign=False`` (legacy behaviour) states are grouped into up to
    12 buckets, one per entry in :data:`utils.plot_style.NETWORK_ORDER`.
    When ``include_sign=True`` states are split by polarity as well
    (``"+"`` for activation, ``"-"`` for deactivation), yielding up to
    24 ``(network, polarity)`` buckets — this is the default in
    ``08d_transformer_depth.py`` since 2026-04 to prevent functionally
    opposite states (e.g. DMN-on vs. DMN-off) from being pooled into a
    single depth profile.

    Parameters
    ----------
    state_means : np.ndarray
        Parcel-space state means with shape ``(n_states, n_parcels)``. These
        come from ``04_combined_hdphmm/.../final/state_means_parcel.npy``,
        which is already back-projected at save time — no PCA inverse
        transform is required.
    active_states : array-like of int
        Subset of state indices to include (usually the ``content_eligible``
        list).
    parcellation : str
        Parcellation name (passed to ``load_parcel_networks``).
    include_sign : bool, default False
        If True, return polarity-aware groups keyed by
        ``(network_name, "+"|"-")``.

    Returns
    -------
    dict
        ``{network_name: [state_ids]}`` when ``include_sign=False``, or
        ``{(network_name, polarity): [state_ids]}`` when
        ``include_sign=True``. Empty buckets are included for deterministic
        iteration by callers.
    """
    from utils.plot_style import (
        NETWORK_ORDER, compute_dominant_networks, load_parcel_networks,
    )

    parcel_networks = load_parcel_networks(parcellation)
    if parcel_networks is None:
        raise RuntimeError(
            f"Could not load parcel networks for {parcellation} — "
            f"D1-net stratification unavailable."
        )

    state_means = np.asarray(state_means)
    active_arr = np.asarray(list(active_states), dtype=int)
    dominant = compute_dominant_networks(
        state_means, active_arr, parcel_networks, include_sign=include_sign,
    )

    if include_sign:
        groups: dict[tuple, list[int]] = {
            (name, polarity): []
            for name in NETWORK_ORDER
            for polarity in ("+", "-")
        }
        for state_id, val in dominant.items():
            if not isinstance(val, tuple):
                logger.debug(
                    "State %d: dominant network payload %r is not a "
                    "(network, polarity) tuple — skipping",
                    state_id, val,
                )
                continue
            net_name, polarity = val
            key = (net_name, polarity)
            if key in groups:
                groups[key].append(int(state_id))
            else:
                logger.debug(
                    "State %d dominant group %s not in NETWORK_ORDER × "
                    "{+,-} — skipping", state_id, key,
                )
        return groups

    groups_flat: dict[str, list[int]] = {name: [] for name in NETWORK_ORDER}
    for state_id, net_name in dominant.items():
        if net_name in groups_flat:
            groups_flat[net_name].append(int(state_id))
        else:
            logger.debug(
                "State %d dominant network '%s' not in NETWORK_ORDER — skipping",
                state_id, net_name,
            )
    return groups_flat


# ---------------------------------------------------------------------------
# LORO fold construction with eligibility remapping
# ---------------------------------------------------------------------------


def build_loro_folds(run_boundaries, eligible_mask=None):
    """Build leave-one-run-out folds expressed in *masked* index space.

    When ``eligible_mask`` is supplied, the returned fold indices address
    into ``X[eligible_mask]`` / ``y[eligible_mask]`` directly. This removes
    the per-fold remap loop that the old 08d performed inside its inner
    layer loop (a significant efficiency win for large state counts).

    Parameters
    ----------
    run_boundaries : sequence of (int, int)
        Contiguous index ranges into the *un-masked* arrays.
    eligible_mask : np.ndarray of bool, optional
        Length-``n_total`` eligibility mask. When ``None``, folds are
        returned directly in unmasked index space.

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        ``[(train_idx, test_idx), ...]`` with one fold per run whose masked
        test set contains at least one sample.
    """
    folds = []
    if eligible_mask is None:
        n_total = run_boundaries[-1][1] if run_boundaries else 0
        all_idx = np.arange(n_total)
        for start, end in run_boundaries:
            test_idx = all_idx[start:end]
            train_idx = np.concatenate([all_idx[:start], all_idx[end:]])
            if len(test_idx) > 0 and len(train_idx) > 0:
                folds.append((train_idx, test_idx))
        return folds

    elig = np.asarray(eligible_mask, dtype=bool)
    masked_idx = np.where(elig)[0]
    if len(masked_idx) == 0:
        return folds
    # Map global index -> position in the masked array.
    lookup = -np.ones(len(elig), dtype=np.int64)
    lookup[masked_idx] = np.arange(len(masked_idx))

    for start, end in run_boundaries:
        run_mask = np.zeros(len(elig), dtype=bool)
        run_mask[start:end] = True
        test_global = np.where(elig & run_mask)[0]
        train_global = np.where(elig & ~run_mask)[0]
        if len(test_global) == 0 or len(train_global) == 0:
            continue
        folds.append((lookup[train_global], lookup[test_global]))
    return folds


# ---------------------------------------------------------------------------
# D1 per-lag partial loader (used by 08d_plots.py for aggregate panels)
# ---------------------------------------------------------------------------


D1_PARTIAL_METRIC_KEYS = (
    "balanced_accuracy",
    "chance_level",
    "null_mean",
    "null_std",
    "p_perm",
    "normalized_effect_size",
)


def load_d1_per_lag_matrix(
    partials_dir: str,
    label: str,
    n_lags: int,
    n_layers: int,
) -> dict:
    """Load all per-lag D1 checkpoint JSONs into stacked (lag × layer) matrices.

    Parameters
    ----------
    partials_dir : str
        Directory containing ``D1_{label}_lag{lag}.json`` checkpoint files.
    label : str
        D1 label, e.g. ``"D1_main"`` or ``"D1_neg_control"``.
    n_lags : int
        Expected number of lags (e.g. 9 for LAGS_TO_TEST).
    n_layers : int
        Expected number of layers for the model (24 for dinov2/w2v-bert, 28
        for llama).

    Returns
    -------
    dict
        Keys:

        * ``balanced_accuracy``, ``chance_level``, ``null_mean``, ``null_std``,
          ``p_perm``, ``normalized_effect_size`` — each a ``(n_lags, n_layers)``
          float array with NaN for missing (lag, layer) cells.
        * ``n_complete_per_lag`` — ``(n_lags,)`` int array of layers present in
          each lag's checkpoint (0 if the lag file is missing).
        * ``n_total`` — total complete cells (int).
        * ``label`` — the input label, echoed back.
    """
    matrices = {
        key: np.full((n_lags, n_layers), np.nan, dtype=np.float64)
        for key in D1_PARTIAL_METRIC_KEYS
    }
    n_complete = np.zeros(n_lags, dtype=np.int64)

    for lag in range(n_lags):
        fpath = os.path.join(partials_dir, f"D1_{label}_lag{lag}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        results = data.get("results", {})
        for layer_str, entry in results.items():
            if entry is None:
                continue
            try:
                layer_idx = int(layer_str)
            except (TypeError, ValueError):
                continue
            if layer_idx < 0 or layer_idx >= n_layers:
                continue
            for key in D1_PARTIAL_METRIC_KEYS:
                v = entry.get(key)
                if v is None:
                    continue
                matrices[key][lag, layer_idx] = float(v)
            n_complete[lag] += 1

    matrices["n_complete_per_lag"] = n_complete
    matrices["n_total"] = int(n_complete.sum())
    matrices["label"] = label
    return matrices
