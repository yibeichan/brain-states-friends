#!/usr/bin/env python3
"""
stats.py - Shared statistical utility functions.

Provides commonly reused statistical routines (FDR correction, RV coefficient)
to avoid duplication across analysis scripts.
"""

import numpy as np


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction (NaN-safe).

    Parameters
    ----------
    p_values : array-like
        Raw p-values. NaN entries are assigned q=1.0 and excluded from
        the multiple-comparison correction count.

    Returns
    -------
    np.ndarray
        FDR-corrected q-values (same length as input).
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    n_valid = valid.sum()
    if n_valid == 0:
        return np.full(n, np.nan)
    # Default: NaN p-values get q=1.0 (not significant)
    q = np.ones(n)
    p_valid = p[valid]
    idx = np.argsort(p_valid)
    sorted_p = p_valid[idx]
    adjusted = np.minimum(1.0, sorted_p * n_valid / np.arange(1, n_valid + 1))
    # Enforce monotonicity (backwards)
    for i in range(n_valid - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    q_valid = np.empty(n_valid)
    q_valid[idx] = adjusted
    q[valid] = q_valid
    return q


def fdr_with_nan(p_arr):
    """Apply BH FDR only over non-NaN entries; leave NaN positions as NaN.

    Thin wrapper around :func:`benjamini_hochberg` that passes through NaN
    positions without inflating the multiple-comparison count.

    Parameters
    ----------
    p_arr : array-like
        Raw p-values (may contain NaN).

    Returns
    -------
    np.ndarray
        FDR-corrected q-values (same shape as input; NaN where input was NaN).
    """
    p_arr = np.asarray(p_arr, dtype=float)
    q_arr = np.full_like(p_arr, np.nan)
    valid = np.isfinite(p_arr)
    if valid.any():
        q_arr[valid] = benjamini_hochberg(p_arr[valid])
    return q_arr


def permutation_pvalue(observed, null_dist, alternative='two-sided'):
    """NaN-safe permutation p-value with Phipson-Smyth finite-sampling correction.

    Handles three failure modes that produce spurious significance:
    1. observed is NaN → returns NaN
    2. null_dist contains NaN entries → excludes them, adjusts denominator
    3. All null entries are NaN → returns NaN

    Parameters
    ----------
    observed : float
        Observed test statistic.
    null_dist : array-like
        Null distribution from permutations.
    alternative : str
        'two-sided' (default): p = P(|null| >= |obs|)
        'greater': p = P(null >= obs)

    Returns
    -------
    float
        Permutation p-value, or NaN if observed or all null values are NaN.
    """
    if not np.isfinite(observed):
        return np.nan
    null = np.asarray(null_dist, dtype=float)
    finite = np.isfinite(null)
    n_finite = finite.sum()
    if n_finite == 0:
        return np.nan
    null_f = null[finite]
    if alternative == 'two-sided':
        count = np.sum(np.abs(null_f) >= np.abs(observed))
    else:
        count = np.sum(null_f >= observed)
    return float((count + 1) / (n_finite + 1))


def bootstrap_mean_ci(values, n_boot=1000, seed=0, ci=0.95):
    """Percentile bootstrap CI for the mean of a 1D sample.

    Uses vectorized resampling (``rng.integers`` on a ``(n_boot, n)`` grid)
    so 1000 × 20 samples cost a single NumPy call rather than a Python
    loop. NaN / None values in ``values`` are dropped before resampling.

    Parameters
    ----------
    values : array-like
        1D sample. Non-finite entries are excluded.
    n_boot : int, default=1000
        Number of bootstrap resamples.
    seed : int, default=0
        PRNG seed for reproducibility. 08-series scripts should pass a
        seed from the project offset block (see individual script
        constants like ``BOOTSTRAP_SEED_D3C``) so resamples are
        decorrelated from 08d/08e permutation nulls.
    ci : float, default=0.95
        Confidence level in (0, 1).

    Returns
    -------
    tuple of (point, ci_low, ci_high)
        Point estimate (``np.mean`` of the cleaned sample) and the
        ``ci``-level percentile bootstrap interval. All three are
        ``None`` when fewer than 2 finite samples are available.
    """
    vals = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=float,
    )
    if len(vals) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot_means = vals[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        float(vals.mean()),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1.0 - alpha))),
    )


def bootstrap_corr_ci(xs, ys, stat_fn, n_boot=1000, seed=0, ci=0.95,
                      min_finite_frac=0.1):
    """Percentile bootstrap CI for a paired-sample correlation.

    Resamples paired indices with replacement and recomputes
    ``stat_fn(xs[idx], ys[idx])`` on each resample. Suited for rank
    correlations where vectorized bootstrap is awkward because the
    statistic function doesn't broadcast (``scipy.stats.spearmanr`` and
    ``kendalltau``).

    Parameters
    ----------
    xs, ys : array-like
        Paired samples; must be the same length.
    stat_fn : callable
        ``(x, y) -> scalar or scipy result tuple``. The first element of
        a tuple return is used as the statistic.
    n_boot, seed, ci : see :func:`bootstrap_mean_ci`.
    min_finite_frac : float, default=0.1
        If fewer than this fraction of bootstrap resamples produce a
        finite statistic (e.g. because tied ranks collapse), the CI is
        returned as ``(point, None, None)`` - the resampled distribution
        is too degenerate to summarize.

    Returns
    -------
    tuple of (point, ci_low, ci_high)
        Point estimate on the original sample plus bootstrap CI. CI
        bounds are ``None`` when ``n < 3`` or when too many resamples
        hit degenerate ties.
    """
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    n = len(x_arr)
    if n != len(y_arr):
        raise ValueError("xs and ys must be paired and equal-length")

    def _as_scalar(result):
        if hasattr(result, "__len__"):
            return float(result[0])
        return float(result)

    point = _as_scalar(stat_fn(x_arr, y_arr))

    if n < 3:
        return point, None, None

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = _as_scalar(stat_fn(x_arr[idx], y_arr[idx]))
        boots[i] = val if np.isfinite(val) else np.nan

    boots = boots[np.isfinite(boots)]
    if len(boots) < max(10, int(min_finite_frac * n_boot)):
        return point, None, None
    alpha = (1.0 - ci) / 2.0
    return (
        point,
        float(np.percentile(boots, 100 * alpha)),
        float(np.percentile(boots, 100 * (1.0 - alpha))),
    )


def partial_spearman(x, y, covariate):
    """Partial Spearman correlation controlling for a covariate.

    Rank-residualizes x and y on covariate, then Pearson-correlates residuals.

    Parameters
    ----------
    x, y : array-like
        Variables to correlate.
    covariate : array-like
        Variable to control for.

    Returns
    -------
    rho : float
        Partial Spearman correlation coefficient (NaN if degenerate).
    p : float
        Two-sided p-value from Pearson on rank-residuals (NaN if degenerate).
    """
    from scipy.stats import rankdata, pearsonr

    x, y, covariate = np.asarray(x, float), np.asarray(y, float), np.asarray(covariate, float)
    r_x = rankdata(x)
    r_y = rankdata(y)
    r_c = rankdata(covariate)

    # Guard: zero-variance covariate after ranking
    if np.std(r_c) < 1e-10:
        return np.nan, np.nan

    def _residualize(target, ctrl):
        X = np.column_stack([np.ones(len(ctrl)), ctrl])
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        return target - X @ beta

    res_x = _residualize(r_x, r_c)
    res_y = _residualize(r_y, r_c)

    # Guard: zero-variance residuals
    if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
        return np.nan, np.nan

    rho, p = pearsonr(res_x, res_y)
    return float(rho), float(p)


def weighted_centroid_index(values, chance=0.5):
    """Above-chance-weighted centroid of an ordered 1-D profile.

    Returns ``Σ_i i · w_i / Σ_i w_i`` with ``w_i = max(values_i − chance, 0)``.
    For a layer-wise decoding profile (e.g. per-layer one-vs-rest AUC, chance
    0.5), this is a stable "preferred depth" estimator: unlike ``argmax`` it
    averages over the whole profile, so it does not collapse to noise when the
    profile is a near-flat ridge (the 08g recurrence×depth use case, where the
    peak layer beats the runner-up by ~0.002 AUC).

    Parameters
    ----------
    values : array-like
        Profile values in index order (index 0 .. n-1).
    chance : float, default=0.5
        Baseline subtracted before weighting. Values at/below chance get
        zero weight.

    Returns
    -------
    float
        Centroid index in ``[0, n-1]``, or NaN if all weights are zero
        (no layer exceeds chance) or the profile is empty.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan
    w = np.clip(v - chance, 0.0, None)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.arange(v.size) @ w / total)


def bootstrap_partial_spearman_ci(x, y, covariate, n_boot=1000, seed=0,
                                  ci=0.95, min_finite_frac=0.1):
    """Percentile bootstrap CI for a partial Spearman correlation.

    Mirrors :func:`bootstrap_corr_ci`, but resamples the *triple*
    ``(x, y, covariate)`` with one shared index per replicate (preserving
    the joint dependence) and recomputes
    ``partial_spearman(x[idx], y[idx], covariate[idx])`` - controlling for
    ``covariate``. ``bootstrap_corr_ci`` cannot back this because the
    covariate is invisible to its two-array ``stat_fn``.

    Ordinal covariates (e.g. integer layer indices) make degenerate
    resamples likely (collapsed residual variance → NaN partial), so the
    same ``min_finite_frac`` guard applies.

    Returns
    -------
    tuple of (point, ci_low, ci_high)
        Point estimate on the original sample plus bootstrap CI. CI bounds
        are ``None`` when ``n < 4`` or when too many resamples are degenerate.
        (``n < 4`` because partialling one covariate costs an extra df.)
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    c_arr = np.asarray(covariate, dtype=float)
    n = len(x_arr)
    if not (n == len(y_arr) == len(c_arr)):
        raise ValueError("x, y, covariate must be paired and equal-length")

    point = float(partial_spearman(x_arr, y_arr, c_arr)[0])

    if n < 4:
        return point, None, None

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = partial_spearman(x_arr[idx], y_arr[idx], c_arr[idx])[0]
        boots[i] = val if np.isfinite(val) else np.nan

    boots = boots[np.isfinite(boots)]
    if len(boots) < max(10, int(min_finite_frac * n_boot)):
        return point, None, None
    alpha = (1.0 - ci) / 2.0
    return (
        point,
        float(np.percentile(boots, 100 * alpha)),
        float(np.percentile(boots, 100 * (1.0 - alpha))),
    )


# ---------------------------------------------------------------------------
# Per-state Mann-Whitney AUC helpers (2026-04-23 08b per-state redesign)
# ---------------------------------------------------------------------------
#
# These primitives implement the two-sample AUC = U_1 / (n_1 * n_0) framework
# used across per-state tests (08b A1/A3). The single-pair helper handles
# validation + NaN + ties; the grid helper amortises rankdata across features
# for K × n_features vectorised evaluation. Call from any script that needs
# per-state per-feature AUCs - not just 08b.


def per_state_auc_mann_whitney(state_mask, feat_values):
    """Mann-Whitney AUC for a single (state, feature) pair.

    Returns ``(auc, sign, n_1, n_0)``. ``auc = U_1 / (n_1 * n_0)`` (symmetric
    around 0.5; AUC > 0.5 means the feature tends to be higher in state-X
    epochs). ``sign`` is the descriptive sign of
    ``mean(y | state) − mean(y | other)``: ``+1``, ``−1``, or ``0`` for an
    exact tie. Non-finite feature values are dropped. Returns
    ``(np.nan, 0, n_1, n_0)`` if either group has fewer than 2 finite values,
    and ``(0.5, 0, n_1, n_0)`` if the feature column is degenerate
    (all equal on finite entries).
    """
    from scipy import stats as _sp_stats

    mask_in = np.asarray(state_mask, dtype=bool)
    vals = np.asarray(feat_values, dtype=np.float64)
    finite = np.isfinite(vals)
    m_in = mask_in & finite
    m_out = (~mask_in) & finite
    n1 = int(m_in.sum())
    n0 = int(m_out.sum())
    if n1 < 2 or n0 < 2:
        return np.nan, 0, n1, n0

    vals_f = vals[finite]
    if np.ptp(vals_f) == 0:
        return 0.5, 0, n1, n0

    ranks = np.full(vals.shape, np.nan)
    ranks[finite] = _sp_stats.rankdata(vals_f, method="average")
    r1_sum = float(np.nansum(ranks[m_in]))
    u1 = r1_sum - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n0)
    mean_in = float(vals[m_in].mean())
    mean_out = float(vals[m_out].mean())
    sign = 1 if mean_in > mean_out else (-1 if mean_in < mean_out else 0)
    return float(auc), int(sign), n1, n0


def per_state_auc_grid(states_arr, feats_mat, target_states, compute_signs=True):
    """Vectorised per-state × per-feature Mann-Whitney AUC grid.

    Ranks each feature column once (NaN-safe) and derives each state's AUC
    via the rank-sum shortcut - K × n_features AUC evaluations but only
    n_features rank sorts. This is the hot path for the 08b A1/A3
    per-permutation null loops.

    Parameters
    ----------
    states_arr : np.ndarray, shape (n_epochs,)
        Integer state labels.
    feats_mat : np.ndarray, shape (n_epochs, n_features)
        Feature matrix. NaN entries are tolerated.
    target_states : sequence of int
        States whose AUCs to compute.
    compute_signs : bool, default True
        When ``False``, ``signs`` is returned as ``None`` and the per-state
        mean computation is skipped (≈ 10% speedup per permutation under
        the null where signs aren't reported).

    Returns
    -------
    aucs : (K, n_features) array
    signs : (K, n_features) int8 array or None
        +1 / −1 / 0 for mean-in > / < / == mean-out.
    """
    from scipy import stats as _sp_stats

    n_ep, n_features = feats_mat.shape
    n_states = len(target_states)
    aucs = np.full((n_states, n_features), np.nan)
    signs = (
        np.zeros((n_states, n_features), dtype=np.int8)
        if compute_signs else None
    )
    state_masks = [(states_arr == s) for s in target_states]

    for fi in range(n_features):
        vals = feats_mat[:, fi]
        finite = np.isfinite(vals)
        if int(finite.sum()) < 4:
            continue
        vals_f = vals[finite]
        if np.ptp(vals_f) == 0:
            for si in range(n_states):
                m = state_masks[si]
                if int((m & finite).sum()) >= 2 and int(((~m) & finite).sum()) >= 2:
                    aucs[si, fi] = 0.5
            continue
        ranks_finite = _sp_stats.rankdata(vals_f, method="average")
        ranks = np.full(n_ep, np.nan)
        ranks[finite] = ranks_finite

        for si, state_id in enumerate(target_states):
            m = state_masks[si]
            m_in = m & finite
            m_out = (~m) & finite
            n1 = int(m_in.sum())
            n0 = int(m_out.sum())
            if n1 < 2 or n0 < 2:
                continue
            r1_sum = float(np.nansum(ranks[m_in]))
            u1 = r1_sum - n1 * (n1 + 1) / 2.0
            aucs[si, fi] = u1 / (n1 * n0)
            if compute_signs:
                mean_in = float(vals[m_in].mean())
                mean_out = float(vals[m_out].mean())
                signs[si, fi] = (
                    1 if mean_in > mean_out else (-1 if mean_in < mean_out else 0)
                )
    return aucs, signs


def two_layer_bh_fdr(p_matrix):
    """Two-layer BH-FDR over a (K, F) p-value matrix.

    Layer 1 ("per-row"): independent BH within each row (e.g. per state
    across features). Layer 2 ("matrix"): BH across all finite entries
    flattened together.

    Parameters
    ----------
    p_matrix : np.ndarray, shape (K, F)
        Raw p-values. NaN entries are carried through.

    Returns
    -------
    p_fdr_per_row : (K, F) array
    p_fdr_matrix : (K, F) array
    """
    p_matrix = np.asarray(p_matrix, dtype=float)
    K, F = p_matrix.shape
    p_per_row = np.full((K, F), np.nan)
    for r in range(K):
        finite = np.isfinite(p_matrix[r])
        if finite.any():
            p_per_row[r, finite] = benjamini_hochberg(p_matrix[r, finite])
    p_matrix_fdr = np.full((K, F), np.nan)
    flat = p_matrix.flatten()
    finite_flat = np.isfinite(flat)
    if finite_flat.any():
        q_flat = benjamini_hochberg(flat[finite_flat])
        out_flat = np.full(flat.shape, np.nan)
        out_flat[finite_flat] = q_flat
        p_matrix_fdr = out_flat.reshape((K, F))
    return p_per_row, p_matrix_fdr


def compute_rv_coefficient(matrices: np.ndarray) -> np.ndarray:
    """RV coefficient between symmetric matrices.

    RV(A, B) = trace(A @ B) / sqrt(trace(A @ A) * trace(B @ B))

    Parameters
    ----------
    matrices : (K, p, p)
        Stack of symmetric matrices.

    Returns
    -------
    rv : (K, K) matrix in [0, 1].
    """
    K = matrices.shape[0]
    flat = matrices.reshape(K, -1)
    gram = flat @ flat.T
    diag = np.diag(gram)
    diag_safe = np.maximum(diag, 1e-10)
    outer = np.sqrt(np.outer(diag_safe, diag_safe))
    rv = gram / outer
    # Mark degenerate entries as NaN
    degenerate = diag < 1e-10
    if np.any(degenerate):
        rv[degenerate, :] = np.nan
        rv[:, degenerate] = np.nan
    np.fill_diagonal(rv, 1.0)
    return np.clip(rv, 0.0, 1.0)
