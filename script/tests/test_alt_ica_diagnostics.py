"""Audit + diagnostic tests for the ICA convergent-validity supplement.

Data-free tests run on the branch alone (synthetic inputs). The data-gated
determinism guard skips when SCRATCH outputs are absent.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # put script/ on path


def test_nojax_unpickler_substitutes_jax_class_only():
    from utils.jax_free_model_io import _NoJaxUnpickler, _HMMStub
    up = _NoJaxUnpickler(io.BytesIO(b""))
    # any module path containing 'hdphmm_jax' -> stub
    assert up.find_class("project.utils.hdphmm_jax", "StickyHDPHMM_JAX") is _HMMStub
    # unrelated classes resolve normally
    assert up.find_class("numpy", "ndarray") is np.ndarray


def test_hungarian_zero_variance_map_no_nan_and_assigns_at_zero():
    from utils.ica_states import match_maps_hungarian
    rng = np.random.default_rng(0)
    P, K = 20, 3
    hmm = rng.normal(size=(K, P))
    hmm[1] = 5.0  # constant (zero-variance) HMM map -> contract: r=0, no NaN
    ica = rng.normal(size=(P, K))
    out = match_maps_hungarian(ica, hmm)
    assert not np.isnan(out["corr"]).any()          # no NaN reaches the cost matrix
    assert out["matched_r"].shape == (K,)
    # the constant map's matched |r| is ~0 (z-scored constant -> all-zero row)
    j = list(out["hmm_idx"]).index(1)
    assert out["matched_r"][j] == pytest.approx(0.0, abs=1e-12)


def test_hungarian_duplicate_hmm_maps_are_one_to_one():
    from utils.ica_states import match_maps_hungarian
    rng = np.random.default_rng(1)
    P, K = 30, 4
    hmm = rng.normal(size=(K, P))
    hmm[2] = hmm[0]                                   # exact duplicate (|r|=1 pair)
    ica = rng.normal(size=(P, K))
    out = match_maps_hungarian(ica, hmm)
    assert len(set(out["ica_idx"].tolist())) == K     # no ICA component reused
    assert len(set(out["hmm_idx"].tolist())) == K     # no HMM state reused


def test_per_rank_pvalues_require_aligned_matched_count():
    from utils.ica_states import spatial_match_pvalues
    rng = np.random.default_rng(2)
    n, n_perm = 5, 200
    null = rng.uniform(0, 1, size=(n_perm, n))
    obs = rng.uniform(0, 1, size=n)
    p = spatial_match_pvalues(obs, null)
    assert p.shape == (n,)
    assert np.all((p >= 1.0 / (1 + n_perm)) & (p <= 1.0))
    # mismatched matched count must not silently misalign
    with pytest.raises((ValueError, IndexError)):
        spatial_match_pvalues(rng.uniform(0, 1, size=n + 1), null)


def _bh_reference(p):
    """Independent Benjamini-Hochberg step-up (numpy only)."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]  # reverse cumulative min
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def test_benjamini_hochberg_matches_reference():
    from utils.stats import benjamini_hochberg
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=37)
    np.testing.assert_allclose(benjamini_hochberg(p), _bh_reference(p), atol=1e-12)


def test_fdr_with_nan_excludes_nan_from_family_size():
    from utils.stats import fdr_with_nan
    # 3 finite p's + 2 NaN: BH denominator must be 3, not 5
    p = np.array([0.01, 0.02, 0.03, np.nan, np.nan])
    q = fdr_with_nan(p)
    assert np.isnan(q[3]) and np.isnan(q[4])             # NaN passthrough
    expected_finite = _bh_reference(np.array([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(q[:3], expected_finite, atol=1e-12)  # family size = 3


def test_build_evidence_rows_counts_and_below_null():
    from sm_alt_ica_diagnostics import build_evidence_rows
    summary = {
        "sub_id": "sub-XX",
        "by_K": {
            "35": {"state_sets": {"eligible": {
                "matched_r": [0.1, 0.5, 0.4],
                "spatial_q": [1.0, 0.01, 0.2],
                "null_mean": 0.3, "null_p95": 0.45}}},
            "41": {"state_sets": {"eligible": {}}},  # empty -> skipped
        },
    }
    rows = build_evidence_rows(summary)
    assert len(rows) == 1
    r = rows[0]
    assert r["sub"] == "sub-XX" and r["K"] == 35
    assert r["n_surv"] == 1 and r["n_total"] == 3
    assert r["mean_r"] == pytest.approx(np.mean([0.1, 0.5, 0.4]))
    assert r["frac_below_null"] == pytest.approx(1 / 3)  # only 0.1 < 0.3


def test_null_self_calibration_is_uniform_not_anticonservative():
    """Hold out each null draw as 'observed' vs the other n_perm-1; p must be
    ~Uniform and NOT systematically conservative (>0.5)."""
    from utils.ica_states import subspace_rotation_null, spatial_match_pvalues
    from scipy import stats as sps
    rng = np.random.default_rng(4)
    n_pcs, P, K = 30, 50, 8
    comps = np.linalg.qr(rng.normal(size=(P, n_pcs)))[0].T   # (n_pcs, P) orthonormal rows
    hmm = rng.normal(size=(K, P))
    null = subspace_rotation_null(comps, hmm, n_components=K, n_perm=400, rng_seed=0)
    n_perm = null.shape[0]
    ps = []
    for j in range(n_perm):
        rest = np.delete(null, j, axis=0)                    # exclude the held-out draw
        ps.extend(spatial_match_pvalues(null[j], rest).tolist())
    ps = np.asarray(ps)
    assert 0.45 <= ps.mean() <= 0.55                         # calibrated, not biased
    assert (ps < 0.5).mean() >= 0.45                         # not anti-conservative
    assert sps.kstest(ps, "uniform").pvalue > 1e-3           # lenient uniformity


def test_empirical_fdr_control_under_global_null():
    """Under the global null (observed drawn from the null model), BH on the
    per-rank p-vector must control FDR <= q. Tests BH applicability to the
    positively-dependent per-rank p's. Under the global null every rejection is
    false, so per-family FDP = 1 if the family makes any rejection else 0, and
    FDR = mean per-family FDP = P(any rejection), which BH controls at <= q."""
    from utils.ica_states import subspace_rotation_null, spatial_match_pvalues
    from utils.stats import fdr_with_nan
    rng = np.random.default_rng(5)
    n_pcs, P, K = 28, 48, 8
    comps = np.linalg.qr(rng.normal(size=(P, n_pcs)))[0].T
    q_level, n_families = 0.05, 200
    per_family_fdp = []
    for f in range(n_families):
        hmm = rng.normal(size=(K, P))
        null = subspace_rotation_null(comps, hmm, n_components=K, n_perm=300,
                                      rng_seed=1000 + f)
        p = spatial_match_pvalues(null[0], null[1:])   # 'observed' is itself a null draw
        q = fdr_with_nan(p)
        per_family_fdp.append(1.0 if int(np.sum(q < q_level)) > 0 else 0.0)
    realized_fdr = float(np.mean(per_family_fdp))
    assert realized_fdr <= q_level + 0.03, f"realized FDR {realized_fdr:.3f} > {q_level}"
