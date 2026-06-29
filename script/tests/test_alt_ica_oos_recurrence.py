import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils.ica_states import (
    wta_labels,
    consensus_projection,
    wta_label_agreement,
)

from utils.ica_oos_recurrence import (
    fo_per_run,
    recurrence_scores,
    continuous_occupancy,
)


def test_wta_labels_per_run_zscore_argmax():
    # Two runs, 3 components, 6 TRs each.  All components have variance so
    # per-run z-scoring is well-defined.  Within each run the data is
    # constructed so that one component has strictly positive z-scores in the
    # second half of the run and the other components have strictly negative
    # z-scores there, making the winner unambiguous.
    #
    # Run 0 (TRs 0-5): cols 0,1 decrease [6→1], col 2 increases [1→6].
    #   Second half (TRs 3-5): z2 > 0, z0=z1 < 0  →  component 2 wins.
    # Run 1 (TRs 6-11): col 0 decreases [9→4], cols 1,2 increase [1→6].
    #   First half (TRs 6-8): z0 > 0, z1=z2 < 0  →  component 0 wins.
    tc = np.array([
        [6.0, 6.0, 1.0],  # run 0 TR 0  (z0=z1>0, z2<0)
        [5.0, 5.0, 2.0],  # run 0 TR 1
        [4.0, 4.0, 3.0],  # run 0 TR 2
        [3.0, 3.0, 4.0],  # run 0 TR 3  (z2>0, z0=z1<0)
        [2.0, 2.0, 5.0],  # run 0 TR 4
        [1.0, 1.0, 6.0],  # run 0 TR 5
        [9.0, 1.0, 1.0],  # run 1 TR 0  (z0>0, z1=z2<0)
        [8.0, 2.0, 2.0],  # run 1 TR 1
        [7.0, 3.0, 3.0],  # run 1 TR 2
        [4.0, 6.0, 6.0],  # run 1 TR 3  (z1=z2>0, z0<0)
        [3.0, 7.0, 7.0],  # run 1 TR 4
        [2.0, 8.0, 8.0],  # run 1 TR 5
    ], dtype=float)
    run_boundaries = [(0, 6), (6, 12)]
    labels = wta_labels(tc, run_boundaries)
    assert labels.shape == (12,)
    # Run 0, second half: component 2 has strictly positive z-score.
    assert set(np.unique(labels[3:6])) == {2}
    # Run 1, first half: component 0 has strictly positive z-score.
    assert set(np.unique(labels[6:9])) == {0}


def test_wta_labels_matches_wta_label_agreement_internal():
    # wta_labels must reproduce the labels wta_label_agreement computes
    # internally (single source of truth after refactor).
    rng = np.random.default_rng(0)
    tc = rng.standard_normal((40, 5))
    run_boundaries = [(0, 20), (20, 40)]
    viterbi = rng.integers(0, 5, size=40)
    agree = wta_label_agreement(tc, viterbi, run_boundaries, n_perm=10, rng_seed=0)
    assert np.array_equal(wta_labels(tc, run_boundaries), agree["ica_labels"])


def test_consensus_projection_matches_independent_lstsq_regression():
    # consensus_projection must regress the full parcel time series
    # (X_full = X_pc @ components) onto the consensus maps. Verify the closed
    # form (components @ M @ pinv(M.T @ M)) against an INDEPENDENT least-squares
    # solve of  X_full.T ~= M @ TC.T,  using arbitrary maps (not icasso output,
    # which itself calls consensus_projection -- that comparison is circular).
    rng = np.random.default_rng(1)
    n_pcs, n_parcels, T, K = 8, 20, 60, 4
    components = rng.standard_normal((n_pcs, n_parcels))
    X_pc = rng.standard_normal((T, n_pcs))
    maps = rng.standard_normal((n_parcels, K))            # full-rank, arbitrary

    tc = X_pc @ consensus_projection(components, maps)     # (T, K)

    X_full = X_pc @ components                             # (T, n_parcels)
    # lstsq solves M (P,K) @ x (K,T) = X_full.T (P,T)  =>  x = TC.T
    tc_ref = np.linalg.lstsq(maps, X_full.T, rcond=None)[0].T
    np.testing.assert_allclose(tc, tc_ref, atol=1e-8)


def test_fo_per_run_counts_fractions():
    # run 0: labels [0,0,1] -> fo {0:2/3, 1:1/3}; run 1: [2,2] -> fo {2:1}
    labels = np.array([0, 0, 1, 2, 2])
    run_boundaries = [(0, 3), (3, 5)]
    fo = fo_per_run(labels, run_boundaries, n_components=3)
    np.testing.assert_allclose(fo[0], [2/3, 1/3, 0.0])
    np.testing.assert_allclose(fo[1], [0.0, 0.0, 1.0])


def test_recurrence_scores_fraction_of_active_runs():
    # comp 0 active (>0.02) in both runs -> 1.0; comp 1 in run0 only -> 0.5;
    # comp 2 in run1 only -> 0.5
    fo = {0: np.array([0.6, 0.4, 0.0]), 1: np.array([0.5, 0.0, 0.5])}
    rec = recurrence_scores(fo, n_components=3, fo_threshold=0.02)
    np.testing.assert_allclose(rec, [1.0, 0.5, 0.5])


def test_recurrence_scores_threshold_excludes_below():
    fo = {0: np.array([0.01, 0.99]), 1: np.array([0.03, 0.97])}
    rec = recurrence_scores(fo, n_components=2, fo_threshold=0.02)
    # comp 0: 0.01 not >0.02 in run0, not >0.02... run1 0.03>0.02 -> 1/2=0.5
    np.testing.assert_allclose(rec, [0.5, 1.0])


def test_continuous_occupancy_zero_variance_component_gets_zero_share():
    rng = np.random.default_rng(2)
    # Two active components + one constant (zero-variance) component.
    # continuous_occupancy z-scores each component per run, which sends the
    # constant component to 0 -> ~0 occupancy share. (Amplitude scale is
    # deliberately removed by z-scoring, so it is NOT a valid ordering signal.)
    tc = np.column_stack([
        rng.standard_normal(40),
        rng.standard_normal(40),
        np.ones(40),
    ])
    occ = continuous_occupancy(tc, [(0, 20), (20, 40)])
    assert occ.shape == (3,)
    np.testing.assert_allclose(occ.sum(), 1.0, atol=1e-10)
    assert occ[2] < 1e-9
    assert occ[0] > occ[2] and occ[1] > occ[2]


# ---------------------------------------------------------------------------
# Data-gated integration tests for sm_alt_ica_oos_recurrence (Task 3)
# ---------------------------------------------------------------------------
import os
import json
from pathlib import Path

from utils.ica_states import consensus_projection

_SCRATCH = os.getenv("SCRATCH_DIR")
_SUB = "sub-01"
_PARC = "atlas-4S156Parcels"
_VT = 0.95


def _movie_dir():
    if _SCRATCH is None:
        return None
    return Path(_SCRATCH) / "output" / "m10_03_projected" / _PARC / _SUB / f"vt{_VT}"


def _ica_dir():
    if _SCRATCH is None:
        return None
    return Path(_SCRATCH) / "output" / "sm_ica_states" / _PARC / _SUB


def _data_ready():
    """Return True iff SCRATCH_DIR is set and the required files exist on disk."""
    if not _SCRATCH:
        return False
    md = _movie_dir()
    id_ = _ica_dir()
    if md is None or id_ is None:
        return False
    ica_maps = id_ / "ica_maps_K42.npy"
    ica_tc = id_ / "ica_timecourses_K42.npy"
    movie_ids = md / "movie_run_ids.json"
    return (ica_maps.exists() and ica_maps.stat().st_size > 0
            and ica_tc.exists() and ica_tc.stat().st_size > 0
            and movie_ids.exists() and movie_ids.stat().st_size > 0)


_SKIP_REASON = "SCRATCH_DIR not set or required sub-01 ICA/movie data not materialized"


def test_wta_labels_reexported_from_ica_oos_recurrence():
    from utils.ica_oos_recurrence import wta_labels as wl_reexport
    from utils.ica_states import wta_labels as wl_source
    assert wl_reexport is wl_source


@pytest.mark.skipif(not _data_ready(), reason=_SKIP_REASON)
def test_friends_projection_reproduces_saved_timecourses():
    """Recompute ICA timecourses from frozen components + saved consensus maps,
    then assert they exactly match the saved ica_timecourses_K42.npy."""
    from sm_alt_ica_oos_recurrence import load_friends_inputs
    inp = load_friends_inputs(_SUB, _PARC, _VT)
    proj = consensus_projection(inp["components"], inp["consensus_maps"])
    recomputed = inp["X_pc"] @ proj
    saved = np.load(_ica_dir() / "ica_timecourses_K42.npy")
    np.testing.assert_allclose(recomputed, saved, atol=1e-8)


@pytest.mark.skipif(not _data_ready(), reason=_SKIP_REASON)
def test_run_subject_emits_well_formed_summary(tmp_path):
    """run_subject must write a JSON with the expected keys/shapes and valid rho."""
    from sm_alt_ica_oos_recurrence import run_subject
    out = run_subject(_SUB, _PARC, _VT, stimulus="movie10",
                      fo_threshold=0.02, out_dir=str(tmp_path))
    assert out["sub_id"] == _SUB
    assert out["K_active"] == 42
    assert len(out["friends_recurrence"]) == out["n_components"]
    assert len(out["movie_occupancy_wta"]) == out["n_components"]
    assert len(out["movie_occupancy_continuous"]) == out["n_components"]
    for arm in ("wta", "continuous"):
        assert -1.0 <= out["overall"][arm]["rho"] <= 1.0
        assert out["overall"][arm]["n"] == out["n_components"]
    assert set(out["per_film"]).issubset({"bourne", "wolf", "figures", "life"})
    assert (tmp_path / "oos_recurrence_summary.json").exists()


# ---------------------------------------------------------------------------
# Phase-randomize unit tests (Task 3 additions)
# ---------------------------------------------------------------------------

def test_phase_randomize_shape_and_real():
    """phase_randomize returns a real array with the same shape as the input."""
    from utils.ica_oos_recurrence import phase_randomize
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    surr = phase_randomize(X, rng)
    assert surr.shape == X.shape
    assert np.isrealobj(surr)


def test_phase_randomize_preserves_power_spectrum():
    """Surrogate must preserve each column's power spectrum to ~1e-8."""
    from utils.ica_oos_recurrence import phase_randomize
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 5))
    surr = phase_randomize(X, np.random.default_rng(99))
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(surr, axis=0)),
        np.abs(np.fft.rfft(X, axis=0)),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# NaN-guard unit tests: json.dump(allow_nan=False) must not crash when a
# Spearman input is constant (rho/p NaN) or the null sd is 0 (z NaN).
# ---------------------------------------------------------------------------

def test_spearman_constant_input_is_json_safe():
    # Spearman on a constant x -> NaN rho/p; _spearman must surface JSON-null
    # so json.dump(allow_nan=False) does not raise.
    from sm_alt_ica_oos_recurrence import _spearman
    with pytest.warns(Warning):  # scipy ConstantInputWarning is expected here
        res = _spearman([1.0, 1.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4])
    assert res["rho"] is None and res["p"] is None
    assert res["n"] == 4
    json.dumps(res, allow_nan=False)  # must not raise


def test_null_summary_zero_sd_is_json_safe():
    # Degenerate null (sd == 0) -> z NaN; _null_summary must surface JSON-null.
    from sm_alt_ica_oos_recurrence import _null_summary
    res = _null_summary(0.3, np.array([0.3, 0.3, 0.3]))
    assert res["z"] is None
    assert res["residual"] == 0.0
    assert res["n_draws"] == 3
    json.dumps(res, allow_nan=False)  # must not raise


@pytest.mark.skipif(not _data_ready(), reason=_SKIP_REASON)
def test_run_subject_null_distribution(tmp_path):
    """run_subject with n_null=20 must populate null sub-dicts and satisfy
    ordering: real > null mean (z>0, residual>0)."""
    from sm_alt_ica_oos_recurrence import run_subject
    out = run_subject(_SUB, _PARC, _VT, stimulus="movie10",
                      fo_threshold=0.02, out_dir=str(tmp_path), n_null=20)
    for arm in ("wta", "continuous"):
        null = out["overall"][arm].get("null")
        assert null is not None, f"missing null key for arm={arm}"
        for k in ("mean", "sd", "z", "p", "n_draws", "residual"):
            assert k in null, f"missing key {k} in null for arm={arm}"
        assert null["n_draws"] == 20
        assert null["residual"] > 0, f"arm={arm}: real should exceed null mean"
        assert null["z"] > 0, f"arm={arm}: z-score should be positive"
    rho_marg = out.get("recurrence_vs_friends_marginal_wta_rho")
    assert rho_marg is not None
    assert 0.9 < rho_marg <= 1.0, f"marginal rho={rho_marg} expected > 0.9"
