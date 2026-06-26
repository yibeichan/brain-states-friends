import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils.ica_states import (
    wta_labels,
    consensus_projection,
    icasso_consensus,
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


def test_consensus_projection_reproduces_icasso_timecourses():
    # consensus_projection(components, maps) @ X_pc must equal the timecourses
    # icasso_consensus returns (same formula, single source of truth).
    rng = np.random.default_rng(1)
    n_pcs, n_parcels, T = 8, 20, 60
    components = rng.standard_normal((n_pcs, n_parcels))
    X_pc = rng.standard_normal((T, n_pcs))
    out = icasso_consensus(components, X_pc, n_components=4, n_restarts=3,
                           rng_seed=0, max_iter=200)
    proj = consensus_projection(components, out["consensus_maps"])
    np.testing.assert_allclose(X_pc @ proj, out["timecourses"], atol=1e-10)


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


def test_continuous_occupancy_sums_to_one_and_orders():
    rng = np.random.default_rng(2)
    # comp 0 has the largest magnitude swings -> largest share
    tc = np.column_stack([
        rng.standard_normal(50) * 5.0,
        rng.standard_normal(50) * 1.0,
        rng.standard_normal(50) * 0.5,
    ])
    occ = continuous_occupancy(tc, [(0, 25), (25, 50)])
    assert occ.shape == (3,)
    np.testing.assert_allclose(occ.sum(), 1.0, atol=1e-10)
    assert occ[0] > occ[1] > occ[2]
