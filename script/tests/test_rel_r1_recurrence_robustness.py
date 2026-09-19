"""Unit tests for sm_rel_r1_recurrence_robustness.

Load-bearing claims: (a) recurrence_at reproduces the main pipeline's
definition (fraction of runs with FO strictly above the threshold);
(b) recurrence is non-increasing in the threshold; (c) churn counts states
that cross the active (>0) and eligibility (>=0.10) lines; (d) the category
mapping for flag-free states depends on recurrence alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import sm_rel_r1_recurrence_robustness as m


def _toy_fo():
    # 3 runs x 4 states
    return {"r1": np.array([0.50, 0.015, 0.00, 0.435]),
            "r2": np.array([0.60, 0.025, 0.00, 0.375]),
            "r3": np.array([0.70, 0.000, 0.03, 0.270])}


def test_recurrence_at_counts_runs_strictly_above_threshold():
    rec = m.recurrence_at(_toy_fo(), 4, 0.02)
    np.testing.assert_allclose(rec, [1.0, 1 / 3, 1 / 3, 1.0])


def test_recurrence_at_excludes_runs_exactly_at_threshold():
    fo = {"r1": np.array([0.02, 0.0201, 0.0199]),
          "r2": np.array([0.02, 0.02, 0.02])}
    np.testing.assert_allclose(m.recurrence_at(fo, 3, 0.02), [0.0, 0.5, 0.0])


def test_recurrence_at_empty_fo_returns_zeros():
    np.testing.assert_array_equal(m.recurrence_at({}, 4, 0.02), np.zeros(4))


def test_recurrence_is_non_increasing_in_threshold():
    fo = _toy_fo()
    prev = m.recurrence_at(fo, 4, 0.0)
    for t in m.THRESHOLDS:
        cur = m.recurrence_at(fo, 4, t)
        assert np.all(cur <= prev + 1e-12)
        prev = cur


def test_mean_fo_averages_runs_equally():
    np.testing.assert_allclose(m.mean_fo(_toy_fo()), [0.6, 0.04 / 3, 0.01, 0.36])


def test_recurrence_category_thresholds():
    assert m.recurrence_category(0.0) == "unused"
    assert m.recurrence_category(0.05) == "rare"
    assert m.recurrence_category(0.10) == "eligible_for_content_analysis"


def test_churn_counts_crossings_only_among_flag_free_states():
    rec_ref = np.array([0.5, 0.09, 0.0, 0.12])
    rec_alt = np.array([0.5, 0.11, 0.02, 0.08])
    flag_free = np.array([True, True, True, False])   # state 3 is flagged (e.g. run-onset)
    c = m.churn(rec_ref, rec_alt, flag_free)
    assert c["active_changed"] == 1                     # state 2: 0 -> 0.02
    assert c["eligible_changed"] == 1                   # state 1 crosses 0.10; state 3 ignored (flagged)
    assert c["gained_eligible"] == [1] and c["lost_eligible"] == []
    assert c["n_eligible_ref"] == 1 and c["n_eligible_alt"] == 2


def test_churn_active_line_counts_flagged_states_but_eligibility_does_not():
    rec_ref = np.array([0.5, 0.0, 0.3])
    rec_alt = np.array([0.5, 0.02, 0.0])
    flag_free = np.array([True, False, False])   # states 1 and 2 are flagged
    c = m.churn(rec_ref, rec_alt, flag_free)
    assert c["active_changed"] == 2                # state 1 becomes active, state 2 becomes inactive
    assert c["eligible_changed"] == 0              # flagged states never enter the eligibility count
    assert c["n_eligible_ref"] == 1 and c["n_eligible_alt"] == 1


def test_rank_stability_over_reference_active_set():
    rec_ref = np.array([0.9, 0.5, 0.1, 0.05, 0.0])
    rec_alt = np.array([0.8, 0.6, 0.02, 0.0, 0.0])      # state 3 drops out at the higher threshold
    s = m.rank_stability(rec_ref, rec_alt)
    assert s["n_reference_active"] == 4 and s["n_both_active"] == 3
    assert s["rho_reference_active"] == pytest.approx(1.0)   # ordering 0 > 1 > 2 > 3 preserved
    assert s["rho_both_active"] == pytest.approx(1.0)


def test_rank_stability_returns_none_below_three_states():
    s = m.rank_stability(np.array([0.5, 0.2, 0.0]), np.array([0.4, 0.0, 0.0]))
    assert s["rho_both_active"] is None and s["n_both_active"] == 1
    assert s["rho_reference_active"] is None and s["n_reference_active"] == 2


def test_range_stats_uses_active_states_only():
    r = m.range_stats(np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
    assert r["n_active"] == 4 and r["min"] == 0.2 and r["max"] == 0.8
    assert r["p10"] == pytest.approx(np.percentile([0.2, 0.4, 0.6, 0.8], 10))


def test_range_stats_all_zero_returns_none_fields():
    r = m.range_stats(np.zeros(5))
    assert r == {"n_active": 0, "min": None, "max": None, "p10": None, "p90": None}
