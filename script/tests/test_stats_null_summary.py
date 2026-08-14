"""Tests for utils.stats.safe_float / null_summary (NaN-safe null summaries)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.stats import null_summary, permutation_pvalue, safe_float  # noqa: E402


def test_safe_float_passthrough_and_none():
    assert safe_float(0.5) == 0.5
    assert safe_float(np.float64(-1.25)) == -1.25
    assert safe_float(float("nan")) is None
    assert safe_float(float("inf")) is None


def test_null_summary_matches_inline_formulas_on_finite_draws():
    rng = np.random.default_rng(0)
    null = rng.standard_normal(1000)
    obs = 2.0
    ns = null_summary(obs, null)
    # exact match with the formulas the published run used (all-finite case)
    assert ns["mean"] == pytest.approx(float(null.mean()))
    assert ns["sd"] == pytest.approx(float(null.std()))  # ddof=0, matches published
    assert ns["z"] == pytest.approx((obs - null.mean()) / null.std())
    assert ns["p"] == (1 + np.sum(null >= obs)) / (1 + len(null))
    assert ns["residual"] == pytest.approx(obs - float(null.mean()))
    assert ns["n_draws"] == 1000 and ns["n_finite"] == 1000


def test_null_summary_excludes_nan_draws_from_p_and_moments():
    null = np.array([0.1, 0.2, np.nan, 0.3, np.nan])
    ns = null_summary(0.25, null)
    assert ns["n_draws"] == 5 and ns["n_finite"] == 3
    # p over finite draws only: (1 + #{finite >= 0.25}) / (1 + 3)
    assert ns["p"] == pytest.approx(2 / 4)
    assert ns["mean"] == pytest.approx(0.2)


def test_null_summary_degenerate_cases_are_none_not_nan():
    ns = null_summary(0.5, np.full(100, 0.3))       # sd == 0
    assert ns["z"] is None and ns["sd"] == 0.0
    ns = null_summary(float("nan"), np.array([0.1, 0.2]))  # NaN observed
    assert ns["p"] is None and ns["z"] is None and ns["residual"] is None
    ns = null_summary(0.5, np.full(3, np.nan))      # all-NaN null
    assert ns["p"] is None and ns["mean"] is None and ns["n_finite"] == 0
