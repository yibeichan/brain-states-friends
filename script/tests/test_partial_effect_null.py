#!/usr/bin/env python3
"""Sanity tests for the C4 partial-effect helpers used in 08b / 08d.

Runs standalone without pytest::

    python script/tests/test_partial_effect_null.py

Covers:

1. ``partial_effect_residualize`` removes a known polynomial drift while
   preserving an orthogonal state-associated signal.
2. Under the null hypothesis (no state-content association) with variable
   epoch counts across permutations, the per-permutation residualize-then-KW
   procedure recovers the correct Type I error rate (~0.05 at alpha=0.05).
   This is the sanity check called out by the Stats re-review (§8.1b of the
   2026-04-20 neg-control redesign doc).

Both tests are synthetic-only; no dependency on project data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

# Make project utils importable when running from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from utils.transformer_analysis import (  # noqa: E402
    build_epoch_run_position_design,
    partial_effect_residualize,
)

# AUC primitives live in utils/stats.py (generic). 08b-specific lag/residualize
# helpers are imported via importlib since the 08b script isn't a package.
from utils.stats import (  # noqa: E402
    per_state_auc_mann_whitney as _per_state_auc,
    per_state_auc_grid as _compute_auc_grid,
)

import importlib.util as _importlib_util  # noqa: E402

_spec = _importlib_util.spec_from_file_location(
    "_m08b",
    os.path.abspath(os.path.join(_HERE, "..", "08b_content_state_correspondence.py")),
)
_m08b = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_m08b)
_epoch_feats_at_lag = _m08b._epoch_feats_at_lag
_maybe_residualize = _m08b._maybe_residualize


def _fake_records(n_epochs, rng):
    """Build synthetic block records with positions spread across runs."""
    n_runs = 6
    records = []
    per_run = n_epochs // n_runs
    for r in range(n_runs):
        run_len = 460
        # Sort positions within run so the per-epoch t_bar is monotone; the
        # design matrix still exercises all three polynomial terms.
        positions = np.sort(rng.integers(0, run_len - 10, size=per_run))
        for p in positions:
            records.append({
                "start_tr": int(p),
                "end_tr": int(p) + 5,
                "episode_length_tr": run_len,
            })
    return records


def test_residualize_removes_polynomial_drift():
    rng = np.random.default_rng(42)
    records = _fake_records(120, rng)
    D = build_epoch_run_position_design(records, degree=3)

    # y = strong linear drift in t_bar + tiny noise — no state signal.
    t_bar = D[:, 1]
    y = 2.0 * t_bar + 0.05 * rng.standard_normal(len(t_bar))

    y_resid = partial_effect_residualize(y, D)

    # Residuals should have near-zero mean and near-zero linear correlation
    # with t_bar (OLS projection removes the drift exactly up to noise).
    assert np.abs(np.mean(y_resid)) < 0.05, f"residual mean {np.mean(y_resid):.3f}"
    r_before = np.corrcoef(t_bar, y)[0, 1]
    r_after = np.corrcoef(t_bar, y_resid)[0, 1]
    assert abs(r_before) > 0.9, f"setup failed: pre-residual r={r_before:.3f}"
    assert abs(r_after) < 0.05, f"post-residual r={r_after:.3f} (drift not removed)"

    print(f"  drift removal: r(t_bar,y) {r_before:+.3f} -> {r_after:+.3f}  OK")


def test_residualize_preserves_orthogonal_state_signal():
    rng = np.random.default_rng(7)
    records = _fake_records(240, rng)
    D = build_epoch_run_position_design(records, degree=3)
    t_bar = D[:, 1]

    # Assign states independent of t_bar (random categorical).
    state = rng.integers(0, 4, size=len(records))

    # Signal: small per-state offset + small cubic drift.
    per_state_offset = np.array([-0.6, -0.2, 0.2, 0.6])
    y = (
        per_state_offset[state]
        + 0.6 * t_bar ** 3
        + 0.1 * rng.standard_normal(len(records))
    )

    # KW on raw y should already find the state effect (because state is
    # orthogonal to t_bar here); residualization shouldn't destroy it.
    def kw_eps(values):
        groups = [values[state == s] for s in range(4)]
        h, _ = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        return float(h / max(n - 1, 1))

    eps_raw = kw_eps(y)
    eps_resid = kw_eps(partial_effect_residualize(y, D))
    # They should be of the same order; residualization may even strengthen
    # the state effect by cleaning drift.
    assert eps_resid > 0.5 * eps_raw, (
        f"residualization gutted the real signal: eps_raw={eps_raw:.4f} "
        f"eps_resid={eps_resid:.4f}"
    )
    print(f"  state signal: eps_raw={eps_raw:.4f}  eps_resid={eps_resid:.4f}  OK")


def test_variable_n_null_type_i_error():
    """Under H0 (no state-feature association), the per-permutation residualize-
    then-KW procedure should have ~5% rejection at alpha=0.05 even when epoch
    counts vary across permutations. We approximate the permutation by
    shuffling state labels within each simulated dataset.
    """
    rng = np.random.default_rng(2024)
    alpha = 0.05
    n_datasets = 300
    rejection_count = 0

    for sim in range(n_datasets):
        # Each dataset has a slightly different epoch count (variable N).
        n_ep = int(rng.integers(100, 160))
        records = _fake_records(n_ep, rng)
        if len(records) < 20:
            continue
        D = build_epoch_run_position_design(records, degree=3)
        t_bar = D[:, 1]

        # Pure-drift feature (no state signal).
        y = 0.8 * t_bar ** 3 + 0.15 * rng.standard_normal(len(records))
        state = rng.integers(0, 4, size=len(records))

        # Observed KW on residualized y.
        y_r = partial_effect_residualize(y, D)
        groups_obs = [y_r[state == s] for s in range(4)]
        if any(len(g) < 2 for g in groups_obs):
            continue
        h_obs, _ = stats.kruskal(*groups_obs)

        # Null: shuffle state labels many times, compute KW on residualized y
        # (residualize IS rebuilt per-permutation in the real pipeline because
        # block boundaries change with the TR-level shift; here the design D
        # is fixed because we only permute state labels — a simplified
        # sanity check. The boundary-shifting case is exercised in the full
        # 08b A3 path and validated there on real data.)
        n_perm = 200
        null_h = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(state)
            groups_null = [y_r[perm == s] for s in range(4)]
            if any(len(g) < 2 for g in groups_null):
                null_h[i] = np.nan
                continue
            h_null, _ = stats.kruskal(*groups_null)
            null_h[i] = h_null
        null_h = null_h[np.isfinite(null_h)]
        if len(null_h) < 20:
            continue

        # p = (1 + # null >= obs) / (1 + len(null))
        p = (1 + np.sum(null_h >= h_obs)) / (1 + len(null_h))
        if p < alpha:
            rejection_count += 1

    rate = rejection_count / n_datasets
    # At alpha=0.05 with n_datasets=300, a 2-SD band is ~0.025. Allow [0.02, 0.10].
    assert 0.02 <= rate <= 0.10, (
        f"Type I error rate {rate:.3f} outside expected [0.02, 0.10] at alpha=0.05"
    )
    print(f"  Type I error rate: {rate:.3f} (expected ~0.05)  OK")


# ── 2026-04-23 per-state redesign tests (§6.5 of the design doc) ─────────────


def test_residualize_does_not_mutate_input():
    """Coding V2: partial_effect_residualize must not mutate caller's arrays."""
    rng = np.random.default_rng(123)
    records = _fake_records(60, rng)
    D = build_epoch_run_position_design(records, degree=3)
    y = rng.standard_normal((len(records), 4))
    y_snapshot = y.copy()
    D_snapshot = D.copy()

    _ = partial_effect_residualize(y, D)
    _ = build_epoch_run_position_design(records, degree=3)  # must not mutate records

    assert np.array_equal(y, y_snapshot), "partial_effect_residualize mutated y"
    assert np.array_equal(D, D_snapshot), "partial_effect_residualize mutated D"
    print("  inputs unchanged after residualize call  OK")


def test_per_state_auc_null_type_i():
    """Test A (§6.5): per-state AUC Type I error rate under H_0 ~ 0.05 (tolerance to 0.1)."""
    rng = np.random.default_rng(321)
    alpha = 0.05
    n_datasets = 200
    reject = 0
    for _ in range(n_datasets):
        n_ep = int(rng.integers(120, 200))
        state_id = 0
        mask = rng.random(n_ep) < 0.15  # ~15% of epochs in state 0
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        y = rng.standard_normal(n_ep)  # no association
        # Permutation null: shuffle state indicator
        obs_auc, _, n1, n0 = _per_state_auc(mask, y)
        if not np.isfinite(obs_auc):
            continue
        null = np.empty(200)
        for i in range(200):
            perm_mask = rng.permutation(mask)
            auc_p, _, _, _ = _per_state_auc(perm_mask, y)
            null[i] = auc_p
        valid = np.isfinite(null)
        p = (1 + np.sum(np.abs(null[valid] - 0.5) >= abs(obs_auc - 0.5))) / (1 + valid.sum())
        if p < alpha:
            reject += 1
    rate = reject / n_datasets
    assert 0.01 <= rate <= 0.12, f"Type I error rate {rate:.3f} outside [0.01, 0.12]"
    print(f"  per-state AUC Type I rate: {rate:.3f} (expected ~0.05)  OK")


def test_per_state_auc_recovery():
    """Test B (§6.5): per-state AUC recovers a planted state-feature association."""
    rng = np.random.default_rng(7)
    n_ep = 400
    # Plant state 2 as high-feature. Other states independent.
    state = rng.integers(0, 5, size=n_ep)
    y = rng.standard_normal(n_ep)
    y[state == 2] += 1.5  # large planted effect

    target_states = list(range(5))
    feats_mat = y.reshape(-1, 1)
    aucs, signs = _compute_auc_grid(state, feats_mat, target_states, compute_signs=True)

    # State 2's AUC should be dramatically > 0.5. Other-state AUCs may drift
    # below 0.5 because their "other" group contains state 2's elevated epochs —
    # that's the expected behavior of a two-sample per-state AUC against the
    # complement. The substantive contrast: state 2 >> every other state.
    assert aucs[2, 0] > 0.7, f"planted state AUC too low: {aucs[2, 0]:.3f}"
    assert signs[2, 0] == 1, "sign should be positive"
    others = np.delete(aucs[:, 0], 2)
    assert np.all(aucs[2, 0] - others > 0.15), (
        f"planted state AUC not well-separated from others: "
        f"planted={aucs[2, 0]:.3f} others={others}"
    )
    print(f"  planted-state AUC={aucs[2, 0]:.3f} >> others {others}  OK")


def test_per_lag_residualization():
    """Test C (§6.5): partial-mode residualization removes known position drift
    without destroying a planted per-state association."""
    rng = np.random.default_rng(2025)
    records = _fake_records(240, rng)
    n = len(records)
    state = rng.integers(0, 4, size=n)
    D = build_epoch_run_position_design(records, degree=3)
    t_bar = D[:, 1]

    # Signal: strong drift + moderate per-state offset on state 3.
    per_state_offset = np.zeros(n)
    per_state_offset[state == 3] = 0.8
    y = per_state_offset + 1.5 * t_bar + 0.2 * rng.standard_normal(n)
    feats_mat = y.reshape(-1, 1)

    # Without residualization: AUC(state=3) > 0.5 but also other states inflated
    # by drift correlation with their non-uniform position coverage.
    aucs_raw, _ = _compute_auc_grid(state, feats_mat, [0, 1, 2, 3], compute_signs=False)

    # With C4 partial mode: drift removed, planted state signal preserved.
    feats_resid = _maybe_residualize(feats_mat.copy(), records, "partial")
    aucs_resid, _ = _compute_auc_grid(state, feats_resid, [0, 1, 2, 3], compute_signs=False)

    assert aucs_resid[3, 0] > 0.6, (
        f"planted state signal gutted by residualization: "
        f"raw={aucs_raw[3, 0]:.3f} resid={aucs_resid[3, 0]:.3f}"
    )
    # For non-planted states, post-residualization AUC should be closer to 0.5.
    for s in [0, 1, 2]:
        # Weak — just confirm no degeneration far from 0.5.
        assert abs(aucs_resid[s, 0] - 0.5) < 0.2
    print(
        f"  planted state: raw AUC={aucs_raw[3, 0]:.3f} "
        f"resid AUC={aucs_resid[3, 0]:.3f}  OK"
    )


def test_mode_symmetry():
    """Raw and partial must be identical on a no-drift feature (residualization
    against a polynomial that explains zero variance is a near-identity map)."""
    rng = np.random.default_rng(11)
    records = _fake_records(180, rng)
    n = len(records)
    state = rng.integers(0, 4, size=n)
    y = rng.standard_normal(n)  # no drift
    feats_mat = y.reshape(-1, 1)

    aucs_raw, _ = _compute_auc_grid(state, feats_mat, [0, 1, 2, 3], compute_signs=False)
    y_resid = _maybe_residualize(feats_mat.copy(), records, "partial")
    aucs_resid, _ = _compute_auc_grid(
        state, y_resid, [0, 1, 2, 3], compute_signs=False,
    )

    # AUCs should agree to within 0.05 (tiny finite-sample drift removed).
    diffs = np.abs(aucs_raw - aucs_resid)
    assert np.all(diffs[np.isfinite(diffs)] < 0.05), (
        f"raw vs partial mode diverged on no-drift data: max diff {np.nanmax(diffs):.3f}"
    )
    print(f"  raw vs partial max |ΔAUC| = {np.nanmax(diffs):.4f}  OK")


if __name__ == "__main__":
    print("test_residualize_removes_polynomial_drift")
    test_residualize_removes_polynomial_drift()
    print("test_residualize_preserves_orthogonal_state_signal")
    test_residualize_preserves_orthogonal_state_signal()
    print("test_variable_n_null_type_i_error")
    test_variable_n_null_type_i_error()
    print("test_residualize_does_not_mutate_input")
    test_residualize_does_not_mutate_input()
    print("test_per_state_auc_null_type_i")
    test_per_state_auc_null_type_i()
    print("test_per_state_auc_recovery")
    test_per_state_auc_recovery()
    print("test_per_lag_residualization")
    test_per_lag_residualization()
    print("test_mode_symmetry")
    test_mode_symmetry()
    print("\nAll tests passed.")
