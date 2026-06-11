#!/usr/bin/env python3
"""Regression test: int8 vs int64 label dtype in batch_loro_ridge_classify.

The 08d D1 + D1confound pipeline allocates several (n_perm, n_eligible)
label arrays inside ``batch_loro_ridge_classify`` (transformer_analysis.py).
Casting state labels from int64 to int8 (HMM states fit comfortably in
[0, 49]) saves ~8× memory on those arrays, but only if the cast is safe —
i.e. the function produces bit-identical observed metrics and
null_balanced_accuracies regardless of integer dtype.

This test locks in that invariant. Run standalone::

    python script/tests/test_batch_ridge_label_dtype.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from utils.transformer_analysis import (  # noqa: E402
    batch_loro_ridge_classify,
    precompute_eligible_null_state_sequences,
)


def _build_synthetic(seed=0):
    rng = np.random.default_rng(seed)

    n_runs = 5
    n_per_run = 40
    n_total = n_runs * n_per_run
    n_classes = 4

    states = rng.integers(0, n_classes, size=n_total)
    eligible_mask = np.ones(n_total, dtype=bool)

    run_boundaries = [
        (i * n_per_run, (i + 1) * n_per_run) for i in range(n_runs)
    ]

    n_features = 8
    class_means = rng.standard_normal((n_classes, n_features)) * 1.5
    X = class_means[states] + rng.standard_normal((n_total, n_features)) * 0.6

    folds = []
    for held_out in range(n_runs):
        test_idx = np.arange(
            held_out * n_per_run, (held_out + 1) * n_per_run,
        )
        train_idx = np.setdiff1d(np.arange(n_total), test_idx)
        folds.append((train_idx, test_idx))

    return X, states, eligible_mask, run_boundaries, folds


def _run(states_dtype, *, n_perm=15, seed=0):
    X, states, eligible_mask, run_boundaries, folds = _build_synthetic(seed=seed)
    states_typed = states.astype(states_dtype)

    null_y = precompute_eligible_null_state_sequences(
        states_typed, run_boundaries, eligible_mask, n_perm, rng_seed=42,
    )
    assert null_y.dtype == states_typed.dtype, (
        f"null preserved dtype expected {states_typed.dtype}, got {null_y.dtype}"
    )

    result = batch_loro_ridge_classify(
        X[eligible_mask], states_typed[eligible_mask], null_y, folds,
    )
    return result


def test_int8_matches_int64():
    res64 = _run(np.int64)
    res8 = _run(np.int8)

    for k in ("balanced_accuracy", "weighted_f1", "cohen_kappa"):
        v64 = res64["observed"][k]
        v8 = res8["observed"][k]
        assert np.isclose(v64, v8, atol=1e-10), (
            f"observed[{k!r}] mismatch: int64={v64} int8={v8}"
        )

    n64 = res64["null_balanced_accuracies"]
    n8 = res8["null_balanced_accuracies"]
    assert len(n64) == len(n8), (
        f"null length mismatch: int64={len(n64)} int8={len(n8)}"
    )
    for i, (a, b) in enumerate(zip(n64, n8)):
        assert np.isclose(a, b, atol=1e-10), (
            f"null[{i}] mismatch: int64={a} int8={b}"
        )

    print(
        f"  observed balanced_accuracy: {res64['observed']['balanced_accuracy']:.4f} "
        f"(int8 == int64)"
    )
    print(f"  null_balanced_accuracies: {len(n64)} perms, all match")
    print("  PASS test_int8_matches_int64")


def test_overflow_guard():
    """If state IDs exceed int8 max, the function must fail loud, not silently
    truncate. Currently we accept ANY integer dtype the caller provides; the
    guard fires when ``classes.max()`` would exceed the input dtype's iinfo
    max — but with int64 input there's no such constraint to trip. We check
    instead that an artificially overflowed int8 input (created by silent
    casting from a too-large int) is REJECTED before producing nonsense.
    """
    X, states, eligible_mask, run_boundaries, folds = _build_synthetic(seed=1)

    states_bad = states.copy()
    states_bad[0] = 200
    silently_truncated = states_bad.astype(np.int8)
    assert silently_truncated[0] != 200, (
        "test setup: expected silent truncation to be observable"
    )

    null_y = precompute_eligible_null_state_sequences(
        silently_truncated, run_boundaries, eligible_mask, 5, rng_seed=42,
    )

    try:
        batch_loro_ridge_classify(
            X[eligible_mask], silently_truncated[eligible_mask], null_y, folds,
        )
    except (AssertionError, ValueError) as e:
        print(f"  PASS test_overflow_guard — rejected with: {type(e).__name__}: {e}")
        return
    raise AssertionError(
        "expected batch_loro_ridge_classify to reject silently-truncated "
        "int8 labels via an assert/ValueError, but it did not"
    )


if __name__ == "__main__":
    test_int8_matches_int64()
    test_overflow_guard()
    print("\nAll tests passed.")
