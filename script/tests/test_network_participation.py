"""Tests for the shared network-participation utility.

The same metric implementation backs the content-eligible Figure 2C and the
supplementary all-category figure, so these tests pin both the metric math and
the category-selection behavior.
"""

import numpy as np
import pandas as pd

from utils.network_participation import (
    ELIGIBLE_CATEGORY,
    compute_network_participation_metrics,
    network_composition_for_state,
    summarize_network_participation,
)


NETWORKS = ["A", "B", "C", "D"]
PARCEL_NETWORKS = ["A", "A", "B", "B", "C", "C", "D", "D"]


def _flags(states, categories, **extra):
    data = {"state": states, "summary_category": categories}
    data.update(extra)
    return pd.DataFrame(data)


# ── composition math ─────────────────────────────────────────────────────────

def test_network_composition_uses_mean_absolute_loading_and_sums_to_one():
    state = np.array([2.0, -2.0, 1.0, 1.0, 0.5, -0.5, 0.0, 0.0])

    composition = network_composition_for_state(state, PARCEL_NETWORKS, NETWORKS)

    np.testing.assert_allclose(composition.sum(), 1.0)
    np.testing.assert_allclose(composition, [2 / 3.5, 1 / 3.5, 0.5 / 3.5, 0.0])


def test_network_share_columns_sum_to_one_per_state():
    flags = {
        "sub-01": _flags(
            [0, 1],
            [ELIGIBLE_CATEGORY, ELIGIBLE_CATEGORY],
            recurrence_score=[0.9, 0.7],
            dominant_network=["A", "D"],
        )
    }
    state_means = {
        "sub-01": np.array(
            [
                [2.0, -2.0, 1.0, 1.0, 0.5, -0.5, 0.3, -0.3],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0, -7.0],
            ]
        )
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    share_cols = [f"network_share_{n}" for n in NETWORKS]
    row_sums = metrics[share_cols].sum(axis=1).values
    np.testing.assert_allclose(row_sums, np.ones(len(metrics)))


def test_concentrated_state_has_lower_entropy_than_distributed_state():
    flags = {
        "sub-01": _flags([0, 1], [ELIGIBLE_CATEGORY, ELIGIBLE_CATEGORY])
    }
    state_means = {
        "sub-01": np.array(
            [
                [10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # concentrated -> A
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # distributed -> uniform
            ]
        )
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    concentrated = metrics.iloc[0]["normalized_network_entropy"]
    distributed = metrics.iloc[1]["normalized_network_entropy"]
    assert concentrated < distributed
    np.testing.assert_allclose(concentrated, 0.0)
    np.testing.assert_allclose(distributed, 1.0)  # uniform over all 4 networks


def test_n_networks_threshold_is_greater_or_equal_10pct():
    # Scores per network: A=9, B=1, C=0, D=0 -> shares [0.9, 0.1, 0, 0].
    # The B share is exactly 0.10 and must be counted (>=, not >).
    flags = {"sub-01": _flags([0], [ELIGIBLE_CATEGORY])}
    state_means = {
        "sub-01": np.array([[9.0, 9.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    row = metrics.iloc[0]
    np.testing.assert_allclose(row["network_share_B"], 0.10)
    assert row["n_networks_ge_10pct"] == 2  # A (0.9) and B (exactly 0.10)


def test_topk_shares_and_networks():
    flags = {"sub-01": _flags([0], [ELIGIBLE_CATEGORY])}
    state_means = {
        "sub-01": np.array([[2.0, -2.0, 1.0, 1.0, 0.5, -0.5, 0.0, 0.0]])
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    row = metrics.iloc[0]
    assert row["top1_network"] == "A"
    np.testing.assert_allclose(row["top1_share"], 2 / 3.5)
    np.testing.assert_allclose(row["top3_share"], 3.5 / 3.5)
    assert row["ordered_top3_networks"] == "A|B|C"
    assert row["unordered_top3_networks"] == "A|B|C"


# ── category selection ───────────────────────────────────────────────────────

def test_default_filter_keeps_only_content_eligible():
    flags = {
        "sub-01": _flags(
            [0, 1, 2],
            [ELIGIBLE_CATEGORY, "unused", "low_confidence"],
        )
    }
    state_means = {"sub-01": np.ones((3, len(PARCEL_NETWORKS)))}

    metrics, summary = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    assert set(metrics["summary_category"]) == {ELIGIBLE_CATEGORY}
    assert summary["n_states"] == 1


def test_all_category_mode_preserves_all_categories():
    cats = [ELIGIBLE_CATEGORY, "unused", "low_confidence", "rare"]
    flags = {"sub-01": _flags([0, 1, 2, 3], cats)}
    state_means = {
        "sub-01": np.array(
            [
                [2.0, -1.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.1],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0],
            ]
        )
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS,
        summary_categories=None,
    )
    assert set(metrics["summary_category"]) == set(cats)
    assert len(metrics) == 4


def test_selected_categories_subset():
    cats = [ELIGIBLE_CATEGORY, "unused", "rare"]
    flags = {"sub-01": _flags([0, 1, 2], cats)}
    state_means = {"sub-01": np.ones((3, len(PARCEL_NETWORKS)))}
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS,
        summary_categories=["unused", "rare"],
    )
    assert set(metrics["summary_category"]) == {"unused", "rare"}


# ── summaries ────────────────────────────────────────────────────────────────

def test_per_category_summary_medians_match_recomputation():
    cats = [ELIGIBLE_CATEGORY, ELIGIBLE_CATEGORY, "unused", "unused"]
    flags = {"sub-01": _flags([0, 1, 2, 3], cats)}
    state_means = {
        "sub-01": np.array(
            [
                [2.0, -2.0, 1.0, 1.0, 0.5, -0.5, 0.0, 0.0],
                [3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.2, 0.2],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    }
    metrics, _ = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS,
        summary_categories=None,
    )
    summary = summarize_network_participation(metrics, NETWORKS)

    for cat, frame in metrics.groupby("summary_category"):
        med = summary["by_category"][cat]["metric_medians"]
        np.testing.assert_allclose(
            med["top1_share"], float(frame["top1_share"].median())
        )
        np.testing.assert_allclose(
            med["normalized_network_entropy"],
            float(frame["normalized_network_entropy"].median()),
        )


def test_unordered_top3_counted_as_sets_not_rank_profiles():
    # Two states share the same top-3 SET (A, B, C) but in different rank order.
    # State 0: A > B > C ; State 1: C > B > A.
    flags = {"sub-01": _flags([0, 1], [ELIGIBLE_CATEGORY, ELIGIBLE_CATEGORY])}
    state_means = {
        "sub-01": np.array(
            [
                [3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0],  # A>B>C
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0],  # C>B>A
            ]
        )
    }
    metrics, summary = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    assert metrics.iloc[0]["ordered_top3_networks"] == "A|B|C"
    assert metrics.iloc[1]["ordered_top3_networks"] == "C|B|A"
    # Both share unordered set A|B|C (network_order canonicalized).
    assert set(metrics["unordered_top3_networks"]) == {"A|B|C"}

    unordered = summary["unordered_top3_combination_counts"]
    ordered = summary["ordered_top3_rank_counts"]
    assert len(unordered) == 1 and unordered[0]["n_states"] == 2
    assert len(ordered) == 2  # two distinct rank profiles


def test_accepts_state_id_column():
    flags = {
        "sub-01": pd.DataFrame(
            {
                "state_id": [0],
                "summary_category": [ELIGIBLE_CATEGORY],
            }
        )
    }
    state_means = {"sub-01": np.ones((1, len(PARCEL_NETWORKS)))}
    metrics, summary = compute_network_participation_metrics(
        flags, state_means, ["sub-01"], PARCEL_NETWORKS, NETWORKS
    )
    assert len(metrics) == 1
    assert summary["n_states"] == 1
