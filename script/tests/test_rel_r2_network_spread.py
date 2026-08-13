"""Tests for sm_rel_r2_network_spread.

Covers the metric implementation (entropy endpoints, vectorization against a
reference loop), the variance-matched sampler's second moments, the two-sided
empirical p, and the back-projection gate's failure mode. No scratch data
needed; everything is synthetic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sm_rel_r2_network_spread as m  # noqa: E402


def _toy_masks(n_networks=4, per=5):
    labels = np.repeat(np.arange(n_networks), per)
    return [labels == i for i in range(n_networks)], n_networks * per


def test_entropy_zero_for_single_network_map():
    masks, n_parcels = _toy_masks()
    v = np.zeros(n_parcels)
    v[masks[0]] = 3.0
    out = m.participation_metrics(v, masks, len(masks))
    assert out["entropy"][0] == pytest.approx(0.0)
    assert out["top1"][0] == pytest.approx(1.0)


def test_entropy_one_for_uniform_map():
    masks, n_parcels = _toy_masks()
    v = np.full(n_parcels, 2.0)
    out = m.participation_metrics(v, masks, len(masks))
    assert out["entropy"][0] == pytest.approx(1.0)
    assert out["top3"][0] == pytest.approx(3.0 / 4.0)
    assert out["n_ge10"][0] == 4


def test_metrics_match_reference_loop():
    masks, n_parcels = _toy_masks(n_networks=5, per=7)
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((20, n_parcels))
    out = m.participation_metrics(maps, masks, len(masks))
    for i in range(20):
        a = np.abs(maps[i])
        scores = np.array([a[mask].mean() for mask in masks])
        c = scores / scores.sum()
        nz = c[c > 0]
        ent = -(nz * np.log(nz)).sum() / np.log(len(masks))
        srt = np.sort(c)[::-1]
        assert out["entropy"][i] == pytest.approx(ent)
        assert out["top1"][i] == pytest.approx(srt[0])
        assert out["top3"][i] == pytest.approx(srt[:3].sum())
        assert out["n_ge10"][i] == (c >= 0.10).sum()


def test_variance_matched_sampler_second_moments():
    rng = np.random.default_rng(1)
    evar = np.array([9.0, 4.0, 1.0])
    draws = rng.standard_normal((200_000, 3)) * np.sqrt(evar)
    np.testing.assert_allclose(draws.var(axis=0), evar, rtol=0.02)


def test_empirical_p_two_sided():
    null = np.arange(999, dtype=float)
    assert m.empirical_p_two_sided(2000.0, null) == pytest.approx(
        2 / 1000)
    assert m.empirical_p_two_sided(499.0, null) == 1.0


def test_backprojection_gate_catches_mismatch():
    rng = np.random.default_rng(2)
    n_pcs, n_parcels, k = 6, 15, 4

    class FakePCA:
        components_ = rng.standard_normal((n_pcs + 2, n_parcels))
        mean_ = np.zeros(n_parcels)
        explained_variance_ = np.linspace(5, 1, n_pcs + 2)

    mu = rng.standard_normal((k, n_pcs))
    good = mu @ FakePCA.components_[:n_pcs] + FakePCA.mean_
    w, evar, mean = m.gate_backprojection(good, mu, FakePCA, "sub-xx")
    assert w.shape == (n_pcs, n_parcels) and len(evar) == n_pcs
    with pytest.raises(RuntimeError, match="back-projection gate failed"):
        m.gate_backprojection(good + 1e-3, mu, FakePCA, "sub-xx")


def test_null_median_distribution_shape_and_range():
    rng = np.random.default_rng(3)
    pool = rng.uniform(0.5, 0.9, size=5000)
    meds = m.null_median_distribution(pool, k=25, n_group=200, rng=rng)
    assert meds.shape == (200,)
    assert pool.min() <= meds.min() and meds.max() <= pool.max()


def test_descriptive_z_normal_case():
    null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = m.descriptive_z(6.0, null)
    assert z == pytest.approx((6.0 - 3.0) / null.std(ddof=1))


def test_descriptive_z_none_for_degenerate_null():
    # Integer-valued metrics (n_ge10) can yield 10,000 identical group
    # medians; z must become None (JSON null), not inf/nan.
    null = np.full(10_000, 4.0)
    assert m.descriptive_z(5.0, null) is None
    assert m.descriptive_z(4.0, null) is None


def test_write_json_rejects_non_finite(tmp_path):
    with pytest.raises(ValueError):
        m.write_json(tmp_path / "bad.json", {"z": float("inf")})
    with pytest.raises(ValueError):
        m.write_json(tmp_path / "bad.json", {"z": float("nan")})


def test_write_json_accepts_none_z(tmp_path):
    import json

    path = tmp_path / "ok.json"
    m.write_json(path, {"z": None, "p": 0.5})
    assert json.loads(path.read_text()) == {"z": None, "p": 0.5}


def test_rng_for_streams_are_disjoint_across_purposes():
    # Regression: 1000*(s+1)+v draw seeds collided with 2000*(s+1)+v group
    # seeds (e.g. sub-02 draws == sub-01 group resampling, seed 2000).
    keys = [("draws", s, v) for s in range(6) for v in range(2)]
    keys += [("group", s, v) for s in range(6) for v in range(2)]
    keys += [("pooled", v) for v in range(2)]
    streams = [m.rng_for(*k).integers(0, 2**63, size=4).tolist() for k in keys]
    assert len({tuple(s) for s in streams}) == len(keys)


def test_rng_for_is_deterministic():
    a = m.rng_for("draws", 1, 0).standard_normal(8)
    b = m.rng_for("draws", 1, 0).standard_normal(8)
    np.testing.assert_array_equal(a, b)
