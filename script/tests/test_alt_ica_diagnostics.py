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
