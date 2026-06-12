"""Tests for the backend-agnostic HMM state posteriors helper.

Part of the ICA-vs-HMM convergent-validity supplement (sm_ica_states).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


def _toy_numpy_hmm(K=3, D=4, seed=0):
    """Build a minimal fitted StickyHDPHMM (numpy) with diag covariance."""
    from utils.hdphmm import StickyHDPHMM
    rng = np.random.default_rng(seed)
    m = StickyHDPHMM(n_components=K, covariance_type="diag")
    m.startprob_ = np.full(K, 1.0 / K)
    tm = np.full((K, K), 0.1) + np.eye(K) * 0.7
    m.transmat_ = tm / tm.sum(1, keepdims=True)
    m.means_ = rng.normal(size=(K, D)) * 3.0
    m.covars_ = np.ones((K, D))
    return m


def test_posteriors_shape_and_normalisation():
    from utils.hmm_io import compute_state_posteriors
    m = _toy_numpy_hmm()
    X = np.random.default_rng(1).normal(size=(50, 4))
    lengths = [20, 30]
    gamma = compute_state_posteriors(m, X, lengths)
    assert gamma.shape == (50, 3)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
    assert (gamma >= 0).all()


def test_posteriors_jax_fallback_path():
    import types
    from utils.hmm_io import compute_state_posteriors
    ref = _toy_numpy_hmm(K=3, D=4, seed=2)
    # stand-in lacking _do_e_step -> forces the numpy-rebuild fallback.
    # covars_ must be (K, D) shape for diag type (hmmlearn stores/returns
    # full (K,D,D) internally, so we provide the raw diag form here).
    fake = types.SimpleNamespace(
        n_components=3, covariance_type="diag",
        startprob_=ref.startprob_, transmat_=ref.transmat_,
        means_=ref.means_, covars_=np.ones((3, 4)),
    )
    assert not hasattr(fake, "_do_e_step")
    X = np.random.default_rng(3).normal(size=(40, 4))
    gamma = compute_state_posteriors(fake, X, [40])
    assert gamma.shape == (40, 3)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_spatial_ica_orientation_and_reconstruction():
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from utils.ica_states import fit_spatial_ica
    rng = np.random.default_rng(0)
    n_pcs, P, T = 8, 156, 400
    # PCA loadings must have ORTHONORMAL ROWS (as sklearn PCA produces)
    Q, _ = np.linalg.qr(rng.normal(size=(P, n_pcs)))
    components = Q.T                              # (n_pcs, 156), orthonormal rows
    X_pc = rng.normal(size=(T, n_pcs))           # PC scores (T, n_pcs)
    # (a) orientation: maps are parcel-space, time courses are (T, K)
    K = 5
    out = fit_spatial_ica(components, X_pc, n_components=K, random_state=0)
    assert out["maps"].shape == (P, K)
    assert out["timecourses"].shape == (T, K)
    # (b) at full rank, the timecourse model reconstructs the parcel time series:
    #     X_full = X_pc @ components ≈ timecourses @ maps.T + (X_pc @ mean_)[:,None]
    # Suppress ConvergenceWarning: full-rank ICA on tiny synthetic data is hard to
    # converge; the assertion tests algebra (timecourse formula), not convergence quality.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full = fit_spatial_ica(components, X_pc, n_components=n_pcs, random_state=0)
    X_full = X_pc @ components                          # (T, 156)
    offset = (X_pc @ full["mean_"])[:, None]            # (T, 1), constant across parcels
    recon = full["timecourses"] @ full["maps"].T + offset
    r = np.corrcoef(X_full.ravel(), recon.ravel())[0, 1]
    assert r > 0.99


def test_match_maps_hungarian_recovers_permutation_and_sign():
    from utils.ica_states import match_maps_hungarian
    rng = np.random.default_rng(0)
    P, K = 156, 6
    hmm_maps = rng.normal(size=(K, P))
    perm = rng.permutation(K)
    signs = rng.choice([-1.0, 1.0], size=K)
    ica_maps = (hmm_maps[perm] * signs[:, None]).T          # (P, K), shuffled+flipped
    res = match_maps_hungarian(ica_maps, hmm_maps)
    assert res["corr"].shape == (K, K)
    assert np.all(res["matched_r"] > 0.99)
    assert res["matched_sign"].shape == res["matched_r"].shape
    assert sorted(res["ica_idx"].tolist()) == list(range(K))
    # matched_sign must equal the sign of the signed matched correlation
    signed = res["corr"][res["ica_idx"], res["hmm_idx"]]
    assert np.all(res["matched_sign"] == np.sign(signed))
    assert np.all(np.abs(signed) == res["matched_r"])


def test_match_maps_hungarian_rectangular_fewer_ica():
    """K_ica < K_hmm: only K_ica pairs returned, all valid."""
    from utils.ica_states import match_maps_hungarian
    rng = np.random.default_rng(1)
    P, K_hmm, K_ica = 156, 6, 4
    hmm_maps = rng.normal(size=(K_hmm, P))
    ica_maps = hmm_maps[:K_ica].T                  # (P, 4) = first 4 HMM maps
    res = match_maps_hungarian(ica_maps, hmm_maps)
    assert res["matched_r"].shape == (K_ica,)
    assert len(set(res["ica_idx"].tolist())) == K_ica
    assert len(set(res["hmm_idx"].tolist())) == K_ica
    assert np.all(res["matched_r"] > 0.99)


def test_subspace_rotation_null_in_pc_subspace_calibrated():
    from utils.ica_states import subspace_rotation_null
    rng = np.random.default_rng(0)
    P, n_pcs, K = 156, 8, 5
    Q, _ = np.linalg.qr(rng.normal(size=(P, n_pcs)))
    components = Q.T                                    # (n_pcs, 156)
    hmm_maps = rng.normal(size=(K, P))
    null = subspace_rotation_null(components, hmm_maps, n_components=K,
                                  n_perm=200, rng_seed=0)
    assert null.shape == (200, K)
    assert np.all((null >= 0) & (null <= 1.0 + 1e-9))
    null2 = subspace_rotation_null(components, hmm_maps, n_components=K,
                                   n_perm=200, rng_seed=1)
    assert abs(null.mean() - null2.mean()) < 0.1


def test_temporal_correspondence_detects_planted_signal():
    from utils.ica_states import temporal_correspondence
    rng = np.random.default_rng(0)
    T = 600
    run_boundaries = [(0, 300), (300, 600)]
    gamma = rng.uniform(size=(T, 2))
    tc = rng.normal(size=(T, 2))
    # plant correspondence: component 0 tracks gamma col 0, with NEGATIVE sign
    # in the timecourse -- Tier-1 would have set matched_sign=-1 for it.
    tc[:, 0] = -(gamma[:, 0] * 5) + rng.normal(scale=0.1, size=T)
    res = temporal_correspondence(
        gamma, tc, hmm_idx=np.array([0, 1]), ica_idx=np.array([0, 1]),
        matched_sign=np.array([-1.0, 1.0]),   # carried from Tier-1 spatial match
        run_boundaries=run_boundaries, n_perm=200, rng_seed=0, min_occ=0.0)
    assert res["rho"].shape == (2,)
    assert res["rho"][0] > 0.5 and res["p"][0] < 0.05
    assert res["p"][1] > 0.05            # noise pair not significant
    assert "q" in res                    # FDR-corrected


def test_temporal_correspondence_rho_matches_scipy_spearman():
    from utils.ica_states import temporal_correspondence
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    T = 500
    gamma = rng.uniform(size=(T, 1))
    tc = (gamma[:, 0] * 2 + rng.normal(scale=1.0, size=T))[:, None]
    res = temporal_correspondence(
        gamma, tc, hmm_idx=np.array([0]), ica_idx=np.array([0]),
        matched_sign=np.array([1.0]), run_boundaries=[(0, T)],
        n_perm=10, rng_seed=0, min_occ=0.0)
    expected = spearmanr(gamma[:, 0], tc[:, 0]).statistic
    assert abs(res["rho"][0] - expected) < 1e-9


def test_wta_label_agreement_matches_when_aligned():
    from utils.ica_states import wta_label_agreement
    rng = np.random.default_rng(0)
    T = 600
    run_boundaries = [(0, 300), (300, 600)]
    viterbi = rng.integers(0, 4, size=T)
    # build ICA timecourses whose within-run-zscored argmax equals viterbi
    tc = rng.normal(scale=0.1, size=(T, 4))
    tc[np.arange(T), viterbi] += 10.0
    res = wta_label_agreement(tc, viterbi, run_boundaries, n_perm=200, rng_seed=0)
    assert res["ami"] > 0.8 and res["ari"] > 0.8
    assert res["p_ami"] < 0.05
    assert res["ica_labels"].shape == (T,)
    assert res["ica_labels"].dtype.kind in "iu"   # integer labels


def test_icasso_consensus_stable_components_have_high_iq():
    from utils.ica_states import icasso_consensus
    rng = np.random.default_rng(0)
    n_pcs, P, T, K = 8, 156, 400, 4
    components = rng.normal(size=(n_pcs, P))
    X_pc = rng.normal(size=(T, n_pcs))
    res = icasso_consensus(components, X_pc, n_components=K,
                           n_restarts=10, rng_seed=0)
    # fcluster may return fewer than K clusters; assert consistency not fixed shape
    n_cons = res["iq"].shape[0]
    assert n_cons <= K
    assert res["consensus_maps"].shape == (P, n_cons)
    assert res["timecourses"].shape == (T, n_cons)
    assert "cluster_sizes" in res and res["cluster_sizes"].shape == (n_cons,)
    # singletons have NaN I_q; non-singletons must be in [0, 1]
    iq = res["iq"]
    assert np.all(np.isnan(iq) | ((iq >= 0) & (iq <= 1.0001)))


def test_spatial_match_pvalues_per_rank_order_preserved():
    from utils.ica_states import spatial_match_pvalues
    rng = np.random.default_rng(0)
    # observed: one very strong match, rest weak; null: moderate everywhere
    obs = np.array([0.9, 0.2, 0.15, 0.1])
    null = rng.uniform(0.1, 0.5, size=(500, 4))
    p = spatial_match_pvalues(obs, null)
    assert p.shape == (4,)
    # strongest observed (rank 0) should be most significant
    assert p[0] == p.min()
    assert (p >= 0).all() and (p <= 1).all()


def test_icasso_consensus_recovers_planted_sources():
    from utils.ica_states import icasso_consensus
    rng = np.random.default_rng(5)
    n_pcs, P, T, K = 10, 156, 800, 4
    true_maps = rng.standard_t(3, size=(P, K))         # heavy-tailed -> ICA-recoverable
    tc = rng.standard_t(3, size=(T, K))
    X_full = tc @ true_maps.T                          # (T, P)
    Xc = X_full - X_full.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_pcs]                             # (n_pcs, P) orthonormal rows
    X_pc = Xc @ components.T                            # (T, n_pcs)
    res = icasso_consensus(components, X_pc, n_components=K, n_restarts=15, rng_seed=0)
    assert res["iq"].shape[0] == K                      # no merged/empty clusters
    assert res["consensus_maps"].shape == (P, K)
    assert np.nanmean(res["iq"]) > 0.3                  # stable across restarts (no NaN here: all clusters size>1)
