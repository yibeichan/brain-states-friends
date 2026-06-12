"""Core math for the ICA alternative state-discovery supplement.

All functions are pure (no I/O). Conventions:
  components : PCA loadings, shape (n_pcs, n_parcels)   [pca.components_[:n_pcs]]
  X_pc       : PC scores / HMM input, shape (T, n_pcs)
  maps       : ICA spatial maps in parcel space, shape (n_parcels, K)
  timecourses: ICA component activations over time, shape (T, K)
"""
import warnings

import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.decomposition import FastICA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from .stats import fdr_with_nan
from .transformer_analysis import circular_shift_states_by_run


def fit_spatial_ica(components, X_pc, n_components, random_state=0,
                    fun="logcosh", max_iter=1000):
    """Spatial ICA within the PC subspace.

    Fit ICA with PARCELS as samples (X_spatial = components.T, shape
    (n_parcels, n_pcs)) so the recovered sources are spatial maps. Derive
    component time courses by projecting the PC scores through the mixing
    (X_full = X_pc @ components ~= (X_pc @ mixing_) @ maps.T + offset).

    Uses whiten="arbitrary-variance" (classical whitening): the ICA solution
    is then an orthogonal rotation of whitened data, which keeps the source
    scaling well-defined. The subspace-rotation null does NOT use FastICA's
    internal whitening (it uses the PCA basis directly), so the whitening mode
    only affects the maps/timecourses, not the null.
    """
    X_spatial = np.asarray(components).T            # (n_parcels, n_pcs)
    n_pcs = X_spatial.shape[1]
    if n_components > n_pcs:
        raise ValueError(
            f"n_components ({n_components}) must be <= n_pcs ({n_pcs})")
    ica = FastICA(n_components=n_components, fun=fun, max_iter=max_iter,
                  whiten="arbitrary-variance", random_state=random_state)
    maps = ica.fit_transform(X_spatial)             # (n_parcels, K) = sources S
    timecourses = np.asarray(X_pc) @ ica.mixing_    # (T, K)
    return {
        "maps": maps,
        "timecourses": timecourses,
        "mixing": ica.mixing_,
        "mean_": ica.mean_,
        "components_": ica.components_,
        "n_iter": int(getattr(ica, "n_iter_", -1)),
    }


def _zscore_rows(M):
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def _corr_matrix(ica_maps, hmm_maps):
    """Pearson corr between every ICA map (cols of ica_maps) and HMM map (rows).

    ica_maps : (P, K_ica)   hmm_maps : (K_hmm, P)  ->  corr (K_ica, K_hmm)
    """
    A = _zscore_rows(np.asarray(ica_maps).T)   # (K_ica, P)
    B = _zscore_rows(np.asarray(hmm_maps))     # (K_hmm, P)
    n_parcels = A.shape[1]
    return (A @ B.T) / n_parcels


def match_maps_hungarian(ica_maps, hmm_maps):
    """Sign-invariant 1-to-1 matching of ICA maps to HMM state maps.

    Assignment maximises |corr|. Returns matched |r| plus the per-pair SIGN
    of the (signed) spatial correlation -- this sign is the canonical ICA flip
    and MUST be reused for the Tier-2 temporal test (do not re-derive it from
    the temporal correlation).

    Returns min(K_ica, K_hmm) matched pairs. When K_ica > K_hmm, only K_hmm
    ICA components are matched (one per HMM state); the remaining ICA
    components have no assigned partner and are not included in the output.

    ica_maps : (P, K_ica)   hmm_maps : (K_hmm, P)
    corr     : (K_ica, K_hmm).  linear_sum_assignment receives corr.T
               (shape K_hmm x K_ica) so row_ind indexes HMM, col_ind indexes ICA.
    """
    corr = _corr_matrix(ica_maps, hmm_maps)           # (K_ica, K_hmm)
    hmm_idx, ica_idx = linear_sum_assignment(-np.abs(corr.T))  # rows=hmm, cols=ica
    signed = corr[ica_idx, hmm_idx]                   # signed r for each matched pair
    matched_sign = np.sign(signed)
    matched_sign[matched_sign == 0] = 1.0
    matched_r = np.abs(signed)
    return {
        "corr": corr,
        "ica_idx": np.asarray(ica_idx),
        "hmm_idx": np.asarray(hmm_idx),
        "matched_r": matched_r,
        "matched_sign": matched_sign,
    }


def _random_orthonormal_frame(n, k, rng):
    """Haar-distributed (n, k) matrix with orthonormal columns (QR of Gaussian)."""
    A = rng.normal(size=(n, k))
    Q, Rup = linalg.qr(A, mode="economic")     # Q: (n, k), orthonormal columns
    Q = Q * np.sign(np.diag(Rup))              # fix sign ambiguity -> Haar
    return Q


def subspace_rotation_null(components, hmm_maps, n_components, n_perm=1000,
                           rng_seed=0):
    """Spec §6 Tier-1 primary null: random K-frame in the full PC subspace.

    components : (n_pcs, P) PCA loadings (orthonormal rows). Their transpose,
                 (P, n_pcs), is an orthonormal basis of the parcel-space PC
                 subspace, so PCA components are already 'whitened' (isotropic).
    A random orthonormal frame Q (n_pcs, K) drawn over the FULL n_pcs space
    yields K random maps ``basis @ Q`` (P, K) in that subspace -- the proper
    'is ICA's rotation special vs a random rotation of the same subspace?'
    null. Re-runs Hungarian per draw (selection is part of the statistic).

    Returns : (n_perm, n_matched) array of matched |r| per draw.
    """
    rng = np.random.default_rng(rng_seed)
    basis = np.asarray(components).T            # (P, n_pcs), orthonormal columns
    n_pcs = basis.shape[1]
    if n_components > n_pcs:
        raise ValueError(f"n_components ({n_components}) must be <= n_pcs ({n_pcs})")
    out = []
    for _ in range(n_perm):
        Q = _random_orthonormal_frame(n_pcs, n_components, rng)  # (n_pcs, K)
        maps_rand = basis @ Q                   # (P, K) random maps in subspace
        out.append(match_maps_hungarian(maps_rand, hmm_maps)["matched_r"])
    return np.vstack(out)


def spatial_match_pvalues(matched_r, null_matched_r):
    """Per-RANK calibrated one-sided (greater) p-values, in original pair order.

    The i-th strongest observed matched |r| is compared against the distribution
    of i-th strongest matched |r| across null draws (each draw sorted descending).
    This is calibrated per rank, unlike pooling the whole null (which is liberal
    for weak pairs and conservative for strong ones). Matches the design's
    'per-rank statistics vs the matched null'.

    matched_r       : (n_matched,) observed matched |r| per pair
    null_matched_r  : (n_perm, n_matched) null matched |r| per draw
    Returns p aligned to the original matched_r order.
    """
    obs = np.asarray(matched_r, dtype=float)
    null = np.asarray(null_matched_r, dtype=float)
    n = obs.shape[0]
    n_perm = null.shape[0]
    order = np.argsort(obs)[::-1]                 # descending
    obs_sorted = obs[order]
    null_sorted = np.sort(null, axis=1)[:, ::-1]  # each draw descending
    p_sorted = np.array([
        (1 + np.sum(null_sorted[:, i] >= obs_sorted[i])) / (1 + n_perm)
        for i in range(n)
    ])
    p = np.empty(n)
    p[order] = p_sorted
    return p


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


def temporal_correspondence(gamma, timecourses, hmm_idx, ica_idx, matched_sign,
                            run_boundaries, n_perm=1000, rng_seed=0,
                            min_occ=0.01, min_shift=1):
    """Spearman(gamma_k, sign-aligned s_i) per matched pair + within-run shift null.

    gamma        : (T, K_hmm) state posteriors
    timecourses  : (T, K_ica) ICA component time courses
    hmm_idx,ica_idx,matched_sign : parallel arrays from match_maps_hungarian.
    Sign alignment: s_i is multiplied by ``matched_sign[j]`` -- the sign carried
    from the TIER-1 SPATIAL match (NOT re-derived from the temporal correlation,
    which would be circular). A true correspondence then gives rho > 0; the
    one-sided 'greater' test is therefore well-posed.
    States with mean(gamma_k) < min_occ are NaN (excluded from FDR).

    Implementation note: Spearman is computed as Pearson on ranks. A within-run
    circular shift permutes positions, and global ranks depend only on value,
    so rankdata(shift(s)) == shift(rankdata(s)). We therefore rank g and s ONCE
    per pair and shift the PRE-RANKED s in the null (no re-ranking per perm).
    """
    n = len(hmm_idx)
    rho = np.full(n, np.nan)
    pval = np.full(n, np.nan)
    occ = gamma.mean(axis=0)
    for j in range(n):
        k = int(hmm_idx[j])
        i = int(ica_idx[j])
        if occ[k] < min_occ:
            continue
        g_rank = rankdata(gamma[:, k])
        s = timecourses[:, i] * float(matched_sign[j])   # sign from Tier-1
        s_rank = rankdata(s)
        r = _pearson(g_rank, s_rank)                      # signed Spearman
        if np.isnan(r):
            continue
        rho[j] = r
        null = np.empty(n_perm)
        for p in range(n_perm):
            s_rank_shift = circular_shift_states_by_run(
                s_rank, run_boundaries, seed=rng_seed + p,
                min_shift=min_shift)
            null[p] = _pearson(g_rank, s_rank_shift)
        pval[j] = (1 + np.sum(null >= r)) / (1 + n_perm)
    q = fdr_with_nan(pval)
    return {"rho": rho, "p": pval, "q": q, "occupancy": occ[np.asarray(hmm_idx)]}


def wta_label_agreement(timecourses, viterbi, run_boundaries,
                        n_perm=1000, rng_seed=0, min_shift=1):
    """Timepoint-label agreement between z-scored-argmax ICA labels and Viterbi.

    Z-score each component time course WITHIN each run (spec section 6 Tier 3)
    before argmax so arbitrary ICA amplitude scaling and per-run drift do not
    bias the winner. Null: circularly shift the completed ICA label sequence
    within each run relative to HMM Viterbi (preserves both label marginals).
    This is a DESCRIPTIVE 'timepoint label agreement', not evidence of shared
    dynamics (ICA components co-express; WTA imposes mutual exclusivity).

    Returns dict with keys:
      ami       : adjusted mutual information (float)
      ari       : adjusted rand index (float)
      p_ami     : permutation p-value for AMI (float)
      ica_labels: integer WTA label per TR, shape (T,)

    Note: matched-occupancy correlation (across Hungarian-matched pairs) is
    intentionally NOT computed here. ICA component indices and HMM state ids
    have no index-level correspondence, so an index-aligned occupancy Pearson
    correlation is meaningless. The orchestrator computes matched occupancy
    using the Tier-1 Hungarian matching from match_maps_hungarian.
    """
    tc = np.asarray(timecourses, dtype=float)
    # z-score each component within each run
    z = np.empty_like(tc)
    for (s0, e0) in run_boundaries:
        seg = tc[s0:e0]
        z[s0:e0] = (seg - seg.mean(0)) / (seg.std(0) + 1e-12)
    ica_lab = z.argmax(axis=1)
    viterbi = np.asarray(viterbi)
    ami = adjusted_mutual_info_score(viterbi, ica_lab)
    ari = adjusted_rand_score(viterbi, ica_lab)
    null_ami = np.empty(n_perm)
    for p in range(n_perm):
        shifted = circular_shift_states_by_run(
            ica_lab, run_boundaries, seed=rng_seed + p, min_shift=min_shift)
        null_ami[p] = adjusted_mutual_info_score(viterbi, shifted)
    p_ami = (1 + np.sum(null_ami >= ami)) / (1 + n_perm)
    return {"ami": float(ami), "ari": float(ari), "p_ami": float(p_ami),
            "ica_labels": ica_lab}


def icasso_consensus(components, X_pc, n_components, n_restarts=25, rng_seed=0,
                     fun="logcosh", max_iter=1000):
    """Restart-consensus ICA (ICASSO-style).

    Run FastICA n_restarts times, pool all maps, cluster on 1-|corr| into
    n_components clusters, take the centrotype (max mean |corr| within cluster)
    as the consensus map, and report I_q = within-cluster minus
    mean |corr| to all out-of-cluster members.

    Returns consensus (centrotype) maps, time courses recomputed to be
    CONSISTENT with those consensus maps (regress the parcel time series onto
    the consensus maps -- do NOT reuse a single restart's timecourses whose
    order/sign need not match the consensus), and I_q per component.
    """
    all_maps = []
    n_nonconv = 0
    for r in range(n_restarts):
        out = fit_spatial_ica(components, X_pc, n_components,
                              random_state=rng_seed + r, fun=fun, max_iter=max_iter)
        all_maps.append(out["maps"].T)              # (K, P) per restart
        if out["n_iter"] == max_iter:
            n_nonconv += 1
    if n_nonconv > 0:
        warnings.warn(
            f"{n_nonconv} of {n_restarts} ICA restarts hit max_iter (non-converged)")
    M = np.vstack(all_maps)                          # (n_restarts*K, P)
    Mz = _zscore_rows(M)
    S = (Mz @ Mz.T) / Mz.shape[1]                    # corr matrix among all maps
    absS = np.abs(S)
    absS = np.clip(absS, 0.0, 1.0)                   # guard against float >1 / <0
    dist = 1.0 - absS
    np.fill_diagonal(dist, 0.0)
    # squareform checks=False: diagonal is already zeroed above, so hollowness holds
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=n_components, criterion="maxclust")
    # Iterate over actual clusters returned (may be < n_components); C1 fix:
    # fcluster(maxclust=n_components) can return fewer clusters, leaving zero-columns
    # in a pre-allocated matrix.  Build lists and stack instead.
    centro_list, iq_list, sizes_list = [], [], []
    unique_labels = np.unique(labels)
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        within = absS[np.ix_(idx, idx)]
        centro = idx[np.argmax(within.mean(axis=1))]    # diagonal is constant -> argmax unaffected
        sign = np.sign(S[centro, idx])
        sign[sign == 0] = 1
        centro_list.append((M[idx] * sign[:, None]).mean(axis=0))
        n_in = len(idx)
        sizes_list.append(int(n_in))
        if n_in < 2:
            iq_list.append(np.nan)                        # singleton: stability undefined
        else:
            w = within[~np.eye(n_in, dtype=bool)].mean()  # off-diagonal only
            out_idx = np.where(labels != c)[0]
            b = absS[np.ix_(idx, out_idx)].mean() if len(out_idx) else 0.0
            iq_list.append(max(0.0, w - b))
    centro_maps = np.column_stack(centro_list)            # (P, n_consensus)
    iq = np.asarray(iq_list)
    if len(unique_labels) < n_components:
        warnings.warn(
            f"icasso_consensus: fcluster returned {len(unique_labels)} clusters "
            f"(< n_components={n_components}); returning {len(unique_labels)} consensus components")
    # Time courses CONSISTENT with the consensus maps: regress the full parcel
    # time series (X_full = X_pc @ components) onto the consensus maps M_cons.
    #   X_full ~= TC @ M_cons.T  =>  TC = X_full @ M_cons @ pinv(M_cons.T @ M_cons)
    M_cons = centro_maps                              # (P, n_consensus)
    proj = np.asarray(components) @ M_cons @ np.linalg.pinv(M_cons.T @ M_cons)  # (n_pcs, n_consensus)
    timecourses = np.asarray(X_pc) @ proj             # (T, n_consensus)
    return {"consensus_maps": centro_maps, "timecourses": timecourses,
            "iq": iq, "cluster_labels": labels,
            "cluster_sizes": np.asarray(sizes_list)}
