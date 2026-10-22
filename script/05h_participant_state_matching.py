#!/usr/bin/env python3
"""05h_participant_state_matching.py - do two participants have matching states?

Each participant's HMM is fitted separately, so nothing forces its states to
match another participant's. All state maps live in the same 156-parcel space, though,
so two subjects' states can be matched by spatial correlation. This script
does that per participant pair (15 pairs) and asks how far the matches go beyond
what the two repertoires would give by chance, beyond canonical large-scale
networks, and whether matched states also occur at the same moments of the
shared stimulus. Per-pair results only; nothing is pooled across pairs.

State map: the back-projected state mean without the PCA mean
(state_means_parcel - pca.mean_, which equals state_means_pca @ W), row
z-scored over the parcels in scope; signed Pearson r. Scopes: all 156 parcels,
and cortex only (the Schaefer parcels of the atlas). Sets: content-eligible (primary)
and all active states (secondary; matches labelled by 05e category and by the
05g_excluded_state_check class); run-onset-anchored states as a positive control.

Matching: rectangular Hungarian on 1 - r (both directions' denominators are
reported), plus each A state's unconstrained best match (collisions counted).

Pair statistic: median matched r. Nulls (Hungarian re-run, maps re-z-scored
in every draw; 1000 draws; A rotated in even draws, B in odd draws):
  state_mixing   the subject's centred state maps mixed by a Haar-random
                 orthogonal matrix that fixes the mean (primary). Keeps the
                 mean map and the parcel-space scatter exactly, so principal
                 axes shared across participants survive; destroys which
                 state is which. Used for the pair test, the per-match
                 thresholds and every specificity test below.
  pc_blocks      Haar rotation of state means within blocks of 5 adjacent
                 PCs of that subject's PCA, back-projected, then given the
                 variant's scope and centring (secondary; pair test only)
A rotation of the maps within their span (keeping the Gram matrix) was the
first primary null. It also rotates the principal axes, so it rejected pairs
with no state correspondence whenever the participants share a dominant
subspace. --mode calibrate measures each null's false-positive rate on such
generators and its power on planted counterparts.
No spin-test null: it needs sphere coordinates for the Schaefer parcels and a
dependency this environment does not have, and it is coarse and liberal for
subcortex.
Specificity (eligible states, all parcels):
  network level     maps replaced by their 13 network/structure means, and the
                    complement (maps minus those means), each with its own
                    state-mixing null
  shared components the top k (1, 3, 5) components of a PCA over the other four
                    participants' state maps projected out of both maps (the
                    state-mixing null re-run at the pre-stated k = 3), with the
                    variance those components explain in the pair's maps; also
                    k = 3 plus the others' mean state map
  amplitude         uncentred cosine on the raw contrast maps (row z-scoring
                    removes each map's global mean and scale), and each matched
                    pair's global mean and SD
Per-state claims use a max-statistic threshold (95th percentile of the largest
matched r per null draw). Many-to-one collisions are counted in both
directions against the same null.
Checks with a different input (occupancy instead of maps), eligible states:
  time locking      Pearson r of matched states' binary occupancy over the runs
                    both participants watched (same run id = same episode half;
                    truncated to the shorter run), against within-run circular
                    shifts of one participant (offsets 1..L-1, conservative).
                    Matched pairs are flagged with a family-wise threshold (max
                    over matched pairs); pointwise flags are kept separately.
                    Matched-pair locking is also compared with the locking of
                    the pairs that the null matchings choose, and
                    recomputed with 10 TRs dropped from both ends of each run.
  timing properties Spearman of recurrence, mean FO and median dwell across
                    matched pairs, against the same statistic for the
                    null matchings
Ceiling: the same matching on the split-half fits, within participant and
across participants, for all active states and for the half-fit states matched
to the final fit's content-eligible states.
Random streams are per (pair, variant). Every pair file records its parameters,
code commit, a hash of the code and a hash of every input file; on rerun a pair
is recomputed when any of them differs, and the run refuses to start with
uncommitted changes under script/. BH runs across pairs within each family:
one per (null, variant) for the pair test, and one per (test, state set) for
the specificity and validation tests; a pair without a test is left out.

Inputs (frozen, under $SCRATCH_DIR/output):
    04_combined_hdphmm/{parc}/{sub}/final/vt0.95/{state_means_parcel.npy,
        state_means_pca.npy, pca_model.pkl, decoded_states.pkl}
    04_combined_hdphmm/{parc}/{sub}/split_half/{A,B}/{state_means_parcel.npy,
        decoded_states.pkl}
    05e_temporal_trend_a4/{parc}/{sub}/vt0.95/state_flags.csv
    05g_excluded_state_check/{parc}/{sub}/vt0.95/state_classes.csv
    $ATLAS_DIR/{parc}/{parc}_dseg.tsv (parcel -> network / structure)
Output:
    {SCRATCH_DIR}/output/05h_participant_state_matching/{parc}/pairs/{subA}_{subB}/summary.json
    {SCRATCH_DIR}/output/05h_participant_state_matching/{parc}/cross_subject_summary.json
"""
import os
import sys
import json
import time
import pickle
import logging
import argparse
import platform
import hashlib
import subprocess
import tempfile
import warnings
import zlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.stats import benjamini_hochberg  # noqa: E402
from utils.plot_style import assign_network  # noqa: E402
from utils.recurrence_utils import active_state_indices  # noqa: E402

logger = logging.getLogger("05h_participant_state_matching")

SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
ELIGIBLE = "eligible_for_content_analysis"
ONSET_FLAGS = ("run_onset", "a_anchored", "b_anchored")
K_SHARED = (1, 3, 5)
K_SHARED_PRIMARY = 3
BLOCK = 5
NULL = "state_mixing"            # primary null (pair test, per-match thresholds, specificity)


# ---------------------------------------------------------------- maps

def zrows(M):
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0                       # zero-variance row -> r = 0 (as 04rb)
    return (M - mu) / sd


def corr_matrix(A, B):
    R = zrows(A) @ zrows(B).T / A.shape[1]
    if not np.all(np.isfinite(R)):
        raise FloatingPointError("non-finite correlation")
    return R


def match(R):
    """Rectangular Hungarian on 1 - r. Returns (rows, cols, r)."""
    ia, ib = linear_sum_assignment(1.0 - R)
    return ia, ib, R[ia, ib]


def many_to_one(R):
    best = R.argmax(axis=1)
    counts = np.bincount(best, minlength=R.shape[1])
    return R.max(axis=1), int(np.sum(counts >= 2))


def rotate_configuration(M, rng):
    """Haar rotation of the centred maps within their own span.

    M (k x P). C = M - mean = U S Vt (rank r). Returns mean + U S Q Vt with Q
    Haar on O(r): same Gram matrix, same span, different assignment of
    directions to states.
    """
    mean = M.mean(axis=0, keepdims=True)
    C = M - mean
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(S > S.max() * 1e-10))
    U, S, Vt = U[:, :r], S[:r], Vt[:r]
    Q = haar(r, rng)
    return mean + (U * S) @ Q @ Vt


def mix_states(M, rng):
    """Random orthogonal mixing of the centred maps across states.

    M (k x P). C = M - mean. Returns mean + Q C with Q Haar on O(k) restricted
    to the complement of the all-ones vector, so the mixed maps stay centred.
    Keeps the mean map and the parcel-space scatter C^T C exactly (principal
    axes, their variances, any subspace shared with other participants) and
    destroys which state is which: each surrogate state is a mixture of the
    participant's states.
    """
    k = M.shape[0]
    mean = M.mean(axis=0, keepdims=True)
    C = M - mean
    H = np.linalg.qr(np.eye(k)[:, 1:] - 1.0 / k)[0]   # orthonormal basis of 1-perp
    return mean + H @ (haar(k - 1, rng) @ (H.T @ C))


def haar(n, rng):
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def rotate_pc_blocks(mu_pc, components, rng, block=BLOCK):
    """Haar rotation of state means within blocks of adjacent PCs."""
    n_pcs = mu_pc.shape[1]
    out = mu_pc.copy()
    for s in range(0, n_pcs, block):
        e = min(s + block, n_pcs)
        if e - s > 1:
            out[:, s:e] = mu_pc[:, s:e] @ haar(e - s, rng)
    return out @ components[:n_pcs]


def collapse_networks(M, parcel_groups):
    out = np.empty_like(M)
    for g in np.unique(parcel_groups):
        m = parcel_groups == g
        out[:, m] = M[:, m].mean(axis=1, keepdims=True)
    return out


def shared_components(maps_list, k):
    """Top-k right singular vectors of the stacked, row-z-scored maps."""
    X = zrows(np.vstack(maps_list))
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


def project_out(M, V):
    return M - (M @ V.T) @ V


# ---------------------------------------------------------------- time locking

def joint_counts(lab_a, lab_b, runs, ka, kb, shifts=None):
    """Joint and marginal occupancy counts over common runs (truncated to the
    shorter run); optional per-run circular shift of B. Labels < 0 mark TRs
    in states outside the matched sets: they count toward T but toward no
    state. Returns (N[a, b], na[a], nb[b])."""
    N = np.zeros(ka * kb)
    na, nb = np.zeros(ka), np.zeros(kb)
    for r in runs:
        a, b = lab_a[r], lab_b[r]
        L = min(a.size, b.size)
        a, b = a[:L], b[:L]
        if shifts is not None:
            b = np.roll(b, shifts[r])
        ok = (a >= 0) & (b >= 0)
        N += np.bincount(a[ok] * kb + b[ok], minlength=ka * kb)
        na += np.bincount(a[a >= 0], minlength=ka)
        nb += np.bincount(b[b >= 0], minlength=kb)
    return N.reshape(ka, kb), na, nb


def occupancy_r(counts, T, pairs):
    """Pearson r of binary occupancy for each (a, b) pair, from counts."""
    N, na, nb = counts
    out = []
    for a, b in pairs:
        den = np.sqrt(na[a] * (T - na[a]) * nb[b] * (T - nb[b]))
        out.append((N[a, b] * T - na[a] * nb[b]) / den if den > 0 else np.nan)
    return np.array(out)


# ---------------------------------------------------------------- loading

def scratch_dir():
    s = os.environ.get("SCRATCH_DIR")
    if not s:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
        s = os.environ.get("SCRATCH_DIR")
    if not s:
        raise SystemExit("SCRATCH_DIR is not set")
    return s


def dseg_path(parc):
    atlas = os.environ.get("ATLAS_DIR")
    if not atlas:
        raise SystemExit("ATLAS_DIR is not set")
    return os.path.join(atlas, parc, f"{parc}_dseg.tsv")


def parcel_groups(parc):
    """Network / structure group per parcel, and a mask of the cortical parcels
    (those assign_network leaves to the Schaefer network label)."""
    df = pd.read_csv(dseg_path(parc), sep="\t")
    groups, cortex = [], []
    for idx in range(len(df)):
        row = df[df["index"] == idx + 1].iloc[0]
        g = assign_network(row["label"])
        cortex.append(g is None)
        groups.append(g if g is not None else row.get("network_label", "Unknown"))
    return np.array(groups), np.array(cortex)


def active_from_decoded(decoded, K):
    return active_state_indices(decoded, K)


def load_subject(sub, parc, vt):
    S = scratch_dir()
    base = os.path.join(S, "output", "04_combined_hdphmm", parc, sub)
    hmm = os.path.join(base, "final", f"vt{vt}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.path.join(hmm, "pca_model.pkl"), "rb") as f:
            pca = pickle.load(f)
    maps = np.load(os.path.join(hmm, "state_means_parcel.npy")) - pca.mean_
    mu_pc = np.load(os.path.join(hmm, "state_means_pca.npy"))
    comps = pca.components_
    if np.abs(maps - mu_pc @ comps[:mu_pc.shape[1]]).max() > 1e-8:
        raise RuntimeError(f"{sub}: contrast map != means_pca @ W")
    input_files = [os.path.join(hmm, f) for f in ("pca_model.pkl", "state_means_parcel.npy",
                                                   "state_means_pca.npy", "decoded_states.pkl")]
    with open(os.path.join(hmm, "decoded_states.pkl"), "rb") as f:
        decoded = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
    K = maps.shape[0]
    flags = pd.read_csv(os.path.join(S, "output", "05e_temporal_trend_a4", parc, sub,
                                     f"vt{vt}", "state_flags.csv")).set_index("state")
    if list(flags.index) != list(range(K)):
        raise RuntimeError(f"{sub}: state_flags rows are not states 0..{K - 1} in order")
    classes_path = os.path.join(S, "output", "05g_excluded_state_check", parc, sub,
                                f"vt{vt}", "state_classes.csv")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"{sub}: run 05g first (missing {classes_path})")
    input_files += [os.path.join(S, "output", "05e_temporal_trend_a4", parc, sub, f"vt{vt}",
                                 "state_flags.csv"), classes_path]
    classes = pd.read_csv(classes_path).set_index("state")["class"].to_dict()
    active = sorted(int(s) for s in active_from_decoded(decoded, K))
    eligible = sorted(int(s) for s in flags.index[flags["summary_category"] == ELIGIBLE])
    if not set(eligible) <= set(active):
        raise RuntimeError(f"{sub}: content-eligible states outside the active set")
    missing = sorted(set(active) - set(eligible) - set(classes))
    if missing:
        raise RuntimeError(f"{sub}: 05g did not classify active states {missing}")
    onset = sorted(s for s in active if any(bool(flags.loc[s, c]) for c in ONSET_FLAGS))
    halves = {}
    for h in ("A", "B"):
        hd = os.path.join(base, "split_half", h)
        hm = np.load(os.path.join(hd, "state_means_parcel.npy"))
        input_files += [os.path.join(hd, "state_means_parcel.npy"), os.path.join(hd, "decoded_states.pkl")]
        with open(os.path.join(hd, "decoded_states.pkl"), "rb") as f:
            hdec = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
        halves[h] = {"maps": hm, "active": sorted(int(s) for s in
                                                  active_from_decoded(hdec, hm.shape[0]))}
    trs = np.bincount(np.concatenate(list(decoded.values())), minlength=K)
    seg_len = {s: [] for s in range(K)}
    for seq in decoded.values():
        change = np.flatnonzero(np.diff(seq)) + 1
        starts = np.concatenate([[0], change])
        lengths = np.diff(np.concatenate([starts, [seq.size]]))
        for s0, ln in zip(seq[starts], lengths):
            seg_len[int(s0)].append(int(ln))
    dwell = np.array([np.median(seg_len[s]) if seg_len[s] else np.nan for s in range(K)])
    return {"sub": sub, "maps": maps, "mu_pc": mu_pc, "comps": comps, "decoded": decoded,
            "K": K, "flags": flags, "classes": classes, "active": active,
            "eligible": eligible, "onset": onset, "halves": halves,
            "rec": flags["recurrence_score"].to_numpy(dtype=float),
            "fo": trs / trs.sum(), "dwell": dwell,
            "max_abs_pca_mean": float(np.abs(pca.mean_).max()), "input_files": input_files}


# ---------------------------------------------------------------- pair analysis

def cos_matrix(A, B):
    """Uncentred cosine between rows (raw contrast maps, amplitude kept)."""
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def pair_null(Ma, Mb, n_perm, rng, kind, extra=None, sim=corr_matrix):
    """Median matched similarity under a null; A rotated in even draws, B in odd.

    Returns (median per draw, max matched per draw, A->B collisions per draw,
    B->A collisions per draw, the null matchings as (ia, ib) per draw).
    """
    out = np.empty(n_perm)
    max_r = np.empty(n_perm)
    coll_ab = np.empty(n_perm)
    coll_ba = np.empty(n_perm)
    matchings = []
    for i in range(n_perm):
        rot_a = i % 2 == 0
        if kind == "state_mixing":
            A = mix_states(Ma, rng) if rot_a else Ma
            B = Mb if rot_a else mix_states(Mb, rng)
        elif kind == "configuration":
            A = rotate_configuration(Ma, rng) if rot_a else Ma
            B = Mb if rot_a else rotate_configuration(Mb, rng)
        else:  # pc_blocks; post maps a back-projected draw into the variant's scope
            muA, cA, muB, cB, post = extra
            A = post(rotate_pc_blocks(muA, cA, rng)) if rot_a else Ma
            B = Mb if rot_a else post(rotate_pc_blocks(muB, cB, rng))
        Rn = sim(A, B)
        ja, jb, r = match(Rn)
        out[i] = np.median(r)
        max_r[i] = r.max()
        coll_ab[i] = many_to_one(Rn)[1]
        coll_ba[i] = many_to_one(Rn.T)[1]
        matchings.append((ja, jb))
    return out, max_r, coll_ab, coll_ba, matchings


def p_greater(obs, null):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= obs)) / (1 + null.size))


def p_less(obs, null):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return float("nan")
    return float((1 + np.sum(null <= obs)) / (1 + null.size))


def variant_rng(args, a, b, key):
    """One stream per (pair, variant), so a variant reruns identically alone."""
    return np.random.default_rng(np.random.SeedSequence(
        [args.seed, int(a[-1]), int(b[-1]), zlib.crc32(key.encode()) & 0x7FFFFFFF]))


def match_block(Ma, Mb, n_perm, rng, sa, sb, sim=corr_matrix):
    """Matching, primary null, max-statistic flags and collisions."""
    R = sim(Ma, Mb)
    ia, ib, r = match(R)
    null, max_r, cab, cba, null_matchings = pair_null(Ma, Mb, n_perm, rng, NULL,
                                                      sim=sim)
    thr = float(np.percentile(max_r, 95))
    out = {"n_a": len(sa), "n_b": len(sb), "n_matched": int(r.size),
           "median_r": float(np.median(r)), "mean_r": float(r.mean()),
           NULL: {"null_mean": float(null.mean()), "p": p_greater(np.median(r), null)},
           "max_null95": thr, "n_pairs_beyond_max_null95": int(np.sum(r > thr)),
           "matches": [{"a": int(sa[i]), "b": int(sb[j]), "r": float(R[i, j]),
                        "beyond_max_null95": bool(R[i, j] > thr)} for i, j in zip(ia, ib)]}
    return out, (R, ia, ib, r, null_matchings)


def analyse_pair(da, db, others, groups, cortex, args):
    a, b = da["sub"], db["sub"]
    res = {"pair": f"{a}_{b}", "variants": {}}
    scopes = {"full": slice(None), "cortex": cortex}
    sets = {"eligible": "eligible", "active": "active", "onset_control": "onset"}
    for set_name, key_set in sets.items():
        sa, sb = da[key_set], db[key_set]
        if len(sa) < 3 or len(sb) < 3:
            continue
        for scope_name, sl in scopes.items():
            for centred in (False, True):
                if set_name == "onset_control" and (scope_name != "full" or centred):
                    continue
                key = f"{set_name}|{scope_name}|{'centred' if centred else 'raw'}"
                rng = variant_rng(args, a, b, key)
                Ma, Mb = da["maps"][sa][:, sl], db["maps"][sb][:, sl]
                if centred:
                    Ma, Mb = Ma - Ma.mean(0, keepdims=True), Mb - Mb.mean(0, keepdims=True)
                v, (R, ia, ib, r, null_matchings) = match_block(Ma, Mb, args.n_perm, rng, sa, sb)
                v["matches_full"] = v.pop("matches")
                # many-to-one collisions, both directions, against the same draws
                _, _, cab, cba, _ = pair_null(Ma, Mb, args.n_perm, variant_rng(args, a, b, key + "|coll"),
                                              NULL)
                obs_ab, obs_ba = many_to_one(R)[1], many_to_one(R.T)[1]
                v["collisions"] = {"a_to_b": obs_ab, "b_to_a": obs_ba,
                                   "null_mean_a_to_b": float(cab.mean()),
                                   "null_mean_b_to_a": float(cba.mean()),
                                   "p_more_a_to_b": p_greater(obs_ab, cab),
                                   "p_more_b_to_a": p_greater(obs_ba, cba),
                                   "p_fewer_a_to_b": p_less(obs_ab, cab),
                                   "p_fewer_b_to_a": p_less(obs_ba, cba)}
                if set_name != "onset_control":
                    # secondary null: rotate in PC space, back-project, then apply
                    # the same scope and centring as the observed maps
                    def post(X, sl=sl, centred=centred):
                        X = X[:, sl]
                        return X - X.mean(0, keepdims=True) if centred else X
                    pcs = (da["mu_pc"][sa], da["comps"], db["mu_pc"][sb], db["comps"], post)
                    nb, _, _, _, _ = pair_null(Ma, Mb, args.n_perm,
                                               variant_rng(args, a, b, key + "|pcb"),
                                               "pc_blocks", pcs)
                    v["pc_blocks"] = {"null_mean": float(nb.mean()),
                                      "p": p_greater(np.median(r), nb)}
                if not centred and set_name != "onset_control":
                    if scope_name == "full":
                        # network-level structure: collapsed maps and their complement
                        Ca, Cb = collapse_networks(Ma, groups), collapse_networks(Mb, groups)
                        vc, _ = match_block(Ca, Cb, args.n_perm,
                                            variant_rng(args, a, b, key + "|collapse"), sa, sb)
                        vw, _ = match_block(Ma - Ca, Mb - Cb, args.n_perm,
                                            variant_rng(args, a, b, key + "|within"), sa, sb)
                        v["network_collapse"] = {k_: vc[k_] for k_ in
                                                 ("median_r", NULL,
                                                  "n_pairs_beyond_max_null95")}
                        v["within_network_residual"] = {k_: vw[k_] for k_ in
                                                        ("median_r", NULL,
                                                         "n_pairs_beyond_max_null95")}
                        # amplitude kept: uncentred cosine on the raw contrast maps
                        vcos, _ = match_block(Ma, Mb, args.n_perm,
                                              variant_rng(args, a, b, key + "|cosine"), sa, sb,
                                              sim=cos_matrix)
                        v["raw_cosine"] = {k_: vcos[k_] for k_ in
                                           ("median_r", NULL,
                                            "n_pairs_beyond_max_null95")}
                        gm_a, gm_b = Ma.mean(1), Mb.mean(1)
                        v["global_mean"] = {
                            "matched_sign_disagreements": int(np.sum(
                                np.sign(gm_a[ia]) != np.sign(gm_b[ib]))),
                            "per_match": [{"a": int(sa[i]), "b": int(sb[j]),
                                           "mean_a": float(gm_a[i]), "mean_b": float(gm_b[j]),
                                           "sd_a": float(Ma[i].std()), "sd_b": float(Mb[j].std())}
                                          for i, j in zip(ia, ib)]}
                    # shared components of the other four participants
                    shared = {}
                    stack = [o["maps"][o[key_set]][:, sl] for o in others]
                    Za, Zb = zrows(Ma), zrows(Mb)
                    for k in K_SHARED:
                        V = shared_components(stack, k)
                        Pa, Pb = project_out(Za, V), project_out(Zb, V)
                        explained = 1.0 - (np.sum(Pa ** 2) + np.sum(Pb ** 2)) / \
                            (np.sum(Za ** 2) + np.sum(Zb ** 2))
                        if k == K_SHARED_PRIMARY:
                            ek, (_, ka_i, kb_i, rk, _) = match_block(
                                Pa, Pb, args.n_perm, variant_rng(args, a, b, key + "|k3"), sa, sb)
                            ek["variance_explained_in_pair"] = float(explained)
                            shared[str(k)] = ek
                        else:
                            rk = match(corr_matrix(Pa, Pb))[2]
                            shared[str(k)] = {"median_r": float(np.median(rk)),
                                              "variance_explained_in_pair": float(explained)}
                    # sensitivity: also remove the others' mean state map
                    Xs = zrows(np.vstack(stack))
                    mvec = Xs.mean(axis=0)
                    V3 = shared_components(stack, K_SHARED_PRIMARY)
                    Vm = np.vstack([V3, mvec / np.linalg.norm(mvec)])
                    Q, _ = np.linalg.qr(Vm.T)
                    Pa, Pb = project_out(Za, Q.T), project_out(Zb, Q.T)
                    em, _ = match_block(Pa, Pb, args.n_perm,
                                        variant_rng(args, a, b, key + "|k3mean"), sa, sb)
                    shared["3_plus_mean"] = {k_: em[k_] for k_ in
                                             ("median_r", NULL,
                                              "n_pairs_beyond_max_null95")}
                    v["shared_removed"] = shared
                if set_name == "active" and scope_name == "full" and not centred:
                    v["match_labels"] = [
                        {"a": int(sa[i]), "b": int(sb[j]), "r": float(R[i, j]),
                         "cat_a": da["flags"].loc[sa[i], "summary_category"],
                         "cat_b": db["flags"].loc[sb[j], "summary_category"],
                         "class_a": da["classes"].get(sa[i], "content-eligible")
                         if sa[i] in da["eligible"] else da["classes"].get(sa[i], "unknown"),
                         "class_b": db["classes"].get(sb[j], "content-eligible")
                         if sb[j] in db["eligible"] else db["classes"].get(sb[j], "unknown")}
                        for i, j in zip(ia, ib)]
                if set_name == "eligible" and scope_name == "full" and not centred:
                    residual = v["shared_removed"][str(K_SHARED_PRIMARY)]["matches"]
                    v["validation"] = validate(da, db, sa, sb, ia, ib, null_matchings, residual,
                                               Ma, Mb, args)
                res["variants"][key] = v
    return res


def label_map(seq, states, K):
    """Position of each TR's state in `states`, or -1 for states outside it."""
    lut = np.full(K, -1, dtype=int)
    lut[np.asarray(states, dtype=int)] = np.arange(len(states))
    return lut[seq]


def locking_matrices(da, db, sa, sb, runs, n_lock, rng, trim=0):
    """Occupancy r for every (A state, B state) pair, observed and per shift draw.

    trim drops that many TRs from both ends of every run (run-edge
    sensitivity). Each draw circularly shifts every run of B by a random
    offset in 1..L-1 (offsets near 0 keep the null close to the data, which
    makes the test conservative).
    """
    la = {r: label_map(da["decoded"][r], sa, da["K"]) for r in runs}
    lb = {r: label_map(db["decoded"][r], sb, db["K"]) for r in runs}
    if trim:
        la = {r: x[trim:-trim] if x.size > 2 * trim + 1 else x[:0] for r, x in la.items()}
        lb = {r: x[trim:-trim] if x.size > 2 * trim + 1 else x[:0] for r, x in lb.items()}
    runs = [r for r in runs if la[r].size and lb[r].size]
    T = sum(min(la[r].size, lb[r].size) for r in runs)
    allpairs = [(i, j) for i in range(len(sa)) for j in range(len(sb))]
    obs = occupancy_r(joint_counts(la, lb, runs, len(sa), len(sb)), T,
                      allpairs).reshape(len(sa), len(sb))
    null = np.empty((n_lock, len(sa), len(sb)))
    for k in range(n_lock):
        shifts = {r: int(rng.integers(1, max(2, min(la[r].size, lb[r].size)))) for r in runs}
        null[k] = occupancy_r(joint_counts(la, lb, runs, len(sa), len(sb), shifts), T,
                              allpairs).reshape(len(sa), len(sb))
    return obs, null, len(runs), T


def lock_flags(obs, null, ia, ib):
    """Family-wise (max over the matched pairs) and pointwise time-locking flags."""
    r = obs[ia, ib]
    max_null = np.nanmax(null[:, ia, ib], axis=1)
    thr = float(np.nanpercentile(max_null, 95))
    pointwise = np.nanpercentile(null[:, ia, ib], 95, axis=0)
    return r, thr, [bool(x > thr) for x in r], [bool(x > t) for x, t in zip(r, pointwise)]


def validate(da, db, sa, sb, ia, ib, null_matchings, residual_matches, Ma, Mb, args):
    """Checks that use occupancy rather than maps, on the eligible full-scope matches."""
    a, b = da["sub"], db["sub"]
    runs = sorted(set(da["decoded"]) & set(db["decoded"]))
    obs, null, n_runs, T = locking_matrices(da, db, sa, sb, runs, args.n_lock,
                                            variant_rng(args, a, b, "lock"))
    r, thr, fwer, pointwise = lock_flags(obs, null, ia, ib)
    stat = float(np.nanmean(r))
    null_stat = np.nanmean(null[:, ia, ib], axis=1)
    # does map matching pick out time-locked pairs better than geometry-relabelled
    # matchings (same observed locking matrix, pairs from the rotation-null draws)?
    relabel = np.array([np.nanmean(obs[ja, jb]) for ja, jb in null_matchings])
    rank_in_row_col = [float((np.sum(obs[i, :] < obs[i, j]) + np.sum(obs[:, j] < obs[i, j]))
                             / (obs.shape[1] + obs.shape[0] - 2)) for i, j in zip(ia, ib)]
    # residual-map matches: their own flags from the same matrices
    pos_a = {s: k for k, s in enumerate(sa)}
    pos_b = {s: k for k, s in enumerate(sb)}
    ra = np.array([pos_a[m["a"]] for m in residual_matches])
    rb = np.array([pos_b[m["b"]] for m in residual_matches])
    _, thr_res, fwer_res, _ = lock_flags(obs, null, ra, rb)
    # run-edge sensitivity: drop 10 TRs from both ends of every run
    obs_t, null_t, _, _ = locking_matrices(da, db, sa, sb, runs, args.n_lock,
                                           variant_rng(args, a, b, "lock_trim"), trim=10)
    stat_t = float(np.nanmean(obs_t[ia, ib]))
    null_t_stat = np.nanmean(null_t[:, ia, ib], axis=1)
    # timing properties of matched pairs, observed and in rotation-null matchings
    props = {"recurrence": (da["rec"][sa], db["rec"][sb]),
             "dwell": (da["dwell"][sa], db["dwell"][sb]),
             "fo": (da["fo"][sa], db["fo"][sb])}
    agree = {}
    for name, (xa, xb) in props.items():
        o = spearmanr(xa[ia], xb[ib]).correlation
        nl = np.array([spearmanr(xa[ja], xb[jb]).correlation for ja, jb in null_matchings])
        agree[name] = {"rho": float(o), "null_mean": float(np.nanmean(nl)),
                       "n_null_finite": int(np.isfinite(nl).sum()), "p": p_greater(o, nl)}
    return {"n_common_runs": n_runs, "n_trs": int(T),
            "locking_mean_r": stat, "locking_null_mean": float(np.nanmean(null_stat)),
            "locking_p": p_greater(stat, null_stat),
            "locking_fwer_threshold": thr,
            "n_matched_time_locked": int(np.sum(fwer)),
            "n_matched_time_locked_pointwise": int(np.sum(pointwise)),
            "per_pair_locking_r": r.tolist(),
            "per_pair_time_locked": fwer,
            "per_pair_time_locked_pointwise": pointwise,
            "matched_vs_relabelled": {"relabelled_mean": float(np.nanmean(relabel)),
                                      "p": p_greater(stat, relabel)},
            "matched_rank_in_row_and_column_median": float(np.median(rank_in_row_col)),
            "residual_matches_time_locked": fwer_res,
            "residual_locking_fwer_threshold": thr_res,
            "trimmed_10": {"locking_mean_r": stat_t,
                           "locking_null_mean": float(np.nanmean(null_t_stat)),
                           "locking_p": p_greater(stat_t, null_t_stat)},
            "rank_agreement": agree}


def ceiling(subjects, args):
    """Split-half fits: all active states, and the half-fit states matched to the
    final fit's content-eligible states (same Hungarian, full scope)."""
    out = {"active": {}, "eligible_matched": {}}
    sub_sets = {}
    for d in subjects:
        hA, hB = d["halves"]["A"], d["halves"]["B"]
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 77, int(d["sub"][-1])]))
        Ma, Mb = hA["maps"][hA["active"]], hB["maps"][hB["active"]]
        r = match(corr_matrix(Ma, Mb))[2]
        null = pair_null(Ma, Mb, args.n_perm_ceiling, rng, NULL)[0]
        out["active"][f"{d['sub']}_A_vs_{d['sub']}_B"] = {"median_r": float(np.median(r)),
                                                           "null_mean": float(null.mean())}
        E = d["maps"][d["eligible"]]
        sel = {}
        for h, hh in (("A", hA), ("B", hB)):
            hm = hh["maps"][hh["active"]]
            ie, ih, _ = match(corr_matrix(E, hm))
            sel[h] = hm[ih[np.argsort(ie)]]
        sub_sets[d["sub"]] = sel
        r = match(corr_matrix(sel["A"], sel["B"]))[2]
        out["eligible_matched"][f"{d['sub']}_A_vs_{d['sub']}_B"] = {"median_r": float(np.median(r))}
    for da, db in combinations(subjects, 2):
        hA, hB = da["halves"]["A"], db["halves"]["B"]
        r = match(corr_matrix(hA["maps"][hA["active"]], hB["maps"][hB["active"]]))[2]
        out["active"][f"{da['sub']}_A_vs_{db['sub']}_B"] = {"median_r": float(np.median(r))}
        r = match(corr_matrix(sub_sets[da["sub"]]["A"], sub_sets[db["sub"]]["B"]))[2]
        out["eligible_matched"][f"{da['sub']}_A_vs_{db['sub']}_B"] = {"median_r": float(np.median(r))}
    return out


# ---------------------------------------------------------------- calibration
# Run before any result under a null is read. Generators without state
# correspondence (G*) must give rate(p < .05) <= 0.075 with the binomial 95% CI
# covering or below 0.05, for the pair test and for the family-wise per-match
# flag; generators with planted counterparts (P*) need pair-test power >= 0.80.

CAL_GENERATORS = ("G1", "G2", "G3", "G4", "G5", "G6", "P1", "P2")
CAL_PAIRS, CAL_DRAWS = 200, 199
CAL_K = (30, 28)
CAL_MAX_FPR, CAL_MIN_POWER, ALPHA = 0.075, 0.80, 0.05


def _sim_subject(W, k, sd_shared, d_id, sd_id, rng, heavy=False):
    """k state maps: coordinates on the shared basis W (d x P) with sd
    `sd_shared` (scalar or per dimension) plus d_id private dimensions."""
    P = W.shape[1]
    G = rng.normal(size=(P, d_id))
    G -= W.T @ (W @ G)
    Bi = np.linalg.qr(G)[0].T

    def coords(n, m):
        if heavy:
            return rng.standard_t(3, size=(n, m)) / np.sqrt(3.0)   # unit sd
        return rng.normal(size=(n, m))
    return (coords(k, W.shape[0]) * sd_shared) @ W + (coords(k, d_id) * sd_id) @ Bi


def _gauss_surrogate(Y, rng):
    """Gaussian states with Y's mean map and sample covariance (Y's state count)."""
    mean = Y.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Y - mean, full_matrices=False)
    g = rng.normal(size=(Y.shape[0], S.size))
    return mean + (g * S) @ Vt / np.sqrt(Y.shape[0] - 1)


def _plant_copies(A, B, rng, frac=0.5, r_target=0.6):
    """Replace frac of B's states by noisy copies of distinct A states
    (expected r with the source = r_target)."""
    B = B.copy()
    n = int(round(frac * B.shape[0]))
    c = np.sqrt(1.0 / r_target ** 2 - 1.0)
    for s, t in zip(rng.choice(A.shape[0], n, replace=False),
                    rng.choice(B.shape[0], n, replace=False)):
        B[t] = A[s] + c * A[s].std() * rng.normal(size=A.shape[1])
    return B


def load_eligible_maps(sub, parc, vt):
    """Contrast maps of the content-eligible states only (no 05g dependency)."""
    hmm = os.path.join(scratch_dir(), "output", "04_combined_hdphmm", parc, sub, "final", f"vt{vt}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.path.join(hmm, "pca_model.pkl"), "rb") as f:
            pca = pickle.load(f)
    maps = np.load(os.path.join(hmm, "state_means_parcel.npy")) - pca.mean_
    flags = pd.read_csv(os.path.join(scratch_dir(), "output", "05e_temporal_trend_a4", parc, sub,
                                     f"vt{vt}", "state_flags.csv")).set_index("state")
    return maps[flags.index[flags["summary_category"] == ELIGIBLE].to_numpy(dtype=int)]


def calibration_pair(gen, i, rng, real, kind):
    """One simulated pair: (pair-test p, any match beyond the max-null 95th pct)."""
    if gen in ("G6", "P2"):
        ordered = [(x, y) for x in SUBJECTS for y in SUBJECTS if x != y]
        x, y = ordered[i % len(ordered)]
        A, B = real[x], _gauss_surrogate(real[y], rng)
    else:
        d = 5 if gen == "G2" else 8
        sd = {"G3": 1.0, "G4": 3.0 / np.sqrt(np.arange(1, d + 1))}.get(gen, 3.0)
        W = np.linalg.qr(rng.normal(size=(156, d)))[0].T
        A, B = (_sim_subject(W, k, sd, 25, 1.0, rng, heavy=(gen == "G5")) for k in CAL_K)
    if gen.startswith("P"):
        B = _plant_copies(A, B, rng)
    r = match(corr_matrix(A, B))[2]
    null, max_r, _, _, _ = pair_null(A, B, CAL_DRAWS, rng, kind)
    return p_greater(np.median(r), null), bool(np.any(r > np.percentile(max_r, 95)))


def calibrate(args):
    from scipy.stats import binomtest
    real = {s: load_eligible_maps(s, args.parcellation, args.vt) for s in SUBJECTS}
    out = {"null": args.cal_null, "n_pairs": CAL_PAIRS, "n_draws": CAL_DRAWS,
           "max_fpr": CAL_MAX_FPR, "min_power": CAL_MIN_POWER, "generators": {}}
    for gen in CAL_GENERATORS:
        ps, fw = [], []
        for i in range(CAL_PAIRS):
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, 55, zlib.crc32(gen.encode()) & 0x7FFFFFFF, i]))
            p_, f_ = calibration_pair(gen, i, rng, real, args.cal_null)
            ps.append(p_)
            fw.append(f_)
        entry = {}
        for name, hits in (("pair_test", np.array(ps) < ALPHA), ("per_match_fwer", np.array(fw))):
            k = int(hits.sum())
            ci = binomtest(k, hits.size).proportion_ci(0.95)
            entry[name] = {"rate": k / hits.size, "ci95": [float(ci.low), float(ci.high)]}
        if gen.startswith("G"):
            entry["pass"] = all(v["rate"] <= CAL_MAX_FPR and v["ci95"][0] <= ALPHA
                                for v in (entry["pair_test"], entry["per_match_fwer"]))
        else:
            entry["pass"] = entry["pair_test"]["rate"] >= CAL_MIN_POWER
        out["generators"][gen] = entry
        logger.info("%s %s: pair %.3f, per-match %.3f, pass %s", args.cal_null, gen,
                    entry["pair_test"]["rate"], entry["per_match_fwer"]["rate"], entry["pass"])
    out["pass"] = all(e["pass"] for e in out["generators"].values())
    out["code_commit"] = git_head()
    write_json(os.path.join(scratch_dir(), "output", "05h_participant_state_matching",
                            args.parcellation, "calibration", f"{args.cal_null}.json"), out)


def run_params(args):
    keep = ("parcellation", "vt", "n_perm", "n_lock", "seed")
    return {k: getattr(args, k) for k in keep}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return None


def git_dirty_scripts():
    """Uncommitted changes under script/ (None when git is unavailable)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--", "."],
                                       text=True).strip()
    except Exception:
        return None


def sha256_files(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def code_sha256():
    here = os.path.dirname(os.path.abspath(__file__))
    return sha256_files([os.path.abspath(__file__)] + [
        os.path.join(here, "utils", f) for f in ("stats.py", "plot_style.py", "recurrence_utils.py")])


def provenance_record(args, data):
    """What a pair result depends on: parameters, code and every input file.
    Identical in every pair file and the cross-pair summary of one run."""
    commit, dirty = git_head(), git_dirty_scripts()
    if commit is None or dirty is None:
        raise SystemExit("cannot read the git state of script/; run from a git checkout")
    if dirty:
        raise SystemExit("uncommitted changes under script/; commit them so the results "
                         "can be traced to code:\n" + dirty)
    inputs = {s: sha256_files(d["input_files"]) for s, d in data.items()}
    inputs["dseg"] = sha256_files([dseg_path(args.parcellation)])
    return {"params": run_params(args), "code_commit": commit,
            "code_sha256": code_sha256(), "inputs_sha256": inputs}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def bh_table(results, all_pairs, getter):
    """BH across pairs for one p-family; pairs where the family is absent are skipped."""
    keys, ps = [], []
    for k in all_pairs:
        try:
            p_ = getter(results[k])
        except KeyError:
            continue
        if p_ is not None and np.isfinite(p_):
            keys.append(k)
            ps.append(p_)
    if not ps:
        return {}
    q = benjamini_hochberg(np.array(ps))
    return {k: {"p": float(p_), "q": float(qq)} for k, p_, qq in zip(keys, ps, q)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_perm_ceiling", type=int, default=200)
    p.add_argument("--n_lock", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="recompute pairs whose summary already exists")
    p.add_argument("--pairs", nargs="*", default=None,
                   help="subset of pairs as subA_subB (default: all 15)")
    p.add_argument("--mode", choices=["run", "calibrate"], default="run")
    p.add_argument("--cal_null", choices=["state_mixing", "configuration"], default=NULL,
                   help="null to calibrate (--mode calibrate)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    if args.mode == "calibrate":
        calibrate(args)
        return
    S = scratch_dir()
    groups, cortex = parcel_groups(args.parcellation)
    data = {s: load_subject(s, args.parcellation, args.vt) for s in SUBJECTS}
    out_root = os.path.join(S, "output", "05h_participant_state_matching", args.parcellation)
    all_pairs = [f"{a}_{b}" for a, b in combinations(SUBJECTS, 2)]
    prov = provenance_record(args, data)
    for key in args.pairs or all_pairs:
        a, b = key.split("_")
        path = os.path.join(out_root, "pairs", key, "summary.json")
        if os.path.exists(path) and not args.force:
            with open(path) as f:
                old = json.load(f)
            if all(old.get(k) == v for k, v in prov.items()):
                continue
            logger.info("%s: existing summary has other params, code or inputs; recomputing", key)
        t0 = time.time()
        others = [data[s] for s in SUBJECTS if s not in (a, b)]
        res = analyse_pair(data[a], data[b], others, groups, cortex, args)
        res.update(prov)
        write_json(path, res)
        logger.info("%s done in %.0f s", key, time.time() - t0)
    if args.pairs:
        return
    results = {}
    for key in all_pairs:
        with open(os.path.join(out_root, "pairs", key, "summary.json")) as f:
            results[key] = json.load(f)
        if any(results[key].get(k) != v for k, v in prov.items()):
            raise RuntimeError(f"{key}: pair summary does not match this run's params, code or inputs")
    variants = sorted({v for k in all_pairs for v in results[k]["variants"]})
    # one BH family per (null, variant) for the pair test, and per (test, state
    # set) for the specificity and validation tests; pairs lacking a test (fewer
    # than 3 states in a set) are left out of its family
    tables = {NULL: {}, "pc_blocks": {}, "families": {}}
    for var in variants:
        for null_name in (NULL, "pc_blocks"):
            tables[null_name][var] = bh_table(
                results, all_pairs, lambda r, var=var, n=null_name: r["variants"][var][n]["p"])
    for set_name in ("eligible", "active"):
        var = f"{set_name}|full|raw"
        families = {
            "network_collapse": lambda r: r["variants"][var]["network_collapse"][NULL]["p"],
            "within_network_residual": lambda r: r["variants"][var]["within_network_residual"][NULL]["p"],
            "raw_cosine": lambda r: r["variants"][var]["raw_cosine"][NULL]["p"],
            "shared_removed_3": lambda r: r["variants"][var]["shared_removed"]["3"][NULL]["p"],
            "shared_removed_3_plus_mean": lambda r: r["variants"][var]["shared_removed"]["3_plus_mean"][NULL]["p"],
            "collisions_more_a_to_b": lambda r: r["variants"][var]["collisions"]["p_more_a_to_b"],
            "collisions_more_b_to_a": lambda r: r["variants"][var]["collisions"]["p_more_b_to_a"],
        }
        if set_name == "eligible":
            families.update({
                "locking": lambda r: r["variants"][var]["validation"]["locking_p"],
                "locking_trimmed_10": lambda r: r["variants"][var]["validation"]["trimmed_10"]["locking_p"],
                "locking_matched_vs_relabelled": lambda r: r["variants"][var]["validation"]["matched_vs_relabelled"]["p"],
                "rank_recurrence": lambda r: r["variants"][var]["validation"]["rank_agreement"]["recurrence"]["p"],
                "rank_fo": lambda r: r["variants"][var]["validation"]["rank_agreement"]["fo"]["p"],
                "rank_dwell": lambda r: r["variants"][var]["validation"]["rank_agreement"]["dwell"]["p"],
            })
        for name, getter in families.items():
            tables["families"][f"{set_name}|{name}"] = bh_table(results, all_pairs, getter)
    summary = {"bh_across_pairs": tables, "ceiling": ceiling(list(data.values()), args),
               "n_states": {s: {"active": len(d["active"]), "eligible": len(d["eligible"]),
                                "onset": len(d["onset"])} for s, d in data.items()},
               "max_abs_pca_mean": {s: d["max_abs_pca_mean"] for s, d in data.items()},
               **prov, "n_perm_ceiling": args.n_perm_ceiling,
               "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    write_json(os.path.join(out_root, "cross_subject_summary.json"), summary)


if __name__ == "__main__":
    main()
