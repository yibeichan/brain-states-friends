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
and cortex only (the 100 Schaefer parcels). Sets: content-eligible (primary)
and all active states (secondary; matches labelled by 05e category and by the
05g_excluded_state_check class); run-onset-anchored states as a positive control.

Matching: rectangular Hungarian on 1 - r (both directions' denominators are
reported), plus each A state's unconstrained best match (collisions counted).

Pair statistic: median matched r. Nulls (Hungarian re-run, maps re-z-scored
in every draw; 1000 draws; A rotated in half the draws, B in the other half):
  configuration  Haar rotation of a subject's centred state maps within their
                 own span; keeps the Gram matrix and the repertoire's subspace
                 and destroys only which direction is which state (primary)
  pc_blocks      Haar rotation of state means within blocks of 5 adjacent
                 PCs of that subject's PCA, back-projected (subspace sharing
                 at a realistic variance spectrum)
No spin-test null: it needs sphere coordinates for the Schaefer parcels and a
dependency this environment does not have, and the plan rated it coarse and
liberal for subcortex; the PC-block and shared-component baselines cover the
subspace and network-level confounds.
Specificity baselines: (a) network collapse, maps replaced by their
network/structure means; (b) projecting out the top k (1, 3, 5) components
of a PCA over the other four subjects' state maps, with the configuration
null re-run at the pre-stated k = 3 ("specific correspondence").
Ceiling: the same pipeline on the split-half fits, within subject (A_half vs
A_otherhalf) and across subjects (half vs half), active states.
Validation independent of the maps: time-locked co-occurrence of matched
states over the runs both subjects watched, against a within-run circular
shift; recurrence / dwell / FO rank agreement of matched pairs, computed in
every configuration-null draw as well.

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
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.stats import benjamini_hochberg  # noqa: E402
from utils.plot_style import assign_network  # noqa: E402

logger = logging.getLogger("05h_participant_state_matching")

SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
ELIGIBLE = "eligible_for_content_analysis"
ONSET_FLAGS = ("run_onset", "a_anchored", "b_anchored")
ACTIVE_FO = 0.02
N_CORTEX = 100
K_SHARED = (1, 3, 5)
K_SHARED_PRIMARY = 3
BLOCK = 5


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


def parcel_groups(parc):
    atlas = os.environ.get("ATLAS_DIR")
    if not atlas:
        raise SystemExit("ATLAS_DIR is not set")
    df = pd.read_csv(os.path.join(atlas, parc, f"{parc}_dseg.tsv"), sep="\t")
    groups = []
    for idx in range(len(df)):
        row = df[df["index"] == idx + 1].iloc[0]
        g = assign_network(row["label"])
        groups.append(g if g is not None else row.get("network_label", "Unknown"))
    return np.array(groups)


def active_from_decoded(decoded, K):
    fo = np.vstack([np.bincount(l, minlength=K) / l.size for l in decoded.values()])
    return np.flatnonzero((fo > ACTIVE_FO).any(axis=0))


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
    with open(os.path.join(hmm, "decoded_states.pkl"), "rb") as f:
        decoded = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
    K = maps.shape[0]
    flags = pd.read_csv(os.path.join(S, "output", "05e_temporal_trend_a4", parc, sub,
                                     f"vt{vt}", "state_flags.csv")).set_index("state")
    classes_path = os.path.join(S, "output", "05g_excluded_state_check", parc, sub,
                                f"vt{vt}", "state_classes.csv")
    classes = (pd.read_csv(classes_path).set_index("state")["class"].to_dict()
               if os.path.exists(classes_path) else {})
    active = active_from_decoded(decoded, K)
    eligible = [int(s) for s in flags.index[flags["summary_category"] == ELIGIBLE]]
    onset = [int(s) for s in active if any(bool(flags.loc[s, c]) for c in ONSET_FLAGS)]
    halves = {}
    for h in ("A", "B"):
        hd = os.path.join(base, "split_half", h)
        hm = np.load(os.path.join(hd, "state_means_parcel.npy"))
        with open(os.path.join(hd, "decoded_states.pkl"), "rb") as f:
            hdec = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
        halves[h] = {"maps": hm, "active": active_from_decoded(hdec, hm.shape[0])}
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
            "K": K, "flags": flags, "classes": classes, "active": [int(s) for s in active],
            "eligible": eligible, "onset": onset, "halves": halves,
            "rec": flags["recurrence_score"].to_numpy(dtype=float),
            "fo": trs / trs.sum(), "dwell": dwell,
            "max_abs_pca_mean": float(np.abs(pca.mean_).max())}


# ---------------------------------------------------------------- pair analysis

def pair_null(Ma, Mb, n_perm, rng, kind, extra=None):
    """Median matched r under a null; A rotated in half the draws, B in the other."""
    out = np.empty(n_perm)
    max_r = np.empty(n_perm)
    collisions = np.empty(n_perm)
    for i in range(n_perm):
        rot_a = i % 2 == 0
        if kind == "configuration":
            A = rotate_configuration(Ma, rng) if rot_a else Ma
            B = Mb if rot_a else rotate_configuration(Mb, rng)
        else:  # pc_blocks
            muA, cA, muB, cB = extra
            A = rotate_pc_blocks(muA, cA, rng) if rot_a else Ma
            B = Mb if rot_a else rotate_pc_blocks(muB, cB, rng)
        Rn = corr_matrix(A, B)
        _, _, r = match(Rn)
        out[i] = np.median(r)
        max_r[i] = r.max()
        collisions[i] = many_to_one(Rn)[1]
    return out, max_r, collisions


def p_greater(obs, null):
    return float((1 + np.sum(null >= obs)) / (1 + null.size))


def analyse_pair(da, db, others, groups, args, rng):
    res = {"pair": f"{da['sub']}_{db['sub']}", "variants": {}}
    scopes = {"full": slice(None), "cortex": slice(0, N_CORTEX)}
    sets = {"eligible": ("eligible", "eligible"), "active": ("active", "active"),
            "onset_control": ("onset", "onset")}
    for set_name, (ka, kb) in sets.items():
        sa, sb = da[ka], db[kb]
        if len(sa) < 3 or len(sb) < 3:
            continue
        for scope_name, sl in scopes.items():
            for centred in (False, True):
                if set_name == "onset_control" and (scope_name != "full" or centred):
                    continue
                Ma, Mb = da["maps"][sa][:, sl], db["maps"][sb][:, sl]
                if centred:
                    Ma, Mb = Ma - Ma.mean(0, keepdims=True), Mb - Mb.mean(0, keepdims=True)
                R = corr_matrix(Ma, Mb)
                ia, ib, r = match(R)
                best_r, collisions = many_to_one(R)
                key = f"{set_name}|{scope_name}|{'centred' if centred else 'raw'}"
                null_c, max_c, coll_c = pair_null(Ma, Mb, args.n_perm, rng, "configuration")
                v = {"n_a": len(sa), "n_b": len(sb), "n_matched": int(r.size),
                     "median_r": float(np.median(r)), "mean_r": float(r.mean()),
                     "configuration": {"null_mean": float(null_c.mean()),
                                       "p": p_greater(np.median(r), null_c)},
                     "n_pairs_beyond_max_null95": int(np.sum(r > np.percentile(max_c, 95))),
                     "many_to_one_collisions": collisions,
                     "many_to_one_collisions_null_mean":
                         float(coll_c.mean()),
                     "many_to_one_collisions_p": p_greater(
                         collisions, coll_c),
                     "best_r_median": float(np.median(best_r)),
                     "max_null95": float(np.percentile(max_c, 95)),
                     "matches_full": [{"a": int(sa[i]), "b": int(sb[j]), "r": float(R[i, j]),
                                       "beyond_max_null95": bool(R[i, j] > np.percentile(max_c, 95))}
                                      for i, j in zip(ia, ib)]}
                if not centred and set_name != "onset_control":
                    pcs = (da["mu_pc"][sa], da["comps"], db["mu_pc"][sb], db["comps"])
                    if scope_name == "full":
                        null_b, _, _ = pair_null(Ma, Mb, args.n_perm, rng, "pc_blocks", pcs)
                        v["pc_blocks"] = {"null_mean": float(null_b.mean()),
                                          "p": p_greater(np.median(r), null_b)}
                    # specificity (a): network collapse
                    if scope_name == "full":
                        Rn = corr_matrix(collapse_networks(Ma, groups), collapse_networks(Mb, groups))
                        v["network_collapse_median_r"] = float(np.median(match(Rn)[2]))
                    # specificity (b): shared components from the other four subjects
                    shared = {}
                    for k in K_SHARED:
                        V = shared_components([o["maps"][o[ka]][:, sl] for o in others], k)
                        Pa, Pb = project_out(zrows(Ma), V), project_out(zrows(Mb), V)
                        ka_i, kb_i, rk = match(corr_matrix(Pa, Pb))
                        entry = {"median_r": float(np.median(rk))}
                        if k == K_SHARED_PRIMARY:
                            nk, mk, _ = pair_null(Pa, Pb, args.n_perm, rng, "configuration")
                            thr_k = float(np.percentile(mk, 95))
                            entry.update({"configuration_p": p_greater(np.median(rk), nk),
                                          "null_mean": float(nk.mean()),
                                          "n_pairs_beyond_max_null95": int(np.sum(rk > thr_k)),
                                          "max_null95": thr_k,
                                          "matches": [{"a": int(sa[i]), "b": int(sb[j]),
                                                       "r": float(r_), "beyond_max_null95": bool(r_ > thr_k)}
                                                      for i, j, r_ in zip(ka_i, kb_i, rk)]})
                        shared[str(k)] = entry
                    v["shared_removed"] = shared
                if set_name == "active" and scope_name == "full" and not centred:
                    v["matches"] = [
                        {"a": int(sa[i]), "b": int(sb[j]), "r": float(R[i, j]),
                         "cat_a": da["flags"].loc[sa[i], "summary_category"],
                         "cat_b": db["flags"].loc[sb[j], "summary_category"],
                         "class_a": da["classes"].get(sa[i], "eligible"),
                         "class_b": db["classes"].get(sb[j], "eligible")}
                        for i, j in zip(ia, ib)]
                if set_name == "eligible" and scope_name == "full" and not centred:
                    v["validation"] = validate(da, db, sa, sb, ia, ib, Ma, Mb, args, rng)
                res["variants"][key] = v
    return res


def validate(da, db, sa, sb, ia, ib, Ma, Mb, args, rng):
    """Map-independent checks on the eligible full-scope matches."""
    # time-locked co-occurrence over common runs
    runs = sorted(set(da["decoded"]) & set(db["decoded"]))
    la = {r: np.where(np.isin(da["decoded"][r], sa),
                      np.searchsorted(sa, da["decoded"][r]), -1) for r in runs}
    lb = {r: np.where(np.isin(db["decoded"][r], sb),
                      np.searchsorted(sb, db["decoded"][r]), -1) for r in runs}
    T = sum(min(da["decoded"][r].size, db["decoded"][r].size) for r in runs)
    pairs = list(zip(ia.tolist(), ib.tolist()))
    N = joint_counts(la, lb, runs, len(sa), len(sb))
    obs = occupancy_r(N, T, pairs)
    null = np.empty((args.n_lock, len(pairs)))
    for k in range(args.n_lock):
        shifts = {r: int(rng.integers(1, max(2, min(da["decoded"][r].size,
                                                    db["decoded"][r].size)))) for r in runs}
        null[k] = occupancy_r(joint_counts(la, lb, runs, len(sa), len(sb), shifts), T, pairs)
    stat = float(np.nanmean(obs))
    per_pair_p95 = np.nanpercentile(null, 95, axis=0)
    # recurrence / dwell / FO rank agreement of matched pairs, observed and null
    props = {"recurrence": (da["rec"][sa], db["rec"][sb]),
             "dwell": (da["dwell"][sa], db["dwell"][sb]),
             "fo": (da["fo"][sa], db["fo"][sb])}
    agree = {}
    for name, (xa, xb) in props.items():
        o = spearmanr(xa[ia], xb[ib]).correlation
        nl = np.empty(args.n_perm)
        for i in range(args.n_perm):
            A = rotate_configuration(Ma, rng) if i % 2 == 0 else Ma
            B = Mb if i % 2 == 0 else rotate_configuration(Mb, rng)
            ja, jb, _ = match(corr_matrix(A, B))
            nl[i] = spearmanr(xa[ja], xb[jb]).correlation
        agree[name] = {"rho": float(o), "null_mean": float(np.nanmean(nl)),
                       "p": float((1 + np.sum(nl >= o)) / (1 + nl.size))}
    return {"n_common_runs": len(runs), "n_trs": int(T),
            "locking_mean_r": stat, "locking_null_mean": float(np.nanmean(null)),
            "locking_p": float((1 + np.sum(np.nanmean(null, axis=1) >= stat)) / (1 + args.n_lock)),
            "n_matched_time_locked": int(np.sum(obs > per_pair_p95)),
            "per_pair_locking_r": obs.tolist(),
            "per_pair_time_locked": [bool(o > t) for o, t in zip(obs, per_pair_p95)],
            "rank_agreement": agree}


def ceiling(subjects, args, rng):
    """Same pipeline on split-half fits (active states, full scope)."""
    out = {}
    for d in subjects:
        hA, hB = d["halves"]["A"], d["halves"]["B"]
        Ma, Mb = hA["maps"][hA["active"]], hB["maps"][hB["active"]]
        r = match(corr_matrix(Ma, Mb))[2]
        null, _, _ = pair_null(Ma, Mb, args.n_perm_ceiling, rng, "configuration")
        out[f"{d['sub']}_A_vs_{d['sub']}_B"] = {"median_r": float(np.median(r)),
                                                 "null_mean": float(null.mean())}
    for da, db in combinations(subjects, 2):
        hA, hB = da["halves"]["A"], db["halves"]["B"]
        r = match(corr_matrix(hA["maps"][hA["active"]], hB["maps"][hB["active"]]))[2]
        out[f"{da['sub']}_A_vs_{db['sub']}_B"] = {"median_r": float(np.median(r))}
    return out


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
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    S = scratch_dir()
    groups = parcel_groups(args.parcellation)
    data = {s: load_subject(s, args.parcellation, args.vt) for s in SUBJECTS}
    out_root = os.path.join(S, "output", "05h_participant_state_matching", args.parcellation)
    all_pairs = [f"{a}_{b}" for a, b in combinations(SUBJECTS, 2)]
    todo = args.pairs or all_pairs
    for key in todo:
        a, b = key.split("_")
        path = os.path.join(out_root, "pairs", key, "summary.json")
        if os.path.exists(path) and not args.force:
            continue
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence(
            [args.seed, int(a[-1]), int(b[-1])]))
        others = [data[s] for s in SUBJECTS if s not in (a, b)]
        res = analyse_pair(data[a], data[b], others, groups, args, rng)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "w") as f:
            json.dump(res, f, indent=1, default=float)
        os.replace(path + ".part", path)
        logger.info("%s done in %.0f s", key, time.time() - t0)
    if args.pairs:
        return
    # cross-pair tables (BH across the 15 pairs within each variant)
    results = {}
    for key in all_pairs:
        with open(os.path.join(out_root, "pairs", key, "summary.json")) as f:
            results[key] = json.load(f)
    table = {}
    for var in results[all_pairs[0]]["variants"]:
        ps = [results[k]["variants"][var]["configuration"]["p"] for k in all_pairs
              if var in results[k]["variants"]]
        q = benjamini_hochberg(np.array(ps))
        table[var] = {k: {"median_r": results[k]["variants"][var]["median_r"],
                          "p_configuration": results[k]["variants"][var]["configuration"]["p"],
                          "q_configuration": float(qq)}
                      for k, qq in zip([k for k in all_pairs if var in results[k]["variants"]], q)}
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 77]))
    summary = {"pairs": table, "ceiling": ceiling(list(data.values()), args, rng),
               "n_states": {s: {"active": len(d["active"]), "eligible": len(d["eligible"]),
                                "onset": len(d["onset"])} for s, d in data.items()},
               "max_abs_pca_mean": {s: d["max_abs_pca_mean"] for s, d in data.items()},
               "params": vars(args),
               "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    with open(os.path.join(out_root, "cross_subject_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
