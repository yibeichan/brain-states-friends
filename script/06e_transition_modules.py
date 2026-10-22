#!/usr/bin/env python3
"""06e_transition_modules.py - modular structure of the state transition graph.

R3 asks how transitions between brain states are organised. 06b reports what
linked states have in common (recurrence assortativity, FC-transition
coupling, network homophily) and finds Louvain communities descriptively,
without a modularity value or a null. This stage asks, per subject, whether
the transition graph is modular beyond what simpler structure produces, and
whether that modularity says anything beyond the pairwise similarity of the
states. The design, nulls and decision rules were fixed before any of this
was run (see the plan and decision records named in the summary JSON).

Graphs (all over the active states, recurrence > 0, as 06b A1):
    M         model transmat_, self-loops removed, P >= 0.01 (= 06b's graphml)
    M02/M005  the same at 0.02 / 0.005
    Mrenorm   off-diagonal rows rescaled to sum to 1, then P >= 0.01
    Mfull     all off-diagonal P
    E         empirical transitions from the decoded runs (within runs,
              normalised over all K like the model), P >= 0.01
Statistic: directed weighted modularity (Leicht-Newman), best of 50 fixed
Louvain seeds at resolution 1, applied identically to every graph.
Nulls:
    N1     rewiring that keeps in/out degree and reciprocal dyads, then
           strength-matched weight reassignment (Rubinov & Sporns 2011)
    N1row  within-row shuffle of off-diagonal entries (for Mfull)
    N2     segment-order shuffle within runs (for E): keeps which states occur
           in which runs and for how long, destroys only their order
    N3     a non-modular chain with the fitted stickiness and popularity,
           simulated with the observed run lengths (for E)
    PRED   similarity-predicted graphs: log P regressed on pairwise
           predictors + strengths; null = prediction + permuted residuals
Also: episode bootstraps of E (stability), LOSO fold graphs (fit
uncertainty), sensitivity graphs (reliable states only; A-B-A flicker
collapsed; near-duplicate states merged; first 10 TRs of runs dropped).

Modes:
    --mode tasks       print the (subject, variant) task table
    --mode run         run one task (--task_id, or SLURM_ARRAY_TASK_ID):
                       null draws in 100-draw chunks, each written atomically;
                       finished chunks are skipped, so a requeued task resumes
    --mode aggregate   observed statistics, p-values, per-subject verdicts,
                       module characterisation, module dwell, cross-subject
                       stimulus locking, and the pre-registered R3 mapping

Inputs (frozen, under $SCRATCH_DIR/output):
    04_combined_hdphmm/{parc}/{sub}/final/vt0.95/{best_model.pkl,
        decoded_states.pkl, state_means_pca.npy, state_covars.npy,
        state_means_parcel.npy, pca_model.pkl}
    04_combined_hdphmm/{parc}/{sub}/loso/season_{s}/{best_model.pkl, decoded_states.pkl}
    05a_recurrence_analysis/{parc}/{sub}/vt0.95/recurrence_summary.json
    05e_temporal_trend_a4/{parc}/{sub}/vt0.95/state_flags.csv
    05f_state_fc/{parc}/{sub}/vt0.95/fc_similarity_corr_rv.npy
    06b_transition_structure/{parc}/{sub}/vt0.95/transition_graph.graphml
Output:
    06e_transition_modules/{parc}/{sub}/vt0.95/nulls/{variant}/chunk_###.npz
    06e_transition_modules/{parc}/{sub}/vt0.95/transition_modules_summary.json,
        partitions.json, characterisation.json
    06e_transition_modules/{parc}/r3_mapping.json

Per-subject inference only; the R3 mapping counts subjects that pass and
carries no pooled test.
"""
import os
import sys
import json
import time
import pickle
import zlib
import logging
import argparse
import platform
import subprocess
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import transition_modules as tm  # noqa: E402
from utils.common import _get_season  # noqa: E402
from utils.recurrence_utils import active_state_indices  # noqa: E402

logger = logging.getLogger("06e_transition_modules")

SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
PARC = "atlas-4S156Parcels"
VT = "0.95"
SEEDS = list(range(50))          # Louvain seeds, fixed before results
RESOLUTION = 1.0
THR = 0.01
CHUNK = 100
ALPHA = 0.05
EFFECT_FLOOR = 0.05              # Q - mean(Q_null) under N1 on M
BOOT_ARI_FLOOR = 0.5
MIN_TRS, MIN_EXITS = 30, 20      # reliable-state rule
PLAN_NOTES = [
    "projects/2026-10-21-brain-states-friends-plan-v2-transition-modules-in-r3-fig-3a-reviewed-r3-mapping-pending-user-approval.md",
    "decisions/2026-10-21-brain-states-friends-r3-transition-modules-pre-registered-r3-mapping-approved-2026-10-21-before-any-run.md",
    "decisions/2026-10-21-brain-states-friends-r3-transition-modules-test-a-replaced-by-a-similarity-predicted-null-2026-10-21-before-any-run.md",
]

# variant -> number of draws
VARIANTS = {
    "N1_M": 1000, "N1_M02": 1000, "N1_M005": 1000, "N1_Mrenorm": 1000,
    "N1row_Mfull": 1000, "N2_E": 1000, "N3_E": 500, "PRED_M": 1000,
    "BOOT_E": 200, "N1_Mreliable": 200, "N1_Eaba": 200, "N1_Emerge90": 200,
    "N1_Edrop10": 200,
}
LOSO_DRAWS = 1000


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class _ModelStub:
    pass


class _NoJaxUnpickler(pickle.Unpickler):
    """Load a pickled StickyHDPHMM(_JAX) without importing jax: any class from
    hdphmm_jax becomes a plain attribute holder."""
    def find_class(self, module, name):
        if "hdphmm_jax" in module:
            return _ModelStub
        return super().find_class(module, name)


def load_transmat(path):
    with open(path, "rb") as f:
        model = _NoJaxUnpickler(f).load()
    P = np.asarray(model.transmat_, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise RuntimeError(f"bad transmat_ in {path}")
    return P


def scratch_dir():
    s = os.environ.get("SCRATCH_DIR")
    if not s:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
        s = os.environ.get("SCRATCH_DIR")
    if not s:
        raise SystemExit("SCRATCH_DIR is not set")
    return s


def out_dir(sub):
    return os.path.join(scratch_dir(), "output", "06e_transition_modules", PARC, sub, f"vt{VT}")


def _p(stage, sub, *parts):
    return os.path.join(scratch_dir(), "output", stage, PARC, sub, f"vt{VT}", *parts)


def load_subject(sub):
    S = scratch_dir()
    hmm = os.path.join(S, "output", "04_combined_hdphmm", PARC, sub, "final", f"vt{VT}")
    P = load_transmat(os.path.join(hmm, "best_model.pkl"))
    K = P.shape[0]
    with open(os.path.join(hmm, "decoded_states.pkl"), "rb") as f:
        decoded = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
    with open(_p("05a_recurrence_analysis", sub, "recurrence_summary.json")) as f:
        rec = np.asarray(json.load(f)["recurrence_scores"], dtype=float)
    nodes = [int(i) for i in np.flatnonzero(rec > 0)]

    G06b = nx.read_graphml(_p("06b_transition_structure", sub, "transition_graph.graphml"))
    W = tm.weight_matrix(P, nodes, THR)
    W06b = np.zeros_like(W)
    loc = {s: k for k, s in enumerate(nodes)}
    if sorted(int(n) for n in G06b.nodes()) != nodes:
        raise RuntimeError(f"{sub}: active states differ from 06b's graph")
    for u, v, d in G06b.edges(data=True):
        W06b[loc[int(u)], loc[int(v)]] = float(d["weight"])
    if not np.allclose(W, W06b, atol=1e-12):
        raise RuntimeError(f"{sub}: rebuilt M differs from 06b transition_graph.graphml")
    networks = [G06b.nodes[str(s)]["dominant_network"] for s in nodes]

    rv = np.load(_p("05f_state_fc", sub, "fc_similarity_corr_rv.npy"))
    flags = pd.read_csv(_p("05e_temporal_trend_a4", sub, "state_flags.csv")).set_index("state")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.path.join(hmm, "pca_model.pkl"), "rb") as f:
            pca = pickle.load(f)
    maps = np.load(os.path.join(hmm, "state_means_parcel.npy")) - pca.mean_
    mu = np.load(os.path.join(hmm, "state_means_pca.npy"))
    var = np.load(os.path.join(hmm, "state_covars.npy"))
    if var.ndim != 2:
        raise RuntimeError(f"{sub}: expected diagonal covariances, got {var.shape}")
    return {"sub": sub, "P": P, "K": K, "decoded": decoded, "rec": rec, "nodes": nodes,
            "networks": networks, "rv": rv, "flags": flags, "maps": maps,
            "mu": mu, "var": var, "hmm_dir": hmm,
            "max_abs_pca_mean": float(np.abs(pca.mean_).max())}


# --------------------------------------------------------------------------
# Graph variants
# --------------------------------------------------------------------------

def empirical_P(decoded, K):
    C = sum(tm.run_transition_counts(decoded, K).values())
    return tm.counts_to_P(C)


def state_exit_counts(decoded, K):
    exits = np.zeros(K, dtype=int)
    for seq in decoded.values():
        lab, _ = tm.run_segments(seq)
        if lab.size > 1:
            np.add.at(exits, lab[:-1], 1)
    return exits


def reliable_nodes(d):
    K = d["K"]
    trs = np.bincount(np.concatenate(list(d["decoded"].values())), minlength=K)
    exits = state_exit_counts(d["decoded"], K)
    fl = d["flags"]
    keep = []
    for s in d["nodes"]:
        bad = any(bool(fl.loc[s, c]) for c in ("sub_hrf", "run_onset", "a_anchored",
                                                 "b_anchored", "unused"))
        if not bad and trs[s] >= MIN_TRS and exits[s] >= MIN_EXITS:
            keep.append(s)
    return keep, trs, exits


def collapse_aba(decoded, max_len=4):
    out = {}
    for r, seq in decoded.items():
        lab, ln = tm.run_segments(seq)
        lab = lab.copy()
        for k in range(1, lab.size - 1):
            if ln[k] <= max_len and lab[k - 1] == lab[k + 1] and lab[k] != lab[k - 1]:
                lab[k] = lab[k - 1]
        out[r] = np.repeat(lab, ln)
    return out


def merge_similar(decoded, maps, nodes, r_thresh=0.9):
    Z = maps[nodes] - maps[nodes].mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    R = Z @ Z.T
    parent = {s: s for s in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if R[a, b] > r_thresh:
                ra, rb = find(nodes[a]), find(nodes[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                pairs.append((nodes[a], nodes[b], float(R[a, b])))
    relabel = np.arange(maps.shape[0])
    for s in nodes:
        relabel[s] = find(s)
    merged = {r: relabel[seq] for r, seq in decoded.items()}
    new_nodes = sorted(set(find(s) for s in nodes))
    return merged, new_nodes, pairs


def drop_onsets(decoded, n=10):
    return {r: seq[n:] for r, seq in decoded.items() if seq.size > n}


def pairwise_predictors(d, nodes):
    """Dyadic predictors over `nodes` (n x n), defined without transitions."""
    idx = np.asarray(nodes)
    Z = d["maps"][idx] - d["maps"][idx].mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    map_r = Z @ Z.T
    rv = d["rv"][np.ix_(idx, idx)]
    bhat = tm.bhattacharyya_diag(d["mu"][idx], d["var"][idx])
    net = np.array([d["networks"][d["nodes"].index(s)] for s in nodes])
    same_net = (net[:, None] == net[None, :]).astype(float)
    rec = d["rec"][idx]
    drec = np.abs(rec[:, None] - rec[None, :])
    runs = sorted(d["decoded"])
    fo = np.stack([np.bincount(d["decoded"][r], minlength=d["K"])[idx] / d["decoded"][r].size
                   for r in runs])
    co_run = spearmanr(fo).correlation if fo.shape[1] > 2 else np.eye(len(idx))
    seasons = np.array([_get_season(r) for r in runs])
    fo_season = np.stack([fo[seasons == s].mean(axis=0) for s in np.unique(seasons)])
    co_season = np.corrcoef(fo_season.T)
    return {"fc_rv": rv, "map_r": map_r, "bhattacharyya": bhat, "same_network": same_net,
            "abs_delta_recurrence": drec, "fo_cooccur_runs": np.asarray(co_run),
            "fo_cooccur_seasons": co_season}


def fc_nodes(d, nodes):
    """Nodes with a defined FC similarity row (05f drops states with < 30 TRs)."""
    keep = list(nodes)
    while True:
        rv = d["rv"][np.ix_(keep, keep)].copy()
        np.fill_diagonal(rv, 0.0)
        n_nan = np.isnan(rv).sum(axis=1)
        if n_nan.max(initial=0) == 0:
            return keep
        keep.pop(int(np.argmax(n_nan)))   # drop the state with the most undefined pairs


def build_variant(d, variant):
    """Observed graph and a draw function for one null variant."""
    P, K, decoded, nodes = d["P"], d["K"], d["decoded"], d["nodes"]
    if variant in ("N1_M", "N1_M02", "N1_M005", "N1_Mrenorm"):
        thr = {"N1_M": THR, "N1_M02": 0.02, "N1_M005": 0.005, "N1_Mrenorm": THR}[variant]
        W = tm.weight_matrix(P, nodes, thr, renorm_rows=(variant == "N1_Mrenorm"))
        return W, nodes, lambda rng: tm.rewire_null(W, rng)
    if variant == "N1row_Mfull":
        W = tm.weight_matrix(P, nodes, None)
        return W, nodes, lambda rng: (tm.row_shuffle_null(W, rng), {})
    if variant in ("N2_E", "N3_E", "BOOT_E"):
        W = tm.weight_matrix(empirical_P(decoded, K), nodes, THR)
        if variant == "N2_E":
            segs = tm.segment_tables(decoded)
            return W, nodes, lambda rng: (tm.weight_matrix(
                tm.counts_to_P(tm.shuffled_counts(segs, K, rng)), nodes, THR), {})
        if variant == "N3_E":
            Q, _, _ = tm.nonmodular_chain(P, nodes)
            loc = {s: k for k, s in enumerate(nodes)}
            runs = [r for r in sorted(decoded) if decoded[r][0] in loc]
            lengths = [decoded[r].size for r in runs]
            starts = [loc[int(decoded[r][0])] for r in runs]
            n = len(nodes)
            return W, nodes, lambda rng: (tm.weight_matrix(
                tm.counts_to_P(tm.simulate_chain_counts(Q, lengths, starts, rng)),
                list(range(n)), THR), {})
        run_counts = tm.run_transition_counts(decoded, K)
        runs = sorted(run_counts)

        def boot(rng):
            pick = rng.choice(len(runs), size=len(runs), replace=True)
            C = sum(run_counts[runs[i]] for i in pick)
            return tm.weight_matrix(tm.counts_to_P(C), nodes, THR), {}
        return W, nodes, boot
    if variant == "PRED_M":
        sub_nodes = fc_nodes(d, nodes)
        W_full = tm.weight_matrix(P, sub_nodes, None)
        preds = pairwise_predictors(d, sub_nodes)
        model = tm.predicted_weight_model(W_full, preds)
        W = tm.weight_matrix(P, sub_nodes, THR)
        n = len(sub_nodes)
        return W, sub_nodes, lambda rng: (tm.predicted_null_graph(model, n, THR, rng),
                                          {"r2": model["r2"]})
    if variant == "N1_Mreliable":
        keep, _, _ = reliable_nodes(d)
        W = tm.weight_matrix(P, keep, THR)
        return W, keep, lambda rng: tm.rewire_null(W, rng)
    if variant in ("N1_Eaba", "N1_Emerge90", "N1_Edrop10"):
        if variant == "N1_Eaba":
            seqs, nd = collapse_aba(decoded), nodes
        elif variant == "N1_Emerge90":
            seqs, nd, _ = merge_similar(decoded, d["maps"], nodes)
        else:
            seqs, nd = drop_onsets(decoded), nodes
        W = tm.weight_matrix(empirical_P(seqs, K), nd, THR)
        return W, nd, lambda rng: tm.rewire_null(W, rng)
    if variant.startswith("LOSO_"):
        fold = os.path.join(os.path.dirname(os.path.dirname(d["hmm_dir"])), "loso",
                            variant.split("_", 1)[1])
        Pf = load_transmat(os.path.join(fold, "best_model.pkl"))
        with open(os.path.join(fold, "decoded_states.pkl"), "rb") as f:
            dec_f = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
        nf = [int(s) for s in active_state_indices(dec_f, Pf.shape[0])]
        W = tm.weight_matrix(Pf, nf, THR)
        return W, nf, lambda rng: tm.rewire_null(W, rng)
    raise ValueError(variant)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

def loso_folds(sub):
    base = os.path.join(scratch_dir(), "output", "04_combined_hdphmm", PARC, sub, "loso")
    return sorted(x for x in os.listdir(base) if x.startswith("season_"))


def task_table():
    rows = []
    for sub in SUBJECTS:
        for v, n in VARIANTS.items():
            rows.append((sub, v, n))
        for f in loso_folds(sub):
            rows.append((sub, f"LOSO_{f}", LOSO_DRAWS))
    return rows


def variant_seed(variant):
    return zlib.crc32(variant.encode()) & 0x7FFFFFFF


def run_task(task_id, base_seed):
    sub, variant, n_draws = task_table()[task_id]
    d = load_subject(sub)
    W_obs, nodes, draw = build_variant(d, variant)
    vdir = os.path.join(out_dir(sub), "nulls", variant)
    os.makedirs(vdir, exist_ok=True)
    q_obs, lab_obs, share_obs = tm.best_modularity(W_obs, SEEDS, RESOLUTION)
    logger.info("%s %s: %d nodes, %d edges, observed Q %.4f", sub, variant,
                len(nodes), int((W_obs > 0).sum()), q_obs)
    sub_num = int(sub.split("-")[1])
    for c in range((n_draws + CHUNK - 1) // CHUNK):
        path = os.path.join(vdir, f"chunk_{c:03d}.npz")
        if os.path.exists(path):
            continue
        rng = np.random.default_rng(np.random.SeedSequence(
            [base_seed, sub_num, variant_seed(variant), c]))
        n = min(CHUNK, n_draws - c * CHUNK)
        q = np.empty(n)
        recip = np.empty(n)
        serr = np.full(n, np.nan)
        acc = np.full(n, np.nan)
        labels = []
        t0 = time.time()
        for i in range(n):
            Wn, info = draw(rng)
            q[i], lab, _ = tm.best_modularity(Wn, SEEDS, RESOLUTION)
            recip[i] = tm.reciprocity(Wn)
            if variant.startswith("N1_") or variant.startswith("LOSO_"):
                serr[i] = tm.strength_error(Wn, W_obs)
                acc[i] = info.get("accepted", np.nan)
            if variant == "BOOT_E":
                labels.append(lab)
        tmp = path + ".part.npz"
        np.savez(tmp, q=q, reciprocity=recip, strength_error=serr, accepted=acc,
                 labels=np.array(labels) if labels else np.zeros((0, 0)))
        os.replace(tmp, path)
        logger.info("%s %s chunk %d: %d draws in %.0f s", sub, variant, c, n, time.time() - t0)
    obs_path = os.path.join(vdir, "observed.json")
    if not os.path.exists(obs_path):
        tmp = obs_path + ".part"
        with open(tmp, "w") as f:
            json.dump({"q": q_obs, "labels": lab_obs.tolist(), "nodes": nodes,
                       "share_seeds_at_best": share_obs,
                       "n_edges": int((W_obs > 0).sum()),
                       "reciprocity": tm.reciprocity(W_obs)}, f)
        os.replace(tmp, obs_path)


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

def load_null(sub, variant, n_draws):
    vdir = os.path.join(out_dir(sub), "nulls", variant)
    n_chunks = (n_draws + CHUNK - 1) // CHUNK
    files = [os.path.join(vdir, f"chunk_{c:03d}.npz") for c in range(n_chunks)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing or not os.path.exists(os.path.join(vdir, "observed.json")):
        raise FileNotFoundError(f"{sub} {variant}: {len(missing)} chunks missing")
    parts = [np.load(f) for f in files]
    with open(os.path.join(vdir, "observed.json")) as f:
        obs = json.load(f)
    null = {k: np.concatenate([p[k] for p in parts]) for k in ("q", "reciprocity",
                                                             "strength_error", "accepted")}
    if variant == "BOOT_E":
        null["labels"] = np.concatenate([p["labels"] for p in parts])
    return obs, null


def null_test(obs_q, null_q):
    p = float((1 + np.sum(null_q >= obs_q)) / (1 + null_q.size))
    return {"q_obs": obs_q, "q_null_mean": float(null_q.mean()),
            "q_null_sd": float(null_q.std(ddof=1)), "effect": float(obs_q - null_q.mean()),
            "z": float((obs_q - null_q.mean()) / null_q.std(ddof=1)), "p": p,
            "n_draws": int(null_q.size)}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return None


def characterise(d, nodes, labels, rng, n_perm):
    """Descriptive: what are the modules made of?"""
    sub_nodes = fc_nodes(d, nodes)
    keep = [nodes.index(s) for s in sub_nodes]
    lab = np.asarray(labels)[keep]
    preds = pairwise_predictors(d, sub_nodes)
    co = (lab[:, None] == lab[None, :]).astype(float)
    mrqap = tm.mrqap_dekker(co, preds, rng, n_perm=n_perm)
    n = len(sub_nodes)
    iu = np.triu_indices(n, 1)
    sizes = np.bincount(lab)
    univariate = {}
    for name, M in preds.items():
        obs = M[iu][co[iu] == 1].mean() - M[iu][co[iu] == 0].mean()
        nulls = np.empty(n_perm)
        for p in range(n_perm):
            lp = rng.permutation(lab)
            cp = lp[:, None] == lp[None, :]
            nulls[p] = M[iu][cp[iu]].mean() - M[iu][~cp[iu]].mean()
        univariate[name] = {"within_minus_between": float(obs),
                            "p_two_sided": float((1 + np.sum(np.abs(nulls) >= abs(obs)))
                                                 / (1 + n_perm))}
    net = np.array([d["networks"][d["nodes"].index(s)] for s in nodes])
    fl = d["flags"]
    lab_all = np.asarray(labels)
    table = []
    for m in np.unique(lab_all):
        members = [s for s, l in zip(nodes, lab_all) if l == m]
        flagged = [s for s in members if fl.loc[s, "summary_category"]
                   != "eligible_for_content_analysis"]
        table.append({"module": int(m), "states": members, "n": len(members),
                      "networks": pd.Series(net[lab_all == m]).value_counts().to_dict(),
                      "mean_recurrence": float(d["rec"][members].mean()),
                      "n_flagged": len(flagged),
                      "categories": fl.loc[members, "summary_category"].value_counts().to_dict()})
    return {"n_states_used": n, "dropped_no_fc": [s for s in nodes if s not in sub_nodes],
            "module_sizes_after_drop": sizes.tolist(),
            "mrqap": mrqap, "univariate": univariate,
            "ari_network": float(adjusted_rand_score(net, lab_all)),
            "ami_network": float(adjusted_mutual_info_score(net, lab_all)),
            "modules": table}


def module_occupancy_tables(d, nodes, labels):
    mod = np.full(d["K"], -1)
    mod[np.asarray(nodes)] = labels
    season_rows, decile_rows = {}, np.zeros((10, int(np.max(labels)) + 1))
    for r, seq in d["decoded"].items():
        m = mod[seq]
        s = _get_season(r)
        season_rows.setdefault(s, np.zeros(int(np.max(labels)) + 1))
        ok = m >= 0
        season_rows[s] += np.bincount(m[ok], minlength=decile_rows.shape[1])
        dec = np.minimum((np.arange(seq.size) * 10) // seq.size, 9)
        for k in range(10):
            mk = m[(dec == k) & ok]
            decile_rows[k] += np.bincount(mk, minlength=decile_rows.shape[1])
    norm = lambda a: (a / a.sum()).tolist()
    return {"by_season": {int(s): norm(v) for s, v in sorted(season_rows.items())},
            "by_run_decile": [norm(row) for row in decile_rows]}


def aggregate(base_seed, n_perm_char, n_dwell, n_lock):
    results, partitions, data = {}, {}, {}
    for sub in SUBJECTS:
        d = load_subject(sub)
        data[sub] = d
        res = {"sub": sub, "n_nodes": len(d["nodes"]),
               "max_abs_pca_mean": d["max_abs_pca_mean"], "tests": {}}
        for v, n in VARIANTS.items():
            obs, null = load_null(sub, v, n)
            entry = null_test(obs["q"], null["q"]) if v != "BOOT_E" else {"q_obs": obs["q"]}
            entry.update({"n_nodes": len(obs["nodes"]), "n_edges": obs["n_edges"],
                          "reciprocity_obs": obs["reciprocity"],
                          "reciprocity_null_mean": float(np.nanmean(null["reciprocity"])),
                          "share_seeds_at_best": obs["share_seeds_at_best"]})
            if np.isfinite(null["strength_error"]).any():
                entry["strength_error_mean"] = float(np.nanmean(null["strength_error"]))
            res["tests"][v] = entry
        # stability: bootstrap partitions vs E's best partition
        obsE, bootE = load_null(sub, "BOOT_E", VARIANTS["BOOT_E"])
        aris = [adjusted_rand_score(obsE["labels"], b) for b in bootE["labels"]]
        res["stability"] = {"boot_ari_median": float(np.median(aris)),
                            "boot_ari_iqr": [float(np.percentile(aris, 25)),
                                             float(np.percentile(aris, 75))]}
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, 7, int(sub[-1])]))
        D = tm.coassignment(bootE["labels"])
        cons, n_it, converged = tm.consensus_partition(D, rng)
        # partitions: M's best partition is the tested and displayed one
        obsM, _ = load_null(sub, "N1_M", VARIANTS["N1_M"])
        res["stability"].update({
            "consensus_iterations": n_it, "consensus_converged": converged,
            "ari_consensus_vs_M": float(adjusted_rand_score(cons, obsM["labels"])),
            "ari_E_vs_M": float(adjusted_rand_score(obsE["labels"], obsM["labels"]))})
        np.save(os.path.join(out_dir(sub), "bootstrap_coassignment.npy"), D)
        # LOSO folds
        folds = {}
        for f in loso_folds(sub):
            o, nl = load_null(sub, f"LOSO_{f}", LOSO_DRAWS)
            folds[f] = null_test(o["q"], nl["q"])
        res["loso"] = folds
        res["loso_n_pass"] = int(sum(v["p"] < ALPHA for v in folds.values()))
        # sensitivity ARIs against M on shared nodes
        for v in ("N1_Mreliable", "N1_Eaba", "N1_Emerge90", "N1_Edrop10"):
            o, _ = load_null(sub, v, VARIANTS[v])
            shared = [s for s in o["nodes"] if s in obsM["nodes"]]
            a = [o["labels"][o["nodes"].index(s)] for s in shared]
            b = [obsM["labels"][obsM["nodes"].index(s)] for s in shared]
            res["tests"][v]["ari_vs_M_partition"] = float(adjusted_rand_score(a, b))
        t = res["tests"]
        res["modular"] = bool(
            t["N1_M"]["p"] < ALPHA and t["N1_M02"]["p"] < ALPHA and t["N2_E"]["p"] < ALPHA
            and t["N1_M"]["effect"] >= EFFECT_FLOOR
            and res["stability"]["boot_ari_median"] >= BOOT_ARI_FLOOR)
        res["test_a_pred_p"] = t["PRED_M"]["p"]
        # (b) module dwell vs segment shuffle
        mod = np.full(d["K"], -1)
        mod[np.asarray(obsM["nodes"])] = obsM["labels"]
        dw_obs = tm.module_dwell(d["decoded"], mod)
        segs = tm.segment_tables(d["decoded"])
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, 8, int(sub[-1])]))
        dw_null = np.array([tm.module_dwell_shuffled(segs, mod, rng) for _ in range(n_dwell)])
        state_dwell = float(np.mean(np.concatenate([tm.run_segments(s)[1]
                                                    for s in d["decoded"].values()])))
        res["test_b_dwell"] = {"module_dwell": dw_obs, "state_dwell": state_dwell,
                               "null_mean": float(dw_null.mean()),
                               "p": float((1 + np.sum(dw_null >= dw_obs)) / (1 + n_dwell))}
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, 9, int(sub[-1])]))
        res["characterisation"] = characterise(d, obsM["nodes"], obsM["labels"], rng, n_perm_char)
        res["occupancy"] = module_occupancy_tables(d, obsM["nodes"], obsM["labels"])
        partitions[sub] = {"nodes": obsM["nodes"], "labels": obsM["labels"],
                           "networks": d["networks"], "consensus_labels": cons.tolist()}
        results[sub] = res
        with open(os.path.join(out_dir(sub), "partitions.json"), "w") as f:
            json.dump(partitions[sub], f, indent=1)

    # (c) stimulus locking across subject pairs
    lock = {}
    for i, a in enumerate(SUBJECTS):
        for b in SUBJECTS[i + 1:]:
            pa, pb = partitions[a], partitions[b]
            cent = {}
            mods = {}
            for s, p in ((a, pa), (b, pb)):
                d = data[s]
                mod = np.full(d["K"], -1)
                mod[np.asarray(p["nodes"])] = p["labels"]
                trs = np.bincount(np.concatenate(list(d["decoded"].values())), minlength=d["K"])
                cs = []
                for m in range(max(p["labels"]) + 1):
                    members = [n for n, l in zip(p["nodes"], p["labels"]) if l == m]
                    w = trs[members] / trs[members].sum()
                    cs.append((d["maps"][members] * w[:, None]).sum(axis=0))
                cent[s] = np.array(cs)
                mods[s] = {r: mod[seq] for r, seq in d["decoded"].items()}
            ia, ib, r_maps = tm.match_modules(cent[a], cent[b])
            runs = sorted(set(mods[a]) & set(mods[b]))
            pairs = list(zip(ia.tolist(), ib.tolist()))
            obs = tm.occupancy_locking(mods[a], mods[b], pairs, runs)
            stat = float(np.nanmean(obs))
            rng = np.random.default_rng(np.random.SeedSequence(
                [base_seed, 10, int(a[-1]), int(b[-1])]))
            null = np.empty(n_lock)
            for k in range(n_lock):
                shifted = {}
                for r in runs:
                    x = mods[b][r]
                    shifted[r] = np.roll(x, int(rng.integers(1, x.size)))
                null[k] = np.nanmean(tm.occupancy_locking(mods[a], shifted, pairs, runs))
            lock[f"{a}_{b}"] = {"mean_r": stat, "null_mean": float(null.mean()),
                                "p": float((1 + np.sum(null >= stat)) / (1 + n_lock)),
                                "n_runs": len(runs), "matched_map_r": r_maps.tolist(),
                                "per_module_r": obs.tolist()}
    for sub in SUBJECTS:
        ps = [v["p"] for k, v in lock.items() if sub in k.split("_")]
        results[sub]["test_c_locking"] = {"n_pairs": len(ps),
                                          "n_pairs_p_lt_alpha": int(sum(p < ALPHA for p in ps))}

    # pre-registered R3 mapping
    modular = [s for s in SUBJECTS if results[s]["modular"]]
    loso_ok = [s for s in modular if results[s]["loso_n_pass"] >= 4]
    new = []
    for s in loso_ok:
        r = results[s]
        a = r["test_a_pred_p"] < ALPHA
        b = r["test_b_dwell"]["p"] < ALPHA
        c = r["test_c_locking"]["n_pairs_p_lt_alpha"] >= r["test_c_locking"]["n_pairs"] / 2
        r["new_criteria"] = {"a_similarity_predicted": a, "b_dwell": b, "c_locking": c}
        if a or b or c:
            new.append(s)
    if len(modular) >= 5 and len(loso_ok) == len(modular) and len(new) >= 4:
        outcome = "modules_lead_R3"
    elif len(modular) >= 5:
        outcome = "modules_as_display_only"
    else:
        outcome = "not_modular_fc_ordering"
    mapping = {"outcome": outcome, "modular_subjects": modular,
               "loso_ok_subjects": loso_ok, "new_criterion_subjects": new,
               "plan_and_decision_notes": PLAN_NOTES, "code_commit": git_head(),
               "frozen_parameters": {"seeds": len(SEEDS), "resolution": RESOLUTION,
                                     "threshold": THR, "alpha": ALPHA,
                                     "effect_floor": EFFECT_FLOOR,
                                     "boot_ari_floor": BOOT_ARI_FLOOR,
                                     "variants": VARIANTS, "loso_draws": LOSO_DRAWS},
               "locking": lock,
               "provenance": {"python": platform.python_version(),
                              "networkx": nx.__version__, "numpy": np.__version__,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    for sub in SUBJECTS:
        results[sub]["code_commit"] = mapping["code_commit"]
        with open(os.path.join(out_dir(sub), "transition_modules_summary.json"), "w") as f:
            json.dump(results[sub], f, indent=1, default=float)
    root = os.path.join(scratch_dir(), "output", "06e_transition_modules", PARC)
    with open(os.path.join(root, "r3_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=1, default=float)
    logger.info("R3 mapping outcome: %s (modular: %s)", outcome, modular)


# --------------------------------------------------------------------------
# Calibration of the N1 test (separate job, run before reading any p-value)
# --------------------------------------------------------------------------

CAL_NULLS = {
    "N1": ("N1_exchangeable", "chain_nonmodular", "sbm_ratio1.5", "sbm_ratio2.0"),
    "N1perm": ("N1perm_exchangeable", "chain_nonmodular", "sbm_ratio1.5", "sbm_ratio2.0"),
    "N2": ("chain_nonmodular", "chain_modular_ratio1.5", "chain_modular_ratio2.0"),
}
CAL_GRAPHS, CAL_CHUNK, CAL_DRAWS, CAL_SEEDS = 200, 20, 200, list(range(10))


def calibration_tasks(cal_null):
    return [(g, c) for g in CAL_NULLS[cal_null] for c in range(CAL_GRAPHS // CAL_CHUNK)]


def _sbm_graph(W_ref, ratio, rng, k=4):
    """Planted modules matched to W_ref's size, density and weight spread:
    within-module edge probability `ratio` times the between-module one."""
    n = W_ref.shape[0]
    density = (W_ref > 0).sum() / (n * (n - 1))
    lab = rng.permutation(np.arange(n) % k)
    same = lab[:, None] == lab[None, :]
    frac_same = (same.sum() - n) / (n * (n - 1))
    p_out = density / (frac_same * ratio + (1 - frac_same))
    A = rng.random((n, n)) < np.where(same, ratio * p_out, p_out)
    np.fill_diagonal(A, False)
    w = W_ref[W_ref > 0]
    return np.where(A, rng.choice(w, size=(n, n)), 0.0)


def run_calibration(task_id, base_seed, cal_null="N1", ref_sub="sub-01"):
    """Rate of p < alpha for one null on graphs with known structure.

    N1 / N1perm: graph-level nulls, tested on graphs that are themselves
    null draws (exchangeable), on graphs re-estimated from a simulated
    non-modular chain with sub-01's stickiness and popularity, and on
    planted-module graphs matched to sub-01's graph. N2 (segment shuffle):
    tested on simulated non-modular and planted-module chains, whose
    sequences are shuffled exactly as the real decoded runs are.
    """
    gen, chunk = calibration_tasks(cal_null)[task_id]
    out = os.path.join(scratch_dir(), "output", "06e_transition_modules", PARC,
                       "calibration", cal_null, gen)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"chunk_{chunk:03d}.npz")
    if os.path.exists(path):
        return
    d = load_subject(ref_sub)
    W_ref = tm.weight_matrix(d["P"], d["nodes"], THR)
    rng = np.random.default_rng(np.random.SeedSequence(
        [base_seed, 99, variant_seed(cal_null), variant_seed(gen), chunk]))
    Q, _, _ = tm.nonmodular_chain(d["P"], d["nodes"])
    n = Q.shape[0]
    loc = {s: k for k, s in enumerate(d["nodes"])}
    runs = [r for r in sorted(d["decoded"]) if d["decoded"][r][0] in loc]
    lengths = [d["decoded"][r].size for r in runs]
    starts = [loc[int(d["decoded"][r][0])] for r in runs]
    null_fn = {"N1": tm.rewire_null, "N1perm": tm.rewire_null_permuted_weights}.get(cal_null)
    p_vals, effects = [], []
    for _ in range(CAL_CHUNK):
        if cal_null == "N2":
            if gen == "chain_nonmodular":
                Qg = Q
            else:
                lab = rng.permutation(np.arange(n) % 4)
                Qg = tm.modular_chain(Q, lab, float(gen.split("ratio")[1]))
            seqs = tm.simulate_chain_sequences(Qg, lengths, starts, rng)
            C = sum(tm.run_transition_counts(seqs, n).values())
            W = tm.weight_matrix(tm.counts_to_P(C), list(range(n)), THR)
            segs = tm.segment_tables(seqs)
            draw = lambda: tm.weight_matrix(tm.counts_to_P(
                tm.shuffled_counts(segs, n, rng)), list(range(n)), THR)
        else:
            if gen.endswith("_exchangeable"):
                W = null_fn(W_ref, rng)[0]
            elif gen == "chain_nonmodular":
                C = tm.simulate_chain_counts(Q, lengths, starts, rng)
                W = tm.weight_matrix(tm.counts_to_P(C), list(range(n)), THR)
            else:
                W = _sbm_graph(W_ref, float(gen.split("ratio")[1]), rng)
            draw = lambda W=W: null_fn(W, rng)[0]
        q_obs = tm.best_modularity(W, CAL_SEEDS)[0]
        qn = np.array([tm.best_modularity(draw(), CAL_SEEDS)[0] for _ in range(CAL_DRAWS)])
        p_vals.append((1 + np.sum(qn >= q_obs)) / (1 + CAL_DRAWS))
        effects.append(q_obs - qn.mean())
    tmp = path + ".part.npz"
    np.savez(tmp, p=np.array(p_vals), effect=np.array(effects))
    os.replace(tmp, path)


def calibration_summary(cal_null="N1"):
    from scipy.stats import binomtest
    root = os.path.join(scratch_dir(), "output", "06e_transition_modules", PARC,
                        "calibration", cal_null)
    out = {}
    for gen in CAL_NULLS[cal_null]:
        files = sorted(os.listdir(os.path.join(root, gen)))
        parts = [np.load(os.path.join(root, gen, f)) for f in files if f.endswith(".npz")]
        p = np.concatenate([x["p"] for x in parts])
        e = np.concatenate([x["effect"] for x in parts])
        k = int(np.sum(p < ALPHA))
        ci = binomtest(k, p.size).proportion_ci(0.95)
        out[gen] = {"n_graphs": int(p.size), "rate_p_lt_alpha": k / p.size,
                    "rate_ci95": [float(ci.low), float(ci.high)],
                    "effect_mean": float(e.mean()), "effect_p95": float(np.percentile(e, 95))}
    with open(os.path.join(root, "calibration_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    logger.info("calibration %s: %s", cal_null, {g: v["rate_p_lt_alpha"] for g, v in out.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["tasks", "run", "aggregate", "calibrate",
                                      "calibrate_tasks", "calibrate_summary"], required=True)
    p.add_argument("--task_id", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_perm_char", type=int, default=5000)
    p.add_argument("--n_dwell", type=int, default=1000)
    p.add_argument("--n_lock", type=int, default=1000)
    p.add_argument("--cal_null", choices=["N1", "N1perm", "N2"], default="N1")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.mode == "tasks":
        for i, (s, v, n) in enumerate(task_table()):
            print(i, s, v, n)
    elif a.mode == "run":
        if a.task_id < 0:
            raise SystemExit("--task_id or SLURM_ARRAY_TASK_ID required")
        run_task(a.task_id, a.seed)
    elif a.mode == "calibrate_tasks":
        for i, (g, c) in enumerate(calibration_tasks(a.cal_null)):
            print(i, g, c)
    elif a.mode == "calibrate":
        if a.task_id < 0:
            raise SystemExit("--task_id or SLURM_ARRAY_TASK_ID required")
        run_calibration(a.task_id, a.seed, a.cal_null)
    elif a.mode == "calibrate_summary":
        calibration_summary(a.cal_null)
    else:
        aggregate(a.seed, a.n_perm_char, a.n_dwell, a.n_lock)


if __name__ == "__main__":
    main()
