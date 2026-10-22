#!/usr/bin/env python3
"""05g_excluded_state_check.py - are the excluded states copies of content-eligible states?

05e_a4 sets aside some active states from content analyses. Its
`low_confidence` flag is a timing flag (median dwell < 3 TRs, under the
hemodynamic response), so it says nothing about a state's map. This script
asks, per subject, what each active non-eligible state is:

  O   run-onset-anchored: 05e_a4 run_onset, a_anchored or b_anchored is set.
      Read from the raw booleans, because 05e_a4's summary category gives
      sub-HRF priority over run onset and so files some run-onset-anchored
      states under low_confidence. Not classified further.
  I   interruption: brief visits to an eligible-like pattern that interrupt
      one parent state (... P P s P P ...), i.e. a copy split off that state.
  B   blend: visits that sit between two different states (... A A s B B ...)
      with a map explained by both better than by either.
  I/B both; D neither; insufficient: fewer than --min_visits uncensored visits.

Visits. Within each run the decoded Viterbi path is collapsed into segments.
A visit's neighbours a and b are the labels of the adjacent segments. Visits
in a run's first or last segment have no a or no b; they are censored from
the sequence measures and counted.

Maps. Every map in this script is an empirical state map: the mean of the
03a PC scores over the state's decoded TRs, back-projected to the 156 parcels
through the PCA loadings (no pca.mean_, which is ~1e-10 here), row z-scored
across parcels. Using data means for every state keeps the short-state maps
and their references on one footing. The model's own back-projected means
(state_means_parcel) are reported alongside as r_max_model.

Measures per state s:
  excess return  share of uncensored visits with a == b, against its value if
                 a and b were independent given s (sum_j P(a=j) P(b=j)),
                 tested by permuting b across s's visits.
  r_parent       r(s, parent), parent = most frequent a == b neighbour.
  ref_nn         95th percentile, over eligible states, of each eligible
                 state's largest r with another eligible state: how similar
                 two distinct states can be in this fit.
  ref_boot       5th percentile of r(parent map, parent map rebuilt from
                 random parent TR windows with the same lengths as s's
                 visits): how low r(s, parent) can fall from estimation noise
                 alone when s is short.
  dR2            OLS of s's map on its two most frequent through-partners
                 (a != b), no intercept, weights unconstrained: R2 of both
                 minus the larger single-partner R2. Null: the same rule
                 applied to every eligible state with enough visits.
  r_switch       r(s, mean map in a window of s's median visit length around
                 direct a<->b switches with nothing in between): how close s
                 is to plain hemodynamic blurring of a switch.

Rules (fixed before results):
  I if excess-return q < alpha (BH within subject, over the classified
      states) AND r_parent >= ref_nn AND r_parent >= ref_boot.
  B if p_through >= 0.5 AND dR2 > 95th percentile of the eligible-state null.
      The null holds one value per eligible state (~15-30), so B is a
      descriptive cut with no multiplicity correction.
  B is labelled "HRF blend" when r_switch >= sqrt(R2 of the blend).

Inputs (frozen, under $SCRATCH_DIR/output):
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/{decoded_states.pkl,
        pca_model.pkl, state_means_pca.npy, state_means_parcel.npy,
        state_covars.npy}
    03a_pca4combined_hmm/{parc}/{sub}/{splits/primary.json, projected/}
    05e_temporal_trend_a4/{parc}/{sub}/vt{vt}/state_flags.csv
Output:
    {SCRATCH_DIR}/output/05g_excluded_state_check/{parc}/{sub}/vt{vt}/
        state_classes.csv, state_classes_summary.json

Per-subject inference only; nothing is pooled.
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
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.hmm_io import load_split, get_projected_dir, load_projected_runs  # noqa: E402
from utils.state_flags_io import load_state_flags  # noqa: E402
from utils.stats import benjamini_hochberg  # noqa: E402

logger = logging.getLogger("05g_excluded_state_check")

ELIGIBLE = "eligible_for_content_analysis"
ONSET_FLAGS = ("run_onset", "a_anchored", "b_anchored")
ACTIVE_FO = 0.02  # active = FO > 0.02 in at least one run (05a / recurrence_utils)


# ---------------------------------------------------------------- sequences

def run_segments(labels):
    """Collapse a 1-D label array into (label, start, length) segments."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [labels.size]]))
    return [(int(labels[s]), int(s), int(n)) for s, n in zip(starts, lengths)]


def visit_table(decoded_by_run):
    """One row per segment: run, state, start, length, prev label, next label.

    prev/next are -1 at a run's first/last segment (censored).
    """
    rows = []
    for run_id, labels in decoded_by_run.items():
        segs = run_segments(labels)
        for i, (lab, start, length) in enumerate(segs):
            prev = segs[i - 1][0] if i > 0 else -1
            nxt = segs[i + 1][0] if i < len(segs) - 1 else -1
            rows.append((run_id, lab, start, length, prev, nxt))
    return pd.DataFrame(rows, columns=["run", "state", "start", "length", "prev", "next"])


def excess_return_test(a, b, n_perm, rng):
    """Return share (a == b), its independence expectation, and a one-sided p.

    a, b: neighbour labels of the uncensored visits. The null permutes b
    across visits, which keeps both marginals and breaks only the pairing.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.size
    obs = float(np.mean(a == b))
    pa = Counter(a.tolist())
    pb = Counter(b.tolist())
    expected = float(sum(pa[j] * pb.get(j, 0) for j in pa) / (n * n))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean(a == rng.permutation(b))
    p = float((1 + np.sum(null >= obs)) / (1 + n_perm))
    return obs, expected, p


# ---------------------------------------------------------------- maps

def zrows(M):
    M = np.atleast_2d(np.asarray(M, dtype=float))
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0  # zero-variance row -> r = 0 with everything (as 04rb)
    return (M - mu) / sd


def corr(x, y):
    """Pearson r between two maps (1-D)."""
    return float((zrows(x) @ zrows(y).T).item() / np.asarray(x).size)


def corr_matrix(A, B):
    return (zrows(A) @ zrows(B).T) / np.asarray(A).shape[1]


def empirical_pc_means(X_by_run, decoded_by_run, K):
    """Mean PC score per state over its decoded TRs, and TR counts."""
    n_pcs = next(iter(X_by_run.values())).shape[1]
    sums = np.zeros((K, n_pcs))
    counts = np.zeros(K, dtype=int)
    for run_id, labels in decoded_by_run.items():
        X = X_by_run[run_id]
        np.add.at(sums, labels, X)
        counts += np.bincount(labels, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]
    return means, counts


def nn_reference(maps_eligible, q=95):
    """q-th percentile of each eligible state's largest r with another one."""
    C = corr_matrix(maps_eligible, maps_eligible)
    np.fill_diagonal(C, -np.inf)
    return float(np.percentile(C.max(axis=1), q)), C.max(axis=1)


def bootstrap_parent_reference(parent_windows, X_by_run, visit_lengths, comps,
                               parent_map, n_boot, rng, q=5):
    """q-th percentile of r(parent map, parent map from s-shaped windows).

    parent_windows: list of (run, start, length) parent segments. For each
    draw and each of s's visit lengths L, pick a parent segment uniformly
    among those at least L long (or the longest one if none is) and a random
    L-window inside it; the draw's map is the mean over all picked TRs.
    """
    # Window sums from one prefix-sum array over the parent's own TRs.
    lens = np.array([w[2] for w in parent_windows])
    seg_rows = [X_by_run[run][start:start + length] for run, start, length in parent_windows]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    csum = np.vstack([np.zeros((1, seg_rows[0].shape[1])),
                      np.cumsum(np.vstack(seg_rows), axis=0)])
    longest = int(np.argmax(lens))
    total = np.zeros((n_boot, csum.shape[1]))
    n_rows = 0
    for L, count in Counter(int(v) for v in visit_lengths).items():
        ok = np.flatnonzero(lens >= L)
        if ok.size:
            j = rng.choice(ok, size=(n_boot, count))
            L_eff = np.full(j.shape, L)
        else:
            j = np.full((n_boot, count), longest)
            L_eff = np.full(j.shape, int(lens[longest]))
        off = (rng.random(j.shape) * (lens[j] - L_eff + 1)).astype(int)
        lo = starts[j] + off
        total += (csum[lo + L_eff] - csum[lo]).sum(axis=1)
        n_rows += int(L_eff[0, 0]) * count
    boot_maps = (total / n_rows) @ comps
    r = corr_matrix(boot_maps, parent_map[None, :]).ravel()
    return float(np.percentile(r, q)), r


def blend_fit(s_map, a_map, b_map):
    """OLS of z(s) on [z(a), z(b)], no intercept, unconstrained weights."""
    y = zrows(s_map).ravel()
    X = zrows(np.vstack([a_map, b_map])).T
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ w
    r2 = 1.0 - float(resid @ resid) / float(y @ y)
    r_a, r_b = corr(s_map, a_map), corr(s_map, b_map)
    return {"r2_blend": r2, "w_a": float(w[0]), "w_b": float(w[1]),
            "r_a": r_a, "r_b": r_b, "dR2": r2 - max(r_a ** 2, r_b ** 2)}


def through_partners(a, b):
    """Most frequent unordered {a, b} pair with a != b, or None."""
    pairs = Counter(tuple(sorted((int(x), int(y)))) for x, y in zip(a, b) if x != y)
    return pairs.most_common(1)[0][0] if pairs else None


def switch_window_map(visits, X_by_run, pair, L, comps):
    """Mean map in an L-TR window centred on direct a<->b switches."""
    a, b = pair
    rows = []
    for run, g in visits.groupby("run", sort=False):
        g = g.sort_values("start")
        lab = g["state"].to_numpy()
        st = g["start"].to_numpy()
        for i in range(len(g) - 1):
            if {lab[i], lab[i + 1]} == {a, b}:
                t = st[i + 1]
                lo = max(0, t - L // 2)
                hi = min(X_by_run[run].shape[0], lo + L)
                rows.append(X_by_run[run][lo:hi])
    if not rows:
        return None, 0
    return np.vstack(rows).mean(axis=0) @ comps, len(rows)


# ---------------------------------------------------------------- classify

def classify(row, alpha):
    if row["run_onset_anchored"]:
        return "O"
    if row["n_uncensored"] < row["min_visits"]:
        return "insufficient"
    is_i = (row["q_excess_return"] < alpha
            and row["r_parent"] >= row["ref_nn"]
            and row["r_parent"] >= row["ref_boot"])
    is_b = (row["p_through"] >= 0.5
            and np.isfinite(row["dR2"])
            and row["dR2"] > row["dR2_null_p95"])
    if is_i and is_b:
        return "I/B"
    if is_i:
        return "I"
    if is_b:
        return "B"
    return "D"


# ---------------------------------------------------------------- driver

def load_inputs(sub_id, parc, vt, scratch_dir):
    hmm_dir = os.path.join(scratch_dir, "output", "04_combined_hdphmm", parc,
                           sub_id, "final", f"vt{vt}")
    pca_base = os.path.join(scratch_dir, "output", "03a_pca4combined_hmm",
                            parc, sub_id)
    with open(os.path.join(hmm_dir, "decoded_states.pkl"), "rb") as f:
        decoded = {k: np.asarray(v, dtype=int) for k, v in pickle.load(f).items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn pickle-version warning; PCA arrays only
        with open(os.path.join(hmm_dir, "pca_model.pkl"), "rb") as f:
            pca = pickle.load(f)
    means_pca = np.load(os.path.join(hmm_dir, "state_means_pca.npy"))
    means_parcel = np.load(os.path.join(hmm_dir, "state_means_parcel.npy"))
    covars = np.load(os.path.join(hmm_dir, "state_covars.npy"))
    n_pcs = means_pca.shape[1]
    comps = pca.components_[:n_pcs]
    gap = float(np.abs(means_parcel - pca.mean_ - means_pca @ comps).max())
    if gap > 1e-8:
        raise RuntimeError(f"{sub_id}: state_means_parcel != means_pca @ W + mean (gap {gap})")

    split = load_split(pca_base)
    projected = get_projected_dir(pca_base)
    X_by_run = {}
    for name in ("train", "valid", "test"):
        for run_id in split[name]:
            X, _ = load_projected_runs([run_id], projected, n_pcs, name)
            X_by_run[run_id] = X
    if set(X_by_run) != set(decoded):
        raise RuntimeError(f"{sub_id}: projected runs and decoded runs differ")
    for run_id, labels in decoded.items():
        if X_by_run[run_id].shape[0] != labels.size:
            raise RuntimeError(f"{sub_id} {run_id}: TR count mismatch")

    flags = load_state_flags(sub_id=sub_id, parcellation=parc,
                             scratch_dir=scratch_dir, vt=vt)
    if flags is None:
        raise FileNotFoundError(f"{sub_id}: state_flags.csv missing")
    return {"decoded": decoded, "X_by_run": X_by_run, "comps": comps,
            "means_parcel_contrast": means_parcel - pca.mean_, "covars": covars,
            "flags": flags.set_index("state"), "n_pcs": n_pcs,
            "max_abs_pca_mean": float(np.abs(pca.mean_).max())}


def active_states(decoded, K):
    fo = np.vstack([np.bincount(l, minlength=K) / l.size for l in decoded.values()])
    return np.flatnonzero((fo > ACTIVE_FO).any(axis=0))


def analyse_subject(sub_id, parc, vt, scratch_dir, args):
    sub_idx = int(sub_id.split("-")[1])
    d = load_inputs(sub_id, parc, vt, scratch_dir)
    decoded, X_by_run, comps, flags = d["decoded"], d["X_by_run"], d["comps"], d["flags"]
    K = d["means_parcel_contrast"].shape[0]
    if len(flags) != K or list(flags.index) != list(range(K)):
        raise RuntimeError(f"{sub_id}: state_flags rows do not match K = {K}")

    active = active_states(decoded, K)
    eligible = [int(s) for s in flags.index[flags["summary_category"] == ELIGIBLE]]
    if not set(eligible) <= set(active):
        raise RuntimeError(f"{sub_id}: eligible states outside the active set")
    unused_active = [int(s) for s in active if bool(flags.loc[s, "unused"])]
    if unused_active:
        logger.warning("%s: active states flagged unused: %s", sub_id, unused_active)
    targets = [int(s) for s in active if s not in set(eligible)]

    pc_means, tr_counts = empirical_pc_means(X_by_run, decoded, K)
    emp_maps = np.full((K, comps.shape[1]), np.nan)
    ok = tr_counts > 0
    emp_maps[ok] = pc_means[ok] @ comps
    model_maps = d["means_parcel_contrast"]

    visits = visit_table(decoded)
    unc = visits[(visits["prev"] >= 0) & (visits["next"] >= 0)]

    ref_nn, _ = nn_reference(emp_maps[eligible], q=95)

    def seq(s):
        g = unc[unc["state"] == s]
        return g["prev"].to_numpy(), g["next"].to_numpy()

    # selection-matched dR2 null: the T rule applied to eligible states
    null_dR2 = []
    for e in eligible:
        a, b = seq(e)
        if a.size < args.min_visits:
            continue
        pair = through_partners(a, b)
        if pair is None:
            continue
        null_dR2.append(blend_fit(emp_maps[e], emp_maps[pair[0]], emp_maps[pair[1]])["dR2"])
    null_dR2 = np.array(null_dR2)
    dR2_p95 = float(np.percentile(null_dR2, 95)) if null_dR2.size else np.nan

    rows = []
    for s in targets:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, sub_idx, s]))
        a, b = seq(s)
        all_v = visits[visits["state"] == s]
        row = {
            "state": s,
            "summary_category": flags.loc[s, "summary_category"],
            "run_onset_anchored": bool(any(bool(flags.loc[s, f]) for f in ONSET_FLAGS)),
            "sub_hrf": bool(flags.loc[s, "sub_hrf"]),
            "recurrence": float(flags.loc[s, "recurrence_score"]),
            "n_trs": int(tr_counts[s]),
            "n_visits": int(len(all_v)),
            "n_censored_start": int((all_v["prev"] < 0).sum()),
            "n_censored_end": int((all_v["next"] < 0).sum()),
            "n_uncensored": int(a.size),
            "median_dwell": float(all_v["length"].median()),
            "min_visits": args.min_visits,
            "ref_nn": ref_nn,
            "dR2_null_p95": dR2_p95,
            "n_dR2_null": int(null_dR2.size),
        }
        elig_r_emp = corr_matrix(emp_maps[[s]], emp_maps[eligible]).ravel()
        elig_r_mod = corr_matrix(model_maps[[s]], model_maps[eligible]).ravel()
        row["closest_eligible"] = int(eligible[int(np.argmax(elig_r_emp))])
        row["r_max"] = float(elig_r_emp.max())
        row["r_max_model"] = float(elig_r_mod.max())
        row["closest_eligible_model"] = int(eligible[int(np.argmax(elig_r_mod))])

        nan_keys = ("return_share", "return_expected", "p_excess_return", "p_through",
                    "parent", "r_parent", "ref_boot", "var_logratio", "var_spearman",
                    "partner_a", "partner_b", "r2_blend", "w_a", "w_b", "r_a", "r_b",
                    "dR2", "r_switch", "n_switches")
        row.update({k: np.nan for k in nan_keys})

        if a.size >= args.min_visits:
            obs, exp, p = excess_return_test(a, b, args.n_perm, rng)
            row.update(return_share=obs, return_expected=exp, p_excess_return=p,
                       p_through=1.0 - obs)
            ret = Counter(int(x) for x, y in zip(a, b) if x == y)
            if ret:
                parent = ret.most_common(1)[0][0]
                row["parent"] = parent
                row["r_parent"] = corr(emp_maps[s], emp_maps[parent])
                pw = visits[visits["state"] == parent][["run", "start", "length"]]
                pw = list(pw.itertuples(index=False, name=None))
                vlen = all_v["length"].to_numpy()
                row["ref_boot"], _ = bootstrap_parent_reference(
                    pw, X_by_run, vlen, comps, emp_maps[parent], args.n_boot, rng, q=5)
                lv_s, lv_p = np.log(d["covars"][s]), np.log(d["covars"][parent])
                row["var_logratio"] = float(np.mean(lv_s - lv_p))
                row["var_spearman"] = float(spearmanr(lv_s, lv_p)[0])
            pair = through_partners(a, b)
            if pair is not None:
                row["partner_a"], row["partner_b"] = pair
                row.update(blend_fit(emp_maps[s], emp_maps[pair[0]], emp_maps[pair[1]]))
                L = max(1, int(round(row["median_dwell"])))
                wmap, n_sw = switch_window_map(visits, X_by_run, pair, L, comps)
                row["n_switches"] = n_sw
                if wmap is not None:
                    row["r_switch"] = corr(emp_maps[s], wmap)
        rows.append(row)

    df = pd.DataFrame(rows)
    testable = df["p_excess_return"].notna() & ~df["run_onset_anchored"]
    df["q_excess_return"] = np.nan
    if testable.any():
        df.loc[testable, "q_excess_return"] = benjamini_hochberg(
            df.loc[testable, "p_excess_return"].to_numpy())
    df["class"] = [classify(r, args.alpha) for r in df.to_dict("records")]
    df["hrf_blend"] = (df["class"].isin(["B", "I/B"])
                       & (df["r_switch"] >= np.sqrt(df["r2_blend"].clip(lower=0))))

    summary = {
        "sub_id": sub_id, "parcellation": parc, "vt": vt,
        "K": K, "n_active": int(active.size), "n_eligible": len(eligible),
        "n_classified": len(targets), "unused_but_active": unused_active,
        "class_counts": df["class"].value_counts().to_dict(),
        "class_by_category": {c: g["class"].value_counts().to_dict()
                              for c, g in df.groupby("summary_category")},
        "n_hrf_blend": int(df["hrf_blend"].sum()),
        "ref_nn": ref_nn, "dR2_null_p95": dR2_p95,
        "dR2_null": null_dR2.tolist(),
        "n_pcs": d["n_pcs"], "max_abs_pca_mean": d["max_abs_pca_mean"],
        "params": {"n_perm": args.n_perm, "n_boot": args.n_boot,
                   "min_visits": args.min_visits, "alpha": args.alpha,
                   "seed": args.seed, "active_fo": ACTIVE_FO,
                   "seed_scheme": "SeedSequence([seed, subject number, state])"},
        "provenance": {"python": platform.python_version(),
                       "numpy": np.__version__, "pandas": pd.__version__,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    return df, summary


def write_atomic(path, write_fn):
    tmp = path + ".part"
    write_fn(tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sub_id", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_perm", type=int, default=5000)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--min_visits", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
        scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        raise SystemExit("SCRATCH_DIR is not set (export it or put it in .env)")

    for sub_id in args.sub_id:
        t0 = time.time()
        df, summary = analyse_subject(sub_id, args.parcellation, args.vt, scratch, args)
        out = os.path.join(scratch, "output", "05g_excluded_state_check",
                           args.parcellation, sub_id, f"vt{args.vt}")
        os.makedirs(out, exist_ok=True)
        write_atomic(os.path.join(out, "state_classes.csv"),
                     lambda f: df.to_csv(f, index=False))

        def dump(f):
            with open(f, "w") as fh:
                json.dump(summary, fh, indent=2, default=float)
        write_atomic(os.path.join(out, "state_classes_summary.json"), dump)
        logger.info("%s: %s (%.0f s) -> %s", sub_id, summary["class_counts"],
                    time.time() - t0, out)


if __name__ == "__main__":
    main()
