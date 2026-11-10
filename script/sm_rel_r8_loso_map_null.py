#!/usr/bin/env python3
"""sm_rel_r8_loso_map_null.py - baselines for the LOSO matched-state map correlations.

04ra matches each leave-one-season-out (LOSO) fold's states to the primary
model's states (Hungarian on 1 - Pearson r of parcel-space state means) and
reports the matched-pair r (pooled median 0.952-0.987 per participant). Each
fold shares five of six seasons with the primary fit, so a high r needs a
baseline. Two are computed from the saved state means (the derangement null
sits near 0 because the state means are close to centred across states; the
best non-match is the informative comparison):

1. Derangement null: within each fold, the matched primary partners are
   permuted with no fixed points (each fold state paired with a different
   matched primary state), and the pooled median r is recomputed
   (n_draws draws; one derangement per fold per draw).
2. Best non-match: for each fold state, the highest r to any primary active
   state other than its Hungarian partner. Its pooled median says how close
   the nearest competitor is.

Gate: the recomputed matched r equals the saved matched_correlations.

Inputs (frozen, under $SCRATCH_DIR/output):
    04ra_loso_struct_comp/{parc}/{sub}/hungarian_matching.json
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/state_means_parcel.npy
    04_combined_hdphmm/{parc}/{sub}/loso/season_*/state_means_parcel.npy
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/recurrence_summary.json (primary active set)
Output:
    {out_dir}/loso_map_null_summary.json
"""
import os
import json
import logging
import argparse
import platform

import numpy as np
from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def corr_rows(a, b):
    """Pearson r between rows of a and rows of b (same standardisation as 04rb hungarian_matching)."""
    def z(m):
        s = m.std(axis=1, keepdims=True)
        return (m - m.mean(axis=1, keepdims=True)) / np.where(s > 0, s, 1.0)
    return (z(a) @ z(b).T) / a.shape[1]


def derangement(n, rng):
    if n < 2:
        raise ValueError(f"no derangement of {n} item(s)")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def run_subject(sub_id, parc, vt, out_dir, n_draws, seed):
    base = os.path.join(SCRATCH_DIR, "output")
    hdir = os.path.join(base, "04_combined_hdphmm", parc, sub_id)
    prim = np.load(os.path.join(hdir, "final", f"vt{vt}", "state_means_parcel.npy"))
    with open(os.path.join(base, "05a_recurrence_analysis", parc, sub_id, f"vt{vt}",
                           "recurrence_summary.json")) as f:
        rec = np.asarray(json.load(f)["recurrence_scores"])
    prim_active = np.flatnonzero(rec > 0)
    with open(os.path.join(base, "04ra_loso_struct_comp", parc, sub_id, "hungarian_matching.json")) as f:
        hm = json.load(f)

    rng = np.random.default_rng(seed)
    folds, matched_all, best_other_all, fold_R = {}, [], [], []
    max_gate = 0.0
    for fold, fd in sorted(hm["per_fold"].items(), key=lambda kv: int(kv[0])):
        if len(prim_active) != fd["n_primary_active"]:
            raise RuntimeError(f"{sub_id} fold {fold}: primary active {len(prim_active)} "
                               f"vs saved {fd['n_primary_active']}")
        fm = np.load(os.path.join(hdir, "loso", f"season_{fold}", "state_means_parcel.npy"))
        fs = np.array([m["fold_state"] for m in fd["matches"]])
        ps = np.array([m["primary_state"] for m in fd["matches"]])
        saved = np.array([m["correlation"] for m in fd["matches"]])
        R = corr_rows(fm[fs], prim[ps])            # matched x matched
        r_match = np.diag(R)
        max_gate = max(max_gate, float(np.max(np.abs(r_match - saved))))
        Rall = corr_rows(fm[fs], prim[prim_active])
        col = {s: j for j, s in enumerate(prim_active)}
        Rmask = Rall.copy()
        Rmask[np.arange(len(fs)), [col[s] for s in ps]] = -np.inf
        best_other = Rmask.max(axis=1)
        matched_all.append(r_match)
        best_other_all.append(best_other)
        fold_R.append(R)
        folds[fold] = {"n_matched": int(len(fs)), "median_matched": float(np.median(r_match)),
                       "median_best_non_match": float(np.median(best_other)),
                       "n_matched_below_best_non_match": int(np.sum(r_match <= best_other))}  # counts ties (<=)
    if max_gate > 1e-5:
        raise RuntimeError(f"{sub_id}: recomputed matched r deviates from saved by {max_gate:.2e}")

    obs = float(np.median(np.concatenate(matched_all)))
    null = np.empty(n_draws)
    for i in range(n_draws):
        vals = [R[np.arange(R.shape[0]), derangement(R.shape[0], rng)] for R in fold_R]
        null[i] = np.median(np.concatenate(vals))
    bo = np.concatenate(best_other_all)
    ma = np.concatenate(matched_all)
    summary = {
        "sub_id": sub_id, "parcellation": parc, "vt": float(vt),
        "n_folds": len(folds), "n_pairs_pooled": int(ma.size),
        "gate": {"matched_r_max_abs_delta": max_gate},
        "observed_pooled_median": obs,
        "observed_pooled_mean": float(ma.mean()),
        "derangement_null": {"n_draws": n_draws, "median": float(np.median(null)),
                             "q95": float(np.quantile(null, 0.95)), "max": float(null.max()),
                             "p_value": float((1 + np.sum(null >= obs)) / (1 + n_draws))},
        "best_non_match": {"pooled_median": float(np.median(bo)),
                           "pooled_median_gap": float(np.median(ma - bo)),
                           "n_pairs_matched_not_above_best_non_match": int(np.sum(ma <= bo))},
        "per_fold": folds, "seed": seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "loso_map_null_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("%s: matched median %.3f | derangement null median %.3f (max %.3f) | best non-match median %.3f",
                sub_id, obs, summary["derangement_null"]["median"], null.max(), np.median(bo))
    return summary


def main():
    if SCRATCH_DIR is None:
        raise ValueError("SCRATCH_DIR must be set in environment or .env file")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    vt = f"{float(a.vt):.2f}"
    out_dir = a.out_dir or os.path.join(SCRATCH_DIR, "output", "sm_rel_r8_loso_map_null",
                                        a.parcellation, a.sub_id, f"vt{vt}")
    run_subject(a.sub_id, a.parcellation, vt, out_dir, a.n_draws, a.seed)


if __name__ == "__main__":
    main()
