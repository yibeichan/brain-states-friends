#!/usr/bin/env python3
"""sm_rel_r7_loso_module_agreement.py - do held-out seasons recover the main transition modules?

06e LOSO2 re-partitions each held-out season (decoded by the model fitted
without it) with Louvain and shows the partition is modular. That does not show
it is the same partition as the main one (M). This script measures agreement.

For each participant and held-out season:
- Map fold states to primary states with the 04ra Hungarian matches
  (well_matched, r >= 0.3).
- Keep fold states that are nodes of the LOSO2 partition and whose matched
  primary state is a node of M.
- ARI between the LOSO2 labels and M's labels on those states, with a
  label-permutation null (fold labels shuffled across the kept states).
- The same ARI for LOSO_ (the fold model's own transition matrix, fitted on
  the five training seasons) as a reference that does not use held-out data.

Inputs (frozen, under $SCRATCH_DIR/output):
    06e_transition_modules/{parc}/{sub}/vt{vt}/partitions.json
    06e_transition_modules/{parc}/{sub}/vt{vt}/nulls/{LOSO2,LOSO}_season_*/observed.json
    04ra_loso_struct_comp/{parc}/{sub}/hungarian_matching.json
Output:
    {out_dir}/loso_module_agreement_summary.json
"""
import os
import json
import logging
import argparse
import platform

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score
from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def agreement(obs, fold_to_primary, primary_label, n_perm, rng):
    """ARI between a fold partition and M over fold states with a matched primary node."""
    pairs = [(fl, primary_label[fold_to_primary[fs]])
             for fs, fl in zip(obs["nodes"], obs["labels"])
             if fs in fold_to_primary and fold_to_primary[fs] in primary_label]
    if len(pairs) < 3:
        return {"n_states": len(pairs), "ari": None, "p_value": None}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    ari = float(adjusted_rand_score(a, b))
    null = np.array([adjusted_rand_score(rng.permutation(a), b) for _ in range(n_perm)])
    return {"n_states": len(pairs), "n_fold_nodes": len(obs["nodes"]),
            "n_modules_fold": int(len(set(a))), "n_modules_main_on_kept": int(len(set(b))),
            "ari": ari, "p_value": float((1 + np.sum(null >= ari)) / (1 + n_perm)),
            "null_q95": float(np.quantile(null, 0.95)), "q_fold": obs.get("q")}


def run_subject(sub_id, parc, vt, out_dir, n_perm, seed):
    base = os.path.join(SCRATCH_DIR, "output")
    mdir = os.path.join(base, "06e_transition_modules", parc, sub_id, f"vt{vt}")
    with open(os.path.join(mdir, "partitions.json")) as f:
        part = json.load(f)
    primary_label = dict(zip(part["nodes"], part["labels"]))
    with open(os.path.join(base, "04ra_loso_struct_comp", parc, sub_id, "hungarian_matching.json")) as f:
        hm = json.load(f)
    rng = np.random.default_rng(seed)
    per_season = {}
    for fold, fd in sorted(hm["per_fold"].items(), key=lambda kv: int(kv[0])):
        f2p = {m["fold_state"]: m["primary_state"] for m in fd["matches"] if m["well_matched"]}
        row = {"n_well_matched": len(f2p)}
        for variant in ("LOSO2", "LOSO"):
            path = os.path.join(mdir, "nulls", f"{variant}_season_{fold}", "observed.json")
            with open(path) as f:
                obs = json.load(f)
            row[variant] = agreement(obs, f2p, primary_label, n_perm, rng)
        per_season[fold] = row
    ari2 = [r["LOSO2"]["ari"] for r in per_season.values() if r["LOSO2"]["ari"] is not None]
    ari1 = [r["LOSO"]["ari"] for r in per_season.values() if r["LOSO"]["ari"] is not None]
    summary = {
        "sub_id": sub_id, "parcellation": parc, "vt": float(vt),
        "n_main_nodes": len(part["nodes"]), "n_main_modules": int(len(set(part["labels"]))),
        "per_season": per_season,
        "loso2_ari_median": float(np.median(ari2)), "loso2_ari_range": [min(ari2), max(ari2)],
        "loso2_n_p_below_0.05": int(sum(r["LOSO2"]["p_value"] < 0.05 for r in per_season.values()
                                        if r["LOSO2"]["p_value"] is not None)),
        "loso_ari_median": float(np.median(ari1)), "loso_ari_range": [min(ari1), max(ari1)],
        "n_perm": n_perm, "seed": seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "loso_module_agreement_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("%s: LOSO2 ARI %s | LOSO ARI %s", sub_id,
                [round(x, 3) for x in ari2], [round(x, 3) for x in ari1])
    return summary


def main():
    if SCRATCH_DIR is None:
        raise ValueError("SCRATCH_DIR must be set in environment or .env file")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    vt = f"{float(a.vt):.2f}"
    out_dir = a.out_dir or os.path.join(SCRATCH_DIR, "output", "sm_rel_r7_loso_module_agreement",
                                        a.parcellation, a.sub_id, f"vt{vt}")
    run_subject(a.sub_id, a.parcellation, vt, out_dir, a.n_perm, a.seed)


if __name__ == "__main__":
    main()
