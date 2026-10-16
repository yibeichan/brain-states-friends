#!/usr/bin/env python3
"""Per-rank view of the ICA-HMM spatial match and the focal ICA components.

Post-hoc analysis of the saved sm_alt_ica_states outputs at each subject's
K_used, over the full active state set (FO > 0.02 in at least one run; the
"all" set's `state_ids`). No ICA re-run. For each subject:

- the per-rank rotation null: the i-th strongest observed matched |r| against
  the 2.5/50/97.5 percentiles of the i-th strongest match across null draws
  (same draws and seed as sm_alt_ica_states);
- each HMM state's best |r| to any ICA map, which separates a poor one-to-one
  assignment from a state that no ICA map resembles;
- focal ICA components: maps whose best |r| to any HMM state is below 0.15,
  with their kurtosis, ICASSO stability (I_q) and the share of map energy in
  their top three parcels.

Output: {ICA_STATES_DIR}/{parcellation}/focal_components.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from scipy.stats import kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import hmm_io  # noqa: E402
from utils.ica_states import (  # noqa: E402
    ICA_STATES_DIR, match_maps_hungarian, subspace_rotation_null,
)
from utils.jax_free_model_io import _load_model_no_jax  # noqa: E402

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
FOCAL_R = 0.15     # best |r| to any HMM state below this -> focal component
TOP_PARCELS = 3


def _z(X):
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def run_subject(sub, parcellation, vt, n_perm, rng_seed):
    fd = os.path.join(SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation,
                      sub, "final", f"vt{vt}")
    model = _load_model_no_jax(os.path.join(fd, "best_model.pkl"))
    with open(os.path.join(fd, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    pca_base = os.path.join(SCRATCH_DIR, "output", "03a_pca4combined_hmm",
                            parcellation, sub)
    n_pcs = int(hmm_io.load_n_pcs_lookup(pca_base)[vt])
    components = pca.components_[:n_pcs]
    state_maps = np.asarray(model.means_)[:, :n_pcs] @ components

    icadir = os.path.join(SCRATCH_DIR, "output", ICA_STATES_DIR, parcellation, sub)
    with open(os.path.join(icadir, "ica_match_summary.json")) as f:
        summary = json.load(f)
    K = int(summary["K_active"])          # K_used
    ids = [int(x) for x in summary["by_K"][str(K)]["state_sets"]["all"]["state_ids"]]
    hmm_maps = state_maps[ids]
    ica_maps = np.load(os.path.join(icadir, f"ica_maps_K{K}.npy"))   # (n_parcels, K)

    m = match_maps_hungarian(ica_maps, hmm_maps)
    obs = np.sort(np.abs(m["matched_r"]))[::-1]
    null = subspace_rotation_null(components, hmm_maps, n_components=K,
                                  n_perm=n_perm, rng_seed=rng_seed, subspace="topk")
    null = np.sort(null, axis=1)[:, ::-1]
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)

    R = np.abs(_z(hmm_maps) @ _z(ica_maps.T).T / hmm_maps.shape[1])   # (states, comps)
    comp_best = R.max(0)
    focal = comp_best < FOCAL_R
    kz = kurtosis(ica_maps, axis=0)
    iq = np.asarray(summary["by_K"][str(K)]["iq"], float)
    e = np.sort(_z(ica_maps.T) ** 2, axis=1)[:, ::-1]
    share = e[:, :TOP_PARCELS].sum(1) / e.sum(1)

    return {
        "K_used": K,
        "n_states": len(ids),
        "per_rank": {"observed": obs.tolist(), "null_p2_5": lo.tolist(),
                     "null_median": med.tolist(), "null_p97_5": hi.tolist()},
        "n_ranks_above_null_p97_5": int((obs > hi).sum()),
        "n_ranks_below_null_p2_5": int((obs < lo).sum()),
        "n_pairs_below_focal_r": int((np.abs(m["matched_r"]) < FOCAL_R).sum()),
        "state_best_r_any_map": {"median": float(np.median(R.max(1))),
                                 "min": float(R.max(1).min())},
        "n_focal_components": int(focal.sum()),
        "focal_kurtosis_median": float(np.median(kz[focal])) if focal.any() else None,
        "other_kurtosis_median": float(np.median(kz[~focal])),
        "focal_top3_energy_share_median": float(np.median(share[focal])) if focal.any() else None,
        "other_top3_energy_share_median": float(np.median(share[~focal])),
        "focal_iq_median": float(np.median(iq[focal])) if focal.any() else None,
        "other_iq_median": float(np.median(iq[~focal])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--subjects", nargs="*", default=[f"sub-0{i}" for i in range(1, 7)])
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--rng_seed", type=int, default=0)
    a = p.parse_args()
    out = {"focal_r": FOCAL_R, "top_parcels": TOP_PARCELS, "n_perm": a.n_perm,
           "rng_seed": a.rng_seed, "subjects": {}}
    for sub in a.subjects:
        out["subjects"][sub] = r = run_subject(sub, a.parcellation, a.vt, a.n_perm, a.rng_seed)
        print(f"{sub} K_used={r['K_used']}: ranks above/below null "
              f"{r['n_ranks_above_null_p97_5']}/{r['n_ranks_below_null_p2_5']}, "
              f"focal {r['n_focal_components']}, state best-any min "
              f"{r['state_best_r_any_map']['min']:.2f}")
    path = os.path.join(SCRATCH_DIR, "output", ICA_STATES_DIR, a.parcellation,
                        "focal_components.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
