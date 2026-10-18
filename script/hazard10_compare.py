#!/usr/bin/env python3
"""Hazard-10 check: do the states move when the CSF/WM regressors are cleaned?

Compares, for one subject, the combined-HMM fits from the corrupt-mask and the
clean-mask runs of `hazard10_e2e_submit.sh` (each a SCRATCH_DIR-style root with
its own output/). The selected fits (best seed) are compared across variants, and
the result is judged against the spread between seeds within each variant: the
states moved if a cross-variant metric is worse than every within-variant seed
pair (see utils/state_compare.py for the metrics).

Every final seed model is back-projected with its variant's PCA and decoded over
all runs, so the comparison sees exactly what the selected fit sees.

Usage:
    python script/hazard10_compare.py --run_root <RUN_ROOT> --sub_id sub-04 \\
        --out <RUN_ROOT>/compare/sub-04.json
"""
import argparse
import itertools
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name  # noqa: E402
from utils.hmm_io import (  # noqa: E402
    back_project_states, decode_all_runs, get_projected_dir, load_split,
)
from utils.recurrence_utils import used_state_indices  # noqa: E402
from utils.state_compare import compare_fits, moved_verdict  # noqa: E402

VARIANTS = ("corrupt", "clean")


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_variant(root, sub_id, parcellation):
    """Selected config, stage-1 vt and every final seed fit for one variant."""
    out = Path(root) / "output"
    combined_base = out / "03a_pca4combined_hmm" / parcellation / sub_id
    sub_dir = out / "04_combined_hdphmm" / parcellation / sub_id
    finals = sorted((sub_dir / "final").glob("vt*/final_results.json"))
    if len(finals) != 1:
        raise FileNotFoundError(f"expected one final/vt*/final_results.json in {sub_dir}, "
                                f"found {len(finals)}")
    final_dir = finals[0].parent
    results = json.loads(finals[0].read_text())
    n_pcs = int(results["data_info"]["n_pcs"])
    best_seed = int(results["final_refit"]["best_seed"])

    stage1 = sub_dir / "stage1_result.json"
    stage1_vt = json.loads(stage1.read_text())["selected_vt"] if stage1.is_file() else None

    pca = _load_pickle(final_dir / "pca_model.pkl")
    split = load_split(str(combined_base))
    projected_dir = get_projected_dir(str(combined_base))

    seeds = {}
    for path in (final_dir / "seeds").glob("seed_*_model.pkl"):
        seed = int(re.match(r"seed_(\d+)_model", path.stem).group(1))
        model = _load_pickle(path)
        means, _ = back_project_states(model, pca, n_pcs)
        seeds[seed] = {
            "means": means,
            "decoded": decode_all_runs(model, split, projected_dir, n_pcs),
            "k_used": used_state_indices(model).size,
        }
    if best_seed not in seeds:
        raise FileNotFoundError(f"best seed {best_seed} model missing in {final_dir}/seeds")

    # The selected fit's saved decode must match its seed model's decode.
    saved = _load_pickle(final_dir / "decoded_states.pkl")
    best = seeds[best_seed]["decoded"]
    decode_matches = set(saved) == set(best) and all(
        np.array_equal(np.asarray(saved[k]), best[k]) for k in saved)

    return {
        "final_dir": str(final_dir),
        "selected_config": results["selected_config_name"],
        "stage1_selected_vt": stage1_vt,
        "n_pcs": n_pcs,
        "best_seed": best_seed,
        "saved_decode_matches_best_seed": bool(decode_matches),
        "seeds": seeds,
    }


def _summary(rows):
    keys = ("median_r", "min_r", "ari", "mapped_agreement", "unmatched_fraction",
            "fo_distance")
    return {k: {"median": float(np.median([r[k] for r in rows])),
                "min": float(np.min([r[k] for r in rows])),
                "max": float(np.max([r[k] for r in rows]))} for k in keys} | {
        "n_pairs": len(rows)}


def run(run_root, sub_id, parcellation):
    fits = {v: load_variant(Path(run_root) / v, sub_id, parcellation) for v in VARIANTS}

    within = {}
    for v, fit in fits.items():
        s = fit["seeds"]
        within[v] = [compare_fits(s[i], s[j]) for i, j in itertools.combinations(sorted(s), 2)]
    c, k = fits["corrupt"], fits["clean"]
    selected = compare_fits(c["seeds"][c["best_seed"]], k["seeds"][k["best_seed"]])
    cross_all = [compare_fits(c["seeds"][i], k["seeds"][j])
                 for i in sorted(c["seeds"]) for j in sorted(k["seeds"])]

    return {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "variants": {v: {key: val for key, val in fit.items() if key != "seeds"}
                     | {"n_seeds": len(fit["seeds"]),
                        "k_used_by_seed": {str(s): int(f["k_used"])
                                           for s, f in sorted(fit["seeds"].items())}}
                     for v, fit in fits.items()},
        "same_selected_config": c["selected_config"] == k["selected_config"],
        "selected_fits": selected,
        "verdict": moved_verdict(selected, within["corrupt"] + within["clean"]),
        "within_variant": {v: _summary(rows) for v, rows in within.items()},
        "cross_variant_all_seed_pairs": _summary(cross_all),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run_root", required=True,
                   help="Directory holding corrupt/ and clean/ variant roots")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    result = run(args.run_root, args.sub_id, normalize_parcellation_name(args.parcellation))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    v = result["verdict"]
    print(f"{args.sub_id}: selected {result['variants']['corrupt']['selected_config']} (corrupt) "
          f"vs {result['variants']['clean']['selected_config']} (clean)")
    for name, e in v["by_metric"].items():
        print(f"  {name:12s} cross={e['cross']:.3f}  within "
              f"[{e['within_min']:.3f}, {e['within_max']:.3f}]  outside={e['outside']}")
    print(f"  moved={v['moved']}  -> {args.out}")


if __name__ == "__main__":
    main()
