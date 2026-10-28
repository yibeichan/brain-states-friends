#!/usr/bin/env python3
"""Do the states move when the CSF/WM regressors are rebuilt over clean masks?

Compares, for one subject, the combined-HMM fits from the corrupt-mask and the
clean-mask runs of `clean_wmcsf_e2e_submit.sh` (each a SCRATCH_DIR-style root with
its own output/). The selected fits (best seed) are compared across variants, and
the result is judged against the spread between seeds within each variant: the
states moved if a verdict metric (median_r, ari or fo_distance; see
utils/state_compare.py) of the selected pair is worse than every within-variant
seed pair. A secondary seed-level permutation test asks whether cross-variant seed
pairs as a whole are worse than within-variant ones.

Every final seed model is back-projected with its variant's PCA and decoded over
all runs, so the comparison sees exactly what the selected fit sees. Stage 1's vt
pick is recomputed from each variant's stage-1 fits with 04's own scoring, so it is
reported even when select ran at a fixed vt.

The JSON is written even when a check fails (fewer final seeds than expected, the
selected fit's saved decode differing from its seed model's, or a non-finite value
in the permutation test, which is then left out), and the script then exits 1.

Usage:
    python script/clean_wmcsf_compare.py --run_root <RUN_ROOT> --sub_id sub-04 \\
        --out <RUN_ROOT>/compare/sub-04.json
"""
import argparse
import importlib.util
import itertools
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name  # noqa: E402
from utils.hmm_io import (  # noqa: E402
    back_project_states, decode_all_runs, get_projected_dir, load_split,
)
from utils.recurrence_utils import used_state_indices  # noqa: E402
from utils.state_compare import (  # noqa: E402
    compare_fits, distribution_test, moved_verdict,
)

VARIANTS = ("corrupt", "clean")


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _hdphmm04():
    """04_combined_hdphmm.py as a module (its file name is not importable).

    It reads SCRATCH_DIR when imported; the functions used here take explicit
    paths, so any value will do.
    """
    os.environ.setdefault("SCRATCH_DIR", tempfile.gettempdir())
    spec = importlib.util.spec_from_file_location(
        "hdphmm04", Path(__file__).parent / "04_combined_hdphmm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage1_vt(sub_dir):
    """Stage 1's vt pick from the stage-1 fits under `sub_dir`/configs.

    Uses 04's own scoring (per-dimension validation LL over the stage-1 grid).
    Returns (vt or None when no stage-1 config was fit, configs scored, grid size).
    """
    m = _hdphmm04()
    grid = m.build_stage1_grid()
    scores = m._collect_config_scores(str(Path(sub_dir) / "configs"), grid, "ll_per_dim")
    vt = float(scores[0]["config"]["variance_threshold"]) if scores else None
    return vt, len(scores), len(grid)


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
    vt1, n_scored, n_grid = stage1_vt(sub_dir)

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
        "stage1_selected_vt": vt1,
        "stage1_configs_scored": f"{n_scored}/{n_grid}",
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


def compare_variants(fits):
    """Verdict, permutation test and summaries for loaded fits of both variants.

    `fits` maps each variant to a dict with `seeds` ({seed: fit}), `best_seed` and
    `selected_config` (plus anything else to report, as load_variant returns).
    """
    # Pool every seed fit of both variants and compare each pair once.
    pooled = [(v, s, f) for v in VARIANTS for s, f in sorted(fits[v]["seeds"].items())]
    labels = np.array([VARIANTS.index(v) for v, _, _ in pooled])
    pairs = {(i, j): compare_fits(pooled[i][2], pooled[j][2])
             for i, j in itertools.combinations(range(len(pooled)), 2)}
    within = {v: [m for (i, j), m in pairs.items()
                  if pooled[i][0] == v and pooled[j][0] == v] for v in VARIANTS}
    cross_all = [m for (i, j), m in pairs.items() if pooled[i][0] != pooled[j][0]]

    c, k = fits["corrupt"], fits["clean"]
    selected = compare_fits(c["seeds"][c["best_seed"]], k["seeds"][k["best_seed"]])

    return {
        "variants": {v: {key: val for key, val in fit.items() if key != "seeds"}
                     | {"n_seeds": len(fit["seeds"]),
                        "k_used_by_seed": {str(s): int(f["k_used"])
                                           for s, f in sorted(fit["seeds"].items())}}
                     for v, fit in fits.items()},
        "same_selected_config": c["selected_config"] == k["selected_config"],
        "selected_fits": selected,
        "verdict": moved_verdict(selected, within["corrupt"] + within["clean"]),
        "distribution_test": _distribution_test_or_error(pairs, labels),
        "within_variant": {v: _summary(rows) for v, rows in within.items()},
        "cross_variant_all_seed_pairs": _summary(cross_all),
    }


def _distribution_test_or_error(pairs, labels):
    """The permutation test, or {"error": ...} if it cannot run; it never decides."""
    try:
        return distribution_test(pairs, labels)
    except ValueError as e:
        return {"error": str(e)}


def run(run_root, sub_id, parcellation):
    fits = {v: load_variant(Path(run_root) / v, sub_id, parcellation) for v in VARIANTS}
    return {"sub_id": sub_id, "parcellation": parcellation} | compare_variants(fits)


def check_failures(result, n_final_seeds=10):
    """Checks that make the comparison unusable, as a list of messages."""
    out = []
    for v, fit in result["variants"].items():
        if fit["n_seeds"] != n_final_seeds:
            out.append(f"{v}: {fit['n_seeds']} final seed models, expected {n_final_seeds}")
        if not fit["saved_decode_matches_best_seed"]:
            out.append(f"{v}: the selected fit's saved decode differs from its seed model's")
    if "error" in result["distribution_test"]:
        out.append(f"permutation test not run: {result['distribution_test']['error']}")
    return out


def write_json(obj, path):
    """Write `obj` to `path` through a temporary file in the same directory."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        mode = os.stat(path).st_mode & 0o777
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--run_root", required=True,
                   help="Directory holding corrupt/ and clean/ variant roots")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--n_final_seeds", type=int, default=10,
                   help="Final seeds each variant must have (default 10)")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    result = run(args.run_root, args.sub_id, normalize_parcellation_name(args.parcellation))
    write_json(result, args.out)

    v = result["verdict"]
    print(f"{args.sub_id}: selected {result['variants']['corrupt']['selected_config']} (corrupt) "
          f"vs {result['variants']['clean']['selected_config']} (clean)")
    for name in ("corrupt", "clean"):
        fit = result["variants"][name]
        print(f"  stage-1 vt ({name}): {fit['stage1_selected_vt']} "
              f"({fit['stage1_configs_scored']} configs scored)")
    for name, e in v["by_metric"].items():
        print(f"  {name:12s} cross={e['cross']:.3f}  within "
              f"[{e['within_min']:.3f}, {e['within_max']:.3f}]  outside={e['outside']}")
    print(f"  moved={v['moved']}")
    d = result["distribution_test"]
    if "error" not in d:
        print(f"  distribution test ({d['n_splits']} splits, exact={d['exact']}):")
        for name, e in d["by_metric"].items():
            print(f"  {name:12s} within median={e['within_median']:.3f}  "
                  f"cross median={e['cross_median']:.3f}  p={e['p']:.4f}")
    print(f"  -> {args.out}")
    failures = check_failures(result, args.n_final_seeds)
    for msg in failures:
        print(f"  CHECK FAILED: {msg}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
