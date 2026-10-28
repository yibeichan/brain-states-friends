#!/usr/bin/env python3
"""Exit 1 unless every stage-1 and stage-2 config of one subject has all its seeds.

Used by clean_wmcsf_e2e_submit.sh between the stage-1/stage-2 fit attempts and the
final refit: the attempts are linked with afterany, so a timed-out attempt does not
cancel the chain, and this check is what stops the final refit when a config is
still incomplete after the last attempt. Reads each config's config_summary.json
under <scratch_dir>/output/04_combined_hdphmm/<parcellation>/<sub_id>/configs.

Usage:
    python script/clean_wmcsf_check_fits.py --scratch_dir <RUN_ROOT>/<variant> \\
        --sub_id sub-02 --fixed_vt 0.95 --n_fit_seeds 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config.combined_hmm_config import (  # noqa: E402
    build_stage1_grid, build_stage2_grid, config_name,
)


def incomplete_configs(configs_dir, fixed_vt, n_fit_seeds):
    """Config names of both stage grids whose summary is missing or short of seeds."""
    names = dict.fromkeys(config_name(c) for c in
                          build_stage1_grid() + build_stage2_grid(fixed_vt))
    out = []
    for name in names:
        path = Path(configs_dir) / name / "config_summary.json"
        if not path.is_file():
            out.append(f"{name}: no config_summary.json")
            continue
        n = json.loads(path.read_text()).get("n_seeds_successful", 0)
        if n != n_fit_seeds:
            out.append(f"{name}: {n} of {n_fit_seeds} seeds")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--scratch_dir", required=True, help="Variant root (its SCRATCH_DIR)")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--fixed_vt", type=float, required=True)
    p.add_argument("--n_fit_seeds", type=int, default=5)
    args = p.parse_args()

    configs = (Path(args.scratch_dir) / "output" / "04_combined_hdphmm" / args.parcellation
               / args.sub_id / "configs")
    bad = incomplete_configs(configs, args.fixed_vt, args.n_fit_seeds)
    for line in bad:
        print(f"INCOMPLETE {args.sub_id} {line}")
    if bad:
        sys.exit(1)
    print(f"{args.sub_id}: every stage-1 and stage-2 config has {args.n_fit_seeds} seeds")


if __name__ == "__main__":
    main()
