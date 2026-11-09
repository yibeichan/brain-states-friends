#!/usr/bin/env python3
"""p7_06e_q_excess.py - observed modularity of graph E and its N2/N3 excess for one subject.

06e_transition_modules.py fixes VT = "0.95", loops over all six subjects, and
its task table and aggregate step need every variant, including LOSO folds
and the bootstrap. The sensitivity analysis needs only the observed Q of the
empirical transition graph E and its excess over the N2 (segment-order
shuffle, 1000 draws) and N3 (non-modular chain, 500 draws) nulls. This
wrapper imports the 06e module, sets its VT, and calls its own run_task,
load_null and null_test on a two-row task table, so the graphs, the Louvain
seeds and the null draws' random streams (SeedSequence([seed, sub_num,
crc32(variant), chunk])) are those of the production run.

Modes:
    run        draw one variant's null chunks (resumable, like 06e --mode run)
    summarise  write q_excess_summary.json from the finished chunks

Usage:
    SCRATCH_DIR=<root> python script/p7_06e_q_excess.py --sub_id sub-06 --vt 0.95 \\
        --mode run --variant N2_E
    SCRATCH_DIR=<root> python script/p7_06e_q_excess.py --sub_id sub-06 --vt 0.95 \\
        --mode summarise
"""
import argparse
import importlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m06e = importlib.import_module("06e_transition_modules")

VARIANTS = ("N2_E", "N3_E")


def configure(sub, vt):
    """Point the 06e module at one subject and variance threshold."""
    if sub not in m06e.SUBJECTS:
        raise SystemExit(f"unknown subject {sub}")
    m06e.VT = vt
    rows = [(sub, v, m06e.VARIANTS[v]) for v in VARIANTS]
    m06e.task_table = lambda: rows
    return rows


def summarise(sub):
    out = {"sub": sub, "vt": m06e.VT, "tests": {}}
    for v in VARIANTS:
        obs, null = m06e.load_null(sub, v, m06e.VARIANTS[v])
        entry = m06e.null_test(obs["q"], null["q"])
        entry.update({"n_nodes": len(obs["nodes"]), "n_edges": obs["n_edges"],
                      "share_seeds_at_best": obs["share_seeds_at_best"]})
        out["tests"][v] = entry
    if out["tests"]["N2_E"]["q_obs"] != out["tests"]["N3_E"]["q_obs"]:
        raise RuntimeError(f"{sub}: N2_E and N3_E observed Q differ")
    out["q_obs_E"] = out["tests"]["N2_E"]["q_obs"]
    out["q_excess_N2"] = out["tests"]["N2_E"]["effect"]
    out["q_excess_N3"] = out["tests"]["N3_E"]["effect"]
    out["code_commit"] = m06e.git_head()
    path = os.path.join(m06e.out_dir(sub), "q_excess_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    logging.info("%s vt%s: Q_E %.4f, excess N2 %.4f (p %.4g), N3 %.4f (p %.4g)", sub, m06e.VT,
                 out["q_obs_E"], out["q_excess_N2"], out["tests"]["N2_E"]["p"],
                 out["q_excess_N3"], out["tests"]["N3_E"]["p"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sub_id", required=True)
    ap.add_argument("--vt", required=True, help='variance threshold string, e.g. "0.95"')
    ap.add_argument("--mode", choices=["run", "summarise"], required=True)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0, help="06e base seed (wrapper default 0)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = configure(a.sub_id, a.vt)
    if a.mode == "run":
        if a.variant is None:
            raise SystemExit("--variant required for --mode run")
        m06e.run_task([r[1] for r in rows].index(a.variant), a.seed)
    else:
        summarise(a.sub_id)


if __name__ == "__main__":
    main()
