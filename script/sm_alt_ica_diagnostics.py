"""Post-hoc diagnostics for the ICA convergent-validity supplement.

Read-only over existing pipeline outputs. Reproduces the sub-03 investigation:
(1) matched-r vs subspace-rotation null evidence table, (2) HMM state-mean
repertoire geometry, (3) cross-analysis corroboration. Needs pipeline outputs
to run; pure functions below are unit-tested without data.
"""
import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # put script/ on path


def build_evidence_rows(summary):
    """Per-(sub,K) evidence over the 'eligible' state set; skips absent/empty K."""
    rows = []
    for K in sorted(summary.get("by_K", {}), key=int):
        ss = summary["by_K"][K].get("state_sets", {}).get("eligible")
        if not ss or not ss.get("matched_r"):
            continue
        r = np.asarray(ss["matched_r"], float)
        q = np.asarray(ss["spatial_q"], float)
        null_mean = float(ss["null_mean"])
        rows.append({
            "sub": summary["sub_id"], "K": int(K),
            "n_surv": int(np.sum(q < 0.05)), "n_total": int(r.size),
            "mean_r": float(r.mean()), "null_mean": null_mean,
            "null_p95": float(ss["null_p95"]),
            "frac_below_null": float(np.mean(r < null_mean)),
        })
    return rows
