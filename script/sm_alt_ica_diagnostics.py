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
from scipy.stats import kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # put script/ on path


def participation_ratio(singular_values):
    s2 = np.asarray(singular_values, float) ** 2
    denom = (s2 ** 2).sum()
    return float((s2.sum() ** 2) / denom) if denom > 0 else 0.0


def _zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def state_mean_geometry(means, components, n_pcs, active_states):
    """Geometry of the HMM state-CONTRAST mean repertoire in the PC subspace."""
    M = np.asarray(means)[active_states][:, :n_pcs]      # (K, n_pcs)
    maps = M @ np.asarray(components)                    # (K, P) parcel-space maps
    sv = np.linalg.svd(M, compute_uv=False)
    norms = np.linalg.norm(M, axis=1)
    K = len(active_states)
    pr = participation_ratio(sv)
    kz = float(np.mean([kurtosis(row, fisher=True) for row in _zscore_rows(maps)]))
    return {
        "n_pcs": int(n_pcs), "k_active": int(K),
        "eff_rank": pr, "eff_rank_norm": pr / float(min(K, n_pcs)),
        "max_norm": float(norms.max()), "med_norm": float(np.median(norms)),
        "mean_excess_kurtosis": kz,
    }


def subspace_affinity(ica_maps, hmm_maps):
    """Principal-angle cosines between col-space(ica_maps) and row-space(hmm_maps)."""
    Qa = np.linalg.qr(np.asarray(ica_maps))[0]           # (P, Kica)
    Qb = np.linalg.qr(np.asarray(hmm_maps).T)[0]         # (P, Khmm)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)    # length min(Kica,Khmm)


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
