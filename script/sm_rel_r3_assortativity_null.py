#!/usr/bin/env python3
"""sm_rel_r3_assortativity_null.py - strength-controlled nulls for recurrence assortativity.

R3 reports, per subject, Newman's numeric assortativity of the recurrence
score over the empirical transition graph (edges with transition probability
>= 0.005), tested against a null that re-pairs recurrence labels across nodes.
Recurrence is nearly the model's stationary occupancy pi (Spearman ~0.97), and
a node's weighted degree ("strength") on that graph is itself a function of
occupancy, so recurrence is a self-derived node attribute: a densely
interconnected high-strength core yields positive assortativity with no
further organization, and the label-permutation null does not control for
it. This script asks whether the reported assortativity survives nulls that
do.

Three quantities on the SAME rebuilt empirical graph (no re-thresholding):
  1. strength-stratified permutation null: recurrence labels are permuted only
     among nodes in the same strength quantile bin, preserving the
     attribute-strength relation the real data necessarily has;
  2. residualized-attribute assortativity: rank(recurrence) with rank(pi)
     regressed out, plus its own label-permutation null;
  3. assortativity of pi itself, and Spearman(recurrence, pi) and
     Spearman(recurrence, strength), to make the near-identity explicit.
The unconditional label-permutation null is recomputed alongside for comparison.

Faithfulness gates: the coefficient recomputed from the empirical transition
graph (rebuilt from 06a's transition_probabilities.npy exactly as
06b_transition_structure.build_transition_graph does for its own
assortativity computation - edge i->j when P[i, j] >= 0.005, over the states
with recurrence > 0) must equal the published point estimate
(recurrence_assortativity.json), and pi recomputed from the saved model must
match stationary_distribution.npy. The saved transition_graph.graphml is a
*different* graph (edge threshold 0.01, built from the model transition
matrix rather than the empirical one - 06b's "A1" topology graph) and is not
used here; it does not reproduce recurrence_assortativity.json.

Inputs (frozen, under $SCRATCH_DIR/output):
    06a_state_temp_dynamics/{parc}/{sub}/vt{vt}/transition_probabilities.npy
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/recurrence_scores.npy
    06b_transition_structure/{parc}/{sub}/vt{vt}/{recurrence_assortativity.json,
        stationary_distribution.npy}
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/best_model.pkl
Output:
    {SCRATCH_DIR}/output/sm_rel_r3_assortativity_null/{parc}/{sub}/vt{vt}/
        assortativity_null_summary.json, null_draws_{unconditional,stratified,residualized}.npy

Per-subject inference only; nothing is pooled.
"""
import os
import sys
import json
import time
import logging
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr
import networkx as nx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.jax_free_model_io import _load_model_no_jax
from utils.stats import null_summary, permutation_pvalue, safe_float
from utils.stationary import stationary_distribution

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

EDGE_THRESHOLD = 0.005

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_empirical_graph(P_empirical, active_states, recurrence_scores, edge_threshold=0.005):
    """Directed graph over active states from the empirical transition matrix,
    mirroring 06b_transition_structure.build_transition_graph: node attr
    recurrence_score; edge i->j (i != j) with weight P[i, j] when P[i, j] >= edge_threshold."""
    G = nx.DiGraph()
    for s in active_states:
        G.add_node(int(s), recurrence_score=float(recurrence_scores[s]))
    for i in active_states:
        for j in active_states:
            if i == j:
                continue
            p = float(P_empirical[i, j])
            if p >= edge_threshold:
                G.add_edge(int(i), int(j), weight=p)
    return G


def assortativity(G, attr):
    """Newman's numeric assortativity: Pearson r of `attr` across directed edge endpoints."""
    return float(nx.numeric_assortativity_coefficient(G, attr))


def node_strength(G):
    """Weighted in-degree plus weighted out-degree per node."""
    return {n: float(G.in_degree(n, weight="weight") + G.out_degree(n, weight="weight"))
            for n in G.nodes}


def strength_bins(strength, n_bins):
    """Assign nodes to contiguous strength-quantile bins (0 = weakest)."""
    order = sorted(strength, key=lambda n: (strength[n], n))
    chunks = np.array_split(np.asarray(order), n_bins)
    return {int(n): b for b, chunk in enumerate(chunks) for n in chunk.tolist()}


def stratified_permutation(values, bins, rng):
    """Permute `values` within each bin only."""
    out = {}
    for b in sorted(set(bins.values())):
        nodes = [n for n in values if bins[n] == b]
        shuffled = rng.permutation([values[n] for n in nodes])
        out.update({n: float(v) for n, v in zip(nodes, shuffled)})
    return out


def rank_residuals(x, y):
    """Residual of rank(x) regressed on rank(y) (with intercept), keyed like x."""
    nodes = sorted(x)
    rx = rankdata([x[n] for n in nodes])
    ry = rankdata([y[n] for n in nodes])
    A = np.column_stack([np.ones_like(ry), ry])
    beta, *_ = np.linalg.lstsq(A, rx, rcond=None)
    res = rx - A @ beta
    return {n: float(r) for n, r in zip(nodes, res)}


def permutation_null(G, attr, values, n_perm, rng, bins=None):
    """Assortativity under label permutation (within `bins` if given); restores `values`."""
    nodes = list(values)
    null = np.empty(n_perm)
    for i in range(n_perm):
        if bins is None:
            perm = {n: float(v) for n, v in zip(nodes, rng.permutation([values[n] for n in nodes]))}
        else:
            perm = stratified_permutation(values, bins, rng)
        nx.set_node_attributes(G, perm, attr)
        null[i] = assortativity(G, attr)
    nx.set_node_attributes(G, values, attr)
    return null


def null_block(observed, null):
    """Summary of one null distribution against an observed coefficient."""
    ns = null_summary(observed, null)
    finite = np.asarray(null, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan))
    return {
        "observed": safe_float(observed),
        "mean": ns["mean"], "sd": ns["sd"],
        "pct2.5": safe_float(lo), "pct97.5": safe_float(hi),
        "delta": ns["residual"], "z": ns["z"],
        "p_greater": ns["p"],
        "p_two_sided": safe_float(permutation_pvalue(observed, null, alternative="two-sided")),
        "p_floor": safe_float(1.0 / (1 + ns["n_finite"])) if ns["n_finite"] else None,
        "n_finite": ns["n_finite"],
    }


def run_subject(sub_id, parcellation, vt, n_perm, n_bins, seed, out_dir, gate_tol=1e-9):
    """Recompute the published assortativity, gate it, and test it against three nulls."""
    base = os.path.join(SCRATCH_DIR, "output")
    tdir = os.path.join(base, "06b_transition_structure", parcellation, sub_id, f"vt{vt}")
    emp_P_path = os.path.join(base, "06a_state_temp_dynamics", parcellation, sub_id,
                              f"vt{vt}", "transition_probabilities.npy")
    rec_path = os.path.join(base, "05a_recurrence_analysis", parcellation, sub_id,
                            f"vt{vt}", "recurrence_scores.npy")
    P_empirical = np.load(emp_P_path)
    recurrence = np.load(rec_path)
    active = np.flatnonzero(recurrence > 0)
    G = build_empirical_graph(P_empirical, active, recurrence, EDGE_THRESHOLD)
    with open(os.path.join(tdir, "recurrence_assortativity.json")) as f:
        pub = json.load(f)

    r_obs = assortativity(G, "recurrence_score")
    delta = abs(r_obs - float(pub["point_estimate"]))
    if not (delta <= gate_tol) or G.number_of_edges() != int(pub["n_edges"]):
        raise RuntimeError(
            f"{sub_id}: recomputed assortativity {r_obs:.9f} / {G.number_of_edges()} edges does not "
            f"reproduce published {pub['point_estimate']:.9f} / {pub['n_edges']} edges (|delta|={delta:.2e})")

    model_path = os.path.join(base, "04_combined_hdphmm", parcellation, sub_id,
                              "final", f"vt{vt}", "best_model.pkl")
    model = _load_model_no_jax(model_path)
    nodes = sorted(G.nodes)
    pi = stationary_distribution(np.asarray(model.transmat_), nodes)
    saved_pi = np.load(os.path.join(tdir, "stationary_distribution.npy"))
    if saved_pi.shape != pi.shape:
        raise RuntimeError(f"{sub_id}: pi has {pi.shape} states but saved has {saved_pi.shape}")
    pi_delta = float(np.max(np.abs(saved_pi - pi)))
    if pi_delta > 1e-6:
        raise RuntimeError(f"{sub_id}: recomputed pi deviates from saved by {pi_delta:.2e}")

    rec = {n: G.nodes[n]["recurrence_score"] for n in nodes}
    pi_d = {n: float(p) for n, p in zip(nodes, pi)}
    strength = node_strength(G)
    bins = strength_bins(strength, n_bins)
    bin_sizes = np.bincount([bins[n] for n in nodes], minlength=n_bins).tolist()

    t0 = time.time()
    rng = np.random.default_rng(seed)
    null_uncond = permutation_null(G, "recurrence_score", rec, n_perm, rng)
    null_strat = permutation_null(G, "recurrence_score", rec, n_perm, rng, bins=bins)

    resid = rank_residuals(rec, pi_d)
    nx.set_node_attributes(G, resid, "recurrence_resid_pi")
    r_resid = assortativity(G, "recurrence_resid_pi")
    null_resid = permutation_null(G, "recurrence_resid_pi", resid, n_perm, rng)

    nx.set_node_attributes(G, pi_d, "pi")
    r_pi = assortativity(G, "pi")
    rec_vec = np.array([rec[n] for n in nodes])
    rho_rec_pi = float(spearmanr(rec_vec, pi).statistic)
    rho_rec_strength = float(spearmanr(rec_vec, [strength[n] for n in nodes]).statistic)

    summary = {
        "sub_id": sub_id, "parcellation": parcellation, "vt": float(vt),
        "n_nodes": len(nodes), "n_edges": G.number_of_edges(),
        "edge_threshold_note": "empirical transition graph rebuilt as in 06b (transition "
                               "probability >= 0.005 over active states); the saved "
                               "transition_graph.graphml is the 0.01 model-matrix topology "
                               "graph and is not used",
        "gate": {"published_assortativity": float(pub["point_estimate"]),
                 "assortativity_abs_delta": delta, "tolerance": gate_tol,
                 "stationary_max_abs_delta": pi_delta},
        "observed": {"r_recurrence": safe_float(r_obs), "r_pi": safe_float(r_pi),
                     "r_recurrence_resid_pi": safe_float(r_resid),
                     "rho_recurrence_pi": safe_float(rho_rec_pi),
                     "rho_recurrence_strength": safe_float(rho_rec_strength)},
        "nulls": {
            "unconditional_label_permutation": null_block(r_obs, null_uncond),
            "strength_stratified_label_permutation": {
                **null_block(r_obs, null_strat),
                "n_bins": n_bins, "bin_sizes": bin_sizes,
                "strength_definition": "weighted in-degree + weighted out-degree on the "
                                       "rebuilt empirical graph"},
            "residualized_attribute_label_permutation": {
                **null_block(r_resid, null_resid),
                "attribute": "residual of rank(recurrence) regressed on rank(pi), with intercept"},
        },
        "n_perm": int(n_perm), "seed": int(seed),
        "seed_rule": "one numpy.random.default_rng(seed) stream, consumed in the order "
                     "unconditional -> stratified -> residualized",
        "runtime_s": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "networkx": nx.__version__},
        "inputs": {"transition_probabilities": emp_P_path, "recurrence_scores": rec_path,
                  "model": model_path},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "assortativity_null_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    np.save(os.path.join(out_dir, "null_draws_unconditional.npy"), null_uncond)
    np.save(os.path.join(out_dir, "null_draws_stratified.npy"), null_strat)
    np.save(os.path.join(out_dir, "null_draws_residualized.npy"), null_resid)
    logger.info("%s: r=%.3f | uncond p=%.4f | stratified p=%.4f | resid r=%.3f p=%.4f | r(pi)=%.3f rho(rec,pi)=%.3f",
                sub_id, r_obs, summary["nulls"]["unconditional_label_permutation"]["p_greater"],
                summary["nulls"]["strength_stratified_label_permutation"]["p_greater"],
                r_resid, summary["nulls"]["residualized_attribute_label_permutation"]["p_greater"],
                r_pi, rho_rec_pi)
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="Strength-controlled nulls for recurrence assortativity.")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_perm", type=int, default=5000)
    p.add_argument("--n_bins", type=int, default=5, help="Strength-quantile bins for the stratified null.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    return p


def main():
    if SCRATCH_DIR is None:
        raise ValueError("SCRATCH_DIR must be set in environment or .env file")
    a = build_parser().parse_args()
    vt = f"{float(a.vt):.2f}"
    out_dir = a.out_dir or os.path.join(SCRATCH_DIR, "output", "sm_rel_r3_assortativity_null",
                                        a.parcellation, a.sub_id, f"vt{vt}")
    run_subject(a.sub_id, a.parcellation, vt, a.n_perm, a.n_bins, a.seed, out_dir)


if __name__ == "__main__":
    main()
