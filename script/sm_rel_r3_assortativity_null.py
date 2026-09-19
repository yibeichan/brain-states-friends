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

Three quantities on the SAME saved graph (no re-thresholding):
  1. strength-stratified permutation null: recurrence labels are permuted only
     among nodes in the same strength quantile bin, preserving the
     attribute-strength relation the real data necessarily has;
  2. residualized-attribute assortativity: rank(recurrence) with rank(pi)
     regressed out, plus its own label-permutation null;
  3. assortativity of pi itself, and Spearman(recurrence, pi) and
     Spearman(recurrence, strength), to make the near-identity explicit.
The unconditional label-permutation null is recomputed alongside for comparison.

Faithfulness gates: the coefficient recomputed from transition_graph.graphml
must equal the published point estimate (recurrence_assortativity.json), and
pi recomputed from the saved model must match stationary_distribution.npy.

Inputs (frozen, under $SCRATCH_DIR/output):
    06b_transition_structure/{parc}/{sub}/vt{vt}/{transition_graph.graphml,
        recurrence_assortativity.json, stationary_distribution.npy}
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_graph(path):
    """Read the saved transition graph with int node ids and float attributes."""
    G = nx.read_graphml(path)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    for n in G.nodes:
        G.nodes[n]["recurrence_score"] = float(G.nodes[n]["recurrence_score"])
    for _, _, d in G.edges(data=True):
        d["weight"] = float(d["weight"])
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
