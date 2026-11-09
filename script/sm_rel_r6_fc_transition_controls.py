#!/usr/bin/env python3
"""sm_rel_r6_fc_transition_controls.py - FC similarity vs transitions, without the shared FC backbone.

06b tests whether state pairs with similar within-state FC transition more
often (Mantel: Spearman between the 05f RV matrix and the symmetrised
empirical transition matrix, 5,000 state-label permutations). Raw RV between
two correlation matrices is high for every pair (0.68-1.00) because all
states share the same grand-mean FC. This script asks whether the association
survives once that shared part and three pair-level confounds are removed.

Per participant:

1. Gate: the 06b raw-RV Mantel rho is reproduced exactly from the same inputs.
2. Residual FC similarity: subtract the mean FC across reliable states from
   each state's correlation matrix and take the unclipped matrix cosine
   (as the residual RV in fig_S04_reliability.py). Two means are used:
   unweighted over reliable states (primary) and the 05f occupancy-weighted
   grand mean (state_delta_fc.npy).
3. Mantel tests on the reliable-state pairs for raw RV and both residuals.
4. Partial Mantel (rank-based) of transitions on each FC similarity,
   controlling for log(TR_i * TR_j), log(min(TR_i, TR_j)) and the Pearson
   correlation of the two states' parcel-space activation maps. Partial
   correlation = Pearson between the residuals of the ranked transition and
   ranked FC vectors after regressing both on the ranked covariates. The null
   permutes the state labels of the residualised FC matrix
   (double semi-partialling, Dekker et al. 2007).

Inputs (frozen, under $SCRATCH_DIR/output):
    06a_state_temp_dynamics/{parc}/{sub}/vt{vt}/transition_probabilities.npy
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/recurrence_summary.json
    05f_state_fc/{parc}/{sub}/vt{vt}/{fc_similarity_corr_rv,state_empirical_corr,state_delta_fc}.npy, metadata.json
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/state_means_parcel.npy
    06b_transition_structure/{parc}/{sub}/vt{vt}/fc_transition_correlation.json (gate)
Output:
    {out_dir}/fc_transition_controls_summary.json
"""
import os
import sys
import json
import logging
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import stats as sp_stats
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import safe_float

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def published_mantel(P_empirical, rv_matrix, active_states, n_perm=5000, seed=42):
    """Copy of 06b test_fc_transition_correlation (observed rho and p), used as the gate."""
    rng = np.random.default_rng(seed)
    n = len(active_states)
    idx = np.array(active_states)
    P_sub = P_empirical[np.ix_(idx, idx)].copy()
    rv_sub = rv_matrix[np.ix_(idx, idx)].copy()
    T_sym = (P_sub + P_sub.T) / 2.0
    np.fill_diagonal(T_sym, 0)
    np.fill_diagonal(rv_sub, 0)
    triu = np.triu_indices(n, k=1)
    t_full, rv_full = T_sym[triu], rv_sub[triu]
    mask = ((t_full > 1e-10) | (rv_full > 1e-10)) & ~np.isnan(rv_full)
    rho_obs = sp_stats.spearmanr(t_full[mask], rv_full[mask]).statistic
    null = np.full(n_perm, np.nan)
    for i in range(n_perm):
        perm = rng.permutation(n)
        rp = rv_sub[np.ix_(perm, perm)][triu]
        m = mask & ~np.isnan(rp)
        if m.sum() >= 10:
            null[i] = sp_stats.spearmanr(t_full[m], rp[m]).statistic
    valid = null[~np.isnan(null)]
    p = (1 + np.sum(np.abs(valid) >= abs(rho_obs))) / (1 + len(valid))
    return float(rho_obs), float(p), int(mask.sum())


def cosine_matrix(mats):
    """Unclipped matrix cosine between every pair of (p, p) matrices."""
    flat = mats.reshape(mats.shape[0], -1)
    gram = flat @ flat.T
    d = np.sqrt(np.diag(gram))
    return gram / np.outer(d, d)


def rank_residuals(y, X):
    """Residuals of rank(y) on [1, rank(X)] (OLS)."""
    ry = sp_stats.rankdata(y)
    if X is None:
        return ry - ry.mean()
    RX = np.column_stack([np.ones(len(y))] + [sp_stats.rankdata(c) for c in X.T])
    beta, *_ = np.linalg.lstsq(RX, ry, rcond=None)
    return ry - RX @ beta


def mantel(T, F, covars, n_perm, seed):
    """(Partial) Mantel on the upper triangle; covars is a list of (n, n) matrices or [].

    Returns rho (Spearman when covars is empty, partial Spearman otherwise) and a
    two-sided permutation p from permuting the state labels of the
    residualised F matrix.
    """
    n = T.shape[0]
    triu = np.triu_indices(n, k=1)
    t = T[triu]
    X = np.column_stack([c[triu] for c in covars]) if covars else None
    rt = rank_residuals(t, X)
    rf = rank_residuals(F[triu], X)
    RF = np.zeros((n, n))
    RF[triu] = rf
    RF = RF + RF.T
    obs = float(np.corrcoef(rt, rf)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = np.corrcoef(rt, RF[np.ix_(perm, perm)][triu])[0, 1]
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)
    return {"rho": obs, "p_value": float(p), "n_pairs": int(len(t)), "n_perm": n_perm,
            "null_median": float(np.median(null)), "null_q975": float(np.quantile(null, 0.975))}


def run_subject(sub_id, parc, vt, out_dir, n_perm, seed, gate_rv_path=None):
    base = os.path.join(SCRATCH_DIR, "output")
    sd = lambda stage: os.path.join(base, stage, parc, sub_id, f"vt{vt}")
    P_emp = np.load(os.path.join(sd("06a_state_temp_dynamics"), "transition_probabilities.npy"))
    with open(os.path.join(sd("05a_recurrence_analysis"), "recurrence_summary.json")) as f:
        rs = json.load(f)
    rec = np.asarray(rs["recurrence_scores"])
    active = [i for i in range(rs["n_states"]) if rec[i] > 0]
    rv = np.load(os.path.join(sd("05f_state_fc"), "fc_similarity_corr_rv.npy"))
    corr = np.load(os.path.join(sd("05f_state_fc"), "state_empirical_corr.npy"))
    delta_w = np.load(os.path.join(sd("05f_state_fc"), "state_delta_fc.npy"))
    with open(os.path.join(sd("05f_state_fc"), "metadata.json")) as f:
        meta = json.load(f)
    n_trs = np.asarray(meta["n_trs_per_state"], dtype=float)
    means = np.load(os.path.join(base, "04_combined_hdphmm", parc, sub_id, "final",
                                 f"vt{vt}", "state_means_parcel.npy"))
    with open(os.path.join(sd("06b_transition_structure"), "fc_transition_correlation.json")) as f:
        pub = json.load(f)

    # Gate: reproduce the published raw-RV Mantel. gate_rv_path names the RV file
    # 06b actually read when 05f was rewritten after 06b ran (sub-06, 2026-11-07);
    # the analysis below always uses the current 05f files.
    rv_gate = np.load(gate_rv_path) if gate_rv_path else rv
    rho_g, p_g, npairs_g = published_mantel(P_emp, rv_gate, active)
    rho_current = published_mantel(P_emp, rv, active, n_perm=1)[0]
    if abs(rho_g - pub["rho"]) > 1e-12 or npairs_g != pub["n_pairs"]:
        raise RuntimeError(f"{sub_id}: gate failed, rho {rho_g} vs published {pub['rho']}, "
                           f"pairs {npairs_g} vs {pub['n_pairs']}")

    # Reliable active states: FC defined (05f marks states below min_trs as NaN rows).
    rel = np.array([s for s in active if not np.all(np.isnan(np.delete(rv[s], s)))])
    P = P_emp[np.ix_(rel, rel)]
    T = (P + P.T) / 2.0
    C = corr[rel]
    fc = {
        "raw_rv": rv[np.ix_(rel, rel)],
        "residual_unweighted": cosine_matrix(C - C.mean(0)),
        "residual_occupancy_weighted": cosine_matrix(delta_w[rel]),
    }
    nt = n_trs[rel]
    m = means[rel]
    covars = {
        "log_tr_product": np.log(np.outer(nt, nt)),
        "log_tr_min": np.log(np.minimum.outer(nt, nt)),
        "activation_map_r": np.corrcoef(m),
    }
    cov_list = list(covars.values())

    results = {}
    for name, F in fc.items():
        results[name] = {
            "mantel": mantel(T, F, [], n_perm, seed),
            "partial_mantel": mantel(T, F, cov_list, n_perm, seed),
        }
    triu = np.triu_indices(len(rel), k=1)
    fc_range = {k: [float(np.nanmin(v[triu])), float(np.nanmax(v[triu]))] for k, v in fc.items()}
    cov_corr = {k: safe_float(sp_stats.spearmanr(T[triu], v[triu]).statistic) for k, v in covars.items()}

    summary = {
        "sub_id": sub_id, "parcellation": parc, "vt": float(vt),
        "gate": {"published_rho": pub["rho"], "reproduced_rho": rho_g,
                 "published_p": pub["p_value"], "reproduced_p": p_g, "n_pairs": npairs_g,
                 "gate_rv_path": gate_rv_path, "raw_rho_on_current_05f": rho_current},
        "n_active": len(active), "n_reliable_active": int(len(rel)),
        "reliable_active_states": rel.tolist(),
        "fc_similarity_range": fc_range,
        "spearman_transition_vs_covariate": cov_corr,
        "covariates": list(covars),
        "results": results,
        "seed": seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fc_transition_controls_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("%s: published rho %.3f | %s", sub_id, pub["rho"],
                {k: (round(v["mantel"]["rho"], 3), round(v["partial_mantel"]["rho"], 3),
                     round(v["partial_mantel"]["p_value"], 4)) for k, v in results.items()})
    return summary


def main():
    if SCRATCH_DIR is None:
        raise ValueError("SCRATCH_DIR must be set in environment or .env file")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--gate_rv_path", default=None,
                   help="RV matrix 06b read, if 05f was rewritten after 06b ran")
    a = p.parse_args()
    vt = f"{float(a.vt):.2f}"
    out_dir = a.out_dir or os.path.join(SCRATCH_DIR, "output", "sm_rel_r6_fc_transition_controls",
                                        a.parcellation, a.sub_id, f"vt{vt}")
    run_subject(a.sub_id, a.parcellation, vt, out_dir, a.n_perm, a.seed, a.gate_rv_path)


if __name__ == "__main__":
    main()
