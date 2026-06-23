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
from scipy.stats import kurtosis, spearmanr

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


def corroboration_metrics(recurrence_summary, transition_summary):
    """Extract scalar corroboration signals from recurrence and transition summaries."""
    rec = np.asarray(recurrence_summary["recurrence_scores"], float)
    active = rec > 0
    return {
        "n_specific": len(recurrence_summary["significant_specific_states"]),
        "mean_recurrence": float(rec[active].mean()) if active.any() else float("nan"),
        "fc_rho": float(transition_summary["A3_fc_transition"]["rho"]),
    }


def spearman_illustrative(x, y):
    """Descriptive Spearman for n=6 ordinal corroboration. NO p-value (under-powered)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ties = (len(np.unique(x)) < len(x)) or (len(np.unique(y)) < len(y))
    rho = float(spearmanr(x, y).statistic)
    return {"rho": rho, "n": int(x.size), "ties_flag": bool(ties),
            "pairs": list(zip(x.tolist(), y.tolist()))}


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


def render_tables_md(evidence_rows, geometry_by_sub, corroboration_by_sub):
    lines = ["# ICA convergence diagnostics", "", "## Tier-1 evidence (eligible set)", "",
             "| sub | K | surviving | mean r | null_mean | null_p95 | frac below null |",
             "|---|---|---|---|---|---|---|"]
    for r in evidence_rows:
        lines.append(f"| {r['sub']} | {r['K']} | {r['n_surv']}/{r['n_total']} | "
                     f"{r['mean_r']:.3f} | {r['null_mean']:.3f} | {r['null_p95']:.3f} | "
                     f"{r['frac_below_null']:.2f} |")
    lines += ["", "## Repertoire geometry", "",
              "| sub | n_pcs | K | eff_rank | eff_rank_norm | max_norm | med_norm | kurtosis | affinity |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sub, g in geometry_by_sub.items():
        aff = g.get("subspace_affinity_mean")
        aff_str = f"{aff:.2f}" if aff is not None else "n/a"
        lines.append(f"| {sub} | {g['n_pcs']} | {g['k_active']} | {g['eff_rank']:.1f} | "
                     f"{g['eff_rank_norm']:.2f} | {g['max_norm']:.2f} | {g['med_norm']:.2f} | "
                     f"{g['mean_excess_kurtosis']:.2f} | {aff_str} |")
    lines += ["", "## Corroboration", "",
              "| sub | n_specific | mean_recurrence | fc_rho |", "|---|---|---|---|"]
    for sub, c in corroboration_by_sub.items():
        lines.append(f"| {sub} | {c['n_specific']} | {c['mean_recurrence']:.3f} | {c['fc_rho']:.3f} |")
    return "\n".join(lines) + "\n"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _subjects(scratch, parc, requested):
    if requested:
        return list(requested)
    base = os.path.join(scratch, "output", "sm_ica_states", parc)
    return sorted(d for d in os.listdir(base)
                  if d.startswith("sub-") and os.path.isdir(os.path.join(base, d)))


def run_subject(scratch, parc, vt, sub):
    """Assemble evidence rows + geometry + corroboration for one subject."""
    from utils.jax_free_model_io import _load_model_no_jax
    import pickle
    icadir = os.path.join(scratch, "output", "sm_ica_states", parc, sub)
    summary = _load_json(os.path.join(icadir, "ica_match_summary.json"))
    rows = build_evidence_rows(summary)

    fd = os.path.join(scratch, "output", "04_combined_hdphmm", parc, sub, "final", f"vt{vt}")
    pca_base = os.path.join(scratch, "output", "03a_pca4combined_hmm", parc, sub)
    model = _load_model_no_jax(os.path.join(fd, "best_model.pkl"))
    with open(os.path.join(fd, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    n_pcs = int(_load_json(os.path.join(pca_base, "n_pcs_lookup.json"))[vt])
    with open(os.path.join(fd, "decoded_states.pkl"), "rb") as f:
        decoded = pickle.load(f)
    active = sorted({int(x) for v in decoded.values() for x in np.unique(v)})
    geom = state_mean_geometry(model.means_, pca.components_[:n_pcs], n_pcs, active)

    # ICA<->HMM subspace affinity at K_active (principal-angle cosines)
    K_active = int(summary["K_active"])
    ica_path = os.path.join(icadir, f"ica_maps_K{K_active}.npy")
    if os.path.exists(ica_path):
        ica_maps = np.load(ica_path)                                  # (P, K_active)
        hmm_active_maps = (np.asarray(model.means_)[active][:, :n_pcs]
                           @ pca.components_[:n_pcs])                  # (n_active, P)
        cos = subspace_affinity(ica_maps, hmm_active_maps)
        geom["subspace_affinity_mean"] = float(np.mean(cos))
        geom["subspace_affinity_min"] = float(np.min(cos))
    else:
        geom["subspace_affinity_mean"] = None
        geom["subspace_affinity_min"] = None

    v = f"vt{vt}"
    rec = _load_json(os.path.join(scratch, "output", "05a_recurrence_analysis", parc, sub, v,
                                  "recurrence_summary.json"))
    tr = _load_json(os.path.join(scratch, "output", "06b_transition_structure", parc, sub, v,
                                 "transition_structure_summary.json"))
    corr = corroboration_metrics(rec, tr)
    return rows, geom, corr


def main(argv=None):
    load_dotenv()
    scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        sys.exit("SCRATCH_DIR is unset; set it in .env")
    ap = argparse.ArgumentParser(description="ICA convergence diagnostics (read-only).")
    ap.add_argument("--parcellation", default="atlas-4S156Parcels")
    ap.add_argument("--vt", default="0.95")
    ap.add_argument("--sub_id", action="append", default=None)
    a = ap.parse_args(argv)

    ev_all, geom_by, corr_by = [], {}, {}
    for sub in _subjects(scratch, a.parcellation, a.sub_id):
        rows, geom, corr = run_subject(scratch, a.parcellation, a.vt, sub)
        ev_all.extend(rows); geom_by[sub] = geom; corr_by[sub] = corr

    convergence = [max((r["n_surv"] for r in ev_all if r["sub"] == s), default=0)
                   for s in geom_by]
    eff_rank = [geom_by[s]["eff_rank_norm"] for s in geom_by]
    spear = spearman_illustrative(convergence, eff_rank) if len(geom_by) > 2 else None

    out_dir = os.path.join(scratch, "output", "sm_ica_states", a.parcellation, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnostics_summary.json"), "w") as f:
        json.dump({"evidence": ev_all, "geometry": geom_by, "corroboration": corr_by,
                   "spearman_convergence_vs_effrank": spear}, f, indent=2)
    with open(os.path.join(out_dir, "tables.md"), "w") as f:
        f.write(render_tables_md(ev_all, geom_by, corr_by))
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
