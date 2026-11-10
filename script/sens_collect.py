#!/usr/bin/env python3
"""sens_collect.py - gather the hyperparameter-sensitivity measures and compare against the published outputs.

Measures, per (side root, subject), read from the files the production scripts write:

    n_active_states_05a        05a recurrence_summary.json n_active_states (recurrence > 0)
    k_used_final               04 final_results.json final_refit.n_active_states
                               (final-iteration usage > 0.01)
    cat_<category>             05e_a4 state_flags_summary.json category_counts
    q_obs_E                    06e graph E observed modularity (N2_E observed.json q)
    q_excess_N2 / q_excess_N3  observed Q minus mean null Q (06e null_test 'effect')
    p_N2 / p_N3                06e null_test p
    fc_transition_rho (+_p)    06b fc_transition_correlation.json rho / p_value
    mfpt_fc_rho (+_p)          06b mfpt_fc_correlation.json rho / p_value
    m10_rho                    m10_05 cross_stimulus_summary.json
                               A1_recurrence_correlation.spearman_rho
    m10_null_mean, m10_delta_rho, m10_delta_p
                               (with --r5_root) sm_rel_r5_phase_null's
                               r5_phase_null_summary.json: null mean, observed rho minus
                               null mean, one-sided empirical p. Its observed rho must
                               equal m10_rho.

Published values come from the published 06e transition_modules_summary.json
(tests.N2_E / tests.N3_E) and the other published JSONs at vt0.95.

Modes:
    table   write the long TSV (config, participant, measure, value, published_value)
    sitable print the supplementary table (Markdown) from that TSV
    gate    compare one side root's outputs with the published files: measures, plus
            exact array/CSV equality for the files listed in GATE_FILES
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

PARC = "atlas-4S156Parcels"
CATS = ("unused", "low_confidence", "run_onset_anchored", "season_temporal",
        "eligible_for_content_analysis", "rare")
CONFIGS = {  # name -> vt string
    "vt0.95_covdiag_nc50_g1": "0.95", "vt0.95_covdiag_nc40_g1": "0.95",
    "vt0.95_covdiag_nc80_g1": "0.95", "vt0.95_covdiag_nc50_g5": "0.95",
    "vt0.90_covdiag_nc50_g1": "0.90", "vt0.99_covdiag_nc50_g1": "0.99",
}
SUBS = [f"sub-0{i}" for i in range(1, 7)]


def _j(path):
    with open(path) as f:
        return json.load(f)


def _p(out, stage, sub, vt, *parts):
    return os.path.join(out, stage, PARC, sub, f"vt{vt}", *parts)


def measures(out, sub, vt, published=False):
    """Measure dict for one subject from an output/ tree."""
    m = {}
    rec = _j(_p(out, "05a_recurrence_analysis", sub, vt, "recurrence_summary.json"))
    m["n_active_states_05a"] = rec["n_active_states"]
    fr = _j(os.path.join(out, "04_combined_hdphmm", PARC, sub, "final", f"vt{vt}",
                         "final_results.json"))
    m["k_used_final"] = fr["final_refit"]["n_active_states"]
    fl = _j(_p(out, "05e_temporal_trend_a4", sub, vt, "state_flags_summary.json"))
    for c in CATS:
        m[f"cat_{c}"] = fl["category_counts"][c]
    if published:
        t = _j(_p(out, "06e_transition_modules", sub, vt,
                  "transition_modules_summary.json"))["tests"]
    else:
        t = _j(_p(out, "06e_transition_modules", sub, vt, "q_excess_summary.json"))["tests"]
    m["q_obs_E"] = t["N2_E"]["q_obs"]
    m["q_excess_N2"] = t["N2_E"]["effect"]
    m["q_excess_N3"] = t["N3_E"]["effect"]
    m["p_N2"] = t["N2_E"]["p"]
    m["p_N3"] = t["N3_E"]["p"]
    fc = _j(_p(out, "06b_transition_structure", sub, vt, "fc_transition_correlation.json"))
    mf = _j(_p(out, "06b_transition_structure", sub, vt, "mfpt_fc_correlation.json"))
    m["fc_transition_rho"], m["fc_transition_p"] = fc["rho"], fc["p_value"]
    m["mfpt_fc_rho"], m["mfpt_fc_p"] = mf["rho"], mf["p_value"]
    x = _j(_p(out, "m10_05_cross_validation", sub, vt, "cross_stimulus_summary.json"))
    m["m10_rho"] = x["A1_recurrence_correlation"]["spearman_rho"]
    return m


R5_N_DRAWS = 10000
R5_TOL = 1e-12


def r5_measures(path, m10_rho):
    """Phase-randomized null measures from one r5_phase_null_summary.json.

    Refuses a summary for another stimulus, with failed draws or a different draw count,
    or whose observed rho is not the m10_05 rho of the same fit.
    The published summaries predate the `stimulus` and `n_finite` fields: a missing
    stimulus is accepted (the rho check ties the file to the Movie10 fit), and a missing
    n_finite is counted from the null_draws.npy beside the summary.
    """
    s = _j(path)
    if s.get("stimulus", "movie10") != "movie10":
        raise ValueError(f"{path}: stimulus {s['stimulus']!r}, expected movie10")
    n = s["null"]
    n_finite = n.get("n_finite")
    if n_finite is None:
        draws = np.load(os.path.join(os.path.dirname(path), "null_draws.npy"))
        n_finite = int(np.isfinite(draws).sum()) if draws.size == n["n_draws"] else -1
    if n["n_draws"] != R5_N_DRAWS or n_finite != n["n_draws"]:
        raise ValueError(f"{path}: {n_finite} finite of {n['n_draws']} draws, "
                         f"expected {R5_N_DRAWS}")
    if not (np.isfinite(m10_rho) and abs(s["observed"]["rho"] - m10_rho) <= R5_TOL):
        raise ValueError(f"{path}: observed rho {s['observed']['rho']!r} is not the "
                         f"m10_05 rho {m10_rho!r}")
    return {"m10_null_mean": n["mean"], "m10_delta_rho": s["delta_rho"],
            "m10_delta_p": s["p_empirical"]}


def r5_published(published_out, sub):
    return os.path.join(published_out, "sm_rel_r5_phase_null", PARC, sub, "vt0.95",
                        "r5_phase_null_summary.json")


def table(root, published_out, out_tsv, configs, r5_root=None):
    rows = []
    for cfg in configs:
        vt = CONFIGS[cfg]
        for sub in SUBS:
            pub = measures(published_out, sub, "0.95", published=True)
            try:
                mine = measures(os.path.join(root, cfg, "output"), sub, vt)
            except FileNotFoundError as e:
                print(f"MISSING {cfg} {sub}: {e}", file=sys.stderr)
                mine = {}
            if r5_root:
                pub.update(r5_measures(r5_published(published_out, sub), pub["m10_rho"]))
                if mine:
                    mine.update(r5_measures(
                        os.path.join(r5_root, cfg, sub, "r5_phase_null_summary.json"),
                        mine["m10_rho"]))
            for k in pub:
                rows.append((cfg, sub, k, mine.get(k, "NA"), pub[k]))
    df = pd.DataFrame(rows, columns=["config", "participant", "measure", "value",
                                     "published_value"])
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}: {len(df)} rows")


GATE_FILES = [  # (stage, relative file); arrays compared exactly
    ("05a_recurrence_analysis", "recurrence_scores.npy"),
    ("05a_recurrence_analysis", "specificity_index.npy"),
    ("05e_temporal_trend_a1", "temporal_trend_metrics.csv"),
    ("05e_temporal_trend_a2", "temporal_position_metrics.csv"),
    ("05e_temporal_trend_a3", "habituation_metrics.csv"),
    ("05e_temporal_trend_a4", "state_flags.csv"),
    ("05f_state_fc", "fc_similarity_corr_rv.npy"),
    ("06a_state_temp_dynamics", "transition_probabilities.npy"),
    ("06b_transition_structure", "mfpt_matrix.npy"),
]


def _equal(a, b):
    if isinstance(a, pd.DataFrame):
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            return False, "shape/columns differ"
        diffs = []
        for c in a.columns:
            x, y = a[c], b[c]
            if x.dtype.kind in "fi" and y.dtype.kind in "fi":
                ok = np.array_equal(x.to_numpy(float), y.to_numpy(float), equal_nan=True)
                if not ok:
                    diffs.append(f"{c} max|d|={np.nanmax(np.abs(x - y)):.3g}")
            elif not x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)):
                diffs.append(f"{c} ({int((x.astype(str) != y.astype(str)).sum())} cells)")
        return not diffs, "; ".join(diffs)
    ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    detail = "" if ok else (f"max|d|={np.nanmax(np.abs(a - b)):.3g}"
                            if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
    return ok, detail


def _load(path):
    return pd.read_csv(path) if path.endswith(".csv") else np.load(path)


def gate(side_out, published_out, subs, vt="0.95", final_only=False):
    n_bad = 0
    for sub in subs:
        if final_only:
            for fn in ("state_means_pca.npy", "state_means_parcel.npy", "state_covars.npy",
                       "state_covars_parcel.npy"):
                a = np.load(os.path.join(side_out, "04_combined_hdphmm", PARC, sub, "final",
                                         f"vt{vt}", fn))
                b = np.load(os.path.join(published_out, "04_combined_hdphmm", PARC, sub,
                                         "final", f"vt{vt}", fn))
                ok, d = _equal(a, b)
                n_bad += not ok
                print(f"{sub} final/{fn}: {'EQUAL' if ok else 'DIFF ' + d}")
            with open(os.path.join(side_out, "04_combined_hdphmm", PARC, sub, "final",
                                   f"vt{vt}", "decoded_states.pkl"), "rb") as f:
                a = pickle.load(f)
            with open(os.path.join(published_out, "04_combined_hdphmm", PARC, sub, "final",
                                   f"vt{vt}", "decoded_states.pkl"), "rb") as f:
                b = pickle.load(f)
            same_keys = sorted(a) == sorted(b)
            n_diff = sum(int(np.sum(np.asarray(a[k]) != np.asarray(b[k]))) for k in b
                         if k in a and len(a[k]) == len(b[k]))
            ok = same_keys and n_diff == 0
            n_bad += not ok
            print(f"{sub} final/decoded_states.pkl: runs {len(a)} vs {len(b)}, "
                  f"same keys {same_keys}, differing TRs {n_diff}")
            fa = _j(os.path.join(side_out, "04_combined_hdphmm", PARC, sub, "final",
                                 f"vt{vt}", "final_results.json"))
            fb = _j(os.path.join(published_out, "04_combined_hdphmm", PARC, sub, "final",
                                 f"vt{vt}", "final_results.json"))
            for k in ("test_ll_per_sample", "n_active_states", "n_total_states"):
                ok = fa["final_refit"][k] == fb["final_refit"][k]
                n_bad += not ok
                print(f"{sub} final_results.final_refit.{k}: {fa['final_refit'][k]} vs "
                      f"{fb['final_refit'][k]} {'EQUAL' if ok else 'DIFF'}")
            continue
        mine = measures(side_out, sub, vt)
        pub = measures(published_out, sub, vt, published=True)
        for k in pub:
            ok = mine[k] == pub[k]
            n_bad += not ok
            print(f"{sub} {k}: rerun {mine[k]!r} published {pub[k]!r} "
                  f"{'EQUAL' if ok else 'DIFF'}")
        for stage, fn in GATE_FILES:
            ok, d = _equal(_load(_p(side_out, stage, sub, vt, fn)),
                           _load(_p(published_out, stage, sub, vt, fn)))
            n_bad += not ok
            print(f"{sub} {stage}/{fn}: {'EQUAL' if ok else 'DIFF ' + d}")
        # m10_03 projection, when regenerated
        proj = _p(side_out, "m10_03_projected", sub, vt)
        if not os.path.islink(os.path.join(side_out, "m10_03_projected")):
            ids = _j(os.path.join(proj, "movie_run_ids.json"))
            n_runs = n_eq = 0
            for runs in ids.values():
                for r in runs:
                    n_runs += 1
                    n_eq += np.array_equal(np.load(os.path.join(proj, f"{r}.npy")),
                                           np.load(_p(published_out, "m10_03_projected",
                                                      sub, vt, f"{r}.npy")))
            n_bad += n_eq != n_runs
            print(f"{sub} m10_03_projected: {n_eq}/{n_runs} runs identical")
    print(f"GATE {'PASS' if n_bad == 0 else 'FAIL'}: {n_bad} mismatches")
    return n_bad


ROW_LABELS = {  # config -> row label, in table order
    "vt0.95_covdiag_nc50_g1": "Baseline (95%, K<sub>max</sub> 50, γ 1)",
    "vt0.95_covdiag_nc40_g1": "K<sub>max</sub> 40",
    "vt0.95_covdiag_nc80_g1": "K<sub>max</sub> 80",
    "vt0.95_covdiag_nc50_g5": "γ 5",
    "vt0.90_covdiag_nc50_g1": "90% variance",
    "vt0.99_covdiag_nc50_g1": "99% variance",
}
PUBLISHED_LABEL = "Published (final model)"
P_FLOORS = {  # p measure -> 1 / (permutations + 1)
    "p_N2": 1 / 1001, "p_N3": 1 / 501, "fc_transition_p": 1 / 5001, "mfpt_fc_p": 1 / 5001,
    "m10_delta_p": 1 / 10001,
}
PARTS = [  # (title, [(value measure, p measure or None, value format)])
    ("A. Active states / content-eligible states",
     [("n_active_states_05a", None, "{:.0f}"),
      ("cat_eligible_for_content_analysis", None, "{:.0f}")]),
    ("B. Transition-module Q in excess of the N2 / N3 nulls",
     [("q_excess_N2", "p_N2", "{:.3f}"), ("q_excess_N3", "p_N3", "{:.3f}")]),
    ("C. FC–transition ρ / MFPT–FC ρ",
     [("fc_transition_rho", "fc_transition_p", "{:.3f}"),
      ("mfpt_fc_rho", "mfpt_fc_p", "{:.3f}")]),
    ("D. Movie10 recurrence–occupancy ρ: observed / Δρ",
     [("m10_rho", None, "{:.3f}"), ("m10_delta_rho", "m10_delta_p", "{:.3f}")]),
]


def _fmt_p(p):
    return f"{p:.2g}"  # two significant figures


def _cell(sel, col, spec):
    out = []
    for key, pkey, fmt in spec:
        s = fmt.format(float(sel.loc[key, col]))
        if pkey is not None:
            p = float(sel.loc[pkey, col])
            if p > P_FLOORS[pkey] * (1 + 1e-9):  # show p only above the permutation floor
                s += f" (p = {_fmt_p(p)})"
        out.append(s)
    return " / ".join(out)


def sitable(tsv):
    """Markdown tables, one per PARTS entry: published row, then one row per config.

    A p value is printed only when it is above its permutation floor (P_FLOORS).
    """
    df = pd.read_csv(tsv, sep="\t")
    if df[["value", "published_value"]].isna().any().any():  # read_csv turns "NA" into NaN
        raise ValueError(f"{tsv}: NA values; every cell must be present")
    lines = []
    for title, spec in PARTS:
        lines += [f"**{title}**", "",
                  "| Configuration | " + " | ".join(SUBS) + " |",
                  "|---|" + "---|" * len(SUBS)]
        sources = [(PUBLISHED_LABEL, list(ROW_LABELS)[0], "published_value")] + \
            [(lab, cfg, "value") for cfg, lab in ROW_LABELS.items()]
        for label, cfg, col in sources:
            cells = [_cell(df[(df.config == cfg) & (df.participant == sub)].set_index("measure"),
                           col, spec) for sub in SUBS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["table", "gate", "sitable"])
    ap.add_argument("--root", help="sensitivity root (table) or side output/ (gate)")
    ap.add_argument("--published_out")
    ap.add_argument("--out_tsv", help="TSV to write (table) or read (sitable)")
    ap.add_argument("--r5_root", help="per-config R5 null outputs: <r5_root>/<config>/<sub>/")
    ap.add_argument("--subs", nargs="+", default=SUBS)
    ap.add_argument("--final_only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    if a.mode == "sitable":
        print(sitable(a.out_tsv))
    elif a.mode == "table":
        table(a.root, a.published_out, a.out_tsv, a.configs, a.r5_root)
    else:
        sys.exit(1 if gate(a.root, a.published_out, a.subs, final_only=a.final_only) else 0)


if __name__ == "__main__":
    main()
