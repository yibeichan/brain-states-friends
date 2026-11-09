#!/usr/bin/env python3
"""p7_collect.py - gather the P7 sensitivity measures and compare against the published outputs.

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

Published values come from the published 06e transition_modules_summary.json
(tests.N2_E / tests.N3_E) and the other published JSONs at vt0.95.

Modes:
    table   write the long TSV (config, participant, measure, value, published_value)
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


def table(root, published_out, out_tsv, configs):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["table", "gate"])
    ap.add_argument("--root", required=True, help="P7 root (table) or side output/ (gate)")
    ap.add_argument("--published_out", required=True)
    ap.add_argument("--out_tsv")
    ap.add_argument("--subs", nargs="+", default=SUBS)
    ap.add_argument("--final_only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    if a.mode == "table":
        table(a.root, a.published_out, a.out_tsv, a.configs)
    else:
        sys.exit(1 if gate(a.root, a.published_out, a.subs, final_only=a.final_only) else 0)


if __name__ == "__main__":
    main()
