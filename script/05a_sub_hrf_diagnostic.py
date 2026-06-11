#!/usr/bin/env python3
"""Sub-HRF state diagnostic: multi-metric comparison + transition classification.

Quick check that reads existing 05a state metrics and 04 transition matrices
to determine whether sub-HRF states are (a) transition bridges between longer
states, (b) fast transients, (c) borderline resolvable, or (d) genuinely
sub-HRF.

Adapted from mario-rSLDS/scripts/05b_sub_hrf_reanalysis.py.

No SLURM needed — runs in seconds on login node.

Usage:
    python script/05a_sub_hrf_diagnostic.py
    python script/05a_sub_hrf_diagnostic.py --vt 0.95
    python script/05a_sub_hrf_diagnostic.py --subjects sub-01 sub-03
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
SCRATCH = Path(os.environ["SCRATCH_DIR"])

ALL_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
ATLAS = "atlas-4S156Parcels"
TR = 1.49
HRF_PEAK_TR = round(5.0 / TR)  # 3 TRs ≈ 4.47s


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sub-HRF state diagnostic report.",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to analyze (default: all with 05a output)",
    )
    parser.add_argument(
        "--vt", type=str, default="0.95",
        help="Variance threshold subdirectory (default: 0.95)",
    )
    parser.add_argument(
        "--parcellation", type=str, default=ATLAS,
        help=f"Parcellation (default: {ATLAS})",
    )
    return parser.parse_args()


def load_state_metrics(recur_dir):
    """Load per-state dwell metrics from 05a CSV output."""
    csv_path = recur_dir / "state_recurrence_dwell_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_transition_matrix(hmm_dir):
    """Load transition matrix from best_model.pkl, or estimate empirically.

    Falls back to computing empirical transition counts from decoded_states.pkl
    if the model pickle requires unavailable dependencies (e.g., JAX).
    """
    model_path = hmm_dir / "best_model.pkl"
    decoded_path = hmm_dir / "decoded_states.pkl"

    # Try model pickle first
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            A = model.transmat_.copy()
            del model
            return A
        except (ImportError, ModuleNotFoundError):
            pass  # fall through to empirical estimation

    # Empirical transition matrix from decoded states
    if not decoded_path.exists():
        return None

    with open(decoded_path, "rb") as f:
        decoded_states = pickle.load(f)

    # Find max state id
    max_state = 0
    for seq in decoded_states.values():
        seq = np.asarray(seq)
        if len(seq) > 0:
            max_state = max(max_state, int(seq.max()))
    n = max_state + 1
    counts = np.zeros((n, n), dtype=float)
    for seq in decoded_states.values():
        seq = np.asarray(seq, dtype=int)
        for t in range(len(seq) - 1):
            counts[seq[t], seq[t + 1]] += 1

    # Row-normalize (add small pseudocount to avoid division by zero)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A = counts / row_sums
    return A


def load_dwell_distributions(hmm_dir, recur_dir):
    """Load per-state block durations from state_blocks CSV or decoded_states.

    Returns dict: state_id -> list of block durations (in TRs).
    """
    # Try the block-level CSV first
    block_path = recur_dir / "state_blocks.csv.gz"
    decoded_path = hmm_dir / "decoded_states.pkl"

    # Fall back to recomputing from decoded_states
    if not decoded_path.exists():
        return None

    with open(decoded_path, "rb") as f:
        decoded_states = pickle.load(f)

    dwells = {}
    for state_seq in decoded_states.values():
        seq = np.asarray(state_seq)
        if len(seq) == 0:
            continue
        changes = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(seq)]))
        for s, e in zip(starts, ends):
            sid = int(seq[s])
            dwells.setdefault(sid, []).append(e - s)

    return dwells


def analyze_metrics(metrics_df, dwells, n_states):
    """Compare sub-HRF definitions across mean, median, and p75."""
    rows = []
    for _, row in metrics_df.iterrows():
        k = int(row["state"])
        cat = row.get("recurrence_score", 0.0)
        status = row.get("decoded_usage_status", "unknown")

        if status == "never_decoded":
            continue

        n_blocks = int(row.get("n_blocks", 0))
        if n_blocks == 0:
            continue

        mean_dwell = row.get("mean_dwell_tr", None)

        # Compute median and p75 from dwell distributions if available
        d = dwells.get(k, []) if dwells else []
        if len(d) > 0:
            median_dwell = float(np.median(d))
            p25 = float(np.percentile(d, 25))
            p75 = float(np.percentile(d, 75))
            frac_resolvable = float(np.mean(np.array(d) >= HRF_PEAK_TR))
        else:
            median_dwell = row.get("median_dwell_tr", mean_dwell)
            p25 = None
            p75 = None
            frac_resolvable = None

        rows.append({
            "state": k,
            "recurrence_score": cat,
            "n_blocks": n_blocks,
            "mean_dwell_tr": mean_dwell,
            "median_dwell_tr": median_dwell,
            "p25_tr": p25,
            "p75_tr": p75,
            "frac_resolvable": frac_resolvable,
            "sub_hrf_mean": mean_dwell is not None and mean_dwell < HRF_PEAK_TR,
            "sub_hrf_median": median_dwell is not None and median_dwell < HRF_PEAK_TR,
            "sub_hrf_p75": p75 is not None and p75 < HRF_PEAK_TR,
        })
    return pd.DataFrame(rows)


def analyze_transitions(A, metrics_df):
    """Describe transition structure of sub-HRF states (no classification).

    Reports self-transition probability, top incoming/outgoing transition
    partners, and bridge score for each sub-HRF state.  Interpretation is
    left to the reader — no hardcoded thresholds are applied.
    """
    rows = []
    for _, row in metrics_df.iterrows():
        k = int(row["state"])
        if not row["sub_hrf_median"]:
            continue
        if k >= A.shape[0]:
            continue

        a_kk = float(A[k, k])

        # Incoming transitions (column k, excluding self)
        incoming = A[:, k].copy()
        incoming[k] = 0
        top_in_state = int(np.argmax(incoming))
        top_in_prob = float(incoming[top_in_state])

        # Outgoing transitions (row k, excluding self)
        outgoing = A[k, :].copy()
        outgoing[k] = 0
        top_out_state = int(np.argmax(outgoing))
        top_out_prob = float(outgoing[top_out_state])

        # Bridge score: geometric mean of top in/out (high = bridge-like)
        bridge_score = float(np.sqrt(top_in_prob * top_out_prob))

        # Whether top in and out go to different states (bridge topology)
        bridge_topology = (top_in_state != top_out_state)

        rows.append({
            "state": k,
            "recurrence_score": float(row.get("recurrence_score", 0.0)),
            "a_kk": a_kk,
            "mean_dwell_tr": row["mean_dwell_tr"],
            "median_dwell_tr": row["median_dwell_tr"],
            "frac_resolvable": row["frac_resolvable"],
            "top_in_state": top_in_state,
            "top_in_prob": top_in_prob,
            "top_out_state": top_out_state,
            "top_out_prob": top_out_prob,
            "bridge_score": bridge_score,
            "bridge_topology": bridge_topology,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    parc = args.parcellation
    vt_sub = f"vt{args.vt}" if args.vt else ""

    subjects = args.subjects or ALL_SUBJECTS

    all_metrics = []
    all_transitions = []

    for sub in subjects:
        recur_dir = SCRATCH / "output" / "05a_recurrence_analysis" / parc / sub
        if vt_sub:
            recur_dir = recur_dir / vt_sub
        hmm_dir = SCRATCH / "output" / "04_combined_hdphmm" / parc / sub / "final"
        if vt_sub:
            hmm_dir = hmm_dir / vt_sub

        if not recur_dir.exists():
            print(f"\n  {sub}: 05a output not found at {recur_dir} — skipping")
            continue

        # Load data
        metrics_df = load_state_metrics(recur_dir)
        if metrics_df is None:
            print(f"\n  {sub}: state_recurrence_dwell_metrics.csv not found — skipping")
            continue

        A = load_transition_matrix(hmm_dir)
        dwells = load_dwell_distributions(hmm_dir, recur_dir)
        n_states = len(metrics_df)

        print(f"\n{'=' * 70}")
        print(f"  {sub}  ({n_states} states)")
        print(f"{'=' * 70}")

        # Multi-metric comparison
        met = analyze_metrics(metrics_df, dwells, n_states)
        met["subject"] = sub

        # Filter to decoded states only
        active = met[met["n_blocks"] > 0]
        n_active = len(active)
        n_mean = int(active["sub_hrf_mean"].sum())
        n_median = int(active["sub_hrf_median"].sum())
        n_p75 = int(active["sub_hrf_p75"].sum()) if "sub_hrf_p75" in active else 0

        print(f"\n  Decoded states: {n_active}")
        print(f"  Sub-HRF counts (decoded states only):")
        print(f"    mean   < {HRF_PEAK_TR} TR:  {n_mean:3d}/{n_active} "
              f"({100 * n_mean / max(n_active, 1):.0f}%)")
        print(f"    median < {HRF_PEAK_TR} TR:  {n_median:3d}/{n_active} "
              f"({100 * n_median / max(n_active, 1):.0f}%)")
        print(f"    p75    < {HRF_PEAK_TR} TR:  {n_p75:3d}/{n_active} "
              f"({100 * n_p75 / max(n_active, 1):.0f}%)")

        # Per-state detail table
        print(f"\n  {'St':>3} {'Cat':>10} {'Blks':>5} {'Mean':>5} {'Med':>5} "
              f"{'P25':>5} {'P75':>5} {'%Res':>5} {'Flag':>6}")
        print(f"  {'-' * 60}")
        for _, r in active.sort_values("state").iterrows():
            flag = ""
            if r["sub_hrf_median"]:
                flag = " *med"
            if r["sub_hrf_mean"] and not r["sub_hrf_median"]:
                flag = " *mean"
            p25_s = f"{r['p25_tr']:5.1f}" if r["p25_tr"] is not None else "  N/A"
            p75_s = f"{r['p75_tr']:5.1f}" if r["p75_tr"] is not None else "  N/A"
            frac_s = f"{r['frac_resolvable']:5.2f}" if r["frac_resolvable"] is not None else "  N/A"
            print(f"  {int(r['state']):3d} {r['recurrence_score']:5.2f} "
                  f"{int(r['n_blocks']):5d} {r['mean_dwell_tr']:5.1f} "
                  f"{r['median_dwell_tr']:5.1f} {p25_s} {p75_s} {frac_s}{flag}")

        # Transition structure analysis
        if A is not None:
            trans = analyze_transitions(A, met)
            trans["subject"] = sub

            if len(trans) > 0:
                print(f"\n  Transition structure of sub-HRF states (median criterion):")
                print(f"  {'St':>3} {'Cat':>10} {'A_kk':>5} {'Brdg':>6} "
                      f"{'%Res':>5} {'In->k':>8} {'k->Out':>8} {'Topo':>5}")
                print(f"  {'-' * 65}")
                for _, r in trans.iterrows():
                    frac_s = f"{r['frac_resolvable']:5.2f}" if r["frac_resolvable"] is not None else "  N/A"
                    topo = "diff" if r["bridge_topology"] else "same"
                    print(f"  {int(r['state']):3d} {r['recurrence_score']:5.2f} "
                          f"{r['a_kk']:5.2f} {r['bridge_score']:6.3f} "
                          f"{frac_s} "
                          f"{int(r['top_in_state']):2d}->{r['top_in_prob']:.2f} "
                          f"{r['top_out_prob']:.2f}->{int(r['top_out_state']):2d} "
                          f"{topo:>5}")
            else:
                print(f"\n  No sub-HRF states (median criterion)")
                trans = pd.DataFrame()

            all_transitions.append(trans)
        else:
            print(f"\n  best_model.pkl not found — skipping transition analysis")

        all_metrics.append(met)

    if not all_metrics:
        print("\nNo subjects processed.")
        return

    # Cross-subject summary
    df_m = pd.concat(all_metrics, ignore_index=True)
    df_t = pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame()

    print(f"\n{'=' * 70}")
    print("  CROSS-SUBJECT SUMMARY")
    print(f"{'=' * 70}")

    # Definition comparison
    print(f"\n  Definition comparison (decoded states with blocks > 0):")
    print(f"  {'Subject':>8} {'Active':>6} {'mean<3':>6} {'med<3':>6} {'p75<3':>6}")
    decoded = df_m[df_m["n_blocks"] > 0]
    for sub in subjects:
        s = decoded[decoded["subject"] == sub]
        if len(s) == 0:
            continue
        print(f"  {sub:>8} {len(s):6d} "
              f"{int(s['sub_hrf_mean'].sum()):6d} "
              f"{int(s['sub_hrf_median'].sum()):6d} "
              f"{int(s['sub_hrf_p75'].sum()):6d}")

    # Transition summary (descriptive, no classification)
    if len(df_t) > 0:
        print(f"\n  Transition structure summary (median-based sub-HRF states):")
        print(f"  {'Subject':>8} {'SubHRF':>6} {'MeanA_kk':>9} {'MeanBrdg':>9} "
              f"{'Mean%Res':>9} {'BridgeTopo':>10}")
        for sub in subjects:
            t = df_t[df_t["subject"] == sub]
            if len(t) == 0:
                continue
            mean_akk = float(t["a_kk"].mean())
            mean_brdg = float(t["bridge_score"].mean())
            frac_res = t["frac_resolvable"].dropna()
            mean_fres = float(frac_res.mean()) if len(frac_res) > 0 else float("nan")
            n_bridge_topo = int(t["bridge_topology"].sum())
            print(f"  {sub:>8} {len(t):6d} {mean_akk:9.2f} {mean_brdg:9.3f} "
                  f"{mean_fres:9.2f} {n_bridge_topo:7d}/{len(t)}")

    # Category breakdown of sub-HRF states
    sub_hrf = decoded[decoded["sub_hrf_median"]]
    if len(sub_hrf) > 0:
        print(f"\n  Recurrence score summary for sub-HRF states (median, all subjects):")
        if "recurrence_score" in sub_hrf.columns:
            print(f"    Mean recurrence score: {sub_hrf['recurrence_score'].mean():.3f}")
        cat_counts = {}  # placeholder for legacy loop
        for cat, count in cat_counts.items():
            print(f"    {cat}: {count}")

    # Save
    out_dir = SCRATCH / "output" / "05a_sub_hrf_diagnostic" / parc
    out_dir.mkdir(parents=True, exist_ok=True)
    df_m.to_csv(out_dir / "metrics_all_subjects.csv", index=False)
    if len(df_t) > 0:
        df_t.to_csv(out_dir / "transitions_all_subjects.csv", index=False)
    print(f"\n  Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
