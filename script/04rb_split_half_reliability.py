#!/usr/bin/env python
"""
04rb: Split-half reliability analysis.

Compares structural invariants between two independently-fit HMM halves
(A and B), each fit on interleaved odd/even episodes. Loads both halves'
results and computes per-half metrics, between-half scalar agreement,
Hungarian-matched recurrence correlations, and network profile consistency.

Prerequisites:
    - 04_combined_hdphmm.py --mode split_half --half A  (and B) completed
    - Outputs: best_model.pkl, decoded_states.pkl, state_means_parcel.npy,
               split_half_results.json in each half directory

Outputs (saved to {SCRATCH_DIR}/output/04rb_split_half/{parcellation}/{sub_id}/):
    - half_invariants.json      per-half scalar metrics
    - split_half_reliability.json   structural comparison, recurrence corr, SB
    - hungarian_matching.json   matched-pair stats, occupancy-stratified
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.plot_style import (
    NETWORK_ORDER, assign_network, load_parcel_networks, compute_dominant_networks,
)
from utils.recurrence_utils import (
    compute_fractional_occupancy,
    compute_recurrence_scores,
)

from dotenv import load_dotenv

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in the .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Path helpers
# =============================================================================

def _get_half_dir(sub_id, parcellation, half):
    """Return path to split-half output directory for a given half."""
    return os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm",
        parcellation, sub_id, "split_half", half,
    )


def _get_output_dir(sub_id, parcellation):
    """Return output directory for reliability results."""
    return os.path.join(
        SCRATCH_DIR, "output", "04rb_split_half",
        parcellation, sub_id,
    )


# =============================================================================
# Loading helpers
# =============================================================================

def load_half_data(half_dir, half_label):
    """Load all required data for one half.

    Returns dict with keys: model, decoded_states, state_means, results_json.
    """
    model_path = os.path.join(half_dir, "best_model.pkl")
    decoded_path = os.path.join(half_dir, "decoded_states.pkl")
    means_path = os.path.join(half_dir, "state_means_parcel.npy")
    results_path = os.path.join(half_dir, "split_half_results.json")

    for p, name in [
        (model_path, "best_model.pkl"),
        (decoded_path, "decoded_states.pkl"),
        (means_path, "state_means_parcel.npy"),
        (results_path, "split_half_results.json"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Half {half_label}: missing {name} at {p}"
            )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(decoded_path, "rb") as f:
        decoded_states = pickle.load(f)
    state_means = np.load(means_path)
    with open(results_path, "r") as f:
        results_json = json.load(f)

    logger.info(
        f"Half {half_label}: loaded model ({model.n_components} components), "
        f"{len(decoded_states)} decoded runs, "
        f"state_means shape {state_means.shape}"
    )
    return {
        "model": model,
        "decoded_states": decoded_states,
        "state_means": state_means,
        "results_json": results_json,
    }


# =============================================================================
# Per-half invariant computation
# =============================================================================

def identify_active_states(decoded_states, n_components, min_state_usage=0.01):
    """Identify active states from decoded sequences.

    A state is active if it occupies > min_state_usage fraction of total TRs.

    Returns sorted array of active state indices.
    """
    total_trs = sum(len(seq) for seq in decoded_states.values())
    if total_trs == 0:
        return np.array([], dtype=int)

    counts = np.zeros(n_components, dtype=float)
    for seq in decoded_states.values():
        counts += np.bincount(seq, minlength=n_components)
    fractions = counts / total_trs

    active = np.where(fractions > min_state_usage)[0]
    return active


def compute_transition_entropy(transmat, active_states):
    """Compute normalized transition entropy from model transition matrix.

    Mean row entropy over active states, normalized by log(K_active).

    Note: This computes entropy **conditional on the active subspace** — rows
    are subsetted to active states and re-normalized. This measures "how random
    are transitions among active states?" rather than full HMM entropy.

    Args:
        transmat: (K, K) transition probability matrix
        active_states: array of active state indices

    Returns:
        float: normalized transition entropy in [0, 1], or NaN if < 2 active states
    """
    K_active = len(active_states)
    if K_active < 2:
        return float("nan")

    sub_mat = transmat[np.ix_(active_states, active_states)]
    # Re-normalize within active subspace
    row_sums = sub_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    sub_mat = sub_mat / row_sums

    row_entropies = np.zeros(K_active)
    for i in range(K_active):
        p = sub_mat[i]
        row_entropies[i] = np.sum(np.where(p > 0, -p * np.log(p), 0.0))

    max_entropy = np.log(K_active)
    return float(np.mean(row_entropies) / max_entropy)


def compute_dwell_stats(decoded_states):
    """Compute dwell time statistics from decoded state sequences.

    Returns dict with median_dwell and iqr_dwell (in TRs).
    """
    dwells = []
    for seq in decoded_states.values():
        if len(seq) == 0:
            continue
        current_state = seq[0]
        current_len = 1
        for t in range(1, len(seq)):
            if seq[t] == current_state:
                current_len += 1
            else:
                dwells.append(current_len)
                current_state = seq[t]
                current_len = 1
        dwells.append(current_len)

    if not dwells:
        return {"median_dwell": float("nan"), "iqr_dwell": float("nan")}

    dwells = np.array(dwells)
    q25, q50, q75 = np.percentile(dwells, [25, 50, 75])
    return {
        "median_dwell": float(q50),
        "iqr_dwell": float(q75 - q25),
    }


def compute_self_transition_prob(transmat, active_states):
    """Mean diagonal (self-transition probability) for active states."""
    if len(active_states) == 0:
        return float("nan")
    diag = np.diag(transmat)
    return float(np.mean(diag[active_states]))


def compute_half_invariants(half_data, fo_threshold=0.01, min_state_usage=0.01,
                            parcel_networks=None):
    """Compute all 6 structural invariants for one half.

    Returns dict of scalar metrics.
    """
    model = half_data["model"]
    decoded = half_data["decoded_states"]
    state_means = half_data["state_means"]
    results_json = half_data["results_json"]

    n_states = results_json["refit"]["n_states"]
    K_active_json = results_json["refit"]["n_active_states"]

    # 1. K_active — from results JSON
    active_states = identify_active_states(decoded, n_states, min_state_usage)
    K_active = len(active_states)
    logger.info(f"  K_active: {K_active} (JSON: {K_active_json})")

    # 2. Recurrence scores
    fo = compute_fractional_occupancy(decoded, n_states)
    recurrence = compute_recurrence_scores(fo, n_states, fo_threshold)

    # 3. Transition entropy
    trans_entropy = compute_transition_entropy(model.transmat_, active_states)

    # 4. Dwell time
    dwell = compute_dwell_stats(decoded)

    # 5. Self-transition probability
    self_trans = compute_self_transition_prob(model.transmat_, active_states)

    # 6. Network composition
    net_composition = {}
    if parcel_networks is not None:
        dominant_nets = compute_dominant_networks(
            state_means, active_states, parcel_networks
        )
        # Count states per network
        for net in NETWORK_ORDER:
            count = sum(1 for n in dominant_nets.values() if n == net)
            if count > 0:
                net_composition[net] = count

    return {
        "K_active": int(K_active),
        "n_states_model": int(n_states),
        "recurrence_scores": recurrence.tolist(),
        "recurrence_mean": float(np.mean(recurrence[active_states])) if K_active > 0 else float("nan"),
        "recurrence_median": float(np.median(recurrence[active_states])) if K_active > 0 else float("nan"),
        "transition_entropy": trans_entropy,
        "median_dwell": dwell["median_dwell"],
        "iqr_dwell": dwell["iqr_dwell"],
        "self_transition_prob": self_trans,
        "network_composition": net_composition,
        "active_states": active_states.tolist(),
        "n_decoded_runs": len(decoded),
    }


# =============================================================================
# Between-half comparison
# =============================================================================

def compare_scalar_invariants(inv_a, inv_b):
    """Compare scalar invariants between halves.

    Returns dict of absolute differences and ratios for each scalar metric.
    """
    metrics = [
        "K_active", "transition_entropy", "median_dwell",
        "iqr_dwell", "self_transition_prob",
        "recurrence_mean", "recurrence_median",
    ]
    comparison = {}
    for m in metrics:
        va, vb = inv_a[m], inv_b[m]
        if np.isnan(va) or np.isnan(vb):
            comparison[m] = {"A": va, "B": vb, "abs_diff": float("nan"), "ratio": float("nan")}
        else:
            abs_diff = abs(va - vb)
            denom = (abs(va) + abs(vb)) / 2.0
            ratio = abs_diff / denom if denom > 0 else float("nan")
            comparison[m] = {
                "A": round(va, 6),
                "B": round(vb, 6),
                "abs_diff": round(abs_diff, 6),
                "ratio": round(ratio, 6),
            }
    return comparison


def compare_sorted_recurrence(inv_a, inv_b):
    """Compare sorted recurrence score distributions descriptively.

    Sorts each half's recurrence vector in descending order and compares the
    value distributions only. State alignment should be assessed separately
    via Hungarian matching; correlating independently sorted vectors is not a
    valid reliability estimate.
    """
    rec_a = np.array(inv_a["recurrence_scores"])
    rec_b = np.array(inv_b["recurrence_scores"])

    # Filter to active states for each half
    active_a = np.array(inv_a["active_states"])
    active_b = np.array(inv_b["active_states"])

    sorted_a = np.sort(rec_a[active_a])[::-1] if len(active_a) > 0 else np.array([])
    sorted_b = np.sort(rec_b[active_b])[::-1] if len(active_b) > 0 else np.array([])

    result = {}

    # KS test on sorted distributions
    if len(sorted_a) > 0 and len(sorted_b) > 0:
        ks_stat, ks_p = ks_2samp(sorted_a, sorted_b)
        result["ks_statistic"] = round(float(ks_stat), 6)
        result["ks_pvalue"] = round(float(ks_p), 6)
    else:
        result["ks_statistic"] = float("nan")
        result["ks_pvalue"] = float("nan")

    result["n_active_A"] = int(len(sorted_a))
    result["n_active_B"] = int(len(sorted_b))
    result["note"] = (
        "Descriptive distribution comparison only. Reliability across halves "
        "should be interpreted from Hungarian-matched states, not from "
        "independently sorted recurrence vectors."
    )

    return result


def hungarian_matching(means_a, means_b, active_a, active_b, match_threshold=0.3):
    """Match states between halves using Hungarian algorithm on correlation distance.

    Args:
        means_a, means_b: (n_states, n_parcels) state mean arrays
        active_a, active_b: arrays of active state indices
        match_threshold: minimum Pearson r for a match to be considered valid

    Returns:
        dict with matched pairs, correlation values, and quality stats
    """
    if len(active_a) == 0 or len(active_b) == 0:
        return {
            "n_matched": 0,
            "n_above_threshold": 0,
            "pairs": [],
            "mean_correlation": float("nan"),
        }

    # Correlation matrix between active states of A and B
    ma = means_a[active_a]  # (Ka, P)
    mb = means_b[active_b]  # (Kb, P)

    # Compute correlation matrix
    # Standardize rows
    ma_z = (ma - ma.mean(axis=1, keepdims=True))
    ma_std = ma.std(axis=1, keepdims=True)
    ma_std = np.where(ma_std > 0, ma_std, 1.0)
    ma_z = ma_z / ma_std

    mb_z = (mb - mb.mean(axis=1, keepdims=True))
    mb_std = mb.std(axis=1, keepdims=True)
    mb_std = np.where(mb_std > 0, mb_std, 1.0)
    mb_z = mb_z / mb_std

    corr_mat = (ma_z @ mb_z.T) / ma_z.shape[1]  # (Ka, Kb)

    # Distance = 1 - correlation (for minimization)
    cost_mat = 1.0 - corr_mat
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    pairs = []
    for ri, ci in zip(row_ind, col_ind):
        r_val = float(corr_mat[ri, ci])
        pairs.append({
            "state_A": int(active_a[ri]),
            "state_B": int(active_b[ci]),
            "correlation": round(r_val, 6),
            "above_threshold": r_val >= match_threshold,
        })

    pairs.sort(key=lambda x: x["correlation"], reverse=True)
    above = [p for p in pairs if p["above_threshold"]]
    all_corrs = [p["correlation"] for p in pairs]

    return {
        "n_matched": len(pairs),
        "n_above_threshold": len(above),
        "match_threshold": match_threshold,
        "mean_correlation": round(float(np.mean(all_corrs)), 6) if all_corrs else float("nan"),
        "mean_correlation_above_threshold": (
            round(float(np.mean([p["correlation"] for p in above])), 6)
            if above else float("nan")
        ),
        "pairs": pairs,
    }


def matched_recurrence_correlation(matching_result, inv_a, inv_b):
    """Compute Spearman correlation of matched recurrence scores.

    Primary reliability metric: for Hungarian-matched state pairs (above
    threshold), correlate their recurrence scores.

    Spearman-Brown is intentionally not reported here because the matched,
    thresholded state pairs are not parallel-form half-test scores.
    """
    rec_a = np.array(inv_a["recurrence_scores"])
    rec_b = np.array(inv_b["recurrence_scores"])

    above_pairs = [p for p in matching_result["pairs"] if p["above_threshold"]]
    if len(above_pairs) < 5:
        return {
            "raw_spearman": float("nan"),
            "raw_pvalue": float("nan"),
            "spearman_brown": float("nan"),
            "n_matched_pairs": len(above_pairs),
            "caveat": (
                "Too few matched pairs (n < 5) for meaningful Spearman correlation. "
                "Hungarian-matched, thresholded state pairs are not suitable "
                "inputs for a Spearman-Brown reliability correction."
            ),
        }

    ra = np.array([rec_a[p["state_A"]] for p in above_pairs])
    rb = np.array([rec_b[p["state_B"]] for p in above_pairs])

    rho, pval = spearmanr(ra, rb)
    rho = float(rho)
    pval = float(pval)

    return {
        "raw_spearman": round(rho, 6),
        "raw_pvalue": round(pval, 6),
        "spearman_brown": float("nan"),
        "n_matched_pairs": len(above_pairs),
        "caveat": (
            "Hungarian-matched, thresholded state pairs are not suitable "
            "inputs for a Spearman-Brown reliability correction."
        ),
    }


def network_profile_consistency(matching_result, inv_a, inv_b, means_a, means_b,
                                parcel_networks):
    """Check if matched state pairs share the same dominant network.

    Returns fraction of above-threshold matched pairs with matching dominant
    networks. This is a coarse descriptive summary only: it discards sign and
    distributed topography, so it should not be treated as a full biological
    equivalence test.
    """
    if parcel_networks is None:
        return {"network_match_fraction": float("nan"), "note": "parcel labels unavailable"}

    active_a = np.array(inv_a["active_states"])
    active_b = np.array(inv_b["active_states"])

    dom_a = compute_dominant_networks(means_a, active_a, parcel_networks)
    dom_b = compute_dominant_networks(means_b, active_b, parcel_networks)

    above_pairs = [p for p in matching_result["pairs"] if p["above_threshold"]]
    if not above_pairs:
        return {"network_match_fraction": float("nan"), "n_pairs": 0}

    n_match = 0
    pair_details = []
    for p in above_pairs:
        net_a = dom_a.get(p["state_A"], "Unknown")
        net_b = dom_b.get(p["state_B"], "Unknown")
        match = net_a == net_b
        if match:
            n_match += 1
        pair_details.append({
            "state_A": p["state_A"],
            "state_B": p["state_B"],
            "network_A": net_a,
            "network_B": net_b,
            "match": match,
            "correlation": p["correlation"],
        })

    return {
        "network_match_fraction": round(n_match / len(above_pairs), 4),
        "n_pairs": len(above_pairs),
        "n_matching": n_match,
        "pair_details": pair_details,
    }


def occupancy_stratified_matching(matching_result, inv_a, inv_b, top_n=10):
    """Report matching quality separately for top-N states by FO vs rest.

    Uses pooled fractional occupancy (mean across halves) to rank states.
    """
    rec_a = np.array(inv_a["recurrence_scores"])
    rec_b = np.array(inv_b["recurrence_scores"])

    above_pairs = [p for p in matching_result["pairs"] if p["above_threshold"]]
    if len(above_pairs) == 0:
        return {
            "top_n": top_n,
            "top_mean_corr": float("nan"),
            "rest_mean_corr": float("nan"),
        }

    # Rank by mean recurrence across halves
    for p in above_pairs:
        p["mean_recurrence"] = (rec_a[p["state_A"]] + rec_b[p["state_B"]]) / 2.0

    sorted_pairs = sorted(above_pairs, key=lambda x: x["mean_recurrence"], reverse=True)
    top_pairs = sorted_pairs[:top_n]
    rest_pairs = sorted_pairs[top_n:]

    top_corrs = [p["correlation"] for p in top_pairs]
    rest_corrs = [p["correlation"] for p in rest_pairs]

    return {
        "top_n": top_n,
        "n_top": len(top_pairs),
        "n_rest": len(rest_pairs),
        "top_mean_corr": round(float(np.mean(top_corrs)), 6) if top_corrs else float("nan"),
        "top_median_corr": round(float(np.median(top_corrs)), 6) if top_corrs else float("nan"),
        "rest_mean_corr": round(float(np.mean(rest_corrs)), 6) if rest_corrs else float("nan"),
        "rest_median_corr": round(float(np.median(rest_corrs)), 6) if rest_corrs else float("nan"),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    sub_id = args.sub_id
    parcellation = normalize_parcellation_name(args.parcellation)
    fo_threshold = args.fo_threshold
    match_threshold = args.match_threshold

    logger.info(f"Split-half reliability: {sub_id}, {parcellation}")
    logger.info(f"  fo_threshold={fo_threshold}, match_threshold={match_threshold}")

    # Load both halves
    half_dir_a = _get_half_dir(sub_id, parcellation, "A")
    half_dir_b = _get_half_dir(sub_id, parcellation, "B")
    logger.info(f"Half A dir: {half_dir_a}")
    logger.info(f"Half B dir: {half_dir_b}")

    data_a = load_half_data(half_dir_a, "A")
    data_b = load_half_data(half_dir_b, "B")

    # Load parcel network mapping (optional, for network composition)
    parcel_networks = load_parcel_networks(parcellation)

    # ── Per-half invariants ──────────────────────────────────────────────────
    logger.info("Computing per-half invariants...")
    logger.info("  Half A:")
    inv_a = compute_half_invariants(data_a, fo_threshold, parcel_networks=parcel_networks)
    logger.info("  Half B:")
    inv_b = compute_half_invariants(data_b, fo_threshold, parcel_networks=parcel_networks)

    half_invariants = {
        "A": inv_a,
        "B": inv_b,
        "fo_threshold": fo_threshold,
        "sub_id": sub_id,
        "parcellation": parcellation,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Between-half comparison ──────────────────────────────────────────────
    logger.info("Comparing scalar invariants...")
    scalar_comparison = compare_scalar_invariants(inv_a, inv_b)

    logger.info("Comparing sorted recurrence distributions...")
    sorted_rec = compare_sorted_recurrence(inv_a, inv_b)

    logger.info("Running Hungarian matching...")
    matching = hungarian_matching(
        data_a["state_means"], data_b["state_means"],
        np.array(inv_a["active_states"]), np.array(inv_b["active_states"]),
        match_threshold=match_threshold,
    )
    logger.info(
        f"  {matching['n_matched']} pairs, "
        f"{matching['n_above_threshold']} above r>{match_threshold}"
    )

    logger.info("Computing matched recurrence correlation...")
    rec_corr = matched_recurrence_correlation(matching, inv_a, inv_b)
    logger.info(
        f"  Raw Spearman: r={rec_corr['raw_spearman']}, "
        f"Spearman-Brown: {rec_corr['spearman_brown']}"
    )

    logger.info("Checking network profile consistency...")
    net_consistency = network_profile_consistency(
        matching, inv_a, inv_b,
        data_a["state_means"], data_b["state_means"],
        parcel_networks,
    )

    logger.info("Computing occupancy-stratified matching...")
    occ_stratified = occupancy_stratified_matching(matching, inv_a, inv_b, top_n=10)

    # ── Assemble results ─────────────────────────────────────────────────────
    reliability = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "fo_threshold": fo_threshold,
        "match_threshold": match_threshold,
        "scalar_comparison": scalar_comparison,
        "sorted_recurrence": sorted_rec,
        "matched_recurrence_correlation": rec_corr,
        "network_profile_consistency": net_consistency,
        "method_notes": [
            "Split-half recurrence reliability is assessed on Hungarian-matched states.",
            "Sorted recurrence output is descriptive only and not a reliability coefficient.",
            "Network profile consistency is a coarse descriptive summary (single "
            "argmax of |activation|) that ignores sign and distributed topography.",
            "Matching uses parcel-space state means only; diagonal covariance "
            "structure is not compared. Covariance-based matching is in 05d/05f.",
            "Dwell times are from Viterbi decoding, which upper-bounds true "
            "dwell durations (blockier state assignments than posterior marginals).",
        ],
        "timestamp": datetime.now().isoformat(),
    }

    hungarian_result = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "match_threshold": match_threshold,
        "matching": matching,
        "occupancy_stratified": occ_stratified,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = _get_output_dir(sub_id, parcellation)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving results to {out_dir}")

    with open(os.path.join(out_dir, "half_invariants.json"), "w") as f:
        json.dump(half_invariants, f, indent=2)

    with open(os.path.join(out_dir, "split_half_reliability.json"), "w") as f:
        json.dump(reliability, f, indent=2)

    with open(os.path.join(out_dir, "hungarian_matching.json"), "w") as f:
        json.dump(hungarian_result, f, indent=2)

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SPLIT-HALF RELIABILITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  K_active:  A={inv_a['K_active']}, B={inv_b['K_active']}")
    logger.info(f"  Trans entropy:  A={inv_a['transition_entropy']:.4f}, B={inv_b['transition_entropy']:.4f}")
    logger.info(f"  Median dwell:  A={inv_a['median_dwell']:.1f}, B={inv_b['median_dwell']:.1f}")
    logger.info(f"  Self-trans:  A={inv_a['self_transition_prob']:.4f}, B={inv_b['self_transition_prob']:.4f}")
    logger.info(f"  Hungarian matched: {matching['n_above_threshold']}/{matching['n_matched']} above r>{match_threshold}")
    logger.info(f"  Mean match corr: {matching['mean_correlation_above_threshold']}")
    logger.info(f"  Recurrence Spearman: r={rec_corr['raw_spearman']} (p={rec_corr['raw_pvalue']})")
    logger.info(f"  Spearman-Brown: {rec_corr['spearman_brown']}")
    if isinstance(net_consistency.get("network_match_fraction"), float) and not np.isnan(net_consistency["network_match_fraction"]):
        logger.info(f"  Network match: {net_consistency['network_match_fraction']:.1%} ({net_consistency['n_matching']}/{net_consistency['n_pairs']})")
    logger.info(f"  Top-10 vs rest corr: {occ_stratified['top_mean_corr']} vs {occ_stratified['rest_mean_corr']}")
    logger.info("=" * 60)
    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split-half reliability analysis for combined HDP-HMM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/04rb_split_half_reliability.py --sub_id sub-01
  python script/04rb_split_half_reliability.py --sub_id sub-01 --match_threshold 0.4
""",
    )
    parser.add_argument("--sub_id", required=True, help="Subject ID (e.g. sub-01)")
    parser.add_argument(
        "--parcellation", default="atlas-4S156Parcels",
        help="Parcellation name (default: atlas-4S156Parcels)",
    )
    parser.add_argument(
        "--fo_threshold", type=float, default=0.01,
        help="FO threshold for recurrence scores (default: 0.01)",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.3,
        help="Minimum Pearson r for Hungarian match quality (default: 0.3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
