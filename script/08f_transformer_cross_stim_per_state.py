#!/usr/bin/env python3
"""
08f_transformer_cross_stim_per_state.py - per-state cross-stimulus signature
consistency (D3c) and language invariance (D4-lang).

Reads D2 outputs from 08d for two stimuli (same subject, same model) and
computes:

* **D3c - per-state similarity.** For each state that is (a) in
  ``content_eligible``, (b) fractional-occupancy ≥
  :data:`~utils.transformer_analysis.INTERSECTION_MIN_FO` in both stimuli,
  (c) D2-selective in both stimuli (``max_minus_median_auc ≥
  :data:`~utils.transformer_analysis.D2_SELECTIVITY_THRESHOLD```), compute
  the Spearman rank correlation ``rho`` of the per-state AUC profiles
  across layers. Only descriptive ``rho`` values are retained per state;
  raw per-state Spearman p-values are intentionally NOT exported (they
  are uninterpretable with 8–28 layers and uncorrected). The inferential
  statistic is the aggregate ``mean_rho`` with bootstrap 95% CI.

* **D3c aggregate.** Mean per-state ρ with bootstrap 95% CI over states,
  Spearman of per-state peak layers across stimuli (exploratory), and the
  structural-realism test ``ρ_s vs recurrence_s`` (exploratory).

* **D4-lang.** Specialization of D3c restricted to
  ``(petitprince_fr, petitprince_en)`` with ``model ∈ {w2v-bert-2.0,
  llama-3.2-3b}``.

This script performs **no feature reloading** - it is a light post-processing
step that consumes ``08d_transformer_depth/.../D2_state_layer_auc.json``.
The FO and selectivity thresholds are imported from
``utils.transformer_analysis`` so 08d, 08e, and 08f share one source of
truth.

Outputs:
``{SCRATCH_DIR}/output/08f_transformer_cross_stim_per_state/{parc}/{sub_id}/{stim_a}_vs_{stim_b}_{model}/``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import normalize_parcellation_name
from utils.stats import bootstrap_corr_ci, bootstrap_mean_ci
from utils.transformer_analysis import (
    D2_SELECTIVITY_THRESHOLD, INTERSECTION_MIN_FO, load_content_eligibility,
    load_recurrence_scores,
)
from utils.transformer_io import MODEL_REGISTRY, validate_stimulus_model

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise RuntimeError("SCRATCH_DIR must be set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("08f_cross_stim_per_state")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 1000
# Reserved slot in the 08-series seed offset block (D1 main=10_000,
# D1 confound=40_000, D2=50_000, D3a=60_000, D3c=70_000). 08f's bootstrap
# is not a permutation null - it's a percentile CI over states - but
# using a reserved seed keeps its RNG decorrelated from 08d/08e nulls
# if those ever share a process with 08f.
BOOTSTRAP_SEED_D3C = 70_000
MIN_INCLUDED_STATES = 5
# Minimum number of layers with finite AUC in BOTH stimuli for a state's
# Spearman ρ to be computed. With fewer than ~8 layers, rank correlations
# are so underpowered that the estimate (and especially its p-value) is
# uninterpretable; we skip the state and log it via skip_counts.
MIN_LAYERS_FOR_RHO = 8

VALID_STIMULI = {
    "friends", "movie10", "harrypotter", "petitprince_fr", "petitprince_en",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_d2(sub_id, parc, stimulus, model_key):
    base = os.path.join(
        SCRATCH_DIR, "output", "08d_transformer_depth", parc, sub_id,
        f"{stimulus}_{model_key}",
    )
    d2_path = os.path.join(base, "D2_state_layer_auc.json")
    if not os.path.exists(d2_path):
        raise FileNotFoundError(
            f"D2 results not found at {d2_path}. Run 08d first for "
            f"{sub_id}/{stimulus}/{model_key}."
        )
    with open(d2_path) as f:
        return json.load(f)





# ---------------------------------------------------------------------------
# D3c core computation
# ---------------------------------------------------------------------------


def _state_profile_to_vector(state_entry, n_layers):
    """Convert a D2 per-state entry's layer_auc map into an ordered vector."""
    vec = np.full(n_layers, np.nan)
    for lyr_str, auc in state_entry.get("layer_auc", {}).items():
        lyr = int(lyr_str)
        if 0 <= lyr < n_layers and auc is not None:
            vec[lyr] = float(auc)
    return vec


def _is_selective(state_entry):
    """Return True if a D2 per-state entry passes the project selectivity gate.

    Uses :data:`~utils.transformer_analysis.D2_SELECTIVITY_THRESHOLD` (=0.05)
    on ``selectivity.max_minus_median`` - the same threshold 08d applies
    when emitting the ``non_selective`` flag in ``D2_state_layer_auc.json``.
    We re-check the threshold explicitly here (instead of only trusting the
    ``non_selective`` flag) so that rerunning 08f against a legacy D2 JSON
    with a different threshold still produces correct results.
    """
    sel = state_entry.get("selectivity", {})
    m_minus_med = sel.get("max_minus_median")
    non_flag = state_entry.get("non_selective", False)
    if non_flag:
        return False
    if m_minus_med is None:
        return False
    return np.isfinite(m_minus_med) and m_minus_med >= D2_SELECTIVITY_THRESHOLD


def _compute_d3c(
    d2_a, d2_b, content_eligible_set, recurrence_scores, n_layers,
):
    """Per-state Spearman ρ between two stimuli's D2 AUC profiles.

    State inclusion rules (all must pass):
        1. Present in both D2 outputs (``d2_a`` and ``d2_b``).
        2. In ``content_eligible_set`` (from 05e_a4 or sub-HRF fallback).
        3. Fractional occupancy ≥ :data:`INTERSECTION_MIN_FO` in BOTH
           stimuli. 08d's D2 only emits states with FO ≥
           ``D2_MIN_FRACTIONAL_OCCUPANCY`` (also 0.01), but we re-check here
           defensively and record per-state ``fo_a`` / ``fo_b`` so the
           downstream audit does not have to trust upstream filtering.
        4. D2-selective in both stimuli
           (``max_minus_median ≥ D2_SELECTIVITY_THRESHOLD``).
        5. At least :data:`MIN_LAYERS_FOR_RHO` layers with finite AUC in
           both stimuli.

    Per-state p-values are intentionally NOT exported: with 8–28 layers
    they are uninterpretable without multiple-comparison correction and
    are descriptive, not inferential. The aggregate ``mean_rho`` +
    bootstrap CI in :func:`_aggregate` is the inferential statistic.
    """
    states_a = {int(k): v for k, v in d2_a.get("states", {}).items()}
    states_b = {int(k): v for k, v in d2_b.get("states", {}).items()}

    states_in_both = set(states_a.keys()) & set(states_b.keys())
    only_a = set(states_a.keys()) - set(states_b.keys())
    only_b = set(states_b.keys()) - set(states_a.keys())
    not_content_eligible = states_in_both - content_eligible_set
    candidate_ids = sorted(states_in_both & content_eligible_set)

    per_state = []
    skip_counts = {
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "not_content_eligible": len(not_content_eligible),
        "low_fo_a": 0,
        "low_fo_b": 0,
        "not_selective_a": 0,
        "not_selective_b": 0,
        "insufficient_layers": 0,
    }
    for sid in candidate_ids:
        ea = states_a[sid]
        eb = states_b[sid]

        # Defensive FO check - 08d's D2 already filters FO ≥ 0.01, but
        # re-checking here keeps 08f's output self-auditing and survives
        # any future change to the upstream threshold.
        fo_a = float(ea.get("fractional_occupancy", 0.0))
        fo_b = float(eb.get("fractional_occupancy", 0.0))
        if fo_a < INTERSECTION_MIN_FO:
            skip_counts["low_fo_a"] += 1
            continue
        if fo_b < INTERSECTION_MIN_FO:
            skip_counts["low_fo_b"] += 1
            continue

        if not _is_selective(ea):
            skip_counts["not_selective_a"] += 1
            continue
        if not _is_selective(eb):
            skip_counts["not_selective_b"] += 1
            continue

        va = _state_profile_to_vector(ea, n_layers)
        vb = _state_profile_to_vector(eb, n_layers)
        mask = np.isfinite(va) & np.isfinite(vb)
        if mask.sum() < MIN_LAYERS_FOR_RHO:
            skip_counts["insufficient_layers"] += 1
            continue

        rho, _ = stats.spearmanr(va[mask], vb[mask])

        # Peak layers come from the D2 `selectivity` sub-dict. Both were
        # produced by 08d's `layer_selectivity` from the same `layer_auc`
        # map, so a state that passed `_is_selective` above will have a
        # finite peak_layer in both stimuli. We still defensively coerce
        # to None on missing data.
        peak_a_raw = ea["selectivity"].get("peak_layer")
        peak_b_raw = eb["selectivity"].get("peak_layer")
        peak_a = int(peak_a_raw) if peak_a_raw is not None else None
        peak_b = int(peak_b_raw) if peak_b_raw is not None else None
        peak_delta = (
            abs(peak_a - peak_b)
            if peak_a is not None and peak_b is not None else None
        )

        recurrence_s = (
            float(recurrence_scores[sid])
            if sid < len(recurrence_scores) else None
        )
        per_state.append({
            "state": int(sid),
            "rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "fo_a": round(fo_a, 4),
            "fo_b": round(fo_b, 4),
            "peak_layer_a": peak_a,
            "peak_layer_b": peak_b,
            "peak_delta": peak_delta,
            "n_layers_compared": int(mask.sum()),
            "recurrence": recurrence_s,
        })

    return per_state, candidate_ids, skip_counts


def _safe_round(value, digits=4):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _aggregate(per_state):
    """Aggregate D3c per-state results into one summary dict.

    The **inferential** statistic is ``mean_rho`` with its bootstrap 95% CI
    (N is the number of included states). The peak-layer and
    structural-realism correlations are **exploratory** point estimates
    with supporting bootstrap CIs - they have no p-values because N is
    typically 5–20 states and ties are guaranteed for small-domain integer
    peak layers. Both Spearman ρ and Kendall τ are reported per the
    project convention (Kendall τ is more robust to ties; see
    ``feedback_kendalltau_ties``).
    """
    rhos = [s["rho"] for s in per_state if s["rho"] is not None]
    if len(rhos) < MIN_INCLUDED_STATES:
        return {
            "n_states": len(rhos),
            "insufficient_states": True,
        }

    # Primary inferential statistic.
    mean_rho, lo, hi = bootstrap_mean_ci(
        rhos, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D3C,
    )

    # Peak-layer rank test - build ONE paired list so peaks_a[i] and
    # peaks_b[i] always refer to the same state. (The old code used two
    # independent list comprehensions, which could silently misalign if
    # one stimulus had a missing peak_layer.)
    peak_pairs = [
        (s["peak_layer_a"], s["peak_layer_b"]) for s in per_state
        if s["peak_layer_a"] is not None and s["peak_layer_b"] is not None
    ]
    peak_spearman = peak_spearman_lo = peak_spearman_hi = None
    peak_kendall = peak_kendall_lo = peak_kendall_hi = None
    if len(peak_pairs) >= MIN_INCLUDED_STATES:
        peaks_a = np.array([p[0] for p in peak_pairs], dtype=float)
        peaks_b = np.array([p[1] for p in peak_pairs], dtype=float)
        # Distinct seed offsets for each bootstrap CI so they remain
        # decorrelated within the same run.
        peak_spearman, peak_spearman_lo, peak_spearman_hi = bootstrap_corr_ci(
            peaks_a, peaks_b,
            lambda x, y: stats.spearmanr(x, y),
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D3C + 1,
        )
        # Kendall τ with method='auto' is the project default for tied
        # ordinal data (integer peak layers tie readily).
        peak_kendall, peak_kendall_lo, peak_kendall_hi = bootstrap_corr_ci(
            peaks_a, peaks_b,
            lambda x, y: stats.kendalltau(x, y, method="auto"),
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D3C + 2,
        )

    # Structural realism (exploratory): is per-state cross-stim consistency
    # related to per-state recurrence score?
    recs = [
        (s["recurrence"], s["rho"]) for s in per_state
        if s["recurrence"] is not None and s["rho"] is not None
    ]
    struct_spearman = struct_spearman_lo = struct_spearman_hi = None
    struct_kendall = struct_kendall_lo = struct_kendall_hi = None
    if len(recs) >= MIN_INCLUDED_STATES:
        rec_arr = np.array([r for r, _ in recs], dtype=float)
        rho_arr = np.array([r for _, r in recs], dtype=float)
        struct_spearman, struct_spearman_lo, struct_spearman_hi = bootstrap_corr_ci(
            rec_arr, rho_arr,
            lambda x, y: stats.spearmanr(x, y),
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D3C + 3,
        )
        struct_kendall, struct_kendall_lo, struct_kendall_hi = bootstrap_corr_ci(
            rec_arr, rho_arr,
            lambda x, y: stats.kendalltau(x, y, method="auto"),
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D3C + 4,
        )

    return {
        "n_states": len(rhos),
        # Primary inferential statistic.
        "mean_rho": _safe_round(mean_rho),
        "mean_rho_ci_low": _safe_round(lo),
        "mean_rho_ci_high": _safe_round(hi),
        # Exploratory: peak-layer rank consistency across stimuli.
        "peak_layer_spearman_rho": _safe_round(peak_spearman),
        "peak_layer_spearman_ci_low": _safe_round(peak_spearman_lo),
        "peak_layer_spearman_ci_high": _safe_round(peak_spearman_hi),
        "peak_layer_kendall_tau": _safe_round(peak_kendall),
        "peak_layer_kendall_ci_low": _safe_round(peak_kendall_lo),
        "peak_layer_kendall_ci_high": _safe_round(peak_kendall_hi),
        "peak_layer_n_pairs": len(peak_pairs),
        # Exploratory: structural realism (recurrence vs cross-stim ρ).
        "recurrence_vs_rho_spearman": _safe_round(struct_spearman),
        "recurrence_vs_rho_spearman_ci_low": _safe_round(struct_spearman_lo),
        "recurrence_vs_rho_spearman_ci_high": _safe_round(struct_spearman_hi),
        "recurrence_vs_rho_kendall_tau": _safe_round(struct_kendall),
        "recurrence_vs_rho_kendall_ci_low": _safe_round(struct_kendall_lo),
        "recurrence_vs_rho_kendall_ci_high": _safe_round(struct_kendall_hi),
        "recurrence_vs_rho_n_pairs": len(recs),
        "exploratory_fields": [
            "peak_layer_spearman_rho", "peak_layer_kendall_tau",
            "recurrence_vs_rho_spearman", "recurrence_vs_rho_kendall_tau",
        ],
    }


def _plot(per_state, out_path, title):
    rhos = [s["rho"] for s in per_state if s["rho"] is not None]
    if not rhos:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rhos, bins=20, color="steelblue", edgecolor="black")
    ax.axvline(np.mean(rhos), color="red", linestyle="--",
               label=f"mean={np.mean(rhos):.3f}")
    ax.set_xlabel("Per-state Spearman ρ across stimuli")
    ax.set_ylabel("Number of states")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-state cross-stimulus signature consistency (08f).",
    )
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--stimulus_a", required=True, choices=sorted(VALID_STIMULI))
    p.add_argument("--stimulus_b", required=True, choices=sorted(VALID_STIMULI))
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--vt", default=None)
    p.add_argument(
        "--d4_lang", action="store_true",
        help="Flag this run as a D4-lang comparison. Enforces PP-FR vs PP-EN + "
             "audio/text model.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    stim_a = args.stimulus_a
    stim_b = args.stimulus_b
    model_key = args.model

    if stim_a == stim_b:
        raise ValueError("stimulus_a and stimulus_b must differ.")

    # Modality guard for both stimuli.
    validate_stimulus_model(stim_a, model_key)
    validate_stimulus_model(stim_b, model_key)

    if args.d4_lang:
        lang_pair = {"petitprince_fr", "petitprince_en"}
        if {stim_a, stim_b} != lang_pair:
            raise ValueError(
                "--d4_lang requires stimulus_a/stimulus_b to be petitprince_fr/_en."
            )
        if model_key not in {"w2v-bert-2.0", "llama-3.2-3b"}:
            raise ValueError(
                "--d4_lang supports only w2v-bert-2.0 and llama-3.2-3b."
            )

    analysis_label = "D4_lang" if args.d4_lang else "D3c"
    logger.info("=" * 60)
    logger.info("08f - %s", analysis_label)
    logger.info("Sub=%s A=%s B=%s model=%s", sub_id, stim_a, stim_b, model_key)
    logger.info("=" * 60)

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "08f_transformer_cross_stim_per_state",
        parc, sub_id, f"{stim_a}_vs_{stim_b}_{model_key}",
    )
    os.makedirs(out_dir, exist_ok=True)

    d2_a = _load_d2(sub_id, parc, stim_a, model_key)
    d2_b = _load_d2(sub_id, parc, stim_b, model_key)

    n_layers = MODEL_REGISTRY[model_key]["n_layers"]
    recurrence_scores = load_recurrence_scores(sub_id, parc, SCRATCH_DIR, vt=args.vt)

    eligibility = load_content_eligibility(sub_id, parc, SCRATCH_DIR, vt=args.vt)
    content_eligible_set = set(int(s) for s in eligibility["content_eligible"])

    per_state, candidate_ids, skip_counts = _compute_d3c(
        d2_a, d2_b, content_eligible_set, recurrence_scores, n_layers,
    )
    n_included = len(per_state)
    aggregate = _aggregate(per_state)

    payload = {
        "sub_id": sub_id,
        "parcellation": parc,
        "stimulus_a": stim_a,
        "stimulus_b": stim_b,
        "model": model_key,
        "vt": args.vt,
        "analysis_label": analysis_label,
        "eligibility_source": eligibility["eligibility_source"],
        "n_candidate_states": len(candidate_ids),
        "n_included_states": n_included,
        "skip_counts": skip_counts,
        "per_state": per_state,
        "aggregate": aggregate,
    }

    out_json = os.path.join(
        out_dir, f"{analysis_label}_{stim_a}_vs_{stim_b}_{model_key}.json",
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    out_png = os.path.join(
        out_dir, f"{analysis_label}_{stim_a}_vs_{stim_b}_{model_key}.png",
    )
    _plot(per_state, out_png,
          f"{analysis_label}: {stim_a} vs {stim_b} ({model_key})")

    logger.info(
        "%s: %d included / %d candidate states. mean ρ = %s",
        analysis_label, n_included, len(candidate_ids),
        aggregate.get("mean_rho"),
    )
    logger.info("Skip counts: %s", skip_counts)
    # Warn if a large fraction of candidate states was dropped by FO /
    # selectivity / insufficient-layers filters (mirrors 08e's >5% warn).
    if candidate_ids:
        n_dropped_post_candidate = len(candidate_ids) - n_included
        drop_frac = n_dropped_post_candidate / len(candidate_ids)
        if drop_frac > 0.5:
            logger.warning(
                "%.0f%% of content-eligible intersection states were dropped "
                "by FO / selectivity / layer filters (%d/%d) - results may "
                "be underpowered",
                100 * drop_frac, n_dropped_post_candidate, len(candidate_ids),
            )
    logger.info("Saved %s", out_json)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
