#!/usr/bin/env python3
"""
08g_transformer_convergence.py - cross-method convergence for the 08d/08e/08f
transformer correspondence sweep.

Three analyses:

* **D5 - transformer / annotation convergence** (Friends only).
  Correlates 08b's per-state binary content AUC (from
  ``analysis_2_decoding_per_state.json``, the univariate decoder) with
  08d D2 per-state transformer peak AUC. Both quantities are
  ROC-AUCs in [0.5, 1] (commensurate units, no aggregation needed).

  Friends-only because 08b only runs on Friends - content annotations
  are Friends-only (te-charnet narrative annotations).

* **Cross-modality dissociation** (Friends + Movie10).
  For each state in the three-way intersection of D2-selective states
  across ``w2v-bert-2.0``, ``dinov2-large``, and ``llama-3.2-3b``,
  classify as audio-/video-/text-preferred or multi-modal. Reports
  counts at three margin thresholds {0.03, 0.05, 0.10} so the
  classification's sensitivity can be inspected without re-running.

* **Recurrence × depth interaction.**
  Tests the structural-realism prediction that recurrent states sit
  deeper in the transformer hierarchy. Computes per-stimulus
  Spearman ρ and Kendall τ + bootstrap CIs against per-state D2 peak
  layer index. An exploratory pooled estimate (averaging each state's
  peak layer across stimuli first) is reported alongside.

Outputs:
``{SCRATCH_DIR}/output/08g_transformer_convergence/{parc}/{sub_id}/``

All output JSONs include ``sub_id``, ``parcellation``, ``vt``,
``eligibility_source``, ``n_candidate_states``, and ``skip_counts``
for traceability - the output path does not embed ``vt``.
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
from utils.stats import (
    bootstrap_corr_ci,
    bootstrap_mean_ci,
    bootstrap_partial_spearman_ci,
    fdr_with_nan,
    partial_spearman,
    permutation_pvalue,
    weighted_centroid_index,
)
from utils.transformer_analysis import (
    BOOTSTRAP_SEED_CROSS_MODALITY,
    BOOTSTRAP_SEED_D5,
    BOOTSTRAP_SEED_RECURRENCE_DEPTH,
    D2_SELECTIVITY_THRESHOLD,
    INTERSECTION_MIN_FO,
    MIN_CONVERGENCE_STATES,
    load_content_eligibility,
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
logger = logging.getLogger("08g_convergence")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Multi-modal classification margin sensitivity sweep. The primary cutoff
#: is the middle value (0.05) - counts at the other two are reported in the
#: payload so reviewers can inspect classification stability under stricter /
#: looser margins without re-running.
MULTIMODAL_MARGINS = (0.03, 0.05, 0.10)
PRIMARY_MULTIMODAL_MARGIN = 0.05

N_PERM_DEFAULT = 1000
BOOTSTRAP_N = 1000

#: Map from modality keyword → TRIBEv2-validated backbone. The 08-series
#: only supports these three at present; ``transformer_io.STIMULUS_MODALITIES``
#: is the source of truth for what stimuli each model can run against.
MODELS_BY_MODALITY = {
    "audio": "w2v-bert-2.0",
    "video": "dinov2-large",
    "text":  "llama-3.2-3b",
}

#: 08g supports cross-modality dissociation only on stimuli that have all
#: three modalities (audio + video + text). Friends and Movie10 are AV
#: TV stimuli with subtitles, HP is text-only, PP-FR/EN are audio+text.
CROSS_MODAL_STIMULI = ("friends", "movie10")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_d2(sub_id, parc, stimulus, model_key):
    """Load 08d D2 ``D2_state_layer_auc.json`` for one (stimulus, model).

    Returns ``None`` (not raises) on missing input - 08g runs three
    independent analyses per call and a missing D2 should skip the affected
    analyses, not crash the script.
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "08d_transformer_depth", parc, sub_id,
        f"{stimulus}_{model_key}",
    )
    path = os.path.join(base, "D2_state_layer_auc.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_08b_per_state(sub_id, parc):
    """Load 08b per-state univariate content decoding from
    ``analysis_2_decoding_per_state.json``.

    Returns a dict mapping ``state_id → {binary_auc_roc, continuous_r_squared,
    n_epochs_in_state, ...}``. The D5 analysis pulls ``binary_auc_roc`` as
    its primary metric (commensurate with 08d D2 peak AUC). ``None`` is
    returned if the file is missing.
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "08b_content_state_correspondence", parc, sub_id,
    )
    path = os.path.join(base, "analysis_2_decoding_per_state.json")
    if not os.path.exists(path):
        logger.warning(
            "08b analysis_2_decoding_per_state.json missing at %s - has 08b "
            "been re-run after the per-state decoder was added?", path,
        )
        return None
    with open(path) as f:
        payload = json.load(f)
    per_state = payload.get("per_state", {})
    if not per_state:
        return None
    return {int(k): v for k, v in per_state.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selective_states(d2_payload):
    """Extract selective states from a 08d D2 JSON, with defensive re-checks.

    Mirrors 08f's ``_is_selective`` defensiveness: re-validates
    ``max_minus_median ≥ D2_SELECTIVITY_THRESHOLD`` and
    ``fractional_occupancy ≥ INTERSECTION_MIN_FO`` instead of trusting the
    ``non_selective`` flag alone. 08d already enforces both upstream, but
    re-checking here keeps 08g robust to threshold changes that don't
    require a 08d rerun.

    Returns ``{state_id → {peak_layer, peak_auc, max_minus_median, fo}}``.
    """
    if d2_payload is None:
        return {}
    out = {}
    for sid_str, entry in d2_payload.get("states", {}).items():
        sel = entry.get("selectivity", {})
        peak_layer = sel.get("peak_layer")
        peak_val = sel.get("peak_value")
        m_minus_med = sel.get("max_minus_median")
        fo = entry.get("fractional_occupancy")

        if peak_layer is None or peak_val is None:
            continue
        if m_minus_med is None or not np.isfinite(m_minus_med):
            continue
        if m_minus_med < D2_SELECTIVITY_THRESHOLD:
            continue
        if fo is None or float(fo) < INTERSECTION_MIN_FO:
            continue

        out[int(sid_str)] = {
            "peak_layer": int(peak_layer),
            "peak_auc": float(peak_val),
            "max_minus_median": float(m_minus_med),
            "fo": float(fo),
        }
    return out


def _eligible_depths(d2_payload):
    """Per-state representational depth for ALL content-eligible states.

    Unlike :func:`_selective_states`, this applies **no** selectivity gate
    (``max_minus_median ≥ 0.05``). The recurrence×depth question - do
    higher-recurrence states peak at deeper layers? - is defined on every
    state's depth, not only those whose layer-AUC profile clears a selectivity
    threshold. With these flat profiles (peak ~0.002 AUC above the runner-up;
    see 2026-06-05 design doc) almost no state is "selective", so gating here
    starves the analysis for no statistical reason.

    Depth is the **AUC-weighted centroid layer** (``weighted_centroid_index``
    over the per-layer one-vs-rest AUC, chance 0.5), which is stable under flat
    profiles where ``argmax`` is noise. ``argmax`` ``peak_layer`` is retained
    only as a secondary descriptive field.

    The FO floor (``INTERSECTION_MIN_FO``) is kept as a *data-quality* filter
    (a state seen in <1% of TRs has too few TRs for a stable layer profile) -
    this is distinct from the FO that recurrence×depth partials out as a
    robustness probe.

    Returns ``{state_id → {centroid, peak_layer, peak_auc, max_minus_median,
    fo}}``.
    """
    if d2_payload is None:
        return {}
    out = {}
    for sid_str, entry in d2_payload.get("states", {}).items():
        layer_auc = entry.get("layer_auc")
        fo = entry.get("fractional_occupancy")
        if not layer_auc:
            continue
        if fo is None or float(fo) < INTERSECTION_MIN_FO:
            continue
        # layer_auc is a {layer_index_str → auc} dict; order by layer index.
        ordered = [layer_auc[k] for k in sorted(layer_auc, key=int)]
        centroid = weighted_centroid_index(ordered, chance=0.5)
        if not np.isfinite(centroid):
            continue
        sel = entry.get("selectivity", {})
        peak_layer = sel.get("peak_layer")
        peak_val = sel.get("peak_value")
        m_minus_med = sel.get("max_minus_median")
        out[int(sid_str)] = {
            "centroid": float(centroid),
            "peak_layer": int(peak_layer) if peak_layer is not None else None,
            "peak_auc": float(peak_val) if peak_val is not None else None,
            "max_minus_median": (
                float(m_minus_med) if m_minus_med is not None else None
            ),
            "fo": float(fo),
        }
    return out


def _load_d1_gradient(sub_id, parc, stimulus, model_key):
    """Aggregate depth-gradient context from 08d ``D1_depth_profile.json``.

    The recurrence×depth correlation is only interpretable where an aggregate
    depth gradient exists at all (Neuro review, 2026-06-05): audio has none, so
    a per-state recurrence×depth result there is vacuous by construction. This
    records, at the lag with the highest mean balanced accuracy, the Spearman
    correlation of per-layer balanced accuracy against layer index plus the
    early→late span. Returns ``None`` if the file is missing.
    """
    base = os.path.join(
        SCRATCH_DIR, "output", "08d_transformer_depth", parc, sub_id,
        f"{stimulus}_{model_key}",
    )
    path = os.path.join(base, "D1_depth_profile.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d1 = json.load(f)
    results = d1.get("results", {})
    best_lag, best_mean, best_ba = None, -np.inf, None
    for lag, layers in results.items():
        idx = sorted(layers, key=int)
        ba = np.array([layers[k]["balanced_accuracy"] for k in idx], dtype=float)
        if ba.mean() > best_mean:
            best_mean, best_lag, best_ba = ba.mean(), lag, ba
    if best_ba is None or len(best_ba) < 3:
        return None
    rho, _ = stats.spearmanr(np.arange(len(best_ba)), best_ba)
    n3 = max(1, len(best_ba) // 3)
    return {
        "d1_best_lag": best_lag,
        "d1_depth_gradient_rho": _safe_round(rho),
        "d1_ba_early": _safe_round(float(best_ba[:n3].mean())),
        "d1_ba_late": _safe_round(float(best_ba[-n3:].mean())),
        "d1_ba_span": _safe_round(float(best_ba.max() - best_ba.min())),
    }


def _safe_round(value, digits=4):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return round(float(value), digits)


def _format_float(value, digits=3):
    """NaN-safe float formatter for plot titles."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# D5 - transformer / annotation convergence (Friends only)
# ---------------------------------------------------------------------------


def _run_d5(sub_id, parc, model_key, eligibility, n_perm, out_dir, vt):
    """Per-state Spearman ρ between 08b binary AUC and 08d D2 peak AUC.

    Friends-only: 08b only runs on Friends because content annotations
    (te-charnet narrative annotations) only exist for Friends.
    """
    logger.info("=" * 40)
    logger.info("D5 - transformer / annotation convergence (model=%s)", model_key)
    logger.info("=" * 40)

    # Modality guard - defensive even though Friends supports all 3 models.
    validate_stimulus_model("friends", model_key)

    out_path = os.path.join(out_dir, f"D5_convergence_{model_key}.json")
    base_payload = {
        "sub_id": sub_id,
        "parcellation": parc,
        "vt": vt,
        "model": model_key,
        "stimulus": "friends",
        "eligibility_source": eligibility["eligibility_source"],
        "08b_metric_source": "analysis_2_decoding_per_state.binary_auc_roc",
        "08d_metric_source": "D2_state_layer_auc.peak_value",
    }

    d2 = _load_d2(sub_id, parc, "friends", model_key)
    if d2 is None:
        logger.warning("D5: Friends D2 missing - writing skip stub")
        payload = {**base_payload, "insufficient_states": True,
                   "skip_reason": "missing_d2"}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    per_state_08b = _load_08b_per_state(sub_id, parc)
    if not per_state_08b:
        logger.warning("D5: 08b per-state decoding missing - writing skip stub")
        payload = {**base_payload, "insufficient_states": True,
                   "skip_reason": "missing_08b_per_state"}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    sel_states = _selective_states(d2)
    content_eligible = set(int(s) for s in eligibility["content_eligible"])

    # Build skip counts for the audit trail.
    in_d2 = set(sel_states)
    in_08b = set(per_state_08b)
    skip_counts = {
        "only_in_08d": len(in_d2 - in_08b),
        "only_in_08b": len(in_08b - in_d2),
        "not_content_eligible": len((in_d2 & in_08b) - content_eligible),
        "missing_08b_auc": 0,
    }

    candidate_ids = sorted(in_d2 & in_08b & content_eligible)
    shared_ids = []
    x = []  # 08b binary AUC
    y = []  # 08d D2 peak AUC
    for sid in candidate_ids:
        b_auc = per_state_08b[sid].get("binary_auc_roc")
        if b_auc is None or not np.isfinite(b_auc):
            skip_counts["missing_08b_auc"] += 1
            continue
        shared_ids.append(sid)
        x.append(float(b_auc))
        y.append(float(sel_states[sid]["peak_auc"]))

    payload = {
        **base_payload,
        "n_candidate_states": len(candidate_ids),
        "n_shared_states": len(shared_ids),
        "shared_state_ids": shared_ids,
        "skip_counts": skip_counts,
    }

    if len(shared_ids) < MIN_CONVERGENCE_STATES:
        logger.warning(
            "D5: only %d shared states (< %d) - writing insufficient stub",
            len(shared_ids), MIN_CONVERGENCE_STATES,
        )
        payload["insufficient_states"] = True
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Observed Spearman ρ + Kendall τ. Per-state p-values not reported -
    # the inferential statistic is the permutation null below; bootstrap
    # CIs supply uncertainty for the observed correlations.
    rho, _ = stats.spearmanr(x_arr, y_arr)
    tau, _ = stats.kendalltau(x_arr, y_arr, method="auto")

    # Bootstrap CIs around the observed correlations (08f pattern).
    rho_pt, rho_lo, rho_hi = bootstrap_corr_ci(
        x_arr, y_arr,
        lambda u, v: stats.spearmanr(u, v),
        n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D5 + 1,
    )
    tau_pt, tau_lo, tau_hi = bootstrap_corr_ci(
        x_arr, y_arr,
        lambda u, v: stats.kendalltau(u, v, method="auto"),
        n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_D5 + 2,
    )

    # Permutation null: shuffle 08b labels under H_0 of no relation. Two-sided
    # because a strong negative correlation is also evidence of spurious
    # matching, not biological convergence.
    perm_rng = np.random.default_rng(BOOTSTRAP_SEED_D5 + 3)
    null_rhos = []
    for _ in range(n_perm):
        x_perm = perm_rng.permutation(x_arr)
        r, _ = stats.spearmanr(x_perm, y_arr)
        if np.isfinite(r):
            null_rhos.append(float(r))
    p_perm = (
        permutation_pvalue(rho, null_rhos, alternative="two-sided")
        if null_rhos else float("nan")
    )

    payload.update({
        "spearman_rho": _safe_round(rho),
        "spearman_rho_ci_low": _safe_round(rho_lo),
        "spearman_rho_ci_high": _safe_round(rho_hi),
        "kendall_tau": _safe_round(tau),
        "kendall_tau_ci_low": _safe_round(tau_lo),
        "kendall_tau_ci_high": _safe_round(tau_hi),
        "p_perm": _safe_round(p_perm),
        "n_permutations_used": len(null_rhos),
        "n_permutations_requested": int(n_perm),
        "exploratory_fields": ["kendall_tau"],
        "inferential_field": "p_perm (Spearman ρ, two-sided permutation null)",
    })

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x_arr, y_arr, alpha=0.7)
    for sid, xv, yv in zip(shared_ids, x_arr, y_arr):
        ax.annotate(str(sid), (xv, yv), fontsize=7, alpha=0.6)
    ax.set_xlabel("08b per-state speech AUC (analysis_2_per_state)")
    ax.set_ylabel("08d D2 peak transformer AUC")
    ax.set_title(
        f"D5 - Friends ({model_key}), ρ={_format_float(rho)}, "
        f"τ={_format_float(tau)}, p_perm={_format_float(p_perm)}"
    )
    fig.savefig(os.path.join(out_dir, f"D5_convergence_{model_key}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "D5 saved: ρ=%s τ=%s p_perm=%s n=%d",
        _format_float(rho), _format_float(tau), _format_float(p_perm),
        len(shared_ids),
    )
    return payload


# ---------------------------------------------------------------------------
# Cross-modality dissociation (Friends + Movie10)
# ---------------------------------------------------------------------------


def _classify_state(aucs, margin):
    """Classify a single state as <modality>_preferred or multi_modal at one margin."""
    best_mod = max(aucs, key=aucs.get)
    best_auc = aucs[best_mod]
    second_best = max(a for m, a in aucs.items() if m != best_mod)
    margin_obs = best_auc - second_best
    if margin_obs >= margin:
        return f"{best_mod}_preferred", margin_obs
    return "multi_modal", margin_obs


def _run_cross_modality(sub_id, parc, stimulus, eligibility, out_dir, vt):
    """Three-way (audio/video/text) modality classification per state."""
    logger.info("=" * 40)
    logger.info("Cross-modality dissociation - stim=%s", stimulus)
    logger.info("=" * 40)

    if stimulus not in CROSS_MODAL_STIMULI:
        logger.warning(
            "Cross-modality only supported on %s - skipping %s",
            CROSS_MODAL_STIMULI, stimulus,
        )
        return None

    # Modality guard for all three model loads.
    for model_key in MODELS_BY_MODALITY.values():
        validate_stimulus_model(stimulus, model_key)

    out_path = os.path.join(
        out_dir, f"cross_modality_dissociation_{stimulus}.json",
    )
    base_payload = {
        "sub_id": sub_id,
        "parcellation": parc,
        "vt": vt,
        "stimulus": stimulus,
        "eligibility_source": eligibility["eligibility_source"],
        "models": dict(MODELS_BY_MODALITY),
        "primary_margin": PRIMARY_MULTIMODAL_MARGIN,
        "margins_reported": list(MULTIMODAL_MARGINS),
    }

    per_modality = {}
    for modality, model_key in MODELS_BY_MODALITY.items():
        d2 = _load_d2(sub_id, parc, stimulus, model_key)
        if d2 is None:
            logger.warning(
                "Missing %s / %s - cannot run cross-modality",
                stimulus, model_key,
            )
            payload = {**base_payload, "insufficient_states": True,
                       "skip_reason": f"missing_d2_{modality}"}
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            return payload
        per_modality[modality] = _selective_states(d2)

    # Defensive content-eligibility filter (08d D2 already filters upstream,
    # but if a user reran 08d for one model with a different eligibility set
    # this would catch the mismatch).
    content_eligible = set(int(s) for s in eligibility["content_eligible"])

    intersection_full = (
        set(per_modality["audio"])
        & set(per_modality["video"])
        & set(per_modality["text"])
    )
    intersection = intersection_full & content_eligible
    skip_counts = {
        "intersection_size_pre_eligibility": len(intersection_full),
        "dropped_not_content_eligible": len(intersection_full - content_eligible),
        "audio_only_selective": len(set(per_modality["audio"]) - intersection_full),
        "video_only_selective": len(set(per_modality["video"]) - intersection_full),
        "text_only_selective": len(set(per_modality["text"]) - intersection_full),
    }

    payload = {
        **base_payload,
        "n_intersection": len(intersection),
        "n_candidate_states": len(intersection),
        "skip_counts": skip_counts,
    }

    if len(intersection) < MIN_CONVERGENCE_STATES:
        logger.warning(
            "Cross-modality: only %d states in 3-way intersection (< %d) - "
            "writing insufficient stub", len(intersection), MIN_CONVERGENCE_STATES,
        )
        payload["insufficient_states"] = True
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    sids_sorted = sorted(intersection)
    classifications_per_margin = {}  # margin -> {state_id -> dict}
    counts_per_margin = {}            # margin -> {category -> int}

    for margin in MULTIMODAL_MARGINS:
        cls_for_margin = {}
        counts = {
            "audio_preferred": 0, "video_preferred": 0,
            "text_preferred": 0, "multi_modal": 0,
        }
        for sid in sids_sorted:
            aucs = {
                m: per_modality[m][sid]["peak_auc"]
                for m in ("audio", "video", "text")
            }
            cls, margin_obs = _classify_state(aucs, margin)
            counts[cls] += 1
            cls_for_margin[int(sid)] = {
                "aucs": {m: round(v, 4) for m, v in aucs.items()},
                "margin_observed": round(margin_obs, 4),
                "classification": cls,
            }
        counts_per_margin[f"{margin:g}"] = counts
        classifications_per_margin[f"{margin:g}"] = cls_for_margin

    primary_key = f"{PRIMARY_MULTIMODAL_MARGIN:g}"
    primary_counts = counts_per_margin[primary_key]
    primary_classifications = classifications_per_margin[primary_key]

    # Bootstrap CI on the primary-margin counts via the shared helper.
    # Each category becomes an indicator vector over states; the bootstrap
    # mean of the indicator times n is the resampled count.
    indicator_by_category = {
        cat: np.array([
            1.0 if primary_classifications[sid]["classification"] == cat else 0.0
            for sid in sids_sorted
        ])
        for cat in primary_counts
    }
    n_states = len(sids_sorted)
    bootstrap_ci = {}
    for i, cat in enumerate(primary_counts):
        # Distinct sub-seed per category so the per-category bootstraps are
        # decorrelated within the run.
        mean_p, lo, hi = bootstrap_mean_ci(
            indicator_by_category[cat],
            n_boot=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED_CROSS_MODALITY + 1 + i,
        )
        bootstrap_ci[cat] = {
            "mean_count": round(float(mean_p) * n_states, 2),
            "ci_low_count": round(float(lo) * n_states, 2),
            "ci_high_count": round(float(hi) * n_states, 2),
        }

    payload.update({
        "counts_at": counts_per_margin,           # all margins
        "counts_primary": primary_counts,         # convenience for downstream
        "bootstrap_ci_primary": bootstrap_ci,
        "classifications_primary": primary_classifications,
        "exploratory_fields": [
            "counts_at",
        ],
        "inferential_field": (
            "bootstrap_ci_primary (counts at PRIMARY_MULTIMODAL_MARGIN, "
            "with CIs over states)"
        ),
    })

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Bar plot at primary margin (other margins live in the JSON sensitivity sweep).
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(primary_counts.keys())
    values = [primary_counts[k] for k in labels]
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#808080"])
    ax.set_ylabel("Number of states")
    ax.set_title(
        f"Cross-modality - {stimulus} (margin={PRIMARY_MULTIMODAL_MARGIN})"
    )
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20)
    fig.savefig(
        os.path.join(out_dir, f"cross_modality_dissociation_{stimulus}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        "Cross-modality saved: counts_primary=%s skip_counts=%s",
        primary_counts, skip_counts,
    )
    return payload


# ---------------------------------------------------------------------------
# Recurrence × depth interaction
# ---------------------------------------------------------------------------


def _per_stim_recurrence_depth(depths, recurrence_scores, eligible_set, drop_log):
    """Build (sid, rec, centroid, peak_auc, fo) rows for one stimulus.

    ``depths`` comes from :func:`_eligible_depths` - ALL content-eligible
    states (no selectivity gate). Depth is the AUC-weighted centroid layer,
    stable under flat profiles. States whose ID exceeds the recurrence vector
    are counted in ``drop_log["sid_out_of_range"]`` (logged loudly).
    """
    pairs = []  # (sid, rec, centroid, peak_auc, fo)
    for sid, info in depths.items():
        if sid not in eligible_set:
            drop_log["not_content_eligible"] += 1
            continue
        if sid >= len(recurrence_scores):
            drop_log["sid_out_of_range"] += 1
            logger.warning(
                "Recurrence × depth: state %d outside recurrence vector "
                "(len=%d) - dropped. Likely a 04 ↔ 05a state-count mismatch.",
                sid, len(recurrence_scores),
            )
            continue
        pairs.append((
            int(sid),
            float(recurrence_scores[sid]),
            float(info["centroid"]),
            float(info["peak_auc"]) if info["peak_auc"] is not None else np.nan,
            float(info["fo"]),
        ))
    return pairs


def _stim_correlation_block(pairs, n_perm, base_seed):
    """Recurrence × centroid-depth statistics for one stimulus.

    PRIMARY: raw Spearman(recurrence, centroid_depth) with a two-sided
    permutation null (shuffle recurrence) - ``p_perm``. The analytic Pearson-p
    of a partial correlation has the wrong df and assumes residual normality
    (stats review 2026-06-05), so inference rests on the permutation null, not
    the bootstrap CI alone.

    ROBUSTNESS PROBE: partial Spearman controlling fractional occupancy (FO),
    with the same permutation null - ``p_perm_partial_fo``. FO is a robustness
    probe, NOT the primary statistic: recurrence and FO are entangled, so
    partialling can suppress true signal (neuro review + USER decision
    2026-06-05, overriding the 2026-06-04 run-plan's binding FO constraint).

    Seed offsets (disjoint within the per-stimulus block): +1 boot ρ, +2 boot
    τ, +3 boot partial, +4 perm ρ, +5 perm partial.
    """
    if len(pairs) < MIN_CONVERGENCE_STATES:
        return {"insufficient_states": True, "n_states": len(pairs)}
    rec = np.array([p[1] for p in pairs], dtype=float)
    depth = np.array([p[2] for p in pairs], dtype=float)
    fo = np.array([p[4] for p in pairs], dtype=float)

    # --- PRIMARY: raw rank correlation + bootstrap CI + permutation null ---
    rho, _ = stats.spearmanr(rec, depth)
    tau, _ = stats.kendalltau(rec, depth, method="auto")
    _, rho_lo, rho_hi = bootstrap_corr_ci(
        rec, depth, lambda u, v: stats.spearmanr(u, v),
        n_boot=BOOTSTRAP_N, seed=base_seed + 1,
    )
    _, tau_lo, tau_hi = bootstrap_corr_ci(
        rec, depth, lambda u, v: stats.kendalltau(u, v, method="auto"),
        n_boot=BOOTSTRAP_N, seed=base_seed + 2,
    )
    perm_rng = np.random.default_rng(base_seed + 4)
    null_rho = []
    for _ in range(n_perm):
        r, _ = stats.spearmanr(perm_rng.permutation(rec), depth)
        if np.isfinite(r):
            null_rho.append(float(r))
    p_perm = (
        permutation_pvalue(rho, null_rho, alternative="two-sided")
        if null_rho else float("nan")
    )

    # --- ROBUSTNESS: FO-partialled correlation + bootstrap CI + perm null ---
    prho, _ = partial_spearman(rec, depth, fo)
    _, prho_lo, prho_hi = bootstrap_partial_spearman_ci(
        rec, depth, fo, n_boot=BOOTSTRAP_N, seed=base_seed + 3,
    )
    perm_rng_p = np.random.default_rng(base_seed + 5)
    null_prho = []
    for _ in range(n_perm):
        pr, _ = partial_spearman(perm_rng_p.permutation(rec), depth, fo)
        if np.isfinite(pr):
            null_prho.append(float(pr))
    p_perm_partial = (
        permutation_pvalue(prho, null_prho, alternative="two-sided")
        if null_prho else float("nan")
    )

    return {
        "n_states": len(pairs),
        # cross-state spread of the centroid depth - a near-zero SD means the
        # depth estimator is compressed toward mid-stack (flat profiles), so a
        # null correlation may be an estimator artifact rather than biology.
        # Interpret alongside the D1 aggregate gradient (neuro review).
        "depth_sd": _safe_round(float(np.std(depth))),
        "depth_mean": _safe_round(float(np.mean(depth))),
        # primary (inferential)
        "spearman_rho": _safe_round(rho),
        "spearman_rho_ci_low": _safe_round(rho_lo),
        "spearman_rho_ci_high": _safe_round(rho_hi),
        "p_perm": _safe_round(p_perm),
        "kendall_tau": _safe_round(tau),
        "kendall_tau_ci_low": _safe_round(tau_lo),
        "kendall_tau_ci_high": _safe_round(tau_hi),
        # robustness probe (FO-partialled)
        "partial_spearman_rho_fo": _safe_round(prho),
        "partial_spearman_rho_fo_ci_low": _safe_round(prho_lo),
        "partial_spearman_rho_fo_ci_high": _safe_round(prho_hi),
        "p_perm_partial_fo": _safe_round(p_perm_partial),
        "n_permutations_used": len(null_rho),
        "n_permutations_used_partial_fo": len(null_prho),
        "n_permutations_requested": int(n_perm),
    }


def _run_recurrence_depth(
    sub_id, parc, model_key, recurrence_scores, eligibility, out_dir, vt, n_perm,
):
    """Per-stimulus recurrence × centroid-depth correlation, plus pooled exploratory.

    Depth = AUC-weighted centroid layer over ALL content-eligible states (no
    selectivity gate; argmax peak_layer is noise at these flat profiles).
    Primary statistic: raw Spearman(recurrence, centroid_depth) with a two-sided
    permutation null. FO-partialled Spearman is a robustness probe. The D1
    aggregate depth-gradient ρ is attached per stimulus - a per-state
    recurrence×depth result is only interpretable where an aggregate depth
    gradient exists (audio has none). See 2026-06-05 design doc.
    """
    logger.info("=" * 40)
    logger.info("Recurrence × depth interaction (model=%s)", model_key)
    logger.info("=" * 40)

    out_path = os.path.join(
        out_dir, f"recurrence_depth_interaction_{model_key}.json",
    )
    base_payload = {
        "schema_version": 2,
        "sub_id": sub_id,
        "parcellation": parc,
        "vt": vt,
        "model": model_key,
        "eligibility_source": eligibility["eligibility_source"],
        "depth_metric": "auc_weighted_centroid_layer (chance=0.5)",
        "selectivity_gate_dropped": True,
        "fo_floor": INTERSECTION_MIN_FO,
        "structural_realism_prediction": (
            "high-recurrence states → deeper centroid layers (positive ρ)"
        ),
    }

    eligible_set = set(int(s) for s in eligibility["content_eligible"])

    per_stim_pairs = {}      # stimulus -> [(sid, rec, centroid, peak_auc, fo)]
    per_stim_results = {}    # stimulus -> dict
    skip_counts = {}         # stimulus -> drop_log
    d1_context = {}          # stimulus -> D1 depth-gradient context
    for stim_idx, stimulus in enumerate(("friends", "movie10")):
        validate_stimulus_model(stimulus, model_key)
        d2 = _load_d2(sub_id, parc, stimulus, model_key)
        if d2 is None:
            logger.info("  %s D2 missing - skipping", stimulus)
            per_stim_results[stimulus] = {"d2_missing": True}
            continue
        depths = _eligible_depths(d2)
        drop_log = {
            "not_content_eligible": 0,
            "sid_out_of_range": 0,
            "n_d2_eligible": len(depths),
        }
        pairs = _per_stim_recurrence_depth(
            depths, recurrence_scores, eligible_set, drop_log,
        )
        per_stim_pairs[stimulus] = pairs
        skip_counts[stimulus] = drop_log
        per_stim_results[stimulus] = _stim_correlation_block(
            pairs, n_perm=n_perm,
            base_seed=BOOTSTRAP_SEED_RECURRENCE_DEPTH + 1000 * stim_idx,
        )
        d1_context[stimulus] = _load_d1_gradient(
            sub_id, parc, stimulus, model_key,
        )

    # Pooled exploratory: each state's MEDIAN centroid + MEDIAN FO across
    # stimuli. Recurrence is subject-level (identical across stimuli).
    sid_to_records = {}
    for stim, pairs in per_stim_pairs.items():
        for sid, rec, centroid, _auc, fo in pairs:
            sid_to_records.setdefault(sid, []).append((stim, rec, centroid, fo))
    pooled_pairs = []  # (sid, rec, median_centroid, median_fo, n_stim)
    for sid, recs in sid_to_records.items():
        rec_val = recs[0][1]
        median_centroid = float(np.median([r[2] for r in recs]))
        median_fo = float(np.median([r[3] for r in recs]))
        pooled_pairs.append((sid, rec_val, median_centroid, median_fo, len(recs)))
    pooled_block = None
    if len(pooled_pairs) >= MIN_CONVERGENCE_STATES:
        rec_arr = np.array([p[1] for p in pooled_pairs], dtype=float)
        depth_arr = np.array([p[2] for p in pooled_pairs], dtype=float)
        fo_arr = np.array([p[3] for p in pooled_pairs], dtype=float)
        rho, _ = stats.spearmanr(rec_arr, depth_arr)
        tau, _ = stats.kendalltau(rec_arr, depth_arr, method="auto")
        prho, _ = partial_spearman(rec_arr, depth_arr, fo_arr)
        # Pooled bootstrap offsets disjoint from per-stim (+1..+5) and the
        # second stimulus block (+1000..+1005): use +11 / +13.
        _, rho_lo, rho_hi = bootstrap_corr_ci(
            rec_arr, depth_arr,
            lambda u, v: stats.spearmanr(u, v),
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_RECURRENCE_DEPTH + 11,
        )
        _, prho_lo, prho_hi = bootstrap_partial_spearman_ci(
            rec_arr, depth_arr, fo_arr,
            n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED_RECURRENCE_DEPTH + 13,
        )
        pooled_block = {
            "n_states": len(pooled_pairs),
            "spearman_rho": _safe_round(rho),
            "spearman_rho_ci_low": _safe_round(rho_lo),
            "spearman_rho_ci_high": _safe_round(rho_hi),
            "kendall_tau": _safe_round(tau),
            "partial_spearman_rho_fo": _safe_round(prho),
            "partial_spearman_rho_fo_ci_low": _safe_round(prho_lo),
            "partial_spearman_rho_fo_ci_high": _safe_round(prho_hi),
            "states_in_both_stimuli": int(
                sum(1 for p in pooled_pairs if p[4] > 1)
            ),
        }

    payload = {
        **base_payload,
        "per_stimulus": per_stim_results,
        "d1_depth_gradient": d1_context,
        "pooled_exploratory": pooled_block,
        "skip_counts": skip_counts,
        "exploratory_fields": [
            "pooled_exploratory", "kendall_tau", "partial_spearman_rho_fo",
        ],
        "inferential_field": (
            "per_stimulus.<stim>.p_perm (raw Spearman recurrence × "
            "centroid-depth, two-sided permutation null)"
        ),
        "robustness_field": (
            "per_stimulus.<stim>.p_perm_partial_fo (FO-partialled, probe only)"
        ),
    }
    # Convenience flag for downstream aggregators - keyed on the emitted
    # inferential statistic (spearman_rho present ⇒ block was computed).
    has_any_stim = any(
        isinstance(r, dict) and r.get("spearman_rho") is not None
        for r in per_stim_results.values()
    )
    if not has_any_stim:
        payload["insufficient_states"] = True

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Plot - recurrence vs centroid depth, one color per stimulus.
    fig, ax = plt.subplots(figsize=(5, 5))
    has_any_points = False
    for stim, color in (("friends", "#1f77b4"), ("movie10", "#ff7f0e")):
        pairs = per_stim_pairs.get(stim, [])
        if pairs:
            ax.scatter(
                [p[1] for p in pairs],
                [p[2] for p in pairs],
                label=f"{stim} (n={len(pairs)})",
                color=color, alpha=0.7,
            )
            has_any_points = True
    ax.set_xlabel("Recurrence score")
    ax.set_ylabel("AUC-weighted centroid layer (depth)")

    # Title shows whichever per-stim ρ / p_perm is available.
    parts = []
    for stim in ("friends", "movie10"):
        block = per_stim_results.get(stim, {})
        if block.get("spearman_rho") is not None:
            parts.append(
                f"{stim}: ρ={_format_float(block['spearman_rho'])} "
                f"p={_format_float(block.get('p_perm'))}"
            )
    title_suffix = "; ".join(parts) if parts else "no data"
    ax.set_title(f"Recurrence × depth ({model_key})\n{title_suffix}")
    if has_any_points:
        ax.legend()
    fig.savefig(
        os.path.join(out_dir, f"recurrence_depth_interaction_{model_key}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        "Recurrence × depth saved: per_stim=%s pooled n=%s",
        {s: r.get("n_states") for s, r in per_stim_results.items()},
        pooled_block.get("n_states") if pooled_block else None,
    )
    return payload


def _write_recurrence_depth_fdr_summary(rd_payloads, out_dir, sub_id, parc, vt):
    """BH-FDR the recurrence×depth permutation p's within this subject.

    Family = the per-(model, stimulus) PRIMARY ``p_perm`` (raw Spearman).
    Single-subject framing → no cross-subject correction. The FO-partialled
    robustness p is corrected as its own parallel family. Writes
    ``recurrence_depth_fdr_summary.json``.
    """
    keys, p_primary, p_partial = [], [], []
    for model_key, payload in rd_payloads.items():
        for stim, block in payload.get("per_stimulus", {}).items():
            if not isinstance(block, dict) or block.get("insufficient_states"):
                continue
            if block.get("p_perm") is None:
                continue
            keys.append({"model": model_key, "stimulus": stim})
            p_primary.append(block["p_perm"])
            ppt = block.get("p_perm_partial_fo")
            p_partial.append(ppt if ppt is not None else np.nan)

    summary = {
        "sub_id": sub_id,
        "parcellation": parc,
        "vt": vt,
        "family": "recurrence×depth per-(model,stimulus) primary p_perm",
        "note": "single-subject; no cross-subject correction",
        "n_tests": len(keys),
        "tests": [],
    }
    if keys:
        q_primary = fdr_with_nan(np.array(p_primary, dtype=float))
        q_partial = fdr_with_nan(np.array(p_partial, dtype=float))
        for k, pp, qp, ppt, qpt in zip(
            keys, p_primary, q_primary, p_partial, q_partial,
        ):
            summary["tests"].append({
                **k,
                "p_perm": _safe_round(pp),
                "q_perm_fdr": _safe_round(float(qp)),
                "p_perm_partial_fo": _safe_round(ppt),
                "q_perm_partial_fo_fdr": _safe_round(float(qpt)),
            })
    out_path = os.path.join(out_dir, "recurrence_depth_fdr_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Recurrence × depth FDR summary: %d tests", len(keys))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Transformer convergence analyses (08g).",
    )
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default=None)
    p.add_argument(
        "--models", nargs="+", default=list(MODELS_BY_MODALITY.values()),
        help="Models to run D5 and recurrence-depth for (default: all three).",
    )
    p.add_argument(
        "--stimuli_cross_modality", nargs="+",
        default=list(CROSS_MODAL_STIMULI),
        choices=list(CROSS_MODAL_STIMULI),
    )
    p.add_argument("--n_permutations", type=int, default=N_PERM_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "08g_transformer_convergence", parc, sub_id,
    )
    os.makedirs(out_dir, exist_ok=True)

    eligibility = load_content_eligibility(sub_id, parc, SCRATCH_DIR, vt=args.vt)
    logger.info(
        "Eligibility source=%s content_eligible=%d",
        eligibility["eligibility_source"],
        len(eligibility["content_eligible"]),
    )

    # Recurrence is subject-level - load once and pass through to recurrence-depth.
    recurrence_scores = load_recurrence_scores(sub_id, parc, SCRATCH_DIR, vt=args.vt)

    rd_payloads = {}
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            logger.warning("Unknown model %s - skipping", model_key)
            continue
        _run_d5(
            sub_id, parc, model_key, eligibility,
            args.n_permutations, out_dir, vt=args.vt,
        )
        rd_payloads[model_key] = _run_recurrence_depth(
            sub_id, parc, model_key, recurrence_scores, eligibility,
            out_dir, vt=args.vt, n_perm=args.n_permutations,
        )

    # BH-FDR across the within-subject recurrence×depth inferential family
    # (models × stimuli). Single-subject framing → no cross-subject correction.
    _write_recurrence_depth_fdr_summary(
        rd_payloads, out_dir, sub_id, parc, args.vt,
    )

    for stimulus in args.stimuli_cross_modality:
        _run_cross_modality(
            sub_id, parc, stimulus, eligibility, out_dir, vt=args.vt,
        )

    logger.info("=" * 60)
    logger.info("08g convergence complete")
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
