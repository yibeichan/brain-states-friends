#!/usr/bin/env python3
"""
08e_transformer_cross_stim_aggregate.py - aggregate Friends → test-stimulus
transformer transfer (D3a).

For each (subject, test_stimulus, model) triple, this script:

1. Validates modality compatibility between the stimulus and the model
   (e.g. rejects ``harrypotter + w2v-bert-2.0`` - no audio stream).
2. Loads Friends raw features and decoded states, fits a per-layer PCA on the
   03a **training split only**.
3. Loads test-stimulus raw features (Friends-HMM-decoded) and projects them
   through the Friends-fit PCA.
4. Restricts analysis to the intersection of ``content_eligible`` states that
   have FO ≥ 1% in **both** Friends and the test stimulus (fixes a silent
   inclusion bug in the old 08d).
5. Fits a single ``RidgeClassifier`` on Friends intersection TRs per layer and
   evaluates on the test-stimulus intersection TRs. Reports balanced accuracy,
   chance level, and per-state recall / precision.
6. Builds a circular-shift null on **test-stimulus** labels per layer and
   reports a permutation p-value.

Outputs are written to
``{SCRATCH_DIR}/output/08e_transformer_cross_stim_aggregate/{parcellation}/{sub_id}/{test_stimulus}_{model}/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re as _re
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import (
    check_checkpoint, feature_key_for_cross_stim_run_id, load_training_split,
    normalize_parcellation_name, resolve_stage_file,
)
from utils.stats import benjamini_hochberg, permutation_pvalue
from utils.transformer_analysis import (
    INTERSECTION_MIN_FO, build_layer_feature_matrix, build_run_boundaries,
    compute_effect_size, load_content_eligibility,
    precompute_eligible_null_state_sequences, stream_pca_features,
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
logger = logging.getLogger("08e_transformer_cross_stim")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PCA_VARIANCE_THRESHOLD = 0.95
# INTERSECTION_MIN_FO is imported from utils.transformer_analysis so that
# 08e, 08f, and 08g share the same source of truth (== 0.01).
N_PERMUTATIONS_DEFAULT = 1000
# Base seed for the null permutation sequence. Matches the 08d sibling
# offsets (D1 main=10_000, D1 confound=40_000, D2=50_000) by allocating
# D3a=60_000 so the null sequences used by different 08-series analyses
# never collide.
NULL_SEED_D3A = 60_000

CROSS_STIM_STAGE_MAP = {
    "movie10": "m10_04_decoded",
    "harrypotter": "hp_04_decoded",
    "petitprince_fr": "pp_04_decoded",
    "petitprince_en": "pp_04_decoded",
}

# Brain-to-feature lag for the transfer decoder. ``friends_best`` uses the lag
# at which this participant's within-Friends decoder (08d D1) peaked for the
# same model, chosen on Friends only and never on the test stimulus. Lag 0 is
# excluded from that peak search, as in 08d (PEAK_LAG_EXCLUDE there): it is
# the synchronous diagnostic, not a candidate peak.
LAG_FRIENDS_BEST = "friends_best"
PEAK_LAG_EXCLUDE = {0}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_friends_decoded(sub_id, parc):
    base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parc, sub_id, "final",
    )
    ds_path = resolve_stage_file(base, "decoded_states.pkl", "Friends decoded states")
    with open(ds_path, "rb") as f:
        return pickle.load(f)


def _load_cross_stim_decoded(sub_id, parc, stimulus):
    stage = CROSS_STIM_STAGE_MAP[stimulus]
    base = os.path.join(SCRATCH_DIR, "output", stage, parc, sub_id)
    ds_path = resolve_stage_file(
        base, "decoded_states.pkl", f"{stimulus} decoded states",
    )
    with open(ds_path, "rb") as f:
        return pickle.load(f)


# Feature loading, drift alignment, and per-layer PCA all live in
# :func:`utils.transformer_analysis.stream_pca_features`. 08e calls it twice:
# once to fit PCA on Friends training runs, once more (with ``pca_models=``
# supplied) to project the test stimulus through the same PCA.


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _compute_intersection(friends_states, test_states, eligible_ids):
    """Return sorted state list meeting FO ≥ 1% in both stimuli + eligibility."""
    eligible_set = set(int(s) for s in eligible_ids)
    f_counts = {}
    for s in friends_states:
        f_counts[int(s)] = f_counts.get(int(s), 0) + 1
    t_counts = {}
    for s in test_states:
        t_counts[int(s)] = t_counts.get(int(s), 0) + 1
    f_total = len(friends_states)
    t_total = len(test_states)

    kept = []
    stats_out = {}
    for s in sorted(eligible_set):
        f_fo = f_counts.get(s, 0) / max(f_total, 1)
        t_fo = t_counts.get(s, 0) / max(t_total, 1)
        if f_fo >= INTERSECTION_MIN_FO and t_fo >= INTERSECTION_MIN_FO:
            kept.append(s)
            stats_out[s] = {
                "friends_fo": round(f_fo, 4),
                "test_fo": round(t_fo, 4),
            }
    return kept, stats_out


def _fit_eval_one_layer(
    layer_idx, friends_X, friends_y, friends_mask, test_X, test_y, test_mask,
    intersect_set, null_friends_seqs, n_classes,
    subsets=None,
):
    """Train Ridge on Friends intersection, test on test-stim intersection.

    Returns a metrics dict or ``None`` if training is degenerate.

    Null model (principled for a cross-stimulus transfer claim): for each
    permutation, the Friends **training** labels are circular-shifted within
    their Friends runs, the Ridge classifier is refit on the shuffled pairs,
    and balanced accuracy is computed against the REAL test labels. This
    tests the null hypothesis that the learned Friends feature→state mapping
    carries no test-stimulus-relevant structure. Mirrors 08d D1's refit-per-
    permutation pattern (``loro_ridge_classifier_cv`` in that context).

    Per-subset emission (Coding review 2026-05-28 § 3 + Stats review). When
    ``subsets`` is given, it is a dict mapping subset_name -> full-length bool
    mask over the test array (each subset mask is already AND'd with
    ``test_mask`` by the caller). The Friends classifier is fit once and
    predictions are evaluated on each subset's TR slice. Each null permutation
    is also re-evaluated on each subset (the same shuffled classifier, masked
    differently) so the null distribution is principled per subset. Per-subset
    ``per_state`` recall/precision is **not** replicated; the pooled file
    carries the full per-state breakdown.
    """
    X_train = friends_X[friends_mask]
    y_train = friends_y[friends_mask]
    # Keep test_X full so subset re-masking is straightforward. Pooled metrics
    # use the same test_mask-restricted slice as before - behavior unchanged
    # when subsets=None.
    X_test_pooled = test_X[test_mask]
    y_test_pooled = test_y[test_mask]

    if (
        len(X_train) == 0
        or len(np.unique(y_train)) < 2
        or len(X_test_pooled) == 0
        or len(np.unique(y_test_pooled)) < 2
    ):
        logger.warning(
            "Layer %d: degenerate train/test split "
            "(n_train=%d, n_test=%d, unique_train=%d, unique_test=%d) - skipping",
            layer_idx, len(X_train), len(X_test_pooled),
            len(np.unique(y_train)) if len(y_train) else 0,
            len(np.unique(y_test_pooled)) if len(y_test_pooled) else 0,
        )
        return None

    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    # Predict on FULL test array so subset masks can re-slice without re-predict.
    y_pred_full = clf.predict(test_X)
    y_pred = y_pred_full[test_mask]
    y_test = y_test_pooled
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Per-state recall + precision over the intersection set (pooled only).
    per_state = {}
    labels_sorted = sorted(intersect_set)
    recall = recall_score(
        y_test, y_pred, labels=labels_sorted, average=None, zero_division=0,
    )
    precision = precision_score(
        y_test, y_pred, labels=labels_sorted, average=None, zero_division=0,
    )
    for lbl, r, p in zip(labels_sorted, recall, precision):
        per_state[int(lbl)] = {
            "recall": round(float(r), 4),
            "precision": round(float(p), 4),
        }

    # Null distribution: refit Ridge on circular-shifted Friends training
    # labels (shift held within each Friends run to preserve within-run
    # autocorrelation), evaluate on REAL test labels. Each row of
    # `null_friends_seqs` is already restricted to the intersection
    # subspace (shape matches X_train), so no secondary masking is needed
    # - this is the label-space-leakage fix from 2026-04-10.
    #
    # When `subsets` is given, each shuffled classifier is also evaluated on
    # each subset slice to build per-subset null distributions. We do NOT
    # cache all 1000 full-length null predictions (would cost ~200 MB for
    # movie10); per-subset null accs are accumulated in-place.
    null_accs = []
    subset_null_accs = (
        {name: [] for name in subsets} if subsets else None
    )
    n_skipped = 0
    for null_y_train in null_friends_seqs:
        if len(np.unique(null_y_train)) < 2:
            n_skipped += 1
            continue
        clf_null = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf_null.fit(X_train, null_y_train)
        y_pred_null_full = clf_null.predict(test_X)
        null_accs.append(float(balanced_accuracy_score(
            y_test, y_pred_null_full[test_mask],
        )))
        if subsets:
            for name, smask in subsets.items():
                y_t = test_y[smask]
                y_p = y_pred_null_full[smask]
                if len(np.unique(y_t)) >= 2:
                    subset_null_accs[name].append(
                        float(balanced_accuracy_score(y_t, y_p))
                    )

    if n_skipped:
        skip_frac = n_skipped / len(null_friends_seqs)
        if skip_frac > 0.05:
            logger.warning(
                "Layer %d: %d/%d null permutations skipped (%.1f%%) - "
                "null distribution may be biased",
                layer_idx, n_skipped, len(null_friends_seqs), 100 * skip_frac,
            )
        else:
            logger.debug(
                "Layer %d: %d/%d null permutations skipped",
                layer_idx, n_skipped, len(null_friends_seqs),
            )

    p = permutation_pvalue(bal_acc, null_accs, alternative="greater")
    eff = compute_effect_size(bal_acc, n_classes)

    result = {
        "balanced_accuracy": round(bal_acc, 4),
        "chance_level": round(eff["chance_level"], 4),
        "normalized_effect_size": round(eff["normalized_effect_size"], 4),
        "p_perm": round(float(p), 4),
        "n_permutations": len(null_accs),
        "n_permutations_skipped": n_skipped,
        "null_mean": round(float(np.mean(null_accs)), 4) if null_accs else None,
        "per_state": per_state,
    }

    if subsets:
        per_subset = {}
        for name, smask in subsets.items():
            if not smask.any():
                continue
            y_t = test_y[smask]
            y_p = y_pred_full[smask]
            n_present = int(len(np.unique(y_t)))
            if n_present < 2:
                continue
            bal_acc_s = float(balanced_accuracy_score(y_t, y_p))
            null_accs_s = subset_null_accs[name]
            p_s = permutation_pvalue(
                bal_acc_s, null_accs_s, alternative="greater",
            ) if null_accs_s else float("nan")
            per_subset[name] = {
                "balanced_accuracy": round(bal_acc_s, 4),
                "chance_level_full": round(eff["chance_level"], 4),
                "chance_level_subset_null_mean": (
                    round(float(np.mean(null_accs_s)), 4) if null_accs_s else None
                ),
                "p_perm": round(float(p_s), 4) if null_accs_s else None,
                "n_permutations": len(null_accs_s),
                "n_test_trs_subset": int(smask.sum()),
                "n_classes_present_subset": n_present,
            }
        result["per_subset"] = per_subset

    return result


def _plot_transfer(results, out_path, title, chance):
    if not results:
        return
    layers = sorted(int(k) for k in results.keys())
    accs = [results[l]["balanced_accuracy"] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, accs, marker="o", markersize=3)
    if chance is not None:
        ax.axhline(chance, color="gray", linestyle=":", label=f"chance={chance:.3f}")
        ax.legend()
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_best_lag_layer(results_per_lag, exclude_lags=None):
    """Return ``(best_lag_key, best_layer_idx, best_acc)`` over a D1 result dict.

    Same rule as ``_find_best_lag_layer`` in 08d_transformer_depth.py (the
    08d module name starts with a digit, so it cannot be imported here):
    the (lag, layer) cell with the highest balanced accuracy, skipping the
    lags in ``exclude_lags``. ``results_per_lag`` maps ``"lag_N"`` keys to
    ``{layer_idx: {"balanced_accuracy": ...}}``.
    """
    exclude = set(int(l) for l in exclude_lags) if exclude_lags else set()
    best = (None, None, -np.inf)
    for lag_key, lag_data in results_per_lag.items():
        try:
            lag_val = int(lag_key.split("_")[1])
        except (IndexError, ValueError):
            lag_val = None
        if lag_val is not None and lag_val in exclude:
            continue
        for layer, entry in lag_data.items():
            acc = entry.get("balanced_accuracy", -np.inf)
            if acc is not None and acc > best[2]:
                best = (lag_key, int(layer), float(acc))
    return best


def friends_d1_path(scratch_dir, parc, sub_id, model_key):
    """Path of the participant's within-Friends depth profile (08d D1)."""
    return os.path.join(
        scratch_dir, "output", "08d_transformer_depth", parc, sub_id,
        f"friends_{model_key}", "D1_depth_profile.json",
    )


def resolve_friends_best_lag(scratch_dir, parc, sub_id, model_key):
    """Return ``(lag, info)`` for the participant's best within-Friends lag.

    Raises if the 08d D1 output is missing or has no candidate lag, rather
    than falling back to lag 0: a silent fallback would reproduce the
    lag-0 transfer this option exists to replace.
    """
    path = friends_d1_path(scratch_dir, parc, sub_id, model_key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"08d D1 output not found: {path}. Run 08d for {sub_id} "
            f"{model_key} first, or pass an explicit --lag."
        )
    with open(path) as f:
        d1 = json.load(f)
    lag_key, layer, acc = find_best_lag_layer(
        d1.get("results", {}), exclude_lags=PEAK_LAG_EXCLUDE,
    )
    if lag_key is None:
        raise ValueError(f"No candidate lag in {path} (lag 0 excluded).")
    lag = int(lag_key.split("_")[1])
    return lag, {
        "friends_d1_path": path,
        "friends_best_layer": layer,
        "friends_best_balanced_accuracy": round(acc, 4),
    }


def parse_lag(value):
    """argparse type: ``friends_best`` or a non-negative integer number of TRs."""
    if value == LAG_FRIENDS_BEST:
        return value
    try:
        lag = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--lag must be {LAG_FRIENDS_BEST!r} or a non-negative integer, "
            f"got {value!r}"
        )
    if lag < 0:
        raise argparse.ArgumentTypeError(f"--lag must be >= 0, got {lag}")
    return lag


def output_subdir(stimulus, model_key, lag_arg):
    """Output folder name for one (stimulus, model, lag) cell.

    ``--lag 0`` keeps the original folder name, so its outputs stay where the
    existing figures read them. Other lags get their own folder, so a rerun
    never overwrites the lag-0 results.
    """
    base = f"{stimulus}_{model_key}"
    if lag_arg == LAG_FRIENDS_BEST:
        return f"{base}_lag-best"
    if lag_arg == 0:
        return base
    return f"{base}_lag-{lag_arg}"


# ── Per-layer checkpoints ─────────────────────────────────────────────
# One layer of a 28-layer language model with 1000 permutations takes over an
# hour, and the D3a JSON is written only after the last layer. Each finished
# layer is therefore saved to its own file, so a job that hits its time limit
# can be resubmitted and continue from the next layer. A layer file is reused
# only when its fingerprint (the settings and inputs that determine the
# layer's result) matches the current run; the files are removed once the
# D3a JSON is written.

PARTIAL_LAYER_DIRNAME = "partial_layers"


def layer_fingerprint(fields):
    """Stable hash of the settings and inputs that determine one layer's result."""
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def partial_layer_path(out_dir, stimulus, model_key, layer_idx):
    return os.path.join(
        out_dir, PARTIAL_LAYER_DIRNAME,
        f"D3a_layer-{layer_idx:03d}_{stimulus}_{model_key}.json",
    )


def save_partial_layer(path, fingerprint, layer_idx, res):
    """Write one finished layer atomically (a killed job leaves no half file)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(
            {"fingerprint": fingerprint, "layer": int(layer_idx), "result": res},
            f,
        )
    os.replace(tmp, path)


def load_partial_layer(path, fingerprint, layer_idx):
    """Return the saved result for this layer, or ``None`` if it can't be reused.

    A missing, unreadable or truncated file, a file for another layer, and a
    file whose fingerprint differs from the current run's all return ``None``,
    so the layer is recomputed.
    """
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("fingerprint") != fingerprint
        or payload.get("layer") != int(layer_idx)
        or not isinstance(payload.get("result"), dict)
    ):
        return None
    return payload["result"]


def clear_partial_layers(out_dir):
    """Remove the per-layer files once the full D3a output has been written."""
    part_dir = os.path.join(out_dir, PARTIAL_LAYER_DIRNAME)
    if os.path.isdir(part_dir):
        shutil.rmtree(part_dir)


def run_layers(layer_indices, compute_layer, out_dir, stimulus, model_key,
               fingerprint, resume=True):
    """Compute every layer, reusing saved layers and saving each new one.

    ``compute_layer(layer_idx)`` returns the layer's result dict, or ``None``
    to skip the layer (no features, degenerate fit). Skipped layers are not
    saved, so they are attempted again on resume.
    """
    results = {}
    for layer_idx in layer_indices:
        path = partial_layer_path(out_dir, stimulus, model_key, layer_idx)
        res = load_partial_layer(path, fingerprint, layer_idx) if resume else None
        if res is not None:
            logger.info("  Layer %d: reused saved result (%s)", layer_idx, path)
        else:
            res = compute_layer(layer_idx)
            if res is None:
                continue
            # Same JSON round trip as a reused layer, so fresh and resumed
            # layers reach the output in the same form (e.g. string keys).
            res = json.loads(json.dumps(res))
            save_partial_layer(path, fingerprint, layer_idx, res)
        results[layer_idx] = res
        logger.info(
            "  Layer %d: bal_acc=%.4f (chance=%.4f, eff=%.3f, p=%.4f)",
            layer_idx, res["balanced_accuracy"], res["chance_level"],
            res["normalized_effect_size"], res["p_perm"],
        )
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-stimulus aggregate transfer (08e).",
    )
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--stimulus", required=True,
                   choices=list(CROSS_STIM_STAGE_MAP.keys()))
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--vt", default=None)
    p.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS_DEFAULT)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if checkpoint output already exists.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore per-layer results saved by an earlier, "
                        "interrupted run of this cell and recompute every layer.")
    p.add_argument("--per_subset", action="store_true",
                   help="Also emit per-subset breakdown (currently honored only "
                        "for --stimulus movie10, which splits by film: wolf / "
                        "figures / bourne / life). Pooled output is unchanged.")
    p.add_argument("--lag", type=parse_lag, default=LAG_FRIENDS_BEST,
                   help="Brain-to-feature lag in TRs for training and test "
                        "features: 'friends_best' (default) uses the "
                        "participant's best within-Friends lag for this model "
                        "from 08d D1 (lag 0 excluded); an integer sets it. "
                        "Outputs go to <stimulus>_<model>_lag-best (or "
                        "_lag-N); --lag 0 writes to the original folder.")
    return p.parse_args()


# Per-subset partition definitions. The values are ordered lists of (label,
# compiled regex) pairs. Each regex is anchored (^...$) and matches the
# canonical run-id form ``<film><integer>(_run-<integer>)?`` so an
# unanticipated run-id like ``figures_remake01`` does NOT silently merge
# into the ``figures`` subset; instead it triggers the orphan assertion
# inside main(). To extend to other stimuli, add an entry keyed by the
# stimulus name (must also appear in CROSS_STIM_STAGE_MAP).
PER_SUBSET_DEFS = {
    "movie10": [
        ("wolf",    _re.compile(r"^wolf\d+(?:_run-\d+)?$")),
        ("figures", _re.compile(r"^figures\d+(?:_run-\d+)?$")),
        ("bourne",  _re.compile(r"^bourne\d+(?:_run-\d+)?$")),
        ("life",    _re.compile(r"^life\d+(?:_run-\d+)?$")),
    ],
}


def subset_partition_for(stimulus, run_ids):
    """Partition ``run_ids`` by the canonical subset patterns for ``stimulus``.

    Returns ``(labels, label_to_run_ids)`` where ``labels`` is the ordered
    list of subset labels and ``label_to_run_ids`` is a dict mapping each
    label to the run_ids in that subset.

    Raises
    ------
    ValueError
        If a run_id matches more than one pattern (partition is not exclusive).
        If a run_id matches no pattern (orphan run_id; partition is incomplete).
        Both failure modes name the offending run_ids so the caller can
        update PER_SUBSET_DEFS.
    """
    if stimulus not in PER_SUBSET_DEFS:
        raise ValueError(
            f"No per-subset partition defined for stimulus={stimulus!r}. "
            f"Available: {sorted(PER_SUBSET_DEFS.keys())}."
        )
    pairs = PER_SUBSET_DEFS[stimulus]
    labels = [lbl for lbl, _ in pairs]
    assigned = {lbl: [] for lbl in labels}
    for r in run_ids:
        r_str = str(r)
        matches = [lbl for lbl, pat in pairs if pat.match(r_str)]
        if len(matches) > 1:
            raise ValueError(
                f"Run id {r_str!r} matches multiple subsets {matches} - "
                f"PER_SUBSET_DEFS[{stimulus!r}] patterns are not mutually exclusive."
            )
        if not matches:
            raise ValueError(
                f"Run id {r_str!r} matches no subset - orphan. Update "
                f"PER_SUBSET_DEFS[{stimulus!r}] to cover this run name."
            )
        assigned[matches[0]].append(r_str)
    return labels, assigned


def main():
    args = parse_args()
    sub_id = args.sub_id
    parc = normalize_parcellation_name(args.parcellation)
    stimulus = args.stimulus
    model_key = args.model

    # Modality guard - MUST be called for both friends and test stimulus (they
    # share the same modality check since both feed the same model).
    validate_stimulus_model("friends", model_key)
    validate_stimulus_model(stimulus, model_key)

    logger.info("=" * 60)
    logger.info("08e - Cross-stimulus aggregate (D3a)")
    logger.info("Sub=%s test_stimulus=%s model=%s", sub_id, stimulus, model_key)
    logger.info("=" * 60)

    if args.lag == LAG_FRIENDS_BEST:
        lag, lag_info = resolve_friends_best_lag(
            SCRATCH_DIR, parc, sub_id, model_key,
        )
        lag_source = LAG_FRIENDS_BEST
    else:
        lag, lag_info, lag_source = args.lag, {}, "explicit"
    logger.info("Lag: %d TRs (source=%s)", lag, lag_source)

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "08e_transformer_cross_stim_aggregate", parc,
        sub_id, output_subdir(stimulus, model_key, args.lag),
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Checkpoint (dual-file when --per_subset is honored) ───────────
    # The pooled output file always counts. When --per_subset is set AND the
    # stimulus has a defined partition (PER_SUBSET_DEFS), the per-subset file
    # must also exist; otherwise the cell is re-run even if the pooled file
    # is present. Without this guard, a previously-completed pooled cell would
    # silently skip the per_subset re-run.
    per_subset_active = args.per_subset and stimulus in PER_SUBSET_DEFS
    checkpoint_files = [f"D3a_transfer_{stimulus}_{model_key}.json"]
    if per_subset_active:
        checkpoint_files.append(f"D3a_per_subset_{stimulus}_{model_key}.json")
    if check_checkpoint(out_dir, checkpoint_files, "D3a", force=args.force):
        return

    # ── States (pre-alignment) ────────────────────────────────────────
    friends_decoded = _load_friends_decoded(sub_id, parc)
    test_decoded = _load_cross_stim_decoded(sub_id, parc, stimulus)

    friends_candidate_ids = sorted(friends_decoded.keys())
    test_candidate_ids = sorted(test_decoded.keys())

    # pp_04_decoded contains both lppFR and lppEN runs in one pickle
    # (one HMM decoded against both Petit Prince stimuli). Filter to the
    # requested stimulus's language so we only look up features that exist
    # under the matching 08c bundle.
    if stimulus == "petitprince_fr":
        test_candidate_ids = [r for r in test_candidate_ids if r.startswith("lppFR_")]
    elif stimulus == "petitprince_en":
        test_candidate_ids = [r for r in test_candidate_ids if r.startswith("lppEN_")]
    friends_n_trs_init = {r: len(friends_decoded[r]) for r in friends_candidate_ids}
    test_n_trs_init = {r: len(test_decoded[r]) for r in test_candidate_ids}

    # ── Features + PCA (streamed per layer) ───────────────────────────
    # First pass fits the per-layer PCA on the Friends training split;
    # the second call reuses the fit PCA to project the test stimulus.
    # `stream_pca_features` also handles TR drift alignment and returns
    # the effective per-run lengths so we can truncate decoded_states.
    #
    # Note on train split usage: PCA is fit on Friends `splits["train"]`
    # ONLY (no leakage), but the downstream Ridge classifier is trained on
    # ALL Friends runs (train + val + test). This is intentional and safe
    # for D3a: the held-out set for the cross-stimulus transfer claim is
    # the test STIMULUS (movie10 / harrypotter / pp_*), not a held-out
    # Friends split. Using all Friends runs for Ridge fitting gives the
    # decoder maximum training data; val/test Friends features just have
    # slightly higher residual PCA reconstruction error than train runs.
    splits = load_training_split(sub_id, parc, SCRATCH_DIR)

    logger.info(
        "Streaming Friends features + fitting per-layer PCA "
        "(variance threshold = %.0f%%)",
        PCA_VARIANCE_THRESHOLD * 100,
    )
    (
        friends_features,
        pca_info,
        pca_models,
        friends_eff,
        friends_dropped,
    ) = stream_pca_features(
        "friends", model_key, friends_candidate_ids, friends_n_trs_init,
        SCRATCH_DIR,
        train_run_ids=splits["train"],
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )

    logger.info(
        "Streaming %s features + projecting through Friends PCA", stimulus,
    )
    # movie10 has two viewings of figures/life per subject but a single
    # 08c feature file per clip - pass the feature_key stripper so both
    # decoded_states entries share the same feature file lookup.
    test_feature_key_fn = (
        (lambda r: feature_key_for_cross_stim_run_id(r, stimulus))
        if stimulus == "movie10" else None
    )
    # Petit Prince audiobook runs have ~9-10s of post-stimulus baseline (BOLD
    # scan continues after the audio ends). 08c's n_trs is derived from audio
    # duration + pre-stimulus silence and does NOT include the post-stimulus
    # tail, so BOLD-derived states are 6-7 TRs longer than features per run
    # (uniform across all 5 subjects, both LLaMA and W2V-BERT - verified
    # 2026-05-25). The 3-TR default drift cap is too strict for this
    # structural offset; raise to 8 TRs for petitprince_* (covers observed
    # 7-TR max + 1-TR safety margin). The trailing TRs are post-stimulus
    # baseline with no audio content to associate with brain states, so
    # truncating states to feature length is appropriate.
    test_max_drift = 8 if stimulus.startswith("petitprince_") else 3
    (
        test_features,
        _,
        _,
        test_eff,
        test_dropped,
    ) = stream_pca_features(
        stimulus, model_key, test_candidate_ids, test_n_trs_init, SCRATCH_DIR,
        pca_models=pca_models, pca_info=pca_info,
        feature_key_fn=test_feature_key_fn,
        max_tr_drift=test_max_drift,
    )

    # Align decoded_states to effective feature lengths.
    for rid in list(friends_decoded.keys()):
        if rid not in friends_eff:
            friends_decoded.pop(rid)
            continue
        eff = friends_eff[rid]
        if len(friends_decoded[rid]) != eff:
            friends_decoded[rid] = np.asarray(friends_decoded[rid])[:eff]
    for rid in list(test_decoded.keys()):
        if rid not in test_eff:
            test_decoded.pop(rid)
            continue
        eff = test_eff[rid]
        if len(test_decoded[rid]) != eff:
            test_decoded[rid] = np.asarray(test_decoded[rid])[:eff]

    if not friends_decoded or not test_decoded:
        logger.error("No runs survived drift alignment - exiting")
        sys.exit(1)

    friends_run_ids = sorted(friends_decoded.keys())
    test_run_ids = sorted(test_decoded.keys())
    # Only Friends boundaries are needed: the null permutes Friends training
    # labels (circular shift within Friends runs). Test labels stay fixed.
    friends_boundaries = build_run_boundaries(friends_run_ids, friends_decoded)

    friends_states_cat = np.concatenate(
        [friends_decoded[r] for r in friends_run_ids]
    )
    test_states_cat = np.concatenate([test_decoded[r] for r in test_run_ids])

    logger.info(
        "Post-alignment: Friends %d runs / %d TRs; %s %d runs / %d TRs",
        len(friends_run_ids), len(friends_states_cat),
        stimulus, len(test_run_ids), len(test_states_cat),
    )

    with open(os.path.join(out_dir, "pca_info.json"), "w") as f:
        json.dump({str(k): v for k, v in pca_info.items()}, f, indent=2)

    # ── Eligibility (Friends-defined; shared state space) ─────────────
    eligibility = load_content_eligibility(sub_id, parc, SCRATCH_DIR, vt=args.vt)
    intersect_ids, fo_stats = _compute_intersection(
        friends_states_cat, test_states_cat, eligibility["content_eligible"],
    )
    logger.info(
        "Eligibility source=%s | %d content_eligible → %d in intersection (FO ≥ %.0f%% both)",
        eligibility["eligibility_source"],
        len(eligibility["content_eligible"]),
        len(intersect_ids),
        INTERSECTION_MIN_FO * 100,
    )
    if len(intersect_ids) < 2:
        logger.error("Intersection contains < 2 states - cannot decode. Exiting.")
        sys.exit(1)

    intersect_set = set(intersect_ids)
    friends_mask = np.array(
        [int(s) in intersect_set for s in friends_states_cat], dtype=bool,
    )
    test_mask = np.array(
        [int(s) in intersect_set for s in test_states_cat], dtype=bool,
    )

    # ── Per-subset masks (movie10 per-film, etc.) ─────────────────────
    # Partition the test run_ids using regex patterns from PER_SUBSET_DEFS
    # (helper `subset_partition_for` raises on orphans / overlaps). Then
    # propagate the per-run assignment to a TR-level mask aligned to
    # test_states_cat, AND'd with test_mask so the downstream classifier
    # eval only sees intersection-eligible TRs within each subset.
    per_subset_masks = None
    subset_meta = None
    subset_labels = None
    if per_subset_active:
        subset_labels, runs_by_subset = subset_partition_for(stimulus, test_run_ids)
        test_run_lens = [len(test_decoded[r]) for r in test_run_ids]
        test_tr_run_id = np.concatenate([
            np.full(L, r, dtype=object) for r, L in zip(test_run_ids, test_run_lens)
        ])
        per_subset_masks = {}
        subset_meta = {}
        for film_key in subset_labels:
            run_set = set(runs_by_subset[film_key])
            m_film = np.array(
                [str(r) in run_set for r in test_tr_run_id], dtype=bool,
            )
            mask = m_film & test_mask
            per_subset_masks[film_key] = mask
            subset_meta[film_key] = {
                "n_runs": int(len(run_set)),
                "n_test_trs_film": int(m_film.sum()),
                "n_test_trs_subset": int(mask.sum()),
            }
        logger.info(
            "Per-subset partition (%s): %s",
            stimulus,
            ", ".join(
                f"{k}={subset_meta[k]['n_test_trs_subset']}TR/"
                f"{subset_meta[k]['n_runs']}runs"
                for k in subset_labels
            ),
        )

    # ── Null sequences for Friends TRAINING labels ────────────────────
    # The principled null for a cross-stimulus transfer claim permutes the
    # Friends training labels (not the test-stimulus labels) - we want to
    # know whether the learned Friends feature→state mapping carries
    # test-stimulus-relevant structure beyond chance. Circular-shift within
    # Friends runs preserves within-run autocorrelation. The classifier is
    # refit on these shuffled Friends labels inside `_fit_eval_one_layer`.
    #
    # The null is generated directly in the **intersection subspace** via
    # `precompute_eligible_null_state_sequences`, so shifted labels can
    # only come from the 29 intersection classes. The prior pattern
    # (shift the full 47-class Friends sequence, then mask by original
    # intersection positions) leaked non-intersection classes into the
    # null training set and depressed null_mean below true chance - see
    # 2026-04-10 null-leakage plan.
    #
    # Base seed matches the 08d sequence: D1 main=10_000, D1 confound=40_000,
    # D2=50_000, D3a=60_000.
    logger.info("Precomputing %d null state sequences (Friends intersection)...",
                args.n_permutations)
    null_friends_seqs = precompute_eligible_null_state_sequences(
        friends_states_cat.astype(int), friends_boundaries, friends_mask,
        args.n_permutations, rng_seed=NULL_SEED_D3A,
    )

    n_layers = MODEL_REGISTRY[model_key]["n_layers"]
    n_classes = len(intersect_ids)

    fingerprint = layer_fingerprint({
        "sub_id": sub_id, "parcellation": parc, "stimulus": stimulus,
        "model": model_key, "vt": args.vt, "lag": lag,
        "n_permutations": args.n_permutations, "null_seed": NULL_SEED_D3A,
        "eligibility_source": eligibility["eligibility_source"],
        "intersection_states": intersect_ids, "n_classes": n_classes,
        "friends_run_ids": friends_run_ids, "test_run_ids": test_run_ids,
        "n_friends_trs": int(friends_mask.sum()),
        "n_test_trs": int(test_mask.sum()),
        "per_subset": bool(per_subset_active), "subsets": subset_labels,
    })

    def compute_layer(layer_idx):
        friends_runs = friends_features.get(layer_idx, {})
        test_runs = test_features.get(layer_idx, {})
        if not friends_runs or not test_runs:
            return None
        try:
            # The same lag for training (Friends) and test features, so the
            # decoder is applied at the lag it was fitted at. Zero-padded
            # first `lag` TRs of each run stay in, as in 08d.
            friends_X = build_layer_feature_matrix(
                friends_runs, friends_run_ids, friends_decoded, lag=lag,
            )
            test_X = build_layer_feature_matrix(
                test_runs, test_run_ids, test_decoded, lag=lag,
            )
        except ValueError as exc:
            logger.debug("Layer %d: %s", layer_idx, exc)
            return None
        return _fit_eval_one_layer(
            layer_idx,
            friends_X, friends_states_cat.astype(int), friends_mask,
            test_X, test_states_cat.astype(int), test_mask,
            intersect_set, null_friends_seqs, n_classes,
            subsets=per_subset_masks,
        )

    results = run_layers(
        range(n_layers), compute_layer, out_dir, stimulus, model_key,
        fingerprint, resume=not args.fresh,
    )

    # Global BH-FDR across layers (mirrors 08d D1 which corrects across the
    # full layers × lags grid; 08e has no lag grid yet, so the correction is
    # over layers only). Attach `p_fdr` next to `p_perm` in each layer's dict.
    if results:
        layer_order = sorted(results.keys())
        raw_p = np.array([results[l]["p_perm"] for l in layer_order])
        fdr_p = benjamini_hochberg(raw_p)
        for l, q in zip(layer_order, fdr_p):
            results[l]["p_fdr"] = round(float(q), 4)

    out_payload = {
        "sub_id": sub_id,
        "test_stimulus": stimulus,
        "model": model_key,
        "lag": lag,
        "lag_source": lag_source,
        **lag_info,
        "eligibility_source": eligibility["eligibility_source"],
        "intersection_states": intersect_ids,
        "intersection_fo": {str(k): v for k, v in fo_stats.items()},
        "n_classes": n_classes,
        "n_friends_trs": int(friends_mask.sum()),
        "n_test_trs": int(test_mask.sum()),
        "per_layer": {str(k): v for k, v in results.items()},
    }
    out_path = os.path.join(
        out_dir, f"D3a_transfer_{stimulus}_{model_key}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    logger.info("Saved %s", out_path)

    chance = 1.0 / n_classes if n_classes else None
    _plot_transfer(
        results,
        os.path.join(out_dir, f"D3a_transfer_{stimulus}_{model_key}.png"),
        f"D3a: Friends → {stimulus} ({model_key})",
        chance,
    )

    # ── Per-subset emission (separate file; pooled file above is unchanged) ──
    # BH-FDR within each subset over layers, mirroring pooled-axis correction.
    # No cross-subset BH (subset axis is small and biologically structured;
    # SI cross-subset robustness check can be computed downstream if needed).
    if per_subset_active and results:
        subset_results = {}
        for film_key in subset_labels:
            per_layer_subset = {}
            for layer_idx, layer_res in results.items():
                ps = layer_res.get("per_subset", {})
                if film_key in ps:
                    per_layer_subset[layer_idx] = ps[film_key]
            if per_layer_subset:
                layer_order = sorted(per_layer_subset.keys())
                raw_p = np.array([
                    per_layer_subset[l]["p_perm"] if per_layer_subset[l]["p_perm"] is not None else 1.0
                    for l in layer_order
                ])
                fdr_p = benjamini_hochberg(raw_p)
                for l, q in zip(layer_order, fdr_p):
                    per_layer_subset[l]["p_fdr"] = round(float(q), 4)
            subset_results[film_key] = per_layer_subset

        per_subset_payload = {
            "sub_id": sub_id,
            "test_stimulus": stimulus,
            "model": model_key,
            "lag": lag,
            "lag_source": lag_source,
            "subset_axis": "film" if stimulus == "movie10" else stimulus,
            "subsets": subset_labels,
            "eligibility_source": eligibility["eligibility_source"],
            "intersection_states": intersect_ids,
            "n_classes_intersection": n_classes,
            "chance_level_full": round(1.0 / n_classes, 4) if n_classes else None,
            "subset_metadata": subset_meta,
            "per_subset": {
                k: {str(l): v for l, v in subset_results[k].items()}
                for k in subset_results
            },
        }
        per_subset_path = os.path.join(
            out_dir, f"D3a_per_subset_{stimulus}_{model_key}.json",
        )
        with open(per_subset_path, "w") as f:
            json.dump(per_subset_payload, f, indent=2)
        logger.info("Saved %s", per_subset_path)

    # Every output is on disk, so the per-layer files are no longer needed.
    clear_partial_layers(out_dir)

    logger.info("=" * 60)
    logger.info("08e_transformer_cross_stim_aggregate complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
