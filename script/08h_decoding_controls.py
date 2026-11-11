#!/usr/bin/env python3
"""
08h_decoding_controls.py - fold, mask and timing controls for the 08d / 08e
transformer-state decoders, at each participant's best within-Friends lag.

08d D1 decodes HMM states from transformer features with leave-one-run-out
(LORO) folds over all content-eligible TRs; 08e trains on Friends and tests
on another stimulus. This script reruns the same decoders (same features,
PCA, labels, RidgeClassifier, circular-shift null and seeds) with one change
at a time, at the lag where the participant's Friends D1 peaked (lag 0
excluded), over every layer unless --layers selects some:

Friends (``--part friends``, one condition per invocation):

    gate      LORO, all content-eligible TRs, observed accuracy only. Must
              equal the published 08d D1 balanced accuracy at that lag for
              every layer, and the timing-only accuracy must equal the
              published D1 confound baseline.
    loeo      Leave-one-episode-out folds: the a/b (and c/d) runs of one
              episode are held out together, so no part of the test episode
              is in training. Null unchanged (circular shift within run).
    opening   LORO with the first 33 + lag brain TRs of each a-run removed
              (the fixed 33-TR window of 05e_temporal_trend_a2, plus the
              lag so that no feature row comes from inside it either).
              05e uses this window as an approximate lower bound on the
              theme position: the cold open comes first, so in most a-runs
              the theme song starts after TR 33 and is not removed here.
    word_trs  Text model only. LORO on TRs whose lagged feature row holds at
              least one word. 08c gives TRs with no transcript tokens a zero
              vector, which after within-run z-scoring is a constant row, so
              the decoder could otherwise read speech presence.
    timing    Features plus the six D1 confound timing regressors vs the
              timing regressors alone. Statistic: delta = acc(features +
              timing) - acc(timing). Null: the same circular-shifted label
              sequences are fit by both decoders and delta recomputed.
              timing_only.p_perm uses this D1-seed null, not the null of
              the published confound baseline.

Per-layer partials make a cell resumable. Each one carries a fingerprint
of the inputs, settings and code that produced it, and is reused only when
the current run's fingerprint matches. p_fdr is Benjamini-Hochberg across
the layers at the best lag (08d's p_fdr is across lags x layers), and is
left out unless every layer is present.

Transfer (``--part transfer``): 08e D3a at the Friends best lag for chosen
layers, (a) unchanged, observed only, as a gate against the published
D3a_transfer file, and (b) restricted to word-bearing TRs in both the Friends
training set and the test stimulus, with the 08e null (Friends training
labels circular-shifted within run, classifier refit, scored on the real test
labels).

Inputs are read from ``$SCRATCH_DIR/output``; nothing is written there.
The 08d PCA cache is mirrored into ``--out_root`` as a tree of symlinks so a
cache hit reads the published PCA and a miss writes only under ``--out_root``.

Outputs:
    {out_root}/08h_decoding_controls/{parc}/{sub}/friends_{model}/{condition}.json
    {out_root}/08h_decoding_controls/{parc}/{sub}/{stimulus}_{model}/transfer.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d08 = _load_module("d08", "08d_transformer_depth.py")
e08 = _load_module("e08", "08e_transformer_cross_stim_aggregate.py")

from utils.common import (  # noqa: E402
    feature_key_for_cross_stim_run_id, load_training_split,
    normalize_parcellation_name,
)
from utils.stats import benjamini_hochberg, permutation_pvalue  # noqa: E402
from utils.transformer_analysis import (  # noqa: E402
    PEAK_LAG_EXCLUDE, batch_loro_ridge_classify, build_layer_feature_matrix,
    build_loro_folds, build_run_boundaries, compute_effect_size,
    d1_eligibility_problems, find_best_lag_layer, friends_d1_path,
    load_content_eligibility, load_or_fit_pca_cache,
    precompute_eligible_null_state_sequences, stream_pca_features,
    transformer_feature_path,
)
from utils.transformer_io import MODEL_REGISTRY  # noqa: E402

SCRATCH_DIR = os.environ["SCRATCH_DIR"]
logger = logging.getLogger("08h_decoding_controls")

OPENING_MASK_TR = 33
TEXT_MODELS = {"llama-3.2-3b"}
FRIENDS_CONDITIONS = ("gate", "loeo", "opening", "word_trs", "timing")
D1_SEED_BASE = 10_000          # 08d D1 main
D1_SEED_LAG_STRIDE = 100003    # 08d: perm_seed_base + lag * 100003
ACC_TOL = 5e-5                 # published accuracies are rounded to 4 dp

#: Source files whose contents enter each per-layer partial's fingerprint:
#: this script, the 08d helpers it calls, and the shared decoding utilities.
#: Whole files are hashed, as in 08e, so any edit to them recomputes every
#: layer instead of reusing a partial computed by other code.
_UTILS_DIR = SCRIPT_DIR / "utils"
CODE_IDENTITY_FILES = (
    Path(__file__).resolve(),
    SCRIPT_DIR / "08d_transformer_depth.py",
    _UTILS_DIR / "transformer_analysis.py",
    _UTILS_DIR / "stats.py",
    _UTILS_DIR / "transformer_io.py",
)
#: Hash taken at import, i.e. of the code this process runs.
CODE_SHA256 = e08.code_identity(CODE_IDENTITY_FILES)


# ---------------------------------------------------------------------------
# Folds and masks
# ---------------------------------------------------------------------------


def episode_key(run_id):
    """``s01e02a`` -> ``s01e02``; the runs of one episode share this key."""
    m = re.search(r"s\d+e\d+", run_id)
    if m is None:
        raise ValueError(f"cannot parse episode from run id {run_id!r}")
    return m.group(0)


def build_episode_folds(run_ids, run_boundaries, eligible_mask):
    """Leave-one-episode-out folds in masked index space (as build_loro_folds)."""
    elig = np.asarray(eligible_mask, dtype=bool)
    lookup = -np.ones(len(elig), dtype=np.int64)
    masked_idx = np.where(elig)[0]
    lookup[masked_idx] = np.arange(len(masked_idx))
    groups = {}
    for rid, (start, end) in zip(run_ids, run_boundaries):
        groups.setdefault(episode_key(rid), []).append((start, end))
    folds, keys = [], []
    for key in sorted(groups):
        in_ep = np.zeros(len(elig), dtype=bool)
        for start, end in groups[key]:
            in_ep[start:end] = True
        test_g = np.where(elig & in_ep)[0]
        train_g = np.where(elig & ~in_ep)[0]
        if len(test_g) == 0 or len(train_g) == 0:
            continue
        folds.append((lookup[train_g], lookup[test_g]))
        keys.append(key)
    return folds, keys


def opening_mask(run_ids, run_boundaries, n_total, lag, mask_tr=OPENING_MASK_TR):
    """True on TRs to KEEP: drops brain TRs t < mask_tr + lag of each a-run."""
    keep = np.ones(n_total, dtype=bool)
    cut = mask_tr + lag
    for rid, (start, end) in zip(run_ids, run_boundaries):
        if rid.endswith("a"):
            keep[start:min(start + cut, end)] = False
    return keep


def lag_shift_bool(x, lag):
    """Shift a per-run bool vector like build_layer_feature_matrix shifts features."""
    if lag <= 0:
        return x.copy()
    out = np.zeros_like(x)
    if lag < len(x):
        out[lag:] = x[:len(x) - lag]
    return out


def word_tr_mask(stimulus, model_key, run_ids, decoded_states, lag,
                 feature_key_fn=None):
    """True where the lagged feature row of a run holds at least one word.

    Read from the layer-0 raw 08c file (pre z-score): 08c writes a zero row
    for a TR with no transcript tokens, identically in every layer.
    Truncated to the aligned run length as stream_pca_features does.
    """
    parts, frac = [], {}
    for rid in run_ids:
        stem = feature_key_fn(rid) if feature_key_fn else rid
        raw = np.load(
            transformer_feature_path(SCRATCH_DIR, stimulus, model_key, 0, stem),
            mmap_mode="r",
        )
        n = len(decoded_states[rid])
        has_word = np.asarray(np.abs(raw[:n]).sum(axis=1) > 0)
        if len(has_word) != n:
            raise RuntimeError(f"{stimulus} {rid}: {len(has_word)} feature rows < {n} TRs")
        frac[rid] = float(has_word.mean())
        parts.append(lag_shift_bool(has_word, lag))
    return np.concatenate(parts), frac


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def mirror_tree(src, dst):
    """Recreate directory ``src`` under ``dst`` with a symlink per file."""
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if f == ".lock":
                continue
            link = os.path.join(dst, rel, f)
            try:
                os.symlink(os.path.join(root, f), link)
            except FileExistsError:
                # Another job for the same cache got there first, or a PCA
                # refit already replaced the link with a real file.
                pass


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp{platform.node()}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def align_decoded(decoded, effective_n_trs):
    for rid in list(decoded.keys()):
        if rid not in effective_n_trs:
            decoded.pop(rid)
            continue
        eff = effective_n_trs[rid]
        if len(decoded[rid]) != eff:
            decoded[rid] = np.asarray(decoded[rid])[:eff]
    return decoded


def environment():
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__}


def summarise(batch, y_elig):
    obs = batch["observed"]
    null = batch["null_balanced_accuracies"]
    n_classes = int(len(np.unique(y_elig)))
    eff = compute_effect_size(obs["balanced_accuracy"], n_classes)
    out = {
        "balanced_accuracy": round(obs["balanced_accuracy"], 4),
        "n_classes": n_classes,
        "chance_level": round(eff["chance_level"], 4),
        "normalized_effect_size": round(eff["normalized_effect_size"], 4),
        "n_permutations": len(null),
    }
    if null:
        out["p_perm"] = round(float(permutation_pvalue(
            obs["balanced_accuracy"], null, alternative="greater")), 4)
        out["null_mean"] = round(float(np.mean(null)), 4)
        out["null_q95"] = round(float(np.quantile(null, 0.95)), 4)
    return out


# ---------------------------------------------------------------------------
# Friends conditions
# ---------------------------------------------------------------------------


def parse_layers_spec(spec):
    """Check the syntax of a --layers value; returns its items, or None.

    Items are layer numbers or "best". Run in main(), before any data is
    loaded, so a typo fails at once rather than after the PCA load.
    """
    if spec is None:
        return None
    items = [item.strip() for item in spec.split(",")]
    for item in items:
        if item != "best" and not item.isdigit():
            raise ValueError(f"--layers item {item!r} is not a layer number or 'best'")
    return items


def select_layers(spec, n_layers, best_layer):
    """Layers to evaluate: all of them for spec None, else the comma list in spec.

    Items are layer numbers or "best" (the published best layer at the best lag).
    """
    items = parse_layers_spec(spec)
    if items is None:
        return list(range(n_layers))
    out = set()
    for item in items:
        li = best_layer if item == "best" else int(item)
        if not 0 <= li < n_layers:
            raise ValueError(f"layer {li} outside 0..{n_layers - 1}")
        out.add(li)
    return sorted(out)


def n_folds_without_every_class(y, folds):
    """Folds whose training labels lack a class present in ``y``.

    batch_loro_ridge_classify skips such a fold without saying so: its TRs
    get no prediction, and the fast-path null then averages recall over
    every class while the observed (sklearn) accuracy averages over the
    classes left in the evaluated TRs, which biases p. Counted so that
    run_friends can refuse the cell.
    """
    classes = np.unique(y)
    return sum(1 for train, _test in folds
               if len(np.unique(y[train])) < len(classes))


def load_matching_partials(partial_dir, cond, n_layers, fingerprints):
    """Per-layer partials on disk whose fingerprint matches this run's.

    A partial written by other code, another lag, permutation count, mask or
    input (or one with no fingerprint, as written before fingerprints were
    added) is ignored, so its layer is recomputed and the file overwritten.
    """
    results, stale = {}, []
    for li in range(n_layers):
        p = os.path.join(partial_dir, f"{cond}_layer{li:02d}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            res = json.load(f)
        if res.get("fingerprint") == fingerprints[li]:
            results[li] = res
        else:
            stale.append(li)
    return results, stale


def _friends_layer(layer_idx, layer_runs, run_ids, decoded_states, lag, mask,
                   y_full, null_y, folds, timing_X=None, timing_null=None):
    """One layer of one Friends condition. Returns (layer_idx, dict | None)."""
    X = build_layer_feature_matrix(layer_runs, run_ids, decoded_states, lag=lag)
    y = y_full[mask]
    if timing_X is not None:
        X = np.hstack([X, timing_X])
    batch = batch_loro_ridge_classify(X[mask], y, null_y, folds)
    if batch is None:
        return layer_idx, None
    res = summarise(batch, y)
    res["n_features"] = int(X.shape[1])
    if timing_X is not None:
        null_ft = np.asarray(batch["null_balanced_accuracies"])
        null_t = np.asarray(timing_null["null"])
        if len(null_ft) != len(null_t) or len(null_ft) != null_y.shape[0]:
            raise RuntimeError(f"layer {layer_idx}: null lengths differ "
                               f"({len(null_ft)}, {len(null_t)}, {null_y.shape[0]})")
        delta = batch["observed"]["balanced_accuracy"] - timing_null["observed"]
        null_delta = null_ft - null_t
        res["delta_vs_timing"] = round(float(delta), 4)
        res["p_delta"] = round(float(permutation_pvalue(
            delta, null_delta.tolist(), alternative="greater")), 4)
        res["null_delta_mean"] = round(float(null_delta.mean()), 4)
        res["null_delta_q95"] = round(float(np.quantile(null_delta, 0.95)), 4)
    return layer_idx, res


def run_friends(args, parc, out_dir):
    sub_id, model_key, cond = args.sub_id, args.model, args.condition
    if cond == "word_trs" and model_key not in TEXT_MODELS:
        raise SystemExit(f"word_trs applies to text models only, not {model_key}")

    pub_dir = os.path.join(SCRATCH_DIR, "output", "08d_transformer_depth", parc,
                           sub_id, f"friends_{model_key}")
    d1_path = friends_d1_path(SCRATCH_DIR, parc, sub_id, model_key)
    with open(d1_path) as f:
        d1 = json.load(f)
    lag_key, best_layer, best_acc = find_best_lag_layer(
        d1["results"], exclude_lags=PEAK_LAG_EXCLUDE)
    lag = int(lag_key.split("_")[1])
    published_at_lag = {int(k): v for k, v in d1["results"][lag_key].items()}
    logger.info("%s %s: best lag %d (layer %d, acc %.4f)", sub_id, model_key,
                lag, best_layer, best_acc)

    eligibility = load_content_eligibility(sub_id, parc, SCRATCH_DIR, vt=args.vt)
    problems = d1_eligibility_problems(d1_path, eligibility)
    if problems:
        raise RuntimeError(f"{sub_id}: published D1 eligibility differs: {problems}")

    decoded_states, _ = d08._load_decoded_states_friends(sub_id, parc)
    candidate = sorted(decoded_states.keys())
    n_init = {r: len(decoded_states[r]) for r in candidate}
    splits = load_training_split(sub_id, parc, SCRATCH_DIR)
    pub_cache = d08._pca_cache_dir(SCRATCH_DIR, parc, sub_id, model_key, "friends")
    cache_dir = os.path.join(args.out_root, "pca_cache", parc, sub_id, model_key, "friends")
    mirror_tree(pub_cache, cache_dir)
    split_hash, split_rel = d08._pca_split_hash_and_path(sub_id, parc, SCRATCH_DIR)
    features_by_layer, pca_info, _m, eff_n, dropped = load_or_fit_pca_cache(
        "friends", model_key, candidate, n_init, SCRATCH_DIR,
        cache_dir=cache_dir, split_hash=split_hash, sub_id=sub_id,
        parcellation=parc, split_path_rel=split_rel,
        train_run_ids=splits["train"],
        variance_threshold=d08.PCA_VARIANCE_THRESHOLD,
    )
    decoded_states = align_decoded(decoded_states, eff_n)
    run_ids = sorted(decoded_states.keys())
    bounds = build_run_boundaries(run_ids, decoded_states)
    all_states = np.concatenate([decoded_states[r] for r in run_ids])
    y_full = all_states.astype(np.int8)
    elig_set = set(int(s) for s in eligibility["content_eligible"])
    base = np.array([int(s) in elig_set for s in all_states], dtype=bool)
    n_layers = max(features_by_layer.keys()) + 1

    meta = {"best_lag": lag, "best_layer_published": best_layer,
            "best_acc_published": best_acc, "n_runs": len(run_ids),
            "n_dropped_runs": len(dropped), "n_trs_total": int(len(all_states)),
            "n_trs_content_eligible": int(base.sum()),
            "n_content_eligible_states": len(elig_set)}

    n_perm = 0 if cond == "gate" else args.n_permutations
    mask = base
    timing_X = timing_null = None
    if cond == "opening":
        mask = base & opening_mask(run_ids, bounds, len(all_states), lag)
        meta["opening_cutoff_trs"] = OPENING_MASK_TR + lag
    elif cond == "word_trs":
        words, frac = word_tr_mask("friends", model_key, run_ids, decoded_states, lag)
        mask = base & words
        meta["word_tr_fraction_all"] = round(float(words.mean()), 4)
        meta["word_tr_fraction_content_eligible"] = round(float(words[base].mean()), 4)
    if cond == "loeo":
        folds, ep_keys = build_episode_folds(run_ids, bounds, mask)
        meta["n_episodes"] = len(ep_keys)
    else:
        folds = build_loro_folds(bounds, eligible_mask=mask)
    meta["n_folds"] = len(folds)
    meta["n_trs_used"] = int(mask.sum())
    meta["n_classes_used"] = int(len(np.unique(y_full[mask])))
    n_bad = n_folds_without_every_class(y_full[mask], folds)
    if n_bad:
        raise SystemExit(f"{sub_id} {model_key} {cond}: {n_bad} of {len(folds)} folds lack "
                         "a class in training; the decoder would skip them and bias p")

    if n_perm == 0:
        null_y = np.empty((0, int(mask.sum())), dtype=np.int8)
    else:
        null_y = precompute_eligible_null_state_sequences(
            y_full, bounds, mask, n_perm, D1_SEED_BASE + lag * D1_SEED_LAG_STRIDE)

    # Timing-only decoder: the 08d D1 confound design, lag-shifted as in 08d.
    T = d08._build_confound_design_matrix(run_ids, decoded_states)
    if lag > 0:
        T = np.vstack([np.zeros((lag, T.shape[1])), T[:-lag]])
    timing = {}
    if cond in ("gate", "timing"):
        if cond == "gate":
            tb = batch_loro_ridge_classify(T[mask], y_full[mask], null_y, folds)
            with open(os.path.join(pub_dir, "D1_confound_baseline.json")) as f:
                pub_conf = json.load(f)
            timing = {"balanced_accuracy": round(tb["observed"]["balanced_accuracy"], 4),
                      "published": pub_conf["balanced_accuracy"],
                      "published_lag": pub_conf["best_lag"]}
        else:
            tb = batch_loro_ridge_classify(T[mask], y_full[mask], null_y, folds)
            timing = summarise(tb, y_full[mask])
            timing_null = {"observed": tb["observed"]["balanced_accuracy"],
                           "null": tb["null_balanced_accuracies"]}
            timing_X = T
        logger.info("timing-only acc %.4f", tb["observed"]["balanced_accuracy"])

    selected = select_layers(args.layers, n_layers, best_layer)
    if cond == "gate" and len(selected) != n_layers:
        raise SystemExit("the gate compares every layer; --layers is not allowed for it")

    # A partial is reused only when it was computed from the same inputs,
    # settings and code (cf. 08e). Feature files are identified by stat, the
    # PCA by its integer summary only (explained variance differs in the last
    # bits across CPU types).
    run_fp = {
        "sub_id": sub_id, "parcellation": parc, "model": model_key,
        "condition": cond, "vt": args.vt, "lag": lag, "n_permutations": n_perm,
        "seed": D1_SEED_BASE + lag * D1_SEED_LAG_STRIDE,
        "pca_variance_threshold": d08.PCA_VARIANCE_THRESHOLD,
        "run_ids": run_ids,
        "states_mask_sha256": e08.array_identity(
            y_full.astype(np.int64), mask, np.asarray(bounds, dtype=np.int64)),
        "content_eligible": sorted(elig_set),
        "pca_split_hash": split_hash,
        "pca_layers_k_n_train": {str(k): [int(v["K"]), int(v["n_train_samples"])]
                                 for k, v in sorted(pca_info.items())},
        "features_stat_sha256": e08.file_stat_identity(
            e08.feature_file_paths(SCRATCH_DIR, "friends", model_key, run_ids),
            SCRATCH_DIR),
        "published_d1_sha256": e08.layer_fingerprint(d1["results"]),
        "code_sha256": CODE_SHA256,
    }
    fingerprints = {li: e08.layer_fingerprint({**run_fp, "layer": li})
                    for li in range(n_layers)}

    partial_dir = os.path.join(out_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)
    results, stale = load_matching_partials(partial_dir, cond, n_layers, fingerprints)
    todo = [li for li in selected if li not in results]
    stale_todo = [li for li in stale if li in todo]
    if stale_todo:
        logger.warning("%s: recomputing layers %s, whose partials came from another "
                       "run (fingerprint differs or missing)", cond, stale_todo)
    t0 = time.monotonic()
    step = max(args.n_jobs, 1)
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(_friends_layer)(
                li, features_by_layer[li], run_ids, decoded_states, lag, mask,
                y_full, null_y, folds, timing_X, timing_null)
            for li in chunk)
        for li, res in out:
            if res is None:
                continue
            pub = published_at_lag.get(li, {})
            res["published_loro_balanced_accuracy"] = pub.get("balanced_accuracy")
            res["published_loro_p_perm"] = pub.get("p_perm")
            res["fingerprint"] = fingerprints[li]
            write_json_atomic(os.path.join(partial_dir, f"{cond}_layer{li:02d}.json"), res)
        logger.info("%s: layers %s done (%.0fs)", cond, chunk, time.monotonic() - t0)

    # Re-read every matching partial, not only the ones this job started with
    # or computed: another job may have finished layers of this cell meanwhile.
    results, _ = load_matching_partials(partial_dir, cond, n_layers, fingerprints)
    missing = [li for li in selected if li not in results]
    if missing:
        raise RuntimeError(
            f"{cond}: layers {missing} have no partial matching this run's fingerprint: "
            "not computed, or overwritten by another job with other code or settings "
            "writing to the same --out_root")
    order = sorted(results)
    all_layers = order == list(range(n_layers))
    # BH across the layers at the best lag (not 08d's lags x layers family);
    # defined over every layer, so skipped when only a subset is present.
    for key, pkey in (("p_fdr", "p_perm"), ("p_delta_fdr", "p_delta")):
        if all_layers and all(pkey in results[l] for l in order):
            q = benjamini_hochberg(np.array([results[l][pkey] for l in order]))
            for l, v in zip(order, q):
                results[l][key] = round(float(v), 4)

    gate = None
    if cond == "gate":
        deltas = {l: abs(results[l]["balanced_accuracy"]
                         - results[l]["published_loro_balanced_accuracy"]) for l in order}
        max_d = max(deltas.values())
        t_d = abs(timing["balanced_accuracy"] - timing["published"])
        gate = {"max_abs_delta_vs_published": round(max_d, 6),
                "timing_abs_delta_vs_published": round(t_d, 6),
                "timing_published_lag_matches": timing["published_lag"] == lag,
                "passed": bool(max_d <= ACC_TOL and t_d <= ACC_TOL
                               and timing["published_lag"] == lag
                               and len(order) == len(published_at_lag))}

    payload = {"sub_id": sub_id, "parcellation": parc, "model": model_key,
               "stimulus": "friends", "condition": cond, "vt": args.vt,
               "n_permutations": n_perm, **meta,
               "seed": D1_SEED_BASE + lag * D1_SEED_LAG_STRIDE,
               "timing_only": timing, "gate": gate,
               "layers_requested": args.layers or "all",
               "all_layers_evaluated": all_layers,
               "fdr_family": ("layers at the best lag" if all_layers
                              else "none (not every layer present)"),
               "code_sha256": CODE_SHA256,
               "per_layer": {str(l): results[l] for l in order},
               "environment": environment()}
    write_json_atomic(os.path.join(out_dir, f"{cond}.json"), payload)
    if gate is not None and not gate["passed"]:
        raise SystemExit(f"{sub_id} {model_key}: gate failed {gate}")
    logger.info("wrote %s", os.path.join(out_dir, f"{cond}.json"))


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def transfer_observed(fX, f_y, f_mask, tX, t_y, t_mask):
    """Observed D3a balanced accuracy, as 08e _fit_eval_one_layer computes it."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(fX[f_mask], f_y[f_mask])
    return round(float(balanced_accuracy_score(t_y[t_mask], clf.predict(tX[t_mask]))), 4)


def run_transfer(args, parc, out_dir):
    sub_id, model_key, stimulus = args.sub_id, args.model, args.stimulus
    if model_key not in TEXT_MODELS:
        raise SystemExit("transfer word-TR control applies to text models only")
    lag, lag_info = e08.resolve_friends_best_lag(SCRATCH_DIR, parc, sub_id, model_key)
    pub_path = os.path.join(
        SCRATCH_DIR, "output", "08e_transformer_cross_stim_aggregate", parc, sub_id,
        e08.output_subdir(stimulus, model_key, e08.LAG_FRIENDS_BEST),
        f"D3a_transfer_{stimulus}_{model_key}.json")
    with open(pub_path) as f:
        pub = json.load(f)
    if int(pub["lag"]) != lag:
        raise RuntimeError(f"published D3a lag {pub['lag']} != Friends best lag {lag}")
    pub_layers = {int(k): v for k, v in pub["per_layer"].items()}
    peak_layer = max(pub_layers, key=lambda k: pub_layers[k]["balanced_accuracy"])
    layers = sorted({int(lag_info["friends_best_layer"]), peak_layer})
    logger.info("%s %s: lag %d, layers %s (Friends best %d, transfer peak %d)",
                sub_id, stimulus, lag, layers, lag_info["friends_best_layer"], peak_layer)

    eligibility = e08.require_05e_eligibility(
        sub_id, parc, SCRATCH_DIR, vt=args.vt, loader=load_content_eligibility)
    friends_decoded = e08._load_friends_decoded(sub_id, parc)
    test_decoded = e08._load_cross_stim_decoded(sub_id, parc, stimulus)
    f_ids = sorted(friends_decoded.keys())
    t_ids = sorted(test_decoded.keys())
    if stimulus == "petitprince_fr":
        t_ids = [r for r in t_ids if r.startswith("lppFR_")]
    elif stimulus == "petitprince_en":
        t_ids = [r for r in t_ids if r.startswith("lppEN_")]
    splits = load_training_split(sub_id, parc, SCRATCH_DIR)
    f_feat, pca_info, pca_models, f_eff, _ = stream_pca_features(
        "friends", model_key, f_ids, {r: len(friends_decoded[r]) for r in f_ids},
        SCRATCH_DIR, train_run_ids=splits["train"],
        variance_threshold=e08.PCA_VARIANCE_THRESHOLD)
    key_fn = ((lambda r: feature_key_for_cross_stim_run_id(r, stimulus))
              if stimulus == "movie10" else None)
    t_feat, _, _, t_eff, _ = stream_pca_features(
        stimulus, model_key, t_ids, {r: len(test_decoded[r]) for r in t_ids},
        SCRATCH_DIR, pca_models=pca_models, pca_info=pca_info,
        feature_key_fn=key_fn,
        max_tr_drift=8 if stimulus.startswith("petitprince_") else 3)
    friends_decoded = align_decoded(friends_decoded, f_eff)
    test_decoded = align_decoded(test_decoded, t_eff)
    f_ids = sorted(friends_decoded.keys())
    t_ids = sorted(test_decoded.keys())
    f_bounds = build_run_boundaries(f_ids, friends_decoded)
    f_states = np.concatenate([friends_decoded[r] for r in f_ids]).astype(int)
    t_states = np.concatenate([test_decoded[r] for r in t_ids]).astype(int)

    inter, _fo = e08._compute_intersection(f_states, t_states,
                                           eligibility["content_eligible"])
    if inter != pub["intersection_states"]:
        raise RuntimeError("intersection states differ from the published D3a file")
    inter_set = set(inter)
    f_mask = np.array([s in inter_set for s in f_states], dtype=bool)
    t_mask = np.array([s in inter_set for s in t_states], dtype=bool)
    f_words, _ = word_tr_mask("friends", model_key, f_ids, friends_decoded, lag)
    t_words, _ = word_tr_mask(stimulus, model_key, t_ids, test_decoded, lag,
                              feature_key_fn=key_fn)
    f_mask_w, t_mask_w = f_mask & f_words, t_mask & t_words
    null_w = precompute_eligible_null_state_sequences(
        f_states, f_bounds, f_mask_w, args.n_permutations, rng_seed=e08.NULL_SEED_D3A)
    n_classes = len(inter)

    results = {}
    for li in layers:
        fX = build_layer_feature_matrix(f_feat[li], f_ids, friends_decoded, lag=lag)
        tX = build_layer_feature_matrix(t_feat[li], t_ids, test_decoded, lag=lag)
        full = transfer_observed(fX, f_states, f_mask, tX, t_states, t_mask)
        word = e08._fit_eval_one_layer(li, fX, f_states, f_mask_w, tX, t_states,
                                       t_mask_w, inter_set, null_w, n_classes)
        word.pop("per_state", None)
        results[li] = {
            "published_balanced_accuracy": pub_layers[li]["balanced_accuracy"],
            "published_p_perm": pub_layers[li]["p_perm"],
            "published_null_mean": pub_layers[li]["null_mean"],
            "reproduced_balanced_accuracy": full,
            "word_trs": word,
            "n_test_classes_present_word": int(len(np.unique(t_states[t_mask_w]))),
        }
        logger.info("layer %d: published %.4f reproduced %.4f word-TR %.4f (p %.4f)",
                    li, pub_layers[li]["balanced_accuracy"], full,
                    word["balanced_accuracy"], word["p_perm"])
    max_d = max(abs(r["reproduced_balanced_accuracy"] - r["published_balanced_accuracy"])
                for r in results.values())
    gate = {"max_abs_delta_vs_published": round(max_d, 6), "passed": bool(max_d <= ACC_TOL)}
    payload = {"sub_id": sub_id, "parcellation": parc, "model": model_key,
               "test_stimulus": stimulus, "lag": lag,
               "friends_best_layer": int(lag_info["friends_best_layer"]),
               "transfer_peak_layer_published": peak_layer,
               "n_classes": n_classes, "intersection_states": inter,
               "n_friends_trs": int(f_mask.sum()), "n_friends_trs_word": int(f_mask_w.sum()),
               "n_test_trs": int(t_mask.sum()), "n_test_trs_word": int(t_mask_w.sum()),
               "word_tr_fraction_test": round(float(t_words.mean()), 4),
               "n_permutations": args.n_permutations, "null_seed": e08.NULL_SEED_D3A,
               "gate": gate, "per_layer": {str(k): v for k, v in results.items()},
               "environment": environment()}
    write_json_atomic(os.path.join(out_dir, "transfer.json"), payload)
    if not gate["passed"]:
        raise SystemExit(f"{sub_id} {stimulus}: transfer gate failed {gate}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--part", required=True, choices=["friends", "transfer"])
    p.add_argument("--sub_id", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--condition", choices=FRIENDS_CONDITIONS)
    p.add_argument("--stimulus", choices=["movie10", "harrypotter",
                                          "petitprince_fr", "petitprince_en"])
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--layers", default=None,
                   help="friends only: comma list of layers or 'best' (published best layer "
                        "at the best lag), e.g. 'best,0'; default all layers. Per-layer "
                        "partials with a matching fingerprint are reused; BH across layers "
                        "is skipped unless every layer is present")
    p.add_argument("--out_root", required=True,
                   help="side root; must not be $SCRATCH_DIR/output")
    a = p.parse_args()
    try:
        parse_layers_spec(a.layers)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if a.condition == "gate" and a.layers is not None:
        raise SystemExit("the gate compares every layer; --layers is not allowed for it")
    if os.path.realpath(a.out_root).startswith(
            os.path.realpath(os.path.join(SCRATCH_DIR, "output"))):
        raise SystemExit("--out_root must be outside $SCRATCH_DIR/output")
    parc = normalize_parcellation_name(a.parcellation)
    if a.part == "friends":
        if a.condition is None:
            raise SystemExit("--condition is required for --part friends")
        sub = f"friends_{a.model}"
    else:
        if a.stimulus is None:
            raise SystemExit("--stimulus is required for --part transfer")
        if a.layers is not None:
            raise SystemExit("--layers applies to --part friends only")
        sub = f"{a.stimulus}_{a.model}"
    out_dir = os.path.join(a.out_root, "08h_decoding_controls", parc, a.sub_id, sub)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    (run_friends if a.part == "friends" else run_transfer)(a, parc, out_dir)


if __name__ == "__main__":
    main()
