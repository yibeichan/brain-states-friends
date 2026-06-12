#!/usr/bin/env python
"""Supplementary: ICA as an alternative state-discovery method.

Per subject: spatial ICA on the vt=0.95 PC subspace (sweep K in {15,25,35} plus
K_active), matched against HMM state maps (Tier 1), with temporal correspondence
(Tier 2) and WTA label agreement (Tier 3). Both eligible-only and all-active
HMM state sets are computed.
"""
import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import hmm_io
from utils.ica_states import (
    icasso_consensus, match_maps_hungarian, subspace_rotation_null,
    spatial_match_pvalues, temporal_correspondence, wta_label_agreement,
)
from utils.stats import fdr_with_nan
from utils.transformer_analysis import (
    load_content_eligibility, build_run_boundaries,
)
from utils.common import check_checkpoint

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sm_ica")

K_SWEEP = [15, 25, 35]
MIN_OCC = 0.01
MIN_SHIFT = 1
HMM_MAPS_KIND = "state_contrast_means_no_global_mean"


# ---------------------------------------------------------------------------
# JAX-free model loader
# ---------------------------------------------------------------------------

class _HMMStub:
    """Minimal stub that holds the numpy arrays from a pickled StickyHDPHMM_JAX.

    We intercept deserialization so that jax is never imported. The stub exposes
    the same attributes that hmm_io.back_project_states and
    hmm_io.compute_state_posteriors need: means_, covars_, transmat_,
    startprob_, n_components, covariance_type.
    """
    pass


class _NoJaxUnpickler(pickle.Unpickler):
    """Custom unpickler that substitutes the JAX HMM class with _HMMStub.

    Any class whose module path contains 'hdphmm_jax' is replaced; its
    __setstate__ (which calls jax.random.PRNGKey) is NOT called. Instead we
    populate the stub directly from the raw state dict.
    """
    def find_class(self, module, name):
        if "hdphmm_jax" in module:
            return _HMMStub
        return super().find_class(module, name)


def _load_model_no_jax(path):
    """Load a pickled StickyHDPHMM (or StickyHDPHMM_JAX) without requiring jax.

    When the pickle contains a JAX model, _NoJaxUnpickler substitutes _HMMStub
    and populates it via __setstate__; we then call a jax-free __setstate__
    replacement. When the pickle contains a numpy model, it loads normally.
    """
    with open(path, "rb") as f:
        model = _NoJaxUnpickler(f).load()
    if isinstance(model, _HMMStub):
        # __setstate__ was called by pickle with the state dict; the stub's
        # __dict__ should already be populated. Guard against missing attrs.
        for attr in ("means_", "covars_", "transmat_", "startprob_",
                     "n_components", "covariance_type"):
            if not hasattr(model, attr):
                raise RuntimeError(
                    f"_HMMStub missing attribute '{attr}' after load: "
                    f"check that the model pickle contains this field."
                )
        # Ensure everything is plain numpy (JAX arrays would fail downstream)
        for attr in ("means_", "covars_", "transmat_", "startprob_"):
            val = getattr(model, attr)
            if val is not None and not isinstance(val, np.ndarray):
                setattr(model, attr, np.asarray(val))
        cov = np.asarray(model.covars_)
        if cov.ndim not in (2, 3):
            raise ValueError(
                f"loaded model covars_ has unexpected ndim={cov.ndim} "
                f"(expected 2=diag or 3=full); _normalize_covars was skipped")
        logger.info("Loaded JAX model via _HMMStub (no jax import needed); "
                    "n_components=%d, covariance_type=%s",
                    model.n_components, model.covariance_type)
    return model


def _np(x):  # json-safe
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _json_sanitize(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None for strict JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def _ordered_runs(pca_base):
    """Concatenate train+valid+test run ids in a fixed order with split names.

    NOTE: splits/projected/n_pcs_lookup live under the 03a PCA base, NOT 04.
    """
    split = hmm_io.load_split(pca_base)
    ordered = []
    for name in ("train", "valid", "test"):
        for rid in split.get(name, []):
            ordered.append((rid, name))
    return ordered


def _matched_occupancy_corr(ica_labels, viterbi, ica_idx, hmm_state_ids):
    """Occupancy correlation ACROSS MATCHED PAIRS (proper, label-aware).

    For each matched pair, ICA-component occupancy (fraction of TRs that
    component wins the WTA) vs HMM-state occupancy (fraction of Viterbi TRs in
    the matched state). Pearson across matched pairs. Returns nan if <2 pairs.
    """
    T_ica = len(ica_labels)
    T_hmm = len(viterbi)
    ica_occ = np.array([(ica_labels == int(i)).sum() / T_ica for i in ica_idx])
    hmm_occ = np.array([(viterbi == int(s)).sum() / T_hmm for s in hmm_state_ids])
    if len(ica_occ) < 2 or ica_occ.std() == 0 or hmm_occ.std() == 0:
        return float("nan")
    return float(np.corrcoef(ica_occ, hmm_occ)[0, 1])


def run_subject(sub_id, parcellation, vt, out_dir, n_restarts, n_perm_spatial,
                n_perm_temporal, rng_seed, extra_k=None):
    # 03a base: PCA model, splits, projected data, n_pcs lookup.
    pca_base = os.path.join(
        SCRATCH_DIR, "output", "03a_pca4combined_hmm", parcellation, sub_id)
    # 04 base: fitted model + decoded states + final_results (+ a pca copy).
    hmm_base = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation, sub_id)
    final_dir = os.path.join(hmm_base, "final", f"vt{vt}")

    model = _load_model_no_jax(os.path.join(final_dir, "best_model.pkl"))
    with open(os.path.join(final_dir, "pca_model.pkl"), "rb") as f:   # 04's copy
        pca = pickle.load(f)
    n_pcs = int(hmm_io.load_n_pcs_lookup(pca_base)[vt])               # from 03a
    with open(os.path.join(final_dir, "final_results.json")) as f:
        _fr = json.load(f)
    try:
        K_active = int(_fr["final_refit"]["n_active_states"])
    except KeyError as e:
        raise KeyError(f"final_results.json missing expected key: {e}") from e
    with open(os.path.join(final_dir, "decoded_states.pkl"), "rb") as f:
        decoded = pickle.load(f)

    components = pca.components_[:n_pcs]                 # (n_pcs, 156)

    projected_dir = hmm_io.get_projected_dir(pca_base)
    ordered = _ordered_runs(pca_base)
    X_parts, lengths, vit_parts, run_ids = [], [], [], []
    for rid, name in ordered:
        if rid not in decoded:
            raise KeyError(f"run {rid} in split but absent from decoded_states.pkl")
        Xr, lr = hmm_io.load_projected_runs([rid], projected_dir, n_pcs, name)
        assert len(decoded[rid]) == lr[0], (
            f"run {rid}: decoded length {len(decoded[rid])} != projected TRs {lr[0]}")
        X_parts.append(Xr)
        lengths.extend(lr)
        vit_parts.append(np.asarray(decoded[rid]))
        run_ids.append(rid)
    X_pc = np.vstack(X_parts)                            # (T, n_pcs)
    viterbi = np.concatenate(vit_parts)                 # (T,)
    run_boundaries = build_run_boundaries(run_ids, decoded)

    # Tier-1 matching compares PATTERNS in the zero-mean PC-loading subspace
    # (same space as ICA maps and the subspace-rotation null). Use state-contrast
    # maps (means_ @ components), NOT back_project_states (which adds pca.mean_,
    # a shared grand-mean pattern that distorts correlations and breaks the null).
    state_means_parcel = np.asarray(model.means_)[:, :n_pcs] @ components   # (K,156)
    gamma = hmm_io.compute_state_posteriors(model, X_pc, lengths)           # (T,K)

    elig = load_content_eligibility(sub_id, parcellation, SCRATCH_DIR, vt=vt)
    eligible = sorted(int(s) for s in elig.get("content_eligible", []))
    # 'all active' = states the Viterbi path actually visits (posteriors are
    # dense, so a mean>0 proxy would include every state).
    all_active = sorted(int(s) for s in np.unique(viterbi))

    summary_path = os.path.join(out_dir, "ica_match_summary.json")
    if extra_k:
        # Augment an existing summary with sensitivity K(s) only -- do NOT
        # recompute the primary sweep (deterministic, already verified).
        if not os.path.exists(summary_path):
            raise SystemExit(
                f"--extra_k requires an existing {summary_path}; run the base "
                "sweep first.")
        # NOTE: single-writer assumption -- the launcher maps one array task to
        # one subject, each writing its own summary, so this read-modify-write
        # is never concurrent. Do NOT launch two --extra_k jobs for the same
        # subject at once (no file lock is taken).
        with open(summary_path) as f:
            results = json.load(f)
        # The merged K entries are computed with the CURRENT model/PCA, but the
        # summary's top-level fields come from the existing file. Refuse to merge
        # into a summary written under different conditions, which would silently
        # mix incompatible metadata.
        for fld, cur in (("sub_id", sub_id), ("parcellation", parcellation),
                         ("vt", vt), ("n_pcs", n_pcs), ("K_active", K_active),
                         ("hmm_maps", HMM_MAPS_KIND)):
            prev = results.get(fld)
            if prev is not None and str(prev) != str(cur):
                raise SystemExit(
                    f"--extra_k: existing summary {fld}={prev!r} != current "
                    f"{fld}={cur!r}; refuse to merge incompatible runs.")
        # Fail fast on invalid requested K rather than silently warn+skip below.
        bad = [k for k in extra_k if not (1 <= k <= n_pcs)]
        if bad:
            raise SystemExit(
                f"--extra_k: invalid K {bad} (must be 1 <= K <= n_pcs={n_pcs}).")
        K_values = list(dict.fromkeys(extra_k))  # dedup, preserve order
        logger.info("extra_k merge mode: computing K=%s, merging into %s",
                    K_values, summary_path)
    else:
        results = {"sub_id": sub_id, "parcellation": parcellation, "vt": vt,
                   "n_pcs": n_pcs, "K_active": K_active,
                   "hmm_maps": HMM_MAPS_KIND,
                   "by_K": {}}
        K_values = list(dict.fromkeys(K_SWEEP + [K_active]))  # dedup, preserve order
    for K in K_values:
        if K > n_pcs:
            logger.warning("skip K=%d > n_pcs=%d", K, n_pcs)
            continue
        if extra_k and str(K) in results["by_K"]:
            # Already computed (same deterministic params); keep the existing
            # entry rather than recompute -- also avoids overwriting the
            # read-only git-annex .npy symlink for that K. Only skip if the
            # entry is structurally complete AND its map artifacts exist;
            # otherwise fall through and recompute to repair a partial entry.
            prev = results["by_K"][str(K)]
            maps_ok = all(os.path.lexists(os.path.join(out_dir, f"ica_{n}_K{K}.npy"))
                          for n in ("maps", "timecourses"))
            if prev.get("state_sets") and maps_ok:
                logger.info("skip K=%d: already present in summary", K)
                continue
            logger.warning("recompute K=%d: existing entry incomplete "
                           "(state_sets=%s, maps_ok=%s)",
                           K, bool(prev.get("state_sets")), maps_ok)
        ica = icasso_consensus(components, X_pc, n_components=K,
                               n_restarts=n_restarts, rng_seed=rng_seed)
        ica_maps = ica["consensus_maps"]               # (156, n_cons)
        timecourses = ica["timecourses"]               # (T, n_cons), consensus-consistent
        entry = {"iq": _np(ica["iq"]),
                 "cluster_sizes": _np(ica["cluster_sizes"]),
                 "n_consensus": int(ica["iq"].shape[0]),
                 "min_occ": MIN_OCC,
                 "n_perm_temporal": n_perm_temporal,
                 "n_perm_spatial": n_perm_spatial,
                 "min_shift": MIN_SHIFT,
                 "state_sets": {}}
        for set_name, state_list in (("eligible", eligible), ("all", all_active)):
            if len(state_list) == 0:
                continue
            # state_list holds absolute model state indices (0..K_model-1),
            # the same basis as model.means_ / decoded_states / gamma columns.
            hmm_maps = state_means_parcel[state_list]   # (n_states, 156)
            m = match_maps_hungarian(ica_maps, hmm_maps)
            null = subspace_rotation_null(components, hmm_maps,
                                          n_components=ica_maps.shape[1],
                                          n_perm=n_perm_spatial, rng_seed=rng_seed)
            pvals = spatial_match_pvalues(m["matched_r"], null)
            spatial_q = fdr_with_nan(np.asarray(pvals))
            hmm_state_ids = [int(state_list[h]) for h in m["hmm_idx"]]
            t2 = temporal_correspondence(
                gamma, timecourses,
                hmm_idx=np.array(hmm_state_ids), ica_idx=m["ica_idx"],
                matched_sign=m["matched_sign"],
                run_boundaries=run_boundaries, n_perm=n_perm_temporal,
                rng_seed=rng_seed, min_occ=MIN_OCC, min_shift=MIN_SHIFT)
            t3 = wta_label_agreement(timecourses, viterbi, run_boundaries,
                                     n_perm=n_perm_temporal, rng_seed=rng_seed,
                                     min_shift=MIN_SHIFT)
            ica_labels = t3.pop("ica_labels")           # keep out of JSON (large)
            occ_corr = _matched_occupancy_corr(
                ica_labels, viterbi, m["ica_idx"], hmm_state_ids)
            entry["state_sets"][set_name] = {
                "matched_r": _np(m["matched_r"]),
                "hmm_state_ids": hmm_state_ids,
                "ica_idx": _np(m["ica_idx"]),
                "spatial_p": _np(pvals),
                "spatial_q": _np(spatial_q),
                "null_mean": float(null.mean()),
                "null_p95": float(np.percentile(null, 95)),
                "tier2_rho": _np(t2["rho"]),
                "tier2_p": _np(t2["p"]),
                "tier2_q": _np(t2["q"]),
                "tier2_occupancy": _np(t2["occupancy"]),
                "tier3": {"ami": t3["ami"], "ari": t3["ari"],
                          "p_ami": t3["p_ami"], "matched_occ_corr": occ_corr},
            }
        results["by_K"][str(K)] = entry
        for path, arr in (
                (os.path.join(out_dir, f"ica_maps_K{K}.npy"), ica_maps),
                (os.path.join(out_dir, f"ica_timecourses_K{K}.npy"),
                 timecourses.astype(np.float32))):
            # Atomic write: save to a temp file (same dir/filesystem) then
            # os.replace onto the final path. replace overwrites whatever is
            # there -- including a read-only git-annex symlink or read-only
            # regular file -- in one rename (it needs only directory write
            # permission, and leaves annex content intact). This both avoids the
            # annex-overwrite PermissionError and closes the crash window where
            # an unlink-then-save could leave no artifact. Write via a file
            # handle so np.save does not re-append a second ".npy".
            tmp_arr = path + ".tmp"
            with open(tmp_arr, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp_arr, path)

    # Atomic write: dump to a temp file then replace, so a crash mid-dump can't
    # truncate the existing summary (which is the input in --extra_k mode).
    tmp_path = summary_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(_json_sanitize(results), f, indent=2, allow_nan=False)
    os.replace(tmp_path, summary_path)
    logger.info("wrote %s", summary_path)


def main():
    p = argparse.ArgumentParser(description="ICA alternative state-discovery (supplement).")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95")
    p.add_argument("--n_restarts", type=int, default=25)
    p.add_argument("--n_perm_spatial", type=int, default=1000)
    p.add_argument("--n_perm_temporal", type=int, default=1000)
    p.add_argument("--rng_seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--extra_k", default="",
                   help="Comma-separated sensitivity K(s) to compute and MERGE "
                        "into an existing summary (e.g. '45' or '41,42,43'); "
                        "skips the primary sweep and preserves existing by_K "
                        "entries. (SLURM callers pass colon-delimited via "
                        "--export=ALL,EXTRA_K=41:42:..; the launcher converts "
                        "colons to commas.)")
    a = p.parse_args()
    if SCRATCH_DIR is None:
        raise SystemExit("SCRATCH_DIR must be set in .env")
    out_dir = os.path.join(SCRATCH_DIR, "output", "sm_ica_states",
                           a.parcellation, a.sub_id)
    os.makedirs(out_dir, exist_ok=True)
    try:
        extra_k = [int(k) for k in a.extra_k.split(",") if k.strip()] or None
    except ValueError as e:
        raise SystemExit(
            f"--extra_k: expected comma-separated integers, got {a.extra_k!r} ({e})")
    # In extra_k merge mode the existing summary is the input, so the
    # checkpoint skip is bypassed intentionally.
    if not extra_k and check_checkpoint(
            out_dir, ["ica_match_summary.json"], "sm_ica", force=a.force):
        logger.info("checkpoint exists; use --force to rerun")
        return
    run_subject(a.sub_id, a.parcellation, a.vt, out_dir, a.n_restarts,
                a.n_perm_spatial, a.n_perm_temporal, a.rng_seed, extra_k=extra_k)


if __name__ == "__main__":
    main()
