"""R2 network-spread null: are multi-network state maps inherited from the PCA basis?

Results R2 reports that content-eligible state maps mix several canonical
networks (pooled medians across 159 states: top-1 network share 0.25, top-3
share 0.60, 4 networks >= 10%, normalized network entropy 0.81). Every state
mean, however, lives in the participant's retained PCA subspace, and the PCA
basis itself mixes networks, so an *arbitrary* direction in that subspace could
already produce multi-network maps. This script quantifies that alternative:
it compares the observed network-participation metrics of content-eligible
states against random directions in the same retained subspace, back-projected
to parcel space through the same PCA transform.

Null variants (both reported; variance-matched is primary):
  - variance_matched: u ~ N(0, diag(explained_variance_[:n_pcs])). A random
    pattern with the training data's second-moment structure in the subspace,
    i.e. "a generic signal this pipeline could have produced".
  - isotropic: u ~ N(0, I_{n_pcs}). Every retained direction weighted equally;
    robustness variant that does not privilege high-variance components.

Faithfulness gates (hard aborts, run before any null is drawn):
  1. Back-projection: state_means_pca @ components_[:n_pcs] + mean_ must
     reproduce state_means_parcel.npy to 1e-8. The null relies on this map.
  2. Published medians: pooled content-eligible metrics must reproduce the
     manuscript's R2 values (n = 159; top1 0.25, top3 0.60, n>=10% 4,
     entropy 0.81 after rounding).

Statistic and inference: per participant, the median metric over that
participant's content-eligible states, compared with a null distribution of
medians (each draw: K_elig per-draw values resampled from the participant's
null pool, median taken; n_group draws). Two-sided empirical p by doubled
min-tail with the (count + 1) / (n + 1) correction (Phipson & Smyth, 2010).
A pooled (159-state) comparison mirrors the manuscript's pooled medians.

Outputs (per subject, under
{SCRATCH_DIR}/output/sm_rel_r2_network_spread/{parcellation}/{sub}/{vt}/):
  - r2_network_spread_summary.json
  - null_entropy_{variant}.npy, null_top1_{variant}.npy (per-draw values)
Pooled summary: pooled_summary.json at the parcellation level.

Deterministic seeding: subject index s, variant v in {0: variance_matched,
1: isotropic} use numpy.random.default_rng(1000 * (s + 1) + v) for draws and
default_rng(2000 * (s + 1) + v) for group resampling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import NETWORK_ORDER, assign_network  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("sm_rel_r2_network_spread")

ELIGIBLE_CATEGORY = "eligible_for_content_analysis"

# Published R2 pooled medians (results.md, R2), reproduced as a gate.
PUBLISHED = {"n_states": 159, "top1": 0.25, "top3": 0.60,
             "n_ge10": 4.0, "entropy": 0.81}


def load_parcel_networks(parcellation: str) -> np.ndarray:
    """Parcel -> canonical network labels from the atlas dseg TSV.

    Mirrors utils.plot_style.load_parcel_networks on the main branch
    (subcortical groups via assign_network, cortical via network_label),
    implemented standalone because the supplements branch does not carry
    viz_yabplot. Validated by gate 2 (published medians reproduce).
    """
    atlas_dir = os.getenv("ATLAS_DIR")
    if not atlas_dir:
        raise RuntimeError("ATLAS_DIR not set")
    tsv = pd.read_csv(
        Path(atlas_dir) / parcellation / f"{parcellation}_dseg.tsv", sep="\t"
    ).sort_values("index")
    return np.array(
        [assign_network(row["label"]) or row.get("network_label", "Unknown")
         for _, row in tsv.iterrows()]
    )


def participation_metrics(maps: np.ndarray, network_masks: list[np.ndarray],
                          n_networks_total: int) -> dict[str, np.ndarray]:
    """Vectorized network-participation metrics for (n_maps, n_parcels) maps.

    Mirrors utils/network_participation.py on the main branch: mean |loading|
    per network (mean, not sum, so large networks are not favored), normalized
    to shares; entropy normalized by log(n_networks_total).
    """
    maps = np.atleast_2d(np.asarray(maps, dtype=float))
    a = np.abs(maps)
    scores = np.stack([a[:, m].mean(axis=1) for m in network_masks], axis=1)
    totals = scores.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("degenerate all-zero map encountered")
    comp = scores / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(comp > 0, np.log(comp), 0.0)
    entropy = -(comp * logs).sum(axis=1) / np.log(n_networks_total)
    srt = np.sort(comp, axis=1)[:, ::-1]
    return {
        "top1": srt[:, 0],
        "top3": srt[:, :3].sum(axis=1),
        "n_ge10": (comp >= 0.10).sum(axis=1).astype(float),
        "entropy": entropy,
    }


def empirical_p_two_sided(observed: float, null: np.ndarray) -> float:
    """Doubled min-tail empirical p with the finite-sampling correction."""
    n = len(null)
    p_hi = (np.sum(null >= observed) + 1) / (n + 1)
    p_lo = (np.sum(null <= observed) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(p_hi, p_lo)))


def null_median_distribution(per_draw: np.ndarray, k: int, n_group: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Medians of n_group resampled groups of k values from the draw pool."""
    idx = rng.integers(0, len(per_draw), size=(n_group, k))
    return np.median(per_draw[idx], axis=1)


def load_subject(scratch: str, parc: str, sub: str, vt: str):
    hmm_dir = Path(scratch) / "output" / "04_combined_hdphmm" / parc / sub / "final" / vt
    means_parcel = np.load(hmm_dir / "state_means_parcel.npy")
    means_pca = np.load(hmm_dir / "state_means_pca.npy")
    with open(hmm_dir / "pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    flags = pd.read_csv(
        Path(scratch) / "output" / "05e_temporal_trend_a4" / parc / sub / vt
        / "state_flags.csv"
    )
    state_col = "state" if "state" in flags.columns else "state_id"
    eligible = flags.loc[
        flags["summary_category"] == ELIGIBLE_CATEGORY, state_col
    ].astype(int).to_numpy()
    return means_parcel, means_pca, pca, eligible


def gate_backprojection(means_parcel, means_pca, pca, sub: str, tol=1e-8):
    n_pcs = means_pca.shape[1]
    w = pca.components_[:n_pcs]
    recon = means_pca @ w + pca.mean_
    err = float(np.abs(recon - means_parcel).max())
    if err > tol:
        raise RuntimeError(
            f"{sub}: back-projection gate failed (max err {err:.2e} > {tol}); "
            "the null's subspace map would not match the published maps"
        )
    logger.info("%s: back-projection gate passed (max err %.2e)", sub, err)
    return w, pca.explained_variance_[:n_pcs], pca.mean_


def gate_published_medians(pooled: dict[str, np.ndarray]):
    got = {
        "n_states": len(pooled["entropy"]),
        "top1": round(float(np.median(pooled["top1"])), 2),
        "top3": round(float(np.median(pooled["top3"])), 2),
        "n_ge10": float(np.median(pooled["n_ge10"])),
        "entropy": round(float(np.median(pooled["entropy"])), 2),
    }
    if got != PUBLISHED:
        raise RuntimeError(
            f"published-medians gate failed: got {got}, expected {PUBLISHED}; "
            "metric implementation or state selection does not match R2"
        )
    logger.info("published-medians gate passed: %s", got)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parcellation", default="atlas-4S156Parcels")
    ap.add_argument("--vt", default="vt0.95")
    ap.add_argument("--n_draws", type=int, default=10000)
    ap.add_argument("--n_group", type=int, default=10000)
    ap.add_argument("--subjects", nargs="*",
                    default=[f"sub-0{i}" for i in range(1, 7)])
    args = ap.parse_args()

    load_dotenv()
    scratch = os.getenv("SCRATCH_DIR")
    if not scratch:
        raise RuntimeError("SCRATCH_DIR not set")

    parcel_networks = load_parcel_networks(args.parcellation)
    masks = [parcel_networks == n for n in NETWORK_ORDER]
    n_nets = len(NETWORK_ORDER)

    # Pass 1: load everything, run both gates before any null work.
    data = {}
    pooled = {k: [] for k in ("top1", "top3", "n_ge10", "entropy")}
    for sub in args.subjects:
        means_parcel, means_pca, pca, eligible = load_subject(
            scratch, args.parcellation, sub, args.vt)
        w, evar, pca_mean = gate_backprojection(
            means_parcel, means_pca, pca, sub)
        obs = participation_metrics(means_parcel[eligible], masks, n_nets)
        for k in pooled:
            pooled[k].append(obs[k])
        data[sub] = (obs, eligible, w, evar, pca_mean)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    gate_published_medians(pooled)

    # Pass 2: nulls.
    variants = ("variance_matched", "isotropic")
    pooled_null = {v: {"entropy": [], "top1": []} for v in variants}
    for s_idx, sub in enumerate(args.subjects):
        obs, eligible, w, evar, pca_mean = data[sub]
        k_elig = len(eligible)
        out_dir = (Path(scratch) / "output" / "sm_rel_r2_network_spread"
                   / args.parcellation / sub / args.vt)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "sub_id": sub, "parcellation": args.parcellation, "vt": args.vt,
            "n_pcs": int(w.shape[0]), "n_eligible_states": int(k_elig),
            "n_draws": args.n_draws, "n_group": args.n_group,
            "network_order": list(NETWORK_ORDER),
            "observed": {k: {"median": float(np.median(v)),
                             "iqr": [float(np.percentile(v, 25)),
                                     float(np.percentile(v, 75))]}
                         for k, v in obs.items()},
            "null": {},
        }
        for v_idx, variant in enumerate(variants):
            rng = np.random.default_rng(1000 * (s_idx + 1) + v_idx)
            scale = np.sqrt(evar) if variant == "variance_matched" else 1.0
            draws_pc = rng.standard_normal((args.n_draws, w.shape[0])) * scale
            null_metrics = participation_metrics(
                draws_pc @ w + pca_mean, masks, n_nets)
            np.save(out_dir / f"null_entropy_{variant}.npy",
                    null_metrics["entropy"])
            np.save(out_dir / f"null_top1_{variant}.npy", null_metrics["top1"])
            for k in ("entropy", "top1"):
                pooled_null[variant][k].append((null_metrics[k], k_elig))
            rng_group = np.random.default_rng(2000 * (s_idx + 1) + v_idx)
            variant_summary = {}
            for k in ("entropy", "top1", "top3", "n_ge10"):
                med_null = null_median_distribution(
                    null_metrics[k], k_elig, args.n_group, rng_group)
                obs_med = float(np.median(obs[k]))
                variant_summary[k] = {
                    "null_per_draw_mean": float(null_metrics[k].mean()),
                    "null_per_draw_sd": float(null_metrics[k].std(ddof=1)),
                    "null_median_mean": float(med_null.mean()),
                    "null_median_ci":
                        [float(np.percentile(med_null, 2.5)),
                         float(np.percentile(med_null, 97.5))],
                    "observed_median": obs_med,
                    "delta": obs_med - float(med_null.mean()),
                    "z": float((obs_med - med_null.mean())
                               / med_null.std(ddof=1)),
                    "p_two_sided": empirical_p_two_sided(obs_med, med_null),
                }
            summary["null"][variant] = variant_summary
        with open(out_dir / "r2_network_spread_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "%s: entropy obs %.3f | vm null median %.3f (p=%.4g) | "
            "iso null median %.3f (p=%.4g)", sub,
            summary["observed"]["entropy"]["median"],
            summary["null"]["variance_matched"]["entropy"]["null_median_mean"],
            summary["null"]["variance_matched"]["entropy"]["p_two_sided"],
            summary["null"]["isotropic"]["entropy"]["null_median_mean"],
            summary["null"]["isotropic"]["entropy"]["p_two_sided"])

    # Pooled comparison mirroring the manuscript's 159-state medians: each
    # group draw takes k_elig values from each subject's null pool.
    pooled_summary = {"n_states": int(len(pooled["entropy"])),
                      "observed": {k: float(np.median(v))
                                   for k, v in pooled.items()},
                      "null": {}}
    for variant in variants:
        rng = np.random.default_rng(9000 + variants.index(variant))
        entry = {}
        for k in ("entropy", "top1"):
            groups = []
            for per_draw, k_elig in pooled_null[variant][k]:
                idx = rng.integers(0, len(per_draw),
                                   size=(args.n_group, k_elig))
                groups.append(per_draw[idx])
            med_null = np.median(np.concatenate(groups, axis=1), axis=1)
            obs_med = float(np.median(pooled[k]))
            entry[k] = {
                "null_median_mean": float(med_null.mean()),
                "null_median_ci": [float(np.percentile(med_null, 2.5)),
                                   float(np.percentile(med_null, 97.5))],
                "observed_median": obs_med,
                "z": float((obs_med - med_null.mean()) / med_null.std(ddof=1)),
                "p_two_sided": empirical_p_two_sided(obs_med, med_null),
            }
        pooled_summary["null"][variant] = entry
    pooled_path = (Path(scratch) / "output" / "sm_rel_r2_network_spread"
                   / args.parcellation / "pooled_summary.json")
    with open(pooled_path, "w") as f:
        json.dump(pooled_summary, f, indent=2)
    logger.info("pooled summary -> %s", pooled_path)


if __name__ == "__main__":
    main()
