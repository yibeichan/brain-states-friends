#!/usr/bin/env python3
"""sm_alt_ica_oos_recurrence.py - ICA out-of-stimulus recurrence (R5 analogue).

Per subject: does a Friends-fit ICA component's recurrence rank predict its
occupancy on out-of-stimulus data (Movie10), with the ICA applied FROZEN (no
refit)? WTA primary + continuous robustness. Per-subject inference only; no
group-level statistic.

Inputs (frozen, from earlier pipeline):
    03a/04: PCA model, n_pcs lookup, projected Friends runs, decoded states.
    sm_ica_states: K_active consensus maps (ica_maps_K{K_active}.npy).
    m10_03: Movie10 PC-score projections (movie_run_ids.json + {run_id}.npy).

Output:
    {SCRATCH_DIR}/output/sm_ica_oos_recurrence/{parc}/{sub}/oos_recurrence_summary.json
"""
import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils import hmm_io
from utils.ica_states import consensus_projection, wta_labels
from utils.ica_oos_recurrence import fo_per_run, recurrence_scores, continuous_occupancy, phase_randomize
from utils.transformer_analysis import build_run_boundaries
from sm_alt_ica_states import _ordered_runs

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in environment or .env file")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MOVIE_FILMS = ("bourne", "wolf", "figures", "life")


def load_friends_inputs(sub_id, parcellation, vt):
    """Load frozen PCA components, K_active consensus maps, Friends X_pc, and run
    boundaries.  Mirrors sm_alt_ica_states.run_subject exactly: same split order,
    same projected-dir, same n_pcs slice.
    """
    pca_base = os.path.join(
        SCRATCH_DIR, "output", "03a_pca4combined_hmm", parcellation, sub_id)
    final_dir = os.path.join(
        SCRATCH_DIR, "output", "04_combined_hdphmm", parcellation, sub_id,
        "final", f"vt{vt}")
    ica_dir = os.path.join(
        SCRATCH_DIR, "output", "sm_ica_states", parcellation, sub_id)

    # --- PCA model + n_pcs (from 04's copy; n_pcs from 03a lookup) ---
    with open(os.path.join(final_dir, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    # n_pcs_lookup keys are strings like "0.95"; vt may be float or string.
    n_pcs_lookup = hmm_io.load_n_pcs_lookup(pca_base)
    n_pcs = int(n_pcs_lookup[str(vt)])  # keys are strings (e.g. "0.95"); stringify vt to index
    components = pca.components_[:n_pcs]            # (n_pcs, n_parcels)

    # --- Consensus maps (ICA) ---
    with open(os.path.join(ica_dir, "ica_match_summary.json")) as f:
        K_active = int(json.load(f)["K_active"])
    consensus_maps = np.load(
        os.path.join(ica_dir, f"ica_maps_K{K_active}.npy"))   # (n_parcels, K_active)

    # --- Friends projected runs + run boundaries (mirror sm_alt_ica_states) ---
    with open(os.path.join(final_dir, "decoded_states.pkl"), "rb") as f:
        decoded = pickle.load(f)
    projected_dir = hmm_io.get_projected_dir(pca_base)
    ordered = _ordered_runs(pca_base)               # [(run_id, split_name), ...]

    X_parts, run_ids = [], []
    for rid, split_name in ordered:
        if rid not in decoded:
            raise KeyError(
                f"run {rid} in split but absent from decoded_states.pkl")
        Xr, _ = hmm_io.load_projected_runs([rid], projected_dir, n_pcs, split_name)
        X_parts.append(Xr)
        run_ids.append(rid)

    X_pc = np.vstack(X_parts)                       # (T, n_pcs)
    run_boundaries = build_run_boundaries(run_ids, decoded)

    return {
        "components": components,
        "consensus_maps": consensus_maps,
        "X_pc": X_pc,
        "run_boundaries": run_boundaries,
        "n_pcs": n_pcs,
        "K_active": K_active,
    }


def load_movie_timecourses(sub_id, parcellation, vt, n_pcs, proj):
    """Apply the frozen Friends projection to each Movie10 PC-score run.

    Returns a tuple (per_film, raw_runs) where:
      - per_film is a dict {film: (timecourses, run_boundaries)} where timecourses
        has shape (T_film, n_consensus) and run_boundaries is a list of (start, end)
        tuples covering the concatenated film runs.
      - raw_runs is a list of (film, rid, X_raw) for each run in order, where X_raw
        is the raw PC-score matrix (T_run, n_pcs) BEFORE projection.
    """
    mdir = os.path.join(
        SCRATCH_DIR, "output", "m10_03_projected",
        parcellation, sub_id, f"vt{vt}")
    with open(os.path.join(mdir, "movie_run_ids.json")) as f:
        movie_run_ids = json.load(f)        # {film: [run_id, ...]}

    per_film = {}
    raw_runs = []  # list of (film, rid, X_raw) in order
    for film, run_ids in movie_run_ids.items():
        tc_parts, decoded_like, ids = [], {}, []
        for rid in run_ids:
            p = os.path.join(mdir, f"{rid}.npy")
            if not os.path.exists(p):
                logger.warning("missing %s; skipping", p)
                continue
            X = np.load(p)[:, :n_pcs]
            raw_runs.append((film, rid, X))
            tc_parts.append(X @ proj)
            decoded_like[rid] = np.empty(X.shape[0])    # length proxy for boundaries
            ids.append(rid)
        if not tc_parts:
            logger.warning("no runs found for film %s; skipping", film)
            continue
        tc = np.vstack(tc_parts)
        per_film[film] = (tc, build_run_boundaries(ids, decoded_like))

    return per_film, raw_runs


def _occupancies(tc, run_boundaries, n_components, fo_threshold):
    """Compute mean WTA fractional occupancy and continuous occupancy over runs."""
    labels = wta_labels(tc, run_boundaries)
    fo = fo_per_run(labels, run_boundaries, n_components)
    wta_mean = np.mean(np.vstack(list(fo.values())), axis=0)   # (n_components,)
    cont = continuous_occupancy(tc, run_boundaries)             # (n_components,)
    return wta_mean, cont


def _spearman(x, y):
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(len(x))}


def _pool_occupancy_from_raw(raw_runs, proj, n_components, fo_threshold):
    """Project raw per-run X arrays and pool to compute occupancy.

    Parameters
    ----------
    raw_runs : list of (film, rid, X_raw) where X_raw is (T_run, n_pcs)
    proj : (n_pcs, n_components)
    n_components : int
    fo_threshold : float

    Returns
    -------
    wta_mean, cont : each (n_components,)
    """
    tc_parts, rb, off = [], [], 0
    for _film, _rid, X in raw_runs:
        tc = X @ proj
        rb.append((off, off + tc.shape[0]))
        tc_parts.append(tc)
        off += tc.shape[0]
    all_tc = np.vstack(tc_parts)
    return _occupancies(all_tc, rb, n_components, fo_threshold)


def run_subject(sub_id, parcellation, vt, stimulus, fo_threshold, out_dir, n_null=100):
    """Run the OOS recurrence analysis for one subject.

    Parameters
    ----------
    sub_id : str
    parcellation : str
    vt : float or str
        Variance threshold used to select n_pcs (e.g. 0.95 or "0.95").
    stimulus : str
        Currently only "movie10" is supported.
    fo_threshold : float
        Minimum fractional occupancy for a component to count as active in a run.
    out_dir : str
        Directory to write oos_recurrence_summary.json.
    n_null : int
        Number of phase-randomized null draws for the overall pooled arm.
        If 0, the null is skipped and null keys are omitted from the summary.

    Returns
    -------
    dict : the summary written to disk.
    """
    if stimulus != "movie10":
        raise NotImplementedError("Phase 1 covers movie10 only; HP/PP are Phase 2.")

    inp = load_friends_inputs(sub_id, parcellation, vt)
    n_components = inp["consensus_maps"].shape[1]
    proj = consensus_projection(inp["components"], inp["consensus_maps"])
    # proj shape: (n_pcs, n_components)

    # x-axis: Friends WTA recurrence (fraction of runs each component is active)
    friends_tc = inp["X_pc"] @ proj                 # (T, n_components)
    friends_labels = wta_labels(friends_tc, inp["run_boundaries"])
    friends_fo = fo_per_run(friends_labels, inp["run_boundaries"], n_components)
    recurrence = recurrence_scores(friends_fo, n_components, fo_threshold)
    # recurrence shape: (n_components,)

    # Marginal caveat: Spearman(recurrence, friends mean WTA-FO)
    friends_wta_mean = np.mean(np.vstack(list(friends_fo.values())), axis=0)
    rho_marginal = float(spearmanr(recurrence, friends_wta_mean).statistic)

    # y-axis: Movie10 occupancy per component
    per_film_tc, raw_runs = load_movie_timecourses(
        sub_id, parcellation, vt, inp["n_pcs"], proj)

    # Pool all movie runs into one big array for the overall correlation
    all_tc_parts, all_rb, off = [], [], 0
    for tc, rb in per_film_tc.values():
        all_tc_parts.append(tc)
        all_rb.extend([(s + off, e + off) for s, e in rb])
        off += tc.shape[0]
    all_tc = np.vstack(all_tc_parts)

    wta_all, cont_all = _occupancies(all_tc, all_rb, n_components, fo_threshold)
    wta_real_rho = spearmanr(recurrence, wta_all).statistic
    cont_real_rho = spearmanr(recurrence, cont_all).statistic

    # Phase-randomized null distribution for overall pooled arm
    def _null_stats(n_draws):
        wta_null, cont_null = [], []
        for draw in range(n_draws):
            rng_w = np.random.default_rng(draw)
            rng_c = np.random.default_rng(10_000 + draw)
            raw_ph_w = [(f, r, phase_randomize(X, rng_w)) for f, r, X in raw_runs]
            raw_ph_c = [(f, r, phase_randomize(X, rng_c)) for f, r, X in raw_runs]
            w_occ, _ = _pool_occupancy_from_raw(raw_ph_w, proj, n_components, fo_threshold)
            _, c_occ = _pool_occupancy_from_raw(raw_ph_c, proj, n_components, fo_threshold)
            wta_null.append(float(spearmanr(recurrence, w_occ).statistic))
            cont_null.append(float(spearmanr(recurrence, c_occ).statistic))
        return np.array(wta_null), np.array(cont_null)

    def _null_summary(real, null_arr):
        m, s = float(null_arr.mean()), float(null_arr.std())
        z = float((real - m) / s) if s > 0 else float("nan")
        p = float((1 + np.sum(null_arr >= real)) / (1 + len(null_arr)))
        return {"mean": m, "sd": s, "z": z, "p": p,
                "n_draws": int(len(null_arr)), "residual": float(real - m)}

    overall_wta = _spearman(recurrence, wta_all)
    overall_cont = _spearman(recurrence, cont_all)

    if n_null > 0:
        wta_null_arr, cont_null_arr = _null_stats(n_null)
        overall_wta["null"] = _null_summary(wta_real_rho, wta_null_arr)
        overall_cont["null"] = _null_summary(cont_real_rho, cont_null_arr)

    summary = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "vt": vt,
        "stimulus": stimulus,
        "K_active": inp["K_active"],
        "n_components": n_components,
        "fo_threshold": fo_threshold,
        "n_movie_runs": int(sum(len(rb) for _, rb in per_film_tc.values())),
        "friends_recurrence": recurrence.tolist(),
        "movie_occupancy_wta": wta_all.tolist(),
        "movie_occupancy_continuous": cont_all.tolist(),
        "recurrence_vs_friends_marginal_wta_rho": rho_marginal,
        "overall": {
            "wta": overall_wta,
            "continuous": overall_cont,
        },
        "per_film": {},
    }

    for film, (tc, rb) in per_film_tc.items():
        wta_m, cont_m = _occupancies(tc, rb, n_components, fo_threshold)
        summary["per_film"][film] = {
            "wta": _spearman(recurrence, wta_m),
            "continuous": _spearman(recurrence, cont_m),
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "oos_recurrence_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    logger.info(
        "%s: overall WTA rho=%.3f (n=%d); wrote %s",
        sub_id, summary["overall"]["wta"]["rho"], n_components, out_path)
    if n_null > 0:
        null_info = summary["overall"]["wta"]["null"]
        logger.info(
            "%s: WTA null mean=%.3f sd=%.3f z=%.2f p=%.4f residual=%.3f",
            sub_id, null_info["mean"], null_info["sd"],
            null_info["z"], null_info["p"], null_info["residual"])
    return summary


def main():
    p = argparse.ArgumentParser(
        description="ICA out-of-stimulus recurrence (R5 analogue).")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95",
                   help="Variance threshold for n_pcs selection (e.g. 0.95)")
    p.add_argument("--stimulus", default="movie10", choices=["movie10"])
    p.add_argument("--fo_threshold", type=float, default=0.02)
    p.add_argument("--n_null", type=int, default=100,
                   help="Number of phase-randomized null draws (0 to skip).")
    a = p.parse_args()

    out_dir = os.path.join(
        SCRATCH_DIR, "output", "sm_ica_oos_recurrence",
        a.parcellation, a.sub_id)
    run_subject(a.sub_id, a.parcellation, a.vt, a.stimulus, a.fo_threshold,
                out_dir, n_null=a.n_null)


if __name__ == "__main__":
    main()
