#!/usr/bin/env python3
"""sm_rel_r5_phase_null.py - phase-randomized null for the R5 recurrence-occupancy correlation.

R5 reports, per subject, Spearman rho between each state's Friends recurrence
score and its mean fractional occupancy when the frozen Friends PCA+HMM is
decoded on Movie10. Both sides inherit low-level temporal structure, so a raw
rho overstates the stimulus-specific correspondence. This script tests the
observed rho against a phase-randomized (Prichard-Theiler) surrogate null.

Surrogate construction (utils.ica_oos_recurrence.phase_randomize): each Movie10
run's PC-score matrix is Fourier-transformed along time and multiplied by ONE
random phase vector shared across all principal components at each frequency
(DC and, for even T, Nyquist held real). A phase rotation common to every
component preserves the full cross-spectrum, so each component's power spectrum
(hence its autocorrelation) AND the cross-component covariance survive exactly;
only stimulus-specific phase and higher-order structure is destroyed. The
Friends side (recurrence scores) is never resampled.

What the null therefore preserves / destroys:
    preserved  - per-component power spectrum, per-component variance,
                 cross-component covariance, run count, run lengths
    destroyed  - stimulus-locked phase alignment and all higher-order structure

Faithfulness gate: the standalone diagonal-Gaussian Viterbi implemented here is
rebuilt from the saved model parameters, so it must reproduce the published R5
rho exactly. The run aborts if it does not, because a null is only meaningful
if it acts on the same statistic the manuscript reports.

Seeds are deterministic and derived from the draw index (draw s uses
np.random.default_rng(SEED_BASE + s)), so a longer run is a strict superset of
a shorter one and any draw can be reproduced individually.

The ICA analogue of this null is not computed here; it is already produced by
sm_alt_ica_oos_recurrence.py, which calls the same phase_randomize helper.

Inputs (frozen, from the main pipeline, all under $SCRATCH_DIR/output):
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/best_model.pkl
    03a_pca4combined_hmm/{parc}/{sub}/n_pcs_lookup.json
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/recurrence_scores.npy
    m10_03_projected/{parc}/{sub}/vt{vt}/{movie_run_ids.json, {run_id}.npy}
    m10_05_cross_validation/{parc}/{sub}/vt{vt}/cross_stimulus_summary.json

Output:
    {SCRATCH_DIR}/output/sm_rel_r5_phase_null/{parc}/{sub}/vt{vt}/r5_phase_null_summary.json

Per-subject inference only; no statistic is pooled across subjects.
"""
import os
import sys
import json
import time
import logging
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils import hmm_io
from utils.jax_free_model_io import _load_model_no_jax
from utils.ica_oos_recurrence import phase_randomize

load_dotenv()
SCRATCH_DIR = os.getenv("SCRATCH_DIR")
if SCRATCH_DIR is None:
    raise ValueError("SCRATCH_DIR must be set in environment or .env file")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED_BASE = 0
VAR_FLOOR = 1e-12
LOG_FLOOR = 1e-300


def _safe_float(x):
    """Convert to float, mapping NaN/inf -> None (json.dump uses allow_nan=False)."""
    x = float(x)
    return x if np.isfinite(x) else None


def hmm_params(model, n_pcs):
    """Extract diagonal-Gaussian emission and transition parameters in PC space.

    Preconditions: model.covariance_type must be "diag", and model.means_.shape[1]
    must equal n_pcs (no truncation or padding).

    covars_ is stored either as (K, n_pcs) diagonals or as (K, n_pcs, n_pcs)
    full matrices whose diagonal is the fitted variance; both are reduced to
    (K, n_pcs) variances here and floored to keep the log-density finite.
    """
    cov_type = getattr(model, "covariance_type", None)
    if cov_type != "diag":
        raise ValueError(
            f"covariance_type={cov_type!r}: this standalone decoder reduces "
            "covariances to diagonals, which is only valid for 'diag' fits")
    if n_pcs != np.asarray(model.means_).shape[1]:
        raise ValueError(
            f"n_pcs lookup says {n_pcs} but model means have "
            f"{np.asarray(model.means_).shape[1]} dims; refusing to decode in a mismatched subspace")
    means = np.asarray(model.means_)[:, :n_pcs]
    cov = np.asarray(model.covars_)
    if cov.ndim == 3:
        cov = np.stack([np.diag(c) for c in cov])
    var = np.maximum(cov[:, :n_pcs], VAR_FLOOR)
    log_start = np.log(np.asarray(model.startprob_) + LOG_FLOOR)
    log_trans = np.log(np.asarray(model.transmat_) + LOG_FLOOR)
    return means, var, log_start, log_trans


def viterbi(X, means, var, log_start, log_trans):
    """Most probable state path for one run under a diagonal-Gaussian HMM.

    The emission log-density is expanded rather than looped per state:
    -0.5 * sum_d ((x_d - mu_kd)^2 / var_kd) + const_k factorizes into three
    matrix products, which keeps the per-run cost dominated by the T x K x K
    transition recursion below.
    """
    inv = 1.0 / var
    const = -0.5 * np.log(2 * np.pi * var).sum(1)
    quad_mean = (means * means * inv).sum(1)
    log_emit = -0.5 * ((X * X) @ inv.T - 2 * (X @ (means * inv).T)
                       + quad_mean[None, :]) + const[None, :]

    n_t, n_k = log_emit.shape
    score = np.empty((n_t, n_k))
    back = np.empty((n_t, n_k), dtype=int)
    score[0] = log_start + log_emit[0]
    for t in range(1, n_t):
        m = score[t - 1][:, None] + log_trans
        back[t] = m.argmax(0)
        score[t] = m.max(0) + log_emit[t]

    path = np.empty(n_t, dtype=int)
    path[-1] = score[-1].argmax()
    for t in range(n_t - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def mean_fractional_occupancy(paths, n_states):
    """Per-state fractional occupancy averaged over runs (runs weighted equally)."""
    per_run = np.vstack([np.bincount(p, minlength=n_states) / len(p) for p in paths])
    return per_run.mean(0)


def load_movie_runs(sub_id, parcellation, vt, n_pcs):
    """Load Movie10 PC-score matrices, truncated to the subject's n_pcs."""
    mdir = os.path.join(SCRATCH_DIR, "output", "m10_03_projected",
                        parcellation, sub_id, f"vt{vt}")
    with open(os.path.join(mdir, "movie_run_ids.json")) as f:
        run_ids = json.load(f)
    runs = []
    for ids in run_ids.values():
        for rid in ids:
            path = os.path.join(mdir, f"{rid}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Projected Movie10 run missing: {path} (run {rid}); "
                    "a partial m10_03 output would silently change the null's run ensemble")
            arr = np.load(path)
            if arr.shape[1] < n_pcs:
                raise ValueError(
                    f"{path}: expected >= {n_pcs} columns, found {arr.shape[1]}")
            arr = arr[:, :n_pcs]
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}: non-finite values in projected run")
            runs.append(arr)
    if not runs:
        raise FileNotFoundError(f"No projected Movie10 runs found under {mdir}")
    return runs


def check_gate(observed, reference, gate_tol, sub_id):
    """Fail closed: observed must be finite and within gate_tol of reference."""
    if not np.isfinite(observed):
        raise RuntimeError(
            f"{sub_id}: standalone Viterbi rho is not finite ({observed!r}); "
            "refusing to draw a null for a degenerate statistic")
    gate = abs(observed - reference)
    if not (gate <= gate_tol):
        raise RuntimeError(
            f"{sub_id}: standalone Viterbi rho={observed:.6f} does not reproduce "
            f"the published R5 rho={reference:.6f} (|delta|={gate:.2e} > {gate_tol:.0e}). "
            "Refusing to report a null for a statistic the manuscript does not use.")
    return gate


def published_reference(sub_id, parcellation, vt):
    """Published R5 rho plus the input counts the standalone rebuild must match."""
    path = os.path.join(SCRATCH_DIR, "output", "m10_05_cross_validation",
                        parcellation, sub_id, f"vt{vt}", "cross_stimulus_summary.json")
    with open(path) as f:
        j = json.load(f)
    rho = j["A1_recurrence_correlation"]["spearman_rho"]
    if rho is None or not np.isfinite(float(rho)):
        raise RuntimeError(f"{sub_id}: published spearman_rho is degenerate ({rho!r})")
    return {"rho": float(rho), "n_movie_runs": int(j["n_movie_runs"]),
            "n_active_states": int(j["A1_recurrence_correlation"]["n_active_states"])}


def run_subject(sub_id, parcellation, vt, n_null, out_dir, gate_tol):
    """Compute the observed R5 rho and its phase-randomized null for one subject."""
    base = os.path.join(SCRATCH_DIR, "output")
    model_path = os.path.join(base, "04_combined_hdphmm", parcellation, sub_id,
                              "final", f"vt{vt}", "best_model.pkl")
    model = _load_model_no_jax(model_path)
    pca_base = os.path.join(base, "03a_pca4combined_hmm", parcellation, sub_id)
    n_pcs = int(hmm_io.load_n_pcs_lookup(pca_base)[str(vt)])
    n_states = int(model.n_components)

    recurrence = np.load(os.path.join(base, "05a_recurrence_analysis", parcellation,
                                      sub_id, f"vt{vt}", "recurrence_scores.npy"))
    active = np.where(recurrence > 0)[0]
    runs = load_movie_runs(sub_id, parcellation, vt, n_pcs)
    means, var, log_start, log_trans = hmm_params(model, n_pcs)

    def rho_for(run_list):
        paths = [viterbi(x, means, var, log_start, log_trans) for x in run_list]
        occ = mean_fractional_occupancy(paths, n_states)
        return spearmanr(recurrence[active], occ[active])

    observed, observed_p = rho_for(runs)

    # Faithfulness gate: this standalone Viterbi must reproduce the published
    # statistic, or the null is testing a different quantity than R5 reports.
    ref = published_reference(sub_id, parcellation, vt)
    if len(runs) != ref["n_movie_runs"] or len(active) != ref["n_active_states"]:
        raise RuntimeError(
            f"{sub_id}: input drift vs published run: loaded {len(runs)} runs / "
            f"{len(active)} active states, published {ref['n_movie_runs']} / "
            f"{ref['n_active_states']}")
    gate = check_gate(observed, ref["rho"], gate_tol, sub_id)

    logger.info("%s: observed rho=%+.4f (gate |delta|=%.2e, %d active states, "
                "%d Movie10 runs, %d PCs) - drawing %d surrogates",
                sub_id, observed, gate, len(active), len(runs), n_pcs, n_null)

    t0 = time.time()
    null = np.empty(n_null)
    for s in range(n_null):
        rng = np.random.default_rng(SEED_BASE + s)
        null[s] = rho_for([phase_randomize(x, rng) for x in runs]).statistic
        if (s + 1) % 100 == 0:
            logger.info("%s: %d/%d draws (%.1f s elapsed)",
                        sub_id, s + 1, n_null, time.time() - t0)

    null_mean, null_sd = float(null.mean()), float(null.std())
    lo, hi = np.percentile(null, [2.5, 97.5])
    summary = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "vt": float(vt),
        "n_states_total": n_states,
        "n_states_active": int(len(active)),
        "n_pcs": n_pcs,
        "n_movie_runs": len(runs),
        "observed": {"rho": _safe_float(observed), "p": _safe_float(observed_p)},
        "gate": {"published_rho": _safe_float(ref["rho"]),
                 "abs_delta": _safe_float(gate),
                 "tolerance": gate_tol},
        "null": {
            "kind": "phase_randomized_prichard_theiler_shared_phase",
            "preserves": ["per-component power spectrum",
                          "per-component variance",
                          "cross-component covariance",
                          "run count", "run lengths"],
            "destroys": ["stimulus-locked phase alignment",
                         "higher-order temporal structure"],
            "randomized_side": "movie10_pc_scores",
            "n_draws": int(n_null),
            "seed_base": SEED_BASE,
            "seed_rule": "draw s uses numpy.random.default_rng(seed_base + s)",
            "mean": _safe_float(null_mean),
            "sd": _safe_float(null_sd),
            "pct2.5": _safe_float(lo),
            "pct97.5": _safe_float(hi),
        },
        "delta_rho": _safe_float(observed - null_mean),
        "z": _safe_float((observed - null_mean) / null_sd if null_sd > 0 else np.nan),
        "p_empirical": _safe_float((1 + np.sum(null >= observed)) / (1 + n_null)),
        "p_floor": _safe_float(1.0 / (1 + n_null)),
        "null_share_of_observed": _safe_float(null_mean / observed if observed else np.nan),
        "runtime_s": round(time.time() - t0, 1),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {"model": model_path, "recurrence_dir": os.path.join(
            base, "05a_recurrence_analysis", parcellation, sub_id, f"vt{vt}")},
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "r5_phase_null_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    np.save(os.path.join(out_dir, "null_draws.npy"), null)

    logger.info("%s: null mean=%+.4f sd=%.4f | delta_rho=%+.4f z=%+.2f p=%.4f "
                "| null is %.0f%% of observed -> %s",
                sub_id, null_mean, null_sd, summary["delta_rho"], summary["z"],
                summary["p_empirical"], 100 * summary["null_share_of_observed"], out_path)
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Phase-randomized null for the R5 recurrence-occupancy correlation.")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95",
                   help="Variance threshold for n_pcs selection (e.g. 0.95)")
    p.add_argument("--n_null", type=int, default=1000,
                   help="Number of phase-randomized surrogate draws.")
    p.add_argument("--gate_tol", type=float, default=1e-12,
                   help="Max |observed - published| rho before aborting.")
    a = p.parse_args()

    out_dir = os.path.join(SCRATCH_DIR, "output", "sm_rel_r5_phase_null",
                           a.parcellation, a.sub_id, f"vt{a.vt}")
    run_subject(a.sub_id, a.parcellation, a.vt, a.n_null, out_dir, a.gate_tol)


if __name__ == "__main__":
    main()
