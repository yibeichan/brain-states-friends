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

The ICA analogue of this null is produced by sm_alt_ica_oos_recurrence.py,
which calls the same phase_randomize helper; Harry Potter and Le Petit Prince
are now covered on the HMM side too (--stimulus), not just for Movie10.

Inputs (frozen, from the main pipeline, all under $SCRATCH_DIR/output):
    04_combined_hdphmm/{parc}/{sub}/final/vt{vt}/best_model.pkl
    03a_pca4combined_hmm/{parc}/{sub}/n_pcs_lookup.json
    05a_recurrence_analysis/{parc}/{sub}/vt{vt}/recurrence_scores.npy
    m10_03_projected/{parc}/{sub}/vt{vt}/{movie_run_ids.json, {run_id}.npy}
    m10_05_cross_validation/{parc}/{sub}/vt{vt}/cross_stimulus_summary.json
    hp_03_projected/{parc}/{sub}/vt{vt}/{hp_run_ids.json, {run_id}.npy}
    hp_05_cross_validation/{parc}/{sub}/vt{vt}/cross_stimulus_summary.json
    pp_03_projected/{parc}/{sub}/vt{vt}/{pp_run_ids.json, {run_id}.npy}
    pp_05_cross_validation/{parc}/{sub}/vt{vt}/cross_stimulus_summary.json

Output:
    Movie10 (legacy path, unchanged):
        {SCRATCH_DIR}/output/sm_rel_r5_phase_null/{parc}/{sub}/vt{vt}/r5_phase_null_summary.json
    Harry Potter / Le Petit Prince (nested under the stimulus name):
        {SCRATCH_DIR}/output/sm_rel_r5_phase_null/{stimulus}/{parc}/{sub}/vt{vt}/r5_phase_null_summary.json
    A subject with no data for the requested stimulus (e.g. sub-04 on HP/PP)
    gets a clean skip: {out_dir}/skipped.json instead, exit code 0.

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
from utils.stats import null_summary, safe_float

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


def occupancy_stats(paths, n_states, recurrence, active, groups=None):
    """Spearman(recurrence, mean FO) over `active`, overall and per run-group.

    `paths` are Viterbi paths in run order; `groups` maps a group name to the
    indices of its runs. Returns (overall SpearmanResult, {group: rho} or None).
    """
    occ = mean_fractional_occupancy(paths, n_states)
    overall = spearmanr(recurrence[active], occ[active])
    if not groups:
        return overall, None
    by_group = {}
    for g, idx in groups.items():
        occ_g = mean_fractional_occupancy([paths[i] for i in idx], n_states)
        by_group[g] = float(spearmanr(recurrence[active], occ_g[active]).statistic)
    return overall, by_group


def default_out_dir(stimulus, parcellation, sub_id, vt):
    """Published Movie10 path is kept verbatim; other stimuli nest under their name."""
    root = os.path.join(SCRATCH_DIR, "output", "sm_rel_r5_phase_null")
    if stimulus != "movie10":
        root = os.path.join(root, stimulus)
    return os.path.join(root, parcellation, sub_id, f"vt{vt}")


def write_skip(out_dir, sub_id, stimulus, reason):
    """Record a clean skip so a SLURM array over six subjects has one file per cell."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "skipped.json")
    with open(path, "w") as f:
        json.dump({"sub_id": sub_id, "stimulus": stimulus, "skipped": True,
                   "reason": reason}, f, indent=2)
    return path


class NoStimulusDataError(Exception):
    """No projected data or cross-stimulus summary for (subject, stimulus).

    Signals a CLEAN skip (e.g. sub-04 has no Harry Potter / Petit Prince
    scans), distinct from a genuine pipeline error such as a missing run
    inside an existing projection, which must still hard-fail.
    """


STIMULI = {
    "movie10": {"proj_dir": "m10_03_projected", "run_ids_file": "movie_run_ids.json",
                "summary_dir": "m10_05_cross_validation", "n_runs_key": "n_movie_runs"},
    "harrypotter": {"proj_dir": "hp_03_projected", "run_ids_file": "hp_run_ids.json",
                    "summary_dir": "hp_05_cross_validation", "n_runs_key": "n_hp_runs"},
    "petitprince": {"proj_dir": "pp_03_projected", "run_ids_file": "pp_run_ids.json",
                    "summary_dir": "pp_05_cross_validation", "n_runs_key": "n_pp_runs"},
}


def output_base():
    """Root of the frozen main-pipeline outputs."""
    return os.path.join(SCRATCH_DIR, "output")


def _proj_dir(sub_id, parcellation, vt, stimulus, base):
    return os.path.join(base, STIMULI[stimulus]["proj_dir"], parcellation, sub_id, f"vt{vt}")


def _summary_path(sub_id, parcellation, vt, stimulus, base):
    return os.path.join(base, STIMULI[stimulus]["summary_dir"], parcellation, sub_id,
                        f"vt{vt}", "cross_stimulus_summary.json")


def ensure_stimulus_available(sub_id, parcellation, vt, stimulus, base=None):
    """Raise NoStimulusDataError unless both the projection and the summary exist.

    Called before any model loading so a subject who never did the stimulus
    exits cleanly instead of failing on an unrelated input.
    """
    base = base or output_base()
    pdir = _proj_dir(sub_id, parcellation, vt, stimulus, base)
    if not os.path.isdir(pdir):
        raise NoStimulusDataError(f"{sub_id}: no projected {stimulus} data at {pdir}")
    spath = _summary_path(sub_id, parcellation, vt, stimulus, base)
    if not os.path.exists(spath):
        raise NoStimulusDataError(f"{sub_id}: no {stimulus} cross-stimulus summary at {spath}")


def load_stimulus_runs(sub_id, parcellation, vt, n_pcs, stimulus, base=None):
    """Load one stimulus's projected PC-score runs, truncated to n_pcs.

    Returns (runs, groups): `runs` is a list of (T, n_pcs) arrays in registry
    order; `groups` maps each run-group name (film, language, ...) to the
    indices of its runs within `runs`.
    """
    spec = STIMULI[stimulus]
    base = base or output_base()
    mdir = _proj_dir(sub_id, parcellation, vt, stimulus, base)
    if not os.path.isdir(mdir):
        raise NoStimulusDataError(f"{sub_id}: no projected {stimulus} data at {mdir}")
    with open(os.path.join(mdir, spec["run_ids_file"])) as f:
        run_ids = json.load(f)
    runs, groups = [], {}
    for group, ids in run_ids.items():
        groups[group] = []
        for rid in ids:
            path = os.path.join(mdir, f"{rid}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Projected {stimulus} run missing: {path} (run {rid}); "
                    "a partial projection output would silently change the null's run ensemble")
            arr = np.load(path)
            if arr.shape[1] < n_pcs:
                raise ValueError(f"{path}: expected >= {n_pcs} columns, found {arr.shape[1]}")
            arr = arr[:, :n_pcs]
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}: non-finite values in projected run")
            groups[group].append(len(runs))
            runs.append(arr)
    if not runs:
        raise FileNotFoundError(f"No projected {stimulus} runs found under {mdir}")
    return runs, groups


def load_movie_runs(sub_id, parcellation, vt, n_pcs):
    """Backward-compatible Movie10 loader (runs only, no group map)."""
    return load_stimulus_runs(sub_id, parcellation, vt, n_pcs, "movie10")[0]


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


def published_reference(sub_id, parcellation, vt, stimulus="movie10", base=None):
    """Published A1 rho plus the input counts the standalone rebuild must match."""
    spec = STIMULI[stimulus]
    base = base or output_base()
    path = _summary_path(sub_id, parcellation, vt, stimulus, base)
    if not os.path.exists(path):
        raise NoStimulusDataError(f"{sub_id}: no {stimulus} cross-stimulus summary at {path}")
    with open(path) as f:
        j = json.load(f)
    rho = j["A1_recurrence_correlation"]["spearman_rho"]
    if rho is None or not np.isfinite(float(rho)):
        raise RuntimeError(f"{sub_id}: published spearman_rho is degenerate ({rho!r})")
    return {"rho": float(rho), "n_runs": int(j[spec["n_runs_key"]]),
            "n_active_states": int(j["A1_recurrence_correlation"]["n_active_states"])}


def normalize_vt(vt):
    """Canonical two-decimal vt string ('0.95'): lookup keys and vt{...} dirs use it."""
    return f"{float(vt):.2f}"


def build_parser():
    p = argparse.ArgumentParser(
        description="Phase-randomized null for the recurrence-occupancy correlation "
                    "(R5: Movie10; extension: Harry Potter, Le Petit Prince).")
    p.add_argument("--sub_id", required=True)
    p.add_argument("--parcellation", default="atlas-4S156Parcels")
    p.add_argument("--vt", default="0.95",
                   help="Variance threshold for n_pcs selection (e.g. 0.95)")
    p.add_argument("--stimulus", default="movie10", choices=sorted(STIMULI),
                   help="Held-out stimulus whose PC scores are phase-randomized.")
    p.add_argument("--per_group", action="store_true",
                   help="Also record rho per run-group (film / language) for every draw.")
    p.add_argument("--n_null", type=int, default=10000,
                   help="Number of phase-randomized surrogate draws "
                        "(10000 = the published run).")
    p.add_argument("--gate_tol", type=float, default=1e-12,
                   help="Max |observed - published| rho before aborting.")
    p.add_argument("--out_dir", default=None,
                   help="Override output directory. Use for test/exploratory runs "
                        "so the published outputs are not overwritten in place.")
    return p


def run_subject(sub_id, parcellation, vt, n_null, out_dir, gate_tol,
                stimulus="movie10", per_group=False):
    """Observed rho and its phase-randomized null for one subject and stimulus."""
    ensure_stimulus_available(sub_id, parcellation, vt, stimulus)
    base = output_base()
    model_path = os.path.join(base, "04_combined_hdphmm", parcellation, sub_id,
                              "final", f"vt{vt}", "best_model.pkl")
    model = _load_model_no_jax(model_path)
    pca_base = os.path.join(base, "03a_pca4combined_hmm", parcellation, sub_id)
    n_pcs = int(hmm_io.load_n_pcs_lookup(pca_base)[str(vt)])
    n_states = int(model.n_components)

    recurrence = np.load(os.path.join(base, "05a_recurrence_analysis", parcellation,
                                      sub_id, f"vt{vt}", "recurrence_scores.npy"))
    active = np.where(recurrence > 0)[0]
    runs, groups = load_stimulus_runs(sub_id, parcellation, vt, n_pcs, stimulus)
    group_names = list(groups) if per_group else []
    means, var, log_start, log_trans = hmm_params(model, n_pcs)

    def decode(run_list):
        return [viterbi(x, means, var, log_start, log_trans) for x in run_list]

    observed_res, observed_groups = occupancy_stats(
        decode(runs), n_states, recurrence, active, groups if per_group else None)
    observed, observed_p = observed_res.statistic, observed_res.pvalue

    ref = published_reference(sub_id, parcellation, vt, stimulus)
    if len(runs) != ref["n_runs"] or len(active) != ref["n_active_states"]:
        raise RuntimeError(
            f"{sub_id}/{stimulus}: input drift vs published run: loaded {len(runs)} runs / "
            f"{len(active)} active states, published {ref['n_runs']} / "
            f"{ref['n_active_states']}")
    gate = check_gate(observed, ref["rho"], gate_tol, sub_id)

    logger.info("%s/%s: observed rho=%+.4f (gate |delta|=%.2e, %d active states, "
                "%d runs, %d PCs) - drawing %d surrogates",
                sub_id, stimulus, observed, gate, len(active), len(runs), n_pcs, n_null)

    t0 = time.time()
    null = np.empty(n_null)
    null_groups = np.empty((n_null, len(group_names))) if per_group else None
    for s in range(n_null):
        rng = np.random.default_rng(SEED_BASE + s)
        res, by_group = occupancy_stats(decode([phase_randomize(x, rng) for x in runs]),
                                        n_states, recurrence, active,
                                        groups if per_group else None)
        null[s] = res.statistic
        if per_group:
            null_groups[s] = [by_group[g] for g in group_names]
        if (s + 1) % 100 == 0:
            logger.info("%s/%s: %d/%d draws (%.1f s elapsed)",
                        sub_id, stimulus, s + 1, n_null, time.time() - t0)

    ns = null_summary(observed, null)
    if ns["n_finite"] < n_null:
        logger.warning("%s: %d/%d null draws were non-finite and are excluded "
                       "from moments and p", sub_id, n_null - ns["n_finite"], n_null)
    lo, hi = np.percentile(null[np.isfinite(null)], [2.5, 97.5])
    summary = {
        "sub_id": sub_id,
        "parcellation": parcellation,
        "vt": float(vt),
        "stimulus": stimulus,
        "n_states_total": n_states,
        "n_states_active": int(len(active)),
        "n_pcs": n_pcs,
        "n_runs": len(runs),
        **({"n_movie_runs": len(runs)} if stimulus == "movie10" else {}),
        "observed": {"rho": safe_float(observed), "p": safe_float(observed_p),
                     **({"rho_by_group": {g: safe_float(v) for g, v in observed_groups.items()}}
                        if per_group else {})},
        "gate": {"published_rho": safe_float(ref["rho"]),
                 "abs_delta": safe_float(gate),
                 "tolerance": gate_tol},
        "null": {
            "kind": "phase_randomized_prichard_theiler_shared_phase",
            "preserves": ["per-component power spectrum",
                          "per-component variance",
                          "cross-component covariance",
                          "run count", "run lengths"],
            "destroys": ["stimulus-locked phase alignment",
                         "higher-order temporal structure"],
            "randomized_side": f"{stimulus}_pc_scores",
            "n_draws": int(n_null),
            "n_finite": ns["n_finite"],
            "seed_base": SEED_BASE,
            "seed_rule": "draw s uses numpy.random.default_rng(seed_base + s)",
            "mean": ns["mean"],
            "sd": ns["sd"],
            "pct2.5": safe_float(lo),
            "pct97.5": safe_float(hi),
        },
        "delta_rho": ns["residual"],
        "z": ns["z"],
        "p_empirical": ns["p"],
        "p_floor": safe_float(1.0 / (1 + ns["n_finite"])) if ns["n_finite"] else None,
        "null_share_of_observed": safe_float(
            ns["mean"] / observed if (ns["mean"] is not None and observed) else float("nan")),
        **({"groups": group_names} if per_group else {}),
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
    if per_group:
        np.save(os.path.join(out_dir, "null_draws_by_group.npy"), null_groups)

    def _fmt(v, spec):
        return format(v, spec) if v is not None else "n/a"

    logger.info("%s/%s: null mean=%s sd=%s | delta_rho=%s z=%s p=%s "
                "| null is %s%% of observed -> %s",
                sub_id, stimulus, _fmt(ns["mean"], "+.4f"), _fmt(ns["sd"], ".4f"),
                _fmt(ns["residual"], "+.4f"), _fmt(ns["z"], "+.2f"),
                _fmt(ns["p"], ".4f"),
                _fmt(100 * summary["null_share_of_observed"]
                     if summary["null_share_of_observed"] is not None else None, ".0f"),
                out_path)
    return summary


def main():
    a = build_parser().parse_args()
    vt = normalize_vt(a.vt)
    out_dir = a.out_dir or default_out_dir(a.stimulus, a.parcellation, a.sub_id, vt)
    try:
        run_subject(a.sub_id, a.parcellation, vt, a.n_null, out_dir, a.gate_tol,
                    stimulus=a.stimulus, per_group=a.per_group)
    except NoStimulusDataError as e:
        logger.warning("%s; skipping %s/%s", e, a.sub_id, a.stimulus)
        write_skip(out_dir, a.sub_id, a.stimulus, str(e))
        sys.exit(0)


if __name__ == "__main__":
    main()
