#!/usr/bin/env python3
"""
sm_sim_prior_state_count.py - Prior-predictive occupied-state count for the
finite (weak-limit) sticky-HDP-HMM.

What this computes
------------------
A fit-free Monte-Carlo characterization of how many distinct states the model's
*prior alone* favors at the production hyperparameters, before any fMRI data are
observed. No data are read; this is purely a property of the prior.

Faithfulness to the model
-------------------------
The prior construction mirrors the combined HMM exactly (utils/hdphmm.py):

  * Global base measure beta ~ truncated stick-breaking GEM(gamma), renormalized
    (StickyHDPHMM._sample_stick_breaking).
  * Start distribution = beta (StickyHDPHMM._init_params).
  * Each transition row i ~ Dirichlet(alpha * beta + kappa * rho * e_i): the
    HDP base measure plus the sticky self-transition pseudo-count on the
    diagonal. This is the prior whose mean is the initial transition matrix and
    whose pseudo-counts enter the EM transition update
    (effective_counts[i] = E[N(i->.)] + alpha*beta + kappa*rho*e_i).

A state counts as "occupied" when its fractional occupancy exceeds usage_thresh
(default 0.01 = 1%), matching the usage-based active-state definition.

Two estimators (they agree at realistic sequence lengths):
  1. ASYMPTOTIC: occupancy -> stationary distribution of the sampled transition
     matrix; count states with stationary probability > usage_thresh. Exact in
     the long-sequence limit. Primary estimator (fast, many draws).
  2. FINITE-T: simulate Markov chains of a given length (restarting from the
     start distribution at each run boundary) and count usage > usage_thresh.
     Validates estimator 1.

Production hyperparameters (config/combined_hmm_config.py):
  gamma = 1, kappa = 10, alpha = 1, rho = 1, truncation K_max = 50.

Usage
-----
  python script/sm_sim_prior_state_count.py                 # production setting
  python script/sm_sim_prior_state_count.py --gamma 5       # sensitivity
  python script/sm_sim_prior_state_count.py --n_draws 5000 --seed 0

Output JSON goes to $SCRATCH_DIR/prior_state_count/ (or ./output/ if SCRATCH_DIR
is unset); override with --out.
"""
import argparse
import json
import os

import numpy as np


def stick_breaking(gamma, K, rng):
    """Truncated GEM(gamma) base measure, renormalized.

    Mirrors StickyHDPHMM._sample_stick_breaking in utils/hdphmm.py.
    """
    betas = rng.beta(1.0, gamma, size=K)
    w = np.zeros(K)
    stick = 1.0
    eps = np.finfo(float).eps
    for i in range(K):
        draw = min(betas[i], 1.0 - eps)
        w[i] = stick * draw
        stick *= (1.0 - draw)
        if stick < eps:
            break
    s = w.sum()
    return w / s if s > eps else np.ones(K) / K


def mean_transition(beta, kappa, alpha, rho):
    """Deterministic prior-mean transition matrix: row i = (alpha*beta + kappa*rho*e_i) normalized."""
    K = beta.shape[0]
    trans = np.tile(alpha * beta, (K, 1))
    trans[np.arange(K), np.arange(K)] += kappa * rho
    return trans / trans.sum(axis=1, keepdims=True)


def sample_prior(gamma, kappa, alpha, rho, K, rng):
    """Draw beta, start distribution, and a sampled transition matrix from the prior."""
    beta = stick_breaking(gamma, K, rng)
    startprob = beta.copy()
    trans = np.zeros((K, K))
    for i in range(K):
        conc = alpha * beta.copy()
        conc[i] += kappa * rho
        # A zero-concentration coordinate (unused truncation slot, beta_j = 0) is a
        # degenerate Dirichlet coordinate with exactly zero mass; sample over the
        # positive support and leave the rest at zero.
        support = conc > 0
        trans[i, support] = rng.dirichlet(conc[support])
    return beta, startprob, trans


def stationary(trans):
    """Stationary distribution (left eigenvector for eigenvalue 1)."""
    vals, vecs = np.linalg.eig(trans.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:, idx])
    v = np.abs(v)
    return v / v.sum()


def simulate_chain_usage(startprob, trans, T, n_runs, rng):
    """Finite-T occupancy: n_runs runs of length ~T/n_runs, each restarting from startprob.

    Actual simulated length is (T // n_runs) * n_runs, which may be slightly below T
    (integer division); negligible at the validation lengths used here.
    """
    K = trans.shape[0]
    counts = np.zeros(K)
    cdf_start = np.cumsum(startprob)
    cdf_trans = np.cumsum(trans, axis=1)
    run_len = max(1, T // n_runs)
    for _ in range(n_runs):
        s = int(np.searchsorted(cdf_start, rng.random_sample()))
        for _t in range(run_len):
            counts[s] += 1
            s = int(np.searchsorted(cdf_trans[s], rng.random_sample()))
    return counts / counts.sum()


def _summary(a):
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "sd": float(np.std(a, ddof=1)),
        "p2.5": float(np.percentile(a, 2.5)),
        "p97.5": float(np.percentile(a, 97.5)),
        "min": int(np.min(a)),
        "max": int(np.max(a)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=50, help="truncation capacity K_max")
    ap.add_argument("--usage_thresh", type=float, default=0.01,
                    help="fractional-occupancy threshold for 'occupied' (1%%)")
    ap.add_argument("--n_draws", type=int, default=3000,
                    help="prior draws for the asymptotic estimator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_lengths", type=int, nargs="*", default=[91547, 137913],
                    help="sequence lengths (TRs) for finite-T validation; empty to skip")
    ap.add_argument("--val_n_runs", type=int, nargs="*", default=[194, 292],
                    help="run counts matching --val_lengths")
    ap.add_argument("--val_chains", type=int, default=5,
                    help="chains per length for finite-T validation (0 to skip)")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON path (default: $SCRATCH_DIR/prior_state_count/"
                         "prior_state_count.json, or ./output/... if unset)")
    args = ap.parse_args()

    if args.out is None:
        # This sim is fit-free and stateless (reads no pipeline outputs), so unlike the
        # data-dependent scripts it tolerates an unset SCRATCH_DIR and falls back to ./output.
        base = os.getenv("SCRATCH_DIR") or "output"
        args.out = os.path.join(base, "prior_state_count", "prior_state_count.json")

    rng = np.random.RandomState(args.seed)  # legacy API for portability (numpy < 1.17)

    # --- Estimator 1: asymptotic (stationary-based), many draws ---
    active_counts = np.empty(args.n_draws, dtype=int)
    n_beta_above = np.empty(args.n_draws, dtype=int)
    for d in range(args.n_draws):
        beta, _start, trans = sample_prior(args.gamma, args.kappa, args.alpha,
                                            args.rho, args.K, rng)
        pi = stationary(trans)
        active_counts[d] = int((pi > args.usage_thresh).sum())
        n_beta_above[d] = int((beta > args.usage_thresh).sum())

    result = {
        "hyperparameters": {
            "gamma": args.gamma, "kappa": args.kappa, "alpha": args.alpha,
            "rho": args.rho, "K_max": args.K, "usage_thresh": args.usage_thresh,
        },
        "n_draws": args.n_draws,
        "seed": args.seed,
        "asymptotic_active_count": _summary(active_counts),
        "beta_components_above_thresh": _summary(n_beta_above),
    }

    # --- Estimator 2: finite-T Monte-Carlo validation ---
    val = []
    if args.val_chains and args.val_lengths:
        for T, n_runs in zip(args.val_lengths, args.val_n_runs):
            chain_counts = []
            for _c in range(args.val_chains):
                _beta, startprob, trans = sample_prior(args.gamma, args.kappa,
                                                       args.alpha, args.rho, args.K, rng)
                usage = simulate_chain_usage(startprob, trans, T, n_runs, rng)
                chain_counts.append(int((usage > args.usage_thresh).sum()))
            val.append({"T": T, "n_runs": n_runs, "chains": args.val_chains,
                        "active_count_mean": float(np.mean(chain_counts)),
                        "active_count_values": chain_counts})
    result["finite_T_validation"] = val

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    a = result["asymptotic_active_count"]
    print("=" * 64)
    print("Prior-predictive occupied-state count (sticky-HDP-HMM prior)")
    print(f"  gamma={args.gamma} kappa={args.kappa} alpha={args.alpha} "
          f"rho={args.rho} K_max={args.K} thresh={args.usage_thresh:.0%}")
    print(f"  draws={args.n_draws} seed={args.seed}")
    print("-" * 64)
    print(f"  occupied states (usage > {args.usage_thresh:.0%}):")
    print(f"    mean   {a['mean']:.1f}")
    print(f"    median {a['median']:.0f}")
    print(f"    95% interval [{a['p2.5']:.0f}, {a['p97.5']:.0f}]   range [{a['min']}, {a['max']}]")
    if val:
        print("-" * 64)
        print("  finite-T validation:")
        for v in val:
            print(f"    T={v['T']:>7} runs={v['n_runs']:>4}  "
                  f"occupied~{v['active_count_mean']:.1f}  {v['active_count_values']}")
    print("=" * 64)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
