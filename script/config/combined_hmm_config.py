#!/usr/bin/env python3
"""
combined_hmm_config.py - Hyperparameter grid for the combined cross-season weak-limit HMM.

Two-stage model selection (2026-03-13 improvement):

  Stage 1 - Select variance threshold (PCA dimension):
    Metric: per-dimension LL = valid_ll_per_sample / n_pcs.
    Fixed: nc=60, gamma=5, cov=diag, kappa=10, alpha=1, rho=1.
    Sweep: vt ∈ {0.80, 0.85, 0.90, 0.95, 0.99}.
    → 5 configs × 5 seeds = 25 fits per subject.

  Stage 2 - Select K, gamma, and covariance type:
    Metric: BIC with effective K (training LL, complexity penalty).
    Fixed: vt from Stage 1.
    Sweep: nc ∈ {40, 60, 80, 100} × gamma ∈ {1, 5, 10} × cov ∈ {full, diag}
           (cov=full restricted to low-dimensional vt per COVARIANCE_RULES).
    → ~18-24 configs × 5 seeds = 90-120 fits per subject.

  Final refit: best config from Stage 2, train+valid, 10 seeds, test eval.
"""

# =============================================================================
# Constants
# =============================================================================

VARIANCE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]

COVARIANCE_RULES = {
    0.80: ['full', 'diag'],   # Full ratio 12.6-18.9; diagonal 69-104
    0.85: ['full', 'diag'],   # Full ratio 6.1-9.1; diagonal 47-71
    0.90: ['diag'],           # Full ratio 2.8-4.2 (marginal); diagonal 31-47
    0.95: ['diag'],           # Full ratio <2 (not viable); diagonal 18-27
    0.99: ['diag'],           # Full ratio <1 (not viable); diagonal 11-16
}

# Stage 2 grid dimensions
N_COMPONENTS_OPTIONS = [20, 40, 50, 60, 80, 100]
GAMMA_OPTIONS = [1, 5, 10]

# Stage 1: fixed hyperparameters for vt sweep
STAGE1_FIXED = {
    'n_components': 60,
    'gamma': 5,
    'covariance_type': 'diag',
}

# Parameters shared across both stages (not grid-searched)
FIXED_PARAMS = {
    'alpha': 1.0,
    'kappa': 10,
    'rho': 1,
    'n_iter': 10000,
    'min_iter': 1000,
    'tol': 1e-6,
    'min_state_usage': 0.01,
}

# Expected grid sizes (verified by assertion in each builder)
N_CONFIGS_STAGE1 = 5   # len(VARIANCE_THRESHOLDS)
# Stage 2 count depends on selected_vt; see build_stage2_grid()

# Legacy constant - matches old 28-config grid (gamma=10 fixed, no gamma sweep).
# Kept for backward compatibility with code that imports N_CONFIGS.
N_CONFIGS = 28


# =============================================================================
# Grid builders
# =============================================================================

def build_stage1_grid():
    """Build the Stage 1 grid: sweep vt with fixed nc/gamma/cov.

    Returns:
        List of 5 config dicts (one per variance threshold).
    """
    configs = []
    for vt in VARIANCE_THRESHOLDS:
        configs.append({
            'variance_threshold': vt,
            'covariance_type': STAGE1_FIXED['covariance_type'],
            'n_components': STAGE1_FIXED['n_components'],
            'gamma': STAGE1_FIXED['gamma'],
            **FIXED_PARAMS,
        })
    assert len(configs) == N_CONFIGS_STAGE1, (
        f"Expected {N_CONFIGS_STAGE1} Stage 1 configs, got {len(configs)}"
    )
    return configs


def build_stage2_grid(selected_vt):
    """Build the Stage 2 grid: sweep nc × gamma × cov at fixed vt.

    Args:
        selected_vt: float, variance threshold selected in Stage 1.

    Returns:
        List of config dicts for Stage 2.
    """
    allowed_covs = COVARIANCE_RULES.get(selected_vt)
    if allowed_covs is None:
        raise ValueError(
            f"Unknown variance threshold {selected_vt}. "
            f"Must be one of {VARIANCE_THRESHOLDS}"
        )

    configs = []
    for gamma in GAMMA_OPTIONS:
        for cov in allowed_covs:
            for nc in N_COMPONENTS_OPTIONS:
                configs.append({
                    'variance_threshold': selected_vt,
                    'covariance_type': cov,
                    'n_components': nc,
                    'gamma': gamma,
                    **FIXED_PARAMS,
                })
    return configs


def build_config_grid():
    """Legacy: build the old 28-config grid (gamma=10 fixed).

    Kept for backward compatibility with existing fit-mode results.
    New runs should use build_stage1_grid() + build_stage2_grid().
    """
    configs = []
    for vt in VARIANCE_THRESHOLDS:
        for cov in COVARIANCE_RULES[vt]:
            for nc in N_COMPONENTS_OPTIONS:
                configs.append({
                    'variance_threshold': vt,
                    'covariance_type': cov,
                    'n_components': nc,
                    'gamma': 10,
                    **FIXED_PARAMS,
                })
    assert len(configs) == N_CONFIGS, (
        f"Expected {N_CONFIGS} legacy configs, got {len(configs)}. "
        "Check VARIANCE_THRESHOLDS, COVARIANCE_RULES, and N_COMPONENTS_OPTIONS."
    )
    return configs


def config_name(cfg):
    """Deterministic directory name for a config dict.

    Includes gamma in the name for Stage 2 configs (gamma is grid-searched).
    """
    return (
        f"vt{cfg['variance_threshold']:.2f}_cov{cfg['covariance_type']}"
        f"_nc{cfg['n_components']}_g{int(cfg['gamma'])}"
    )


def legacy_config_name(cfg):
    """Old config_name format without gamma (for reading existing results)."""
    return (
        f"vt{cfg['variance_threshold']:.2f}_cov{cfg['covariance_type']}"
        f"_nc{cfg['n_components']}"
    )
