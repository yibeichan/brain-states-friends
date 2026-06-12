"""
hdphmm_jax.py - JAX/GPU-accelerated Sticky HDP-HMM.

Drop-in replacement for StickyHDPHMM (hdphmm.py) that runs the E-step,
M-step, and most of the HDP posterior update on GPU via JAX.

Usage:
    from utils.hdphmm_jax import StickyHDPHMM_JAX as StickyHDPHMM

The class exposes the same API (.fit, .score, .decode, .history, .means_,
.covars_, .transmat_, .startprob_) as the numpy version. Pickled models
convert JAX arrays to numpy so downstream scripts (05a, 05b, 06) never
import JAX.

Requirements:
    pip install "jax[cuda12]>=0.4.30" "dynamax>=0.1.4"

Critical: float64 is mandatory - see the design notes
"""

# Enable float64 BEFORE importing jax (must be set before any JAX operation)
import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "True")

import warnings
import time
import logging

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.special import logsumexp as jax_logsumexp
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import EmpiricalCovariance

# Ensure float64 is active
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

# Device detection
_GPU_DEVICES = jax.devices("gpu") if jax.default_backend() == "gpu" else []
_DEVICE = _GPU_DEVICES[0] if _GPU_DEVICES else jax.devices("cpu")[0]
logger.info("JAX backend: %s, device: %s", jax.default_backend(), _DEVICE)


# =============================================================================
# Low-level JAX primitives (all JIT-compiled)
# =============================================================================

@jit
def _gaussian_log_likelihood_full(X, means, chol_factors, log_dets):
    """Batched Gaussian log-pdf for full covariance.

    Args:
        X:            (N, D) observations
        means:        (K, D) state means
        chol_factors: (K, D, D) lower Cholesky factors of covariances
        log_dets:     (K,) log-determinants of covariances

    Returns:
        (N, K) log emission probabilities
    """
    D = means.shape[1]
    const = -0.5 * D * jnp.log(2.0 * jnp.pi)

    # diff: (N, K, D)
    diff = X[:, None, :] - means[None, :, :]

    # Solve L @ y = diff^T for each state k, then compute ||y||^2
    # vmap over K (axis 0 of chol_factors, axis 1 of diff)
    def _mahal_one_state(L, d):
        # d: (N, D), L: (D, D)
        # solve_triangular(L, d.T, lower=True) -> (D, N)
        solved = solve_triangular(L, d.T, lower=True)  # (D, N)
        return jnp.sum(solved ** 2, axis=0)  # (N,)

    maha = vmap(_mahal_one_state, in_axes=(0, 1), out_axes=1)(
        chol_factors, diff
    )  # (N, K)

    return const - 0.5 * log_dets[None, :] - 0.5 * maha


@jit
def _gaussian_log_likelihood_diag(X, means, covars_diag):
    """Batched Gaussian log-pdf for diagonal covariance.

    Args:
        X:           (N, D) observations
        means:       (K, D) state means
        covars_diag: (K, D) diagonal covariance entries

    Returns:
        (N, K) log emission probabilities
    """
    D = means.shape[1]
    const = -0.5 * D * jnp.log(2.0 * jnp.pi)

    # log determinant for diagonal: sum of log(diag)
    log_dets = jnp.sum(jnp.log(covars_diag), axis=1)  # (K,)

    diff = X[:, None, :] - means[None, :, :]  # (N, K, D)
    maha = jnp.sum(diff ** 2 / covars_diag[None, :, :], axis=2)  # (N, K)

    return const - 0.5 * log_dets[None, :] - 0.5 * maha


def _prepare_cholesky(covars_full, min_covar):
    """Compute Cholesky factors and log-determinants from full covariances.

    Args:
        covars_full: (K, D, D) covariance matrices
        min_covar:   minimum eigenvalue for regularization

    Returns:
        chol_factors: (K, D, D) lower Cholesky factors
        log_dets:     (K,) log-determinants
    """
    K, D, _ = covars_full.shape
    # Add jitter for numerical stability
    jittered = covars_full + min_covar * jnp.eye(D)[None, :, :]

    def _chol_one(cov):
        L = jnp.linalg.cholesky(cov)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return L, log_det

    chol_factors, log_dets = vmap(_chol_one)(jittered)
    return chol_factors, log_dets


# ---------------------------------------------------------------------------
# Forward-backward (custom lax.scan, no dynamax dependency for core loop)
# ---------------------------------------------------------------------------

def _forward_one_seq(log_startprob, log_transmat, framelogprob):
    """Forward algorithm for one sequence via lax.scan.

    Args:
        log_startprob: (K,) log initial distribution
        log_transmat:  (K, K) log transition matrix
        framelogprob:  (T, K) log emission probabilities

    Returns:
        log_alpha: (T, K) forward lattice (log domain)
        log_prob:  scalar, sequence log-likelihood
    """
    K = log_startprob.shape[0]

    alpha0 = log_startprob + framelogprob[0]

    def _step(alpha_prev, log_emit):
        # alpha_prev: (K,)  log_emit: (K,)
        # log_alpha_t(j) = logsumexp_i(alpha_prev(i) + log_A(i,j)) + log_emit(j)
        # alpha_prev[:, None] + log_transmat -> (K, K), logsumexp over axis=0 -> (K,)
        alpha = jax_logsumexp(alpha_prev[:, None] + log_transmat, axis=0) + log_emit
        return alpha, alpha

    alpha_final, alphas_rest = lax.scan(_step, alpha0, framelogprob[1:])
    # Stack: first row + rest
    log_alpha = jnp.concatenate([alpha0[None, :], alphas_rest], axis=0)
    log_prob = jax_logsumexp(alpha_final)

    return log_alpha, log_prob


def _backward_one_seq(log_transmat, framelogprob):
    """Backward algorithm for one sequence via lax.scan.

    Args:
        log_transmat: (K, K) log transition matrix
        framelogprob: (T, K) log emission probabilities

    Returns:
        log_beta: (T, K) backward lattice (log domain)
    """
    T, K = framelogprob.shape

    beta_T = jnp.zeros(K)  # log(1) = 0

    def _step(beta_next, log_emit_next):
        # beta_t(i) = logsumexp_j(log_A(i,j) + log_emit(t+1,j) + beta(t+1,j))
        beta = jax_logsumexp(
            log_transmat + log_emit_next[None, :] + beta_next[None, :],
            axis=1
        )
        return beta, beta

    # Scan in reverse: from T-1 down to 1
    _, betas_rev = lax.scan(
        _step, beta_T, framelogprob[1:][::-1]  # reversed time steps
    )
    # betas_rev is in reverse order: [T-2, T-3, ..., 0]
    log_beta = jnp.concatenate([betas_rev[::-1], beta_T[None, :]], axis=0)

    return log_beta


def _xi_sum_one_seq(log_alpha, log_beta, log_transmat, framelogprob, log_prob):
    """Compute expected transition counts for one sequence.

    xi_sum[i,j] = sum_t P(z_t=i, z_{t+1}=j | X)

    Args:
        log_alpha:    (T, K) forward lattice
        log_beta:     (T, K) backward lattice
        log_transmat: (K, K) log transition matrix
        framelogprob: (T, K) log emission probs
        log_prob:     scalar, sequence log-likelihood

    Returns:
        xi_sum: (K, K) expected transition counts (NOT log)
    """
    # log_xi[t, i, j] = alpha[t,i] + log_A[i,j] + emit[t+1,j] + beta[t+1,j] - log_prob
    # Sum over t via logsumexp

    def _xi_one_step(carry, t_data):
        # t_data: (alpha_t, emit_next, beta_next) each (K,)
        alpha_t, emit_next, beta_next = t_data
        # (K, K): log_xi for this timestep
        log_xi_t = (alpha_t[:, None] + log_transmat +
                    emit_next[None, :] + beta_next[None, :] - log_prob)
        return carry, log_xi_t

    _, log_xi_all = lax.scan(
        _xi_one_step, None,
        (log_alpha[:-1], framelogprob[1:], log_beta[1:])
    )  # log_xi_all: (T-1, K, K)

    # logsumexp over time -> (K, K), then exp
    log_xi_sum = jax_logsumexp(log_xi_all, axis=0)
    xi_sum = jnp.exp(log_xi_sum)
    xi_sum = jnp.where(jnp.isfinite(xi_sum), xi_sum, 0.0)
    return jnp.maximum(xi_sum, 0.0)


# JIT-compile the per-sequence functions
_forward_one_seq_jit = jit(_forward_one_seq)
_backward_one_seq_jit = jit(_backward_one_seq)
_xi_sum_one_seq_jit = jit(_xi_sum_one_seq)


# ---------------------------------------------------------------------------
# M-step primitives
# ---------------------------------------------------------------------------

@jit
def _compute_sufficient_stats_full(posteriors, X):
    """Compute Gaussian sufficient statistics for full covariance.

    Args:
        posteriors: (N, K)
        X:          (N, D)

    Returns:
        post:      (K,) state occupancy counts
        obs:       (K, D) weighted observation sums
        obs_outer: (K, D, D) weighted outer product sums
    """
    post = posteriors.sum(axis=0)  # (K,)
    obs = posteriors.T @ X  # (K, D)
    obs_outer = jnp.einsum('tk,td,te->kde', posteriors, X, X)  # (K, D, D)
    return post, obs, obs_outer


@jit
def _compute_sufficient_stats_diag(posteriors, X):
    """Compute Gaussian sufficient statistics for diagonal covariance.

    Args:
        posteriors: (N, K)
        X:          (N, D)

    Returns:
        post:    (K,) state occupancy counts
        obs:     (K, D) weighted observation sums
        obs_sq:  (K, D) weighted squared observation sums
    """
    post = posteriors.sum(axis=0)
    obs = posteriors.T @ X
    obs_sq = posteriors.T @ (X ** 2)
    return post, obs, obs_sq


@jit
def _update_means(obs, post):
    """Update means from sufficient statistics."""
    return obs / jnp.maximum(post[:, None], 1e-10)


@jit
def _update_covars_full(obs_outer, post, means, min_covar):
    """Update full covariances with eigenvalue regularization.

    Vectorized over K states via vmap.
    """
    def _update_one(obs_out_k, post_k, mean_k):
        ExxT = obs_out_k / jnp.maximum(post_k, 1e-10)
        ExExT = mean_k[:, None] * mean_k[None, :]
        cov = ExxT - ExExT
        cov = 0.5 * (cov + cov.T)  # enforce symmetry
        # Eigenvalue regularization
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        eigvals = jnp.clip(eigvals, min_covar, None)
        return (eigvecs * eigvals[None, :]) @ eigvecs.T

    return vmap(_update_one)(obs_outer, post, means)


@jit
def _update_covars_diag(obs_sq, post, means, min_covar):
    """Update diagonal covariances."""
    Ex2 = obs_sq / jnp.maximum(post[:, None], 1e-10)
    Ex_sq = means ** 2
    return jnp.maximum(Ex2 - Ex_sq, min_covar)


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------

def _viterbi_one_seq(log_startprob, log_transmat, framelogprob):
    """Viterbi decoding for one sequence via lax.scan.

    Args:
        log_startprob: (K,)
        log_transmat:  (K, K)
        framelogprob:  (T, K)

    Returns:
        state_sequence: (T,) int array
        logprob: scalar
    """
    T, K = framelogprob.shape

    v0 = log_startprob + framelogprob[0]  # (K,)

    def _step(v_prev, log_emit):
        # v_prev: (K,)  log_emit: (K,)
        candidates = v_prev[:, None] + log_transmat  # (K, K)
        best_prev = jnp.argmax(candidates, axis=0)  # (K,)
        v = jnp.max(candidates, axis=0) + log_emit  # (K,)
        return v, (v, best_prev)

    v_final, (v_all, bp_all) = lax.scan(_step, v0, framelogprob[1:])
    # v_all: (T-1, K), bp_all: (T-1, K)

    # Backtrack
    best_last = jnp.argmax(v_final)
    logprob = v_final[best_last]

    def _backtrack_step(state, bp):
        prev_state = bp[state]
        return prev_state, prev_state

    _, states_rev = lax.scan(_backtrack_step, best_last, bp_all[::-1])
    states = jnp.concatenate([states_rev[::-1], best_last[None]])

    return states, logprob


_viterbi_one_seq_jit = jit(_viterbi_one_seq)


# =============================================================================
# StickyHDPHMM_JAX - Main class
# =============================================================================

class StickyHDPHMM_JAX:
    """JAX-accelerated Sticky HDP-HMM, API-compatible with StickyHDPHMM.

    Parameters match the numpy StickyHDPHMM exactly. See hdphmm.py for docs.
    """

    def __init__(self, n_components=10, alpha=10.0, gamma=10.0, kappa=50.0,
                 rho=0.1, covariance_type='diag', random_state=None, n_iter=100,
                 tol=1e-4, verbose=False, params='mc', init_params='mc',
                 min_state_usage=0.01, min_iter=0, learn_hyperparameters=False,
                 min_covar=1e-3, n_jobs=1):
        self.n_components = n_components
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.rho = rho
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.min_state_usage = min_state_usage
        self.min_iter = min_iter
        self.learn_hyperparameters = learn_hyperparameters
        self.min_covar = min_covar
        self.n_jobs = n_jobs  # Ignored - JAX handles parallelism

        self.n_features = None
        self.state_weights = None
        self.active_states = None
        self.means_ = None
        self.covars_ = None
        self.transmat_ = None
        self.startprob_ = None

        self.history = {
            'log_likelihood': [],
            'state_usage': [],
            'active_states': [],
            'alpha': [],
            'gamma': [],
        }

        # JAX PRNG key
        seed = random_state if isinstance(random_state, int) else 42
        self._rng_key = jax.random.PRNGKey(seed)

        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

    # -----------------------------------------------------------------
    # Initialization (numpy - runs once, cheap)
    # -----------------------------------------------------------------

    def _sample_stick_breaking(self, concentration, n_samples):
        """Stick-breaking construction for DP (numpy, matches original)."""
        if concentration <= 0:
            concentration = 1e-6
        if n_samples <= 0:
            return np.array([])

        betas = np.random.beta(1, concentration, size=n_samples)
        weights = np.zeros(n_samples)
        stick = 1.0
        eps = np.finfo(float).eps

        for i in range(n_samples):
            current_draw = min(betas[i], 1.0 - eps)
            weights[i] = stick * current_draw
            stick *= (1.0 - current_draw)
            if stick < eps:
                break

        weights_sum = np.sum(weights)
        if weights_sum > eps:
            return weights / weights_sum
        return np.ones(n_samples) / n_samples

    def _init_params_method(self, X):
        """Initialize model parameters (numpy, matches original exactly)."""
        n_samples, n_features = X.shape
        self.n_features = n_features

        # 1. HDP structure
        self.state_weights = self._sample_stick_breaking(self.gamma, self.n_components)
        self.transmat_ = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            probs = self.state_weights.copy()
            probs[i] += self.kappa * self.rho
            self.transmat_[i] = probs / (probs.sum() + 1e-40)
        self.transmat_ /= (self.transmat_.sum(axis=1, keepdims=True) + 1e-40)

        self.startprob_ = self.state_weights.copy()
        self.startprob_ /= (self.startprob_.sum() + 1e-40)
        self.active_states = np.ones(self.n_components, dtype=bool)

        # 2. Emission parameters
        if "m" in self.init_params:
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_components,
                random_state=self.random_state, n_init=3, batch_size=10000
            )
            if n_samples >= self.n_components:
                kmeans.fit(X)
                self.means_ = kmeans.cluster_centers_
            else:
                indices = np.random.choice(n_samples, self.n_components, replace=True)
                self.means_ = X[indices]
        else:
            self.means_ = np.zeros((self.n_components, n_features))

        if "c" in self.init_params:
            cv = EmpiricalCovariance()
            cv.fit(X)
            cov = cv.covariance_
            if self.covariance_type == 'full':
                cov += np.eye(n_features) * self.min_covar
                self.covars_ = np.tile(cov, (self.n_components, 1, 1))
            elif self.covariance_type == 'diag':
                diag_cov = np.maximum(np.diag(cov), self.min_covar)
                self.covars_ = np.tile(diag_cov, (self.n_components, 1))
        else:
            if self.covariance_type == 'full':
                self.covars_ = np.tile(np.eye(n_features), (self.n_components, 1, 1))
            elif self.covariance_type == 'diag':
                self.covars_ = np.ones((self.n_components, n_features))

    def _normalize_covars(self, covars):
        """Ensure covars are in native storage format for our covariance_type.

        hmmlearn's covars_ property always returns (K, D, D) via fill_covars(),
        even for diag type (where internal storage is (K, D)). This method
        converts back to native format.
        """
        if covars is None:
            return covars
        if self.covariance_type == 'diag' and covars.ndim == 3:
            # Extract diagonal from (K, D, D) -> (K, D)
            K, D, _ = covars.shape
            return np.array([np.diag(covars[k]) for k in range(K)])
        return covars

    # -----------------------------------------------------------------
    # E-step (JAX - the hot path)
    # -----------------------------------------------------------------

    def _e_step(self, X_seqs, means_j, covars_j, startprob_j, transmat_j):
        """Full E-step: emission LL + forward-backward for all sequences.

        Args:
            X_seqs:      list of JAX arrays, each (T_i, D)
            means_j:     (K, D) JAX array
            covars_j:    (K, D, D) or (K, D) JAX array
            startprob_j: (K,) JAX array
            transmat_j:  (K, K) JAX array

        Returns:
            total_logprob:         float
            posteriors_list:       list of (T_i, K) JAX arrays
            xi_sum:                (K, K) numpy array, total expected transitions
        """
        K = means_j.shape[0]

        log_startprob = jnp.log(jnp.maximum(startprob_j, 1e-40))
        log_transmat = jnp.log(jnp.maximum(transmat_j, 1e-40))

        # Pre-compute Cholesky/log-det for emission likelihood
        if self.covariance_type == 'full':
            chol_factors, log_dets = _prepare_cholesky(covars_j, self.min_covar)
        else:
            chol_factors, log_dets = None, None

        total_logprob = 0.0
        posteriors_list = []
        xi_sum_total = jnp.zeros((K, K))

        for X_seq in X_seqs:
            T = X_seq.shape[0]
            if T == 0:
                continue

            # 1. Emission log-likelihood
            if self.covariance_type == 'full':
                flp = _gaussian_log_likelihood_full(X_seq, means_j, chol_factors, log_dets)
            else:
                flp = _gaussian_log_likelihood_diag(X_seq, means_j, covars_j)

            # Clamp extreme values
            flp = jnp.clip(flp, -1e10, 1e10)

            # 2. Forward
            log_alpha, log_prob = _forward_one_seq_jit(log_startprob, log_transmat, flp)
            total_logprob += float(log_prob)

            # 3. Backward
            log_beta = _backward_one_seq_jit(log_transmat, flp)

            # 4. Posteriors
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - jax_logsumexp(log_gamma, axis=1, keepdims=True)
            posteriors = jnp.exp(log_gamma)
            posteriors = jnp.where(jnp.isfinite(posteriors), posteriors, 1.0 / K)
            posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
            posteriors_list.append(posteriors)

            # 5. Xi
            if T > 1:
                xi = _xi_sum_one_seq_jit(log_alpha, log_beta, log_transmat, flp, log_prob)
                xi_sum_total = xi_sum_total + xi

        return total_logprob, posteriors_list, xi_sum_total

    # -----------------------------------------------------------------
    # M-step (JAX)
    # -----------------------------------------------------------------

    def _m_step(self, X_seqs, posteriors_list):
        """Update emission parameters from sufficient statistics.

        Args:
            X_seqs:          list of (T_i, D) JAX arrays
            posteriors_list: list of (T_i, K) JAX arrays

        Returns:
            means:  (K, D) numpy array
            covars: (K, D, D) or (K, D) numpy array
        """
        # Concatenate for stats computation
        X_cat = jnp.concatenate(X_seqs, axis=0)
        post_cat = jnp.concatenate(posteriors_list, axis=0)

        if self.covariance_type == 'full':
            post, obs, obs_outer = _compute_sufficient_stats_full(post_cat, X_cat)
            means = _update_means(obs, post)
            covars = _update_covars_full(obs_outer, post, means, self.min_covar)
        else:
            post, obs, obs_sq = _compute_sufficient_stats_diag(post_cat, X_cat)
            means = _update_means(obs, post)
            covars = _update_covars_diag(obs_sq, post, means, self.min_covar)

        return np.asarray(means), np.asarray(covars)

    # -----------------------------------------------------------------
    # HDP posterior update (mostly numpy - cheap scalar ops)
    # -----------------------------------------------------------------

    def _hdp_posterior_update(self, posteriors_list, xi_sum, lengths):
        """Update transition matrix, state weights, start probs using HDP.

        This runs on CPU (numpy) because it involves scalar operations and
        conditional logic that don't benefit from GPU acceleration.
        """
        xi_sum_np = np.asarray(xi_sum)

        # 1. Update state activity
        post_cat = np.concatenate([np.asarray(p) for p in posteriors_list], axis=0)
        post_sum = post_cat.sum()
        if post_sum > 1e-10:
            state_usage = post_cat.sum(axis=0) / post_sum
            self.active_states = state_usage > self.min_state_usage
            if np.sum(self.active_states) < 2:
                top2 = np.argsort(state_usage)[-2:]
                self.active_states = np.zeros(self.n_components, dtype=bool)
                self.active_states[top2] = True

        # 2. Optional hyperparameter update
        if self.learn_hyperparameters:
            self._update_hyperparameters(xi_sum_np)

        # 3. Compute effective counts + update transmat
        K = self.n_components
        effective_counts = np.zeros((K, K))
        for i in range(K):
            base_counts = self.alpha * self.state_weights
            sticky_counts = np.zeros(K)
            sticky_counts[i] = self.kappa * self.rho
            effective_counts[i, :] = xi_sum_np[i, :] + base_counts + sticky_counts

        row_sums = effective_counts.sum(axis=1, keepdims=True)
        zero_mask = (row_sums < 1e-40).flatten()
        new_transmat = np.zeros_like(effective_counts)
        new_transmat[zero_mask] = 1.0 / K
        positive = ~zero_mask
        new_transmat[positive] = effective_counts[positive] / row_sums[positive]
        final_sums = new_transmat.sum(axis=1, keepdims=True)
        final_sums = np.maximum(final_sums, 1e-40)
        self.transmat_ = new_transmat / final_sums

        # 4. Update state weights (beta)
        global_counts = effective_counts.sum(axis=0)
        posterior_params = self.gamma * self.state_weights + global_counts
        smoothed = posterior_params + 1e-20
        self.state_weights = smoothed / np.sum(smoothed)

        # 5. Update startprob
        posteriors_np = [np.asarray(p) for p in posteriors_list]
        start_counts = np.zeros(K)
        for p in posteriors_np:
            if len(p) > 0:
                start_counts += p[0]
        start_counts += self.alpha * self.state_weights
        sc_sum = start_counts.sum()
        if sc_sum > 0:
            self.startprob_ = start_counts / sc_sum
        else:
            self.startprob_ = np.ones(K) / K

    def _update_hyperparameters(self, expected_transitions):
        """Update alpha, gamma via auxiliary variable sampling (numpy)."""
        K = self.n_components
        row_sums = expected_transitions.sum(axis=1)

        # Alpha update
        total_tables = 0
        total_customers = 0
        for i in range(K):
            n_i = row_sums[i]
            if n_i < 1:
                continue
            total_customers += n_i
            for j in range(K):
                n_ij = expected_transitions[i, j]
                if n_ij < 1:
                    continue
                # Expected number of tables ~ sum of Bernoulli(alpha*beta_j / (alpha*beta_j + l))
                alpha_beta_j = self.alpha * self.state_weights[j]
                if i == j:
                    alpha_beta_j += self.kappa * self.rho
                for l in range(int(min(n_ij, 100))):
                    prob = alpha_beta_j / (alpha_beta_j + l)
                    total_tables += (np.random.rand() < prob)

        if total_customers > 0 and total_tables > 0:
            eta = np.random.beta(self.alpha + 1, total_customers)
            pi_eta = total_tables / (total_tables + total_customers * (1.0 - np.log(max(eta, 1e-40))))
            if np.random.rand() < pi_eta:
                self.alpha = np.random.gamma(total_tables + 1, 1.0 / (1.0 - np.log(max(eta, 1e-40))))
            else:
                self.alpha = np.random.gamma(total_tables, 1.0 / (1.0 - np.log(max(eta, 1e-40))))
            self.alpha = max(self.alpha, 1e-6)

        # Gamma update (simplified)
        n_active = max(np.sum(self.active_states), 1)
        if n_active > 1:
            eta_g = np.random.beta(self.gamma + 1, n_active)
            self.gamma = np.random.gamma(n_active, 1.0 / (1.0 - np.log(max(eta_g, 1e-40))))
            self.gamma = max(self.gamma, 1e-6)

    # -----------------------------------------------------------------
    # fit()
    # -----------------------------------------------------------------

    def fit(self, X, lengths=None):
        """Fit the Sticky HDP-HMM using JAX-accelerated EM.

        Args:
            X:       (N_total, D) numpy array, concatenated sequences
            lengths: list of int, per-sequence lengths

        Returns:
            self
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2D numpy array.")

        N, D = X.shape
        if lengths is None:
            lengths = [N]

        # Seed numpy RNG for initialization reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize parameters (numpy)
        self._init_params_method(X)
        logger.info("Initialized: K=%d, D=%d, N=%d, %d sequences",
                     self.n_components, D, N, len(lengths))

        # Split X into per-sequence JAX arrays and transfer to device
        X_seqs = []
        pos = 0
        for length in lengths:
            seq = jax.device_put(jnp.array(X[pos:pos+length], dtype=jnp.float64), _DEVICE)
            X_seqs.append(seq)
            pos += length

        # EM loop
        prev_ll = -np.inf
        converged = False
        window_size = 100

        t_start = time.time()

        for iter_idx in range(self.n_iter):
            iter_start = time.time()

            # Transfer current params to JAX
            means_j = jax.device_put(jnp.array(self.means_, dtype=jnp.float64), _DEVICE)
            startprob_j = jax.device_put(jnp.array(self.startprob_, dtype=jnp.float64), _DEVICE)
            transmat_j = jax.device_put(jnp.array(self.transmat_, dtype=jnp.float64), _DEVICE)

            if self.covariance_type == 'full':
                covars_j = jax.device_put(jnp.array(self.covars_, dtype=jnp.float64), _DEVICE)
            else:
                covars_j = jax.device_put(jnp.array(self.covars_, dtype=jnp.float64), _DEVICE)

            # E-step (JAX)
            logprob_total, posteriors_list, xi_sum = self._e_step(
                X_seqs, means_j, covars_j, startprob_j, transmat_j
            )

            # M-step (JAX)
            if 'm' in self.params or 'c' in self.params:
                self.means_, self.covars_ = self._m_step(X_seqs, posteriors_list)

            # HDP update (numpy)
            self._hdp_posterior_update(posteriors_list, xi_sum, lengths)

            # History
            post_cat = np.concatenate([np.asarray(p) for p in posteriors_list], axis=0)
            post_sum = post_cat.sum()
            state_usage = post_cat.sum(axis=0) / max(post_sum, 1e-10)
            active_count = int(np.sum(state_usage > self.min_state_usage))

            self.history['log_likelihood'].append(logprob_total)
            self.history['state_usage'].append(state_usage.tolist())
            self.history['active_states'].append(active_count)
            self.history['alpha'].append(float(self.alpha))
            self.history['gamma'].append(float(self.gamma))

            iter_time = time.time() - iter_start
            if self.verbose and iter_idx % 100 == 0:
                logger.info(
                    "Iter %d: LL=%.4f, active=%d, %.2fs/iter",
                    iter_idx, logprob_total, active_count, iter_time
                )

            # Convergence check (windowed, matches original)
            if iter_idx >= self.min_iter + 2 * window_size:
                ll_hist = self.history['log_likelihood']
                w1_mean = np.mean(ll_hist[-(2*window_size):-window_size])
                w2_mean = np.mean(ll_hist[-window_size:])
                rel_improvement = abs(w2_mean - w1_mean) / max(abs(w2_mean), 1e-10)
                if rel_improvement < self.tol:
                    converged = True
                    logger.info(
                        "Converged at iter %d: rel_improvement=%.2e < tol=%.2e",
                        iter_idx, rel_improvement, self.tol
                    )
                    break

            prev_ll = logprob_total

        total_time = time.time() - t_start
        logger.info(
            "Fit complete: %d iters, %.1f min, final LL=%.4f, active=%d, converged=%s",
            iter_idx + 1, total_time / 60, logprob_total, active_count, converged
        )

        return self

    # -----------------------------------------------------------------
    # score() and decode()
    # -----------------------------------------------------------------

    def score(self, X, lengths=None):
        """Compute log-likelihood of data under the model.

        Args:
            X:       (N, D) numpy array
            lengths: list of int

        Returns:
            total log-likelihood (float)
        """
        if lengths is None:
            lengths = [X.shape[0]]

        means_j = jax.device_put(jnp.array(self.means_, dtype=jnp.float64), _DEVICE)
        startprob_j = jax.device_put(jnp.array(self.startprob_, dtype=jnp.float64), _DEVICE)
        transmat_j = jax.device_put(jnp.array(self.transmat_, dtype=jnp.float64), _DEVICE)
        covars_j = jax.device_put(jnp.array(self.covars_, dtype=jnp.float64), _DEVICE)

        log_sp = jnp.log(jnp.maximum(startprob_j, 1e-40))
        log_tm = jnp.log(jnp.maximum(transmat_j, 1e-40))

        if self.covariance_type == 'full':
            chol, log_dets = _prepare_cholesky(covars_j, self.min_covar)
        else:
            chol, log_dets = None, None

        total_ll = 0.0
        pos = 0
        for length in lengths:
            X_seq = jax.device_put(jnp.array(X[pos:pos+length], dtype=jnp.float64), _DEVICE)
            if self.covariance_type == 'full':
                flp = _gaussian_log_likelihood_full(X_seq, means_j, chol, log_dets)
            else:
                flp = _gaussian_log_likelihood_diag(X_seq, means_j, covars_j)
            flp = jnp.clip(flp, -1e10, 1e10)
            _, log_prob = _forward_one_seq_jit(log_sp, log_tm, flp)
            total_ll += float(log_prob)
            pos += length

        return total_ll

    def decode(self, X, lengths=None):
        """Viterbi decoding.

        Args:
            X:       (N, D) numpy array (single sequence) or concatenated
            lengths: list of int

        Returns:
            logprob:        float, total log-probability
            state_sequence: (N,) numpy int array
        """
        if lengths is None:
            lengths = [X.shape[0]]

        means_j = jax.device_put(jnp.array(self.means_, dtype=jnp.float64), _DEVICE)
        startprob_j = jax.device_put(jnp.array(self.startprob_, dtype=jnp.float64), _DEVICE)
        transmat_j = jax.device_put(jnp.array(self.transmat_, dtype=jnp.float64), _DEVICE)
        covars_j = jax.device_put(jnp.array(self.covars_, dtype=jnp.float64), _DEVICE)

        log_sp = jnp.log(jnp.maximum(startprob_j, 1e-40))
        log_tm = jnp.log(jnp.maximum(transmat_j, 1e-40))

        if self.covariance_type == 'full':
            chol, log_dets = _prepare_cholesky(covars_j, self.min_covar)
        else:
            chol, log_dets = None, None

        total_logprob = 0.0
        all_states = []
        pos = 0

        for length in lengths:
            X_seq = jax.device_put(jnp.array(X[pos:pos+length], dtype=jnp.float64), _DEVICE)
            if self.covariance_type == 'full':
                flp = _gaussian_log_likelihood_full(X_seq, means_j, chol, log_dets)
            else:
                flp = _gaussian_log_likelihood_diag(X_seq, means_j, covars_j)
            flp = jnp.clip(flp, -1e10, 1e10)

            states, logprob = _viterbi_one_seq_jit(log_sp, log_tm, flp)
            total_logprob += float(logprob)
            all_states.append(np.asarray(states))
            pos += length

        return total_logprob, np.concatenate(all_states)

    # -----------------------------------------------------------------
    # Pickle compatibility: convert JAX -> numpy before save
    # -----------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert any JAX arrays to numpy
        for key in ('means_', 'covars_', 'transmat_', 'startprob_',
                     'state_weights', 'active_states'):
            val = state.get(key)
            if val is not None and hasattr(val, '__jax_array__'):
                state[key] = np.asarray(val)
        # Remove JAX PRNG key (not picklable / not needed)
        state.pop('_rng_key', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        seed = self.random_state if isinstance(self.random_state, int) else 42
        self._rng_key = jax.random.PRNGKey(seed)
        # Normalize covars in case they were saved in full format
        self.covars_ = self._normalize_covars(self.covars_)

    # -----------------------------------------------------------------
    # Convenience: prune_and_decode (matches original API)
    # -----------------------------------------------------------------

    def prune_and_decode(self, X, lengths=None, min_usage=None):
        """Decode after masking inactive states in transmat.

        Matches the numpy StickyHDPHMM.prune_and_decode() API.
        """
        if min_usage is None:
            min_usage = self.min_state_usage

        # Identify active states from last usage
        if self.history['state_usage']:
            usage = np.array(self.history['state_usage'][-1])
        else:
            usage = np.ones(self.n_components) / self.n_components

        active_mask = usage > min_usage
        if not np.any(active_mask):
            active_mask[np.argmax(usage)] = True

        # Mask transmat: zero out inactive rows/cols, renormalize
        masked_transmat = self.transmat_.copy()
        masked_transmat[~active_mask, :] = 0.0
        masked_transmat[:, ~active_mask] = 0.0
        row_sums = masked_transmat.sum(axis=1, keepdims=True)
        zero_rows = (row_sums < 1e-40).flatten()
        masked_transmat[zero_rows] = 1.0 / self.n_components
        masked_transmat = masked_transmat / masked_transmat.sum(axis=1, keepdims=True)

        # Temporarily swap transmat, decode, swap back
        orig_transmat = self.transmat_
        self.transmat_ = masked_transmat
        try:
            result = self.decode(X, lengths)
        finally:
            self.transmat_ = orig_transmat

        return result
