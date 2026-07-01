import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import EmpiricalCovariance
import warnings
import traceback
from scipy.special import logsumexp as special_logsumexp
import time
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from hmmlearn import hmm
from hmmlearn._hmmc import (
    forward_log as _hmmc_forward_log,
    backward_log as _hmmc_backward_log,
    compute_log_xi_sum as _hmmc_compute_log_xi_sum,
)


# =============================================================================
# Module-level functions for parallel E-step (must be top-level for pickling)
# =============================================================================

def _forward_single_seq(startprob, transmat, seq_framelogprob):
    """Forward pass for a single sequence using hmmlearn's Cython implementation.

    Module-level for joblib pickling.

    Args:
        startprob: (K,) start probabilities (NOT log)
        transmat:  (K, K) transition matrix (NOT log)
        seq_framelogprob: (T, K) log emission probabilities for this sequence

    Returns:
        fwd: (T, K) forward lattice (log domain)
        seq_logprob: float, sequence log-likelihood
    """
    seq_logprob, fwd = _hmmc_forward_log(startprob, transmat, seq_framelogprob)
    return fwd, float(seq_logprob)


def _backward_single_seq(startprob, transmat, seq_framelogprob):
    """Backward pass for a single sequence using hmmlearn's Cython implementation.

    Module-level for joblib pickling.

    Args:
        startprob: (K,) start probabilities (NOT log)
        transmat:  (K, K) transition matrix (NOT log)
        seq_framelogprob: (T, K) log emission probabilities for this sequence

    Returns:
        bwd: (T, K) backward lattice (log domain)
    """
    return _hmmc_backward_log(startprob, transmat, seq_framelogprob)


def _transitions_single_seq(seq_fwd, seq_bwd, seq_framelogprob, transmat):
    """Compute expected transition counts for one sequence using hmmlearn's Cython.

    Args:
        seq_fwd:          (T, K) forward lattice (log domain)
        seq_bwd:          (T, K) backward lattice (log domain)
        seq_framelogprob: (T, K) log emission probabilities
        transmat:         (K, K) transition matrix (NOT log)

    Returns:
        expected: (K, K) expected transition counts for this sequence
    """
    T, K = seq_framelogprob.shape
    if T <= 1:
        return np.zeros((K, K))

    log_xi_sum = _hmmc_compute_log_xi_sum(
        seq_fwd, transmat, seq_bwd, seq_framelogprob
    )
    with np.errstate(under='ignore'):
        expected = np.exp(log_xi_sum)
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(expected, 0.0)


class WeakLimitHMM(hmm.GaussianHMM):
    """
    A Gaussian-emission hidden Markov model fit by EM, with sticky
    self-transition and hierarchical Dirichlet concentration priors on its
    transitions under a fixed-capacity weak-limit truncation. This borrows the
    prior structure of the sticky hierarchical Dirichlet process HMM
    (Fox et al. 2011) but NOT its nonparametric inference: n_components is a
    fixed truncation cap and the number of occupied states emerges below it.

    This implementation uses a hybrid approach combining EM updates for emission
    parameters with Bayesian inference for transition parameters and hyperparameters.

    Implementation Notes
    --------------------
    This class extends hmmlearn.GaussianHMM but implements custom E-step methods
    (_do_e_step, _do_forward_pass, _do_backward_pass) rather than using the parent
    class methods. This is necessary because:

    1. The HDP posterior update requires access to the forward/backward lattices
       (fwdlattice, bwdlattice) and frame log-probabilities (framelogprob), which
       the parent class's score_samples() method does not expose.

    2. The _compute_expected_transitions() method needs these lattices to compute
       expected transition counts E[N(i->j)] for updating the HDP transition prior.

    3. Custom numerical stability handling is implemented for high-dimensional
       fMRI data (e.g., 156 parcels).

    State Management
    ----------------
    All n_components states are preserved in the final model (no pruning).
    Low-usage states are tracked via the `active_states` attribute but are NOT
    removed. Downstream hierarchical clustering (scripts 05-07) handles state
    organization, where low-usage states naturally form small clusters.

    Parameters
    ----------
    n_components : int, default=10
        Maximum number of states in the model. The HDP prior encourages sparsity,
        so fewer states may be actively used.

    alpha : float, default=10.0
        Row-level concentration parameter. Controls transition sparsity per state.
        Lower values → sparser transitions (fewer likely next states).

    gamma : float, default=10.0
        Global concentration parameter. Controls overall state repertoire size.
        Lower values → fewer distinct states used across the sequence.

    kappa : float, default=50.0
        Sticky parameter. Adds bias to self-transitions for state persistence.

    rho : float, default=0.1
        Sticky scaling factor. Effective sticky bias = kappa * rho.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Covariance type for Gaussian emissions.

    min_state_usage : float, default=0.01
        Threshold for considering a state "active" (for diagnostics only).
        States below this threshold are tracked but NOT removed.

    min_iter : int, default=0
        Minimum iterations before checking convergence.

    learn_hyperparameters : bool, default=False
        If True, alpha and gamma are updated via auxiliary variable sampling.

    random_state : int or RandomState, default=None
        Random state for reproducibility.

    n_iter : int, default=100
        Maximum number of EM iterations.

    tol : float, default=1e-3
        Convergence threshold (relative log-likelihood improvement).

    verbose : bool, default=False
        Print progress during fitting.

    params : str, default='mc'
        Parameters to update in M-step ('m'=means, 'c'=covariances).
        Transitions are updated by HDP logic, not standard M-step.

    init_params : str, default='mc'
        Parameters to initialize. Transitions initialized by HDP prior.
    """
    def __init__(self, n_components=10, alpha=10.0, gamma=10.0, kappa=50.0,
                 rho=0.1, covariance_type='diag', random_state=None, n_iter=100,
                 tol=1e-4, verbose=False, params='mc', init_params='mc',
                 min_state_usage=0.01, min_iter=0, learn_hyperparameters=False,
                 min_covar=1e-3, n_jobs=1):
        # Default covariance_type changed to 'diag' as it's often better for fMRI
        # Default tol adjusted based on previous example
        super().__init__(n_components=n_components,
                         covariance_type=covariance_type,
                         random_state=random_state,
                         n_iter=n_iter, # n_iter of base class is used for internal checks if fit is called
                         tol=tol,       # tol of base class might be used internally
                         verbose=verbose, # verbose of base class
                         params=params,
                         init_params=init_params) # init_params controls base class internal init

        # Override base class parameters with potentially different values if needed
        # self.n_iter = n_iter # This shadows the base class but makes it explicit for our loop
        # self.tol = tol
        # self.verbose = verbose

        # HDP specific parameters
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.rho = rho
        self.learn_hyperparameters = learn_hyperparameters  # Whether to learn alpha/gamma during training
        self.state_weights = None # Holds the base measure (beta)
        self.active_states = None # Boolean mask for active states
        self.min_state_usage = min_state_usage
        self.min_iter = min_iter # Store min_iter for our custom fit loop
        # Enhanced history tracking for diagnostics
        # - state_usage: per-state usage proportions at each iteration (list of arrays)
        # - active_states: count of states with usage > min_state_usage (for quick reference)
        self.history = {
            'log_likelihood': [],
            'state_usage': [],      # Full per-state usage array at each iteration
            'active_states': [],    # Count of states above threshold (backward compat)
            'alpha': [],
            'gamma': []
        }
        self.min_covar = min_covar
        self.n_jobs = n_jobs  # Parallel jobs for E-step (forward/backward across sequences)

        # Initialize n_features as None, will be set during _init_params
        self.n_features = None

        # Ensure minimum number of components is reasonable
        if self.n_components < 1:
             raise ValueError("n_components must be at least 1.")

    def _sample_stick_breaking(self, concentration, n_samples):
        """Sample weights using stick-breaking construction for DP."""
        # Check for non-positive concentration
        if concentration <= 0:
            warnings.warn(f"Concentration parameter is non-positive ({concentration}). Using a small positive value instead.")
            concentration = 1e-6

        # Check for non-positive n_samples
        if n_samples <= 0:
            return np.array([]) # Return empty array if no samples requested

        betas = np.random.beta(1, concentration, size=n_samples)
        weights = np.zeros(n_samples)
        stick = 1.0
        eps = np.finfo(float).eps # Machine epsilon for checks

        for i in range(n_samples):
            # Prevent numerical issues where beta might be exactly 1
            current_draw = min(betas[i], 1.0 - eps)
            weights[i] = stick * current_draw
            stick *= (1.0 - current_draw)
            # If stick becomes essentially zero, stop early
            if stick < eps:
                break

        # Renormalize to ensure sum is exactly 1 due to potential floating point issues
        weights_sum = np.sum(weights)
        if weights_sum > eps:
             return weights / weights_sum
        else:
             # If sum is zero (e.g., n_samples=1 and beta drawn as 1), return uniform
             warnings.warn("Stick-breaking resulted in zero sum weights, returning uniform.")
             return np.ones(n_samples) / n_samples


    def _init_params(self, X):
        """Initialize model parameters with HDP priors and basic emission estimates."""
        # Validate X
        if not isinstance(X, np.ndarray) or X.ndim != 2:
             raise ValueError("Input data X must be a 2D numpy array.")
        n_samples, n_features = X.shape
        if n_samples < self.n_components:
             warnings.warn(f"Number of samples ({n_samples}) is less than number of components ({self.n_components})."
                           " Initialization might be unstable.")
        
        # Set n_features attribute from data
        self.n_features = n_features

        # 1. Initialize HDP structure components first
        self.state_weights = self._sample_stick_breaking(self.gamma, self.n_components)
        self.transmat_ = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            probs = self.state_weights.copy()
            probs[i] += self.kappa * self.rho # Add sticky bias
            probs_sum = probs.sum()
            # Use a small epsilon for robustness against zero sum
            self.transmat_[i] = probs / (probs_sum + 1e-40)

        # Normalize again to be sure rows sum to 1
        self.transmat_ /= (self.transmat_.sum(axis=1, keepdims=True) + 1e-40)

        self.startprob_ = self.state_weights.copy()
        # Ensure startprob sums to 1
        self.startprob_ /= (self.startprob_.sum() + 1e-40)

        self.active_states = np.ones(self.n_components, dtype=bool)

        # 2. Initialize Emission Parameters (means_, covars_)
        # These need to exist before the first E-step.
        # We use a basic initialization similar to hmmlearn's default.
        if self.verbose: print("Initializing emission parameters...")
        try:
            if "m" in self.init_params: # Check if means initialization is requested
                 kmeans = MiniBatchKMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=3, batch_size=10000)
                 if n_samples >= self.n_components:
                      kmeans.fit(X)
                      self.means_ = kmeans.cluster_centers_
                 else:
                      # Fallback: Assign random samples as means
                      warnings.warn("Not enough samples for KMeans, initializing means randomly.")
                      indices = np.random.choice(n_samples, self.n_components, replace=True)
                      self.means_ = X[indices]
            else:
                # If not initializing means, create placeholder
                self.means_ = np.zeros((self.n_components, n_features))

            if "c" in self.init_params: # Check if covariances initialization is requested
                cv = EmpiricalCovariance()
                cv.fit(X)
                cov = cv.covariance_
                min_covar = self.min_covar # Floor covariance diagonals

                if self.covariance_type == 'full':
                    cov += np.eye(n_features) * min_covar
                    self.covars_ = np.tile(cov, (self.n_components, 1, 1))
                elif self.covariance_type == 'diag':
                    diag_cov = np.diag(cov)
                    diag_cov = np.maximum(diag_cov, min_covar)
                    self.covars_ = np.tile(diag_cov, (self.n_components, 1))
                elif self.covariance_type == 'tied':
                    cov += np.eye(n_features) * min_covar
                    self.covars_ = cov
                elif self.covariance_type == 'spherical':
                    mean_var = max(cov.mean(), min_covar)
                    self.covars_ = np.tile(mean_var, (self.n_components,))
            else:
                # If not initializing covars, create placeholder based on type
                if self.covariance_type == 'full':
                    self.covars_ = np.tile(np.eye(n_features), (self.n_components, 1, 1))
                elif self.covariance_type == 'diag':
                    self.covars_ = np.ones((self.n_components, n_features))
                elif self.covariance_type == 'tied':
                    self.covars_ = np.eye(n_features)
                elif self.covariance_type == 'spherical':
                    self.covars_ = np.ones(self.n_components)

            # Manual checks for finiteness
            if not np.all(np.isfinite(self.means_)): raise ValueError("Non-finite values in initial means.")
            if not np.all(np.isfinite(self.covars_)): raise ValueError("Non-finite values in initial covars.")
            if self.verbose: print("Emission parameters initialized.")

        except Exception as e:
             raise RuntimeError(f"Failed to initialize emission parameters: {e}")


    def _update_startprob(self, posteriors, lengths=None):
        """
        Update initial state distribution with HDP prior.

        For multi-sequence data, sums the posterior at the first time step
        of every sequence (not just the first sequence).

        Parameters
        ----------
        posteriors : ndarray, shape (n_samples, n_components)
            State-membership probabilities from the E-step.
        lengths : array-like of int, optional
            Lengths of individual sequences in the concatenated data.
            If None, treats the data as a single sequence.

        Returns
        -------
        startprob_ : ndarray, shape (n_components,)
            Updated initial state distribution
        """
        # Sum expected initial state counts across all sequence starts
        if lengths is None:
            start_counts = posteriors[0]
        else:
            start_counts = np.zeros(self.n_components)
            pos = 0
            for length in lengths:
                start_counts += posteriors[pos]
                pos += length

        # Add HDP prior (alpha acts as pseudocount)
        start_counts += self.alpha * self.state_weights

        # Normalize
        start_counts_sum = np.sum(start_counts)
        if start_counts_sum > 0:
            self.startprob_ = start_counts / start_counts_sum
        else:
             self.startprob_ = np.ones(self.n_components) / self.n_components

        return self.startprob_

    def _update_state_activity(self, posterior_probs):
        """Identify active states based on posterior usage."""
        post_sum = posterior_probs.sum()
        if post_sum < 1e-10:
             warnings.warn("Posterior probabilities sum to zero. State activity not updated.")
             return self.active_states

        state_usage = posterior_probs.sum(axis=0) / post_sum
        new_active_states = state_usage > self.min_state_usage
        active_count = np.sum(new_active_states)

        # Ensure we keep at least min_components states active (e.g., 2)
        min_active_states = getattr(self, 'min_active_states_enforced', 2) # Allow optional override via self.min_active_states_enforced = N
        if self.n_components >= min_active_states and active_count < min_active_states:
            # Find the top min_active_states based on usage
            top_states_indices = np.argsort(state_usage)[-min_active_states:]
            new_active_states = np.zeros(self.n_components, dtype=bool)
            new_active_states[top_states_indices] = True
            if self.verbose:
                print(f"State activity below minimum {min_active_states}. Keeping top {min_active_states} states: {top_states_indices}")

        # Check if activity changed
        if not np.array_equal(self.active_states, new_active_states) and self.verbose:
             print(f"Active states updated: {np.sum(self.active_states)} -> {np.sum(new_active_states)}")

        self.active_states = new_active_states
        return self.active_states

    # NOTE: _prune_inactive_states was removed (Dec 2025).
    # Pruning is no longer performed - all n_components states are preserved.
    # Low-usage states are handled by downstream hierarchical clustering (scripts 05-07).
    # The active_states attribute is still maintained for diagnostics and potential
    # hyperparameter learning, but states are never actually removed from the model.

    def decode(self, X, lengths=None):
        """
        Error-resilient state sequence decoding with proper Viterbi fallback

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : array-like, optional
            Lengths of the sequences in X

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence
        state_sequence : ndarray, shape (n_samples,)
            Viterbi-decoded state sequence
        """
        try:
             # Check for non-finite values in parameters before decoding
             if not np.all(np.isfinite(self.startprob_)):
                 raise ValueError("Non-finite values in startprob_")
             if not np.all(np.isfinite(self.transmat_)):
                 raise ValueError("Non-finite values in transmat_")
             if not np.all(np.isfinite(self.means_)):
                 raise ValueError("Non-finite values in means_")
             # Check internal _covars_ as that's likely used by decode's likelihood calculations
             if not hasattr(self, '_covars_') or not np.all(np.isfinite(self._covars_)):
                 warnings.warn("_covars_ not found or contains non-finite values before decode. Attempting self._check().")
                 try:
                    self._check() # Try to ensure consistency before decode
                    if not hasattr(self, '_covars_') or not np.all(np.isfinite(self._covars_)):
                        raise ValueError("Non-finite values in _covars_ even after _check() before decode.")
                 except Exception as check_e:
                     raise ValueError(f"Failed to run _check() or _covars_ still invalid before decode: {check_e}")

             logprob, state_sequence = super().decode(X, lengths)
             # Check output of decode
             if not np.isfinite(logprob):
                  raise ValueError(f"Decode resulted in non-finite logprob: {logprob}")
             return logprob, state_sequence

        except Exception as e:
            warnings.warn(f"Standard decode failed: {str(e)}. Using manual Viterbi fallback.")

            # Fallback Viterbi Implementation
            framelogprob = self._compute_log_likelihood(X)
            if not np.all(np.isfinite(framelogprob)):
                 warnings.warn("Non-finite values detected in frame log likelihoods during fallback decode.")
                 # Attempt to replace non-finite with very small number? Risky.
                 # Or maybe just fail here? Let's fail more gracefully if possible.
                 framelogprob = np.nan_to_num(framelogprob, nan=-1e10, posinf=-1e10, neginf=-1e10)


            # Use hmmlearn's _viterbi implementation directly if accessible and robust
            # Otherwise, use the manual implementation below.
            try:
                 logprob, state_sequence = self._viterbi(X) # Assuming _viterbi exists and is robust
                 if not np.isfinite(logprob):
                      raise ValueError("Fallback _viterbi resulted in non-finite logprob.")
                 return logprob, state_sequence
            except Exception as viterbi_e:
                 warnings.warn(f"hmmlearn._viterbi fallback also failed: {viterbi_e}. Using simplified manual Viterbi.")


                 # Simplified Manual Viterbi (potentially less stable than hmmlearn's)
                 n_samples, n_features = X.shape
                 n_components = self.n_components

                 # Recompute frame log likelihoods just in case
                 framelogprob = self._compute_log_likelihood(X)
                 framelogprob = np.nan_to_num(framelogprob, nan=-1e10, posinf=-1e10, neginf=-1e10)

                 log_startprob = np.log(self.startprob_ + 1e-10)
                 log_transmat = np.log(self.transmat_ + 1e-10)

                 if not np.all(np.isfinite(log_startprob)): warnings.warn("Non-finite log_startprob in manual viterbi")
                 if not np.all(np.isfinite(log_transmat)): warnings.warn("Non-finite log_transmat in manual viterbi")

                 viterbi_lattice = np.zeros((n_samples, n_components))
                 backpointers = np.zeros((n_samples, n_components), dtype=int)

                 # Initialization
                 viterbi_lattice[0] = log_startprob + framelogprob[0]

                 # Recursion
                 for t in range(1, n_samples):
                     for j in range(n_components):
                         seg_prob = viterbi_lattice[t - 1] + log_transmat[:, j]
                         backpointers[t, j] = np.argmax(seg_prob)
                         viterbi_lattice[t, j] = np.max(seg_prob) + framelogprob[t, j]

                 # Termination
                 logprob = np.max(viterbi_lattice[-1])
                 state_sequence = np.zeros(n_samples, dtype=int)
                 state_sequence[-1] = np.argmax(viterbi_lattice[-1])

                 # Backtracking
                 for t in range(n_samples - 2, -1, -1):
                     state_sequence[t] = backpointers[t + 1, state_sequence[t + 1]]

                 if not np.isfinite(logprob):
                    warnings.warn("Manual Viterbi fallback resulted in non-finite logprob. Returning -inf.")
                    logprob = -np.inf # Return a clear failure indicator

                 return logprob, state_sequence


    def prune_and_decode(self, X, lengths=None, min_usage=None):
        """Decode after zeroing transition probability for inactive states.

        States with final-iteration usage below min_usage are treated as
        inactive. Their transition probabilities (both incoming and outgoing)
        are set to zero before Viterbi decode, preventing "dead" states with
        residual transition mass from capturing individual TRs.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        lengths : array-like, optional
        min_usage : float, optional
            Override for self.min_state_usage (default: self.min_state_usage).

        Returns
        -------
        logprob : float
        state_sequence : ndarray, shape (n_samples,)
        """
        if min_usage is None:
            min_usage = getattr(self, 'min_state_usage', 0.01)

        # Determine active states from final-iteration usage
        usage = None
        if hasattr(self, 'history') and self.history:
            usage_list = self.history.get('state_usage')
            if usage_list:
                usage = np.array(usage_list[-1])

        if usage is None:
            # No usage history available - fall back to standard decode
            return self.decode(X, lengths=lengths)

        active_mask = usage >= min_usage
        # Guard: if all states are below threshold, keep the highest-usage state
        if active_mask.sum() == 0:
            warnings.warn(
                "prune_and_decode: all states below min_usage "
                f"({min_usage}). Keeping top-1 state."
            )
            active_mask[np.argmax(usage)] = True

        # Temporarily modify transition matrix and start probs
        orig_transmat = self.transmat_.copy()
        orig_startprob = self.startprob_.copy()

        try:
            inactive = ~active_mask
            # Zero out rows and columns for inactive states
            self.transmat_[inactive, :] = 0.0
            self.transmat_[:, inactive] = 0.0
            self.startprob_[inactive] = 0.0

            # Re-normalize rows (only active rows have mass)
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)  # avoid /0
            self.transmat_ /= row_sums

            sp_sum = self.startprob_.sum()
            if sp_sum > 0:
                self.startprob_ /= sp_sum
            else:
                # Fallback: uniform over active states
                self.startprob_[active_mask] = 1.0 / active_mask.sum()

            logprob, state_sequence = self.decode(X, lengths=lengths)
            return logprob, state_sequence
        finally:
            # Restore originals
            self.transmat_ = orig_transmat
            self.startprob_ = orig_startprob

    def _compute_expected_transitions(self, fwd_log_probs, bwd_log_probs, framelogprob, transmat, logprob, lengths=None):
        """Compute expected transition counts E[N(i->j)] = sum_t xi(t, i, j).

        Uses hmmlearn's Cython _hmmc.compute_log_xi_sum per sequence,
        with optional parallelism across sequences via joblib.

        Parameters
        ----------
        fwd_log_probs : ndarray, shape (n_samples, n_components)
        bwd_log_probs : ndarray, shape (n_samples, n_components)
        framelogprob : ndarray, shape (n_samples, n_components)
        transmat : ndarray, shape (n_components, n_components)
            Transition matrix (NOT log).
        logprob : float
        lengths : array-like, optional

        Returns
        -------
        expected_transitions : ndarray, shape (n_components, n_components)
        """
        n_samples, n_components = framelogprob.shape
        expected_transitions = np.zeros((n_components, n_components))

        if not np.isfinite(logprob):
            warnings.warn(f"Input logprob is non-finite ({logprob}). Cannot compute transitions.")
            return expected_transitions

        if lengths is None:
            lengths = [n_samples]

        n_jobs = getattr(self, 'n_jobs', 1)

        # Split arrays into per-sequence chunks
        seq_data = []
        pos = 0
        for length in lengths:
            if length <= 1:
                pos += length
                continue
            seq_data.append((
                fwd_log_probs[pos:pos + length],
                bwd_log_probs[pos:pos + length],
                framelogprob[pos:pos + length],
            ))
            pos += length

        if not seq_data:
            return expected_transitions

        if n_jobs != 1 and len(seq_data) > 1:
            # --- Parallel path: Cython xi per sequence via joblib ---
            results = Parallel(n_jobs=n_jobs)(
                delayed(_transitions_single_seq)(
                    s_fwd, s_bwd, s_flp, transmat
                )
                for s_fwd, s_bwd, s_flp in seq_data
            )
            for et in results:
                expected_transitions += et
        else:
            # --- Serial path: Cython xi per sequence ---
            for s_fwd, s_bwd, s_flp in seq_data:
                expected_transitions += _transitions_single_seq(
                    s_fwd, s_bwd, s_flp, transmat
                )

        expected_transitions = np.maximum(expected_transitions, 0)
        if not np.all(np.isfinite(expected_transitions)):
            warnings.warn("Non-finite values in expected transitions. Replacing with zeros.")
            expected_transitions = np.nan_to_num(expected_transitions)

        return expected_transitions


    def _update_hyperparameters(self, transition_counts):
        """Update concentration parameters alpha and gamma using auxiliary variable sampling."""
        transition_counts = np.maximum(transition_counts, 0) # Ensure non-negative

        # --- Alpha Update ---
        customers_per_row = transition_counts.sum(axis=1)
        tables_per_row = np.sum(transition_counts > 1e-10, axis=1)
        total_customers_alpha = customers_per_row.sum()
        total_tables_alpha = tables_per_row.sum()

        if total_customers_alpha > 1e-10 and total_tables_alpha > 0: # Use epsilon for float comparison
            try:
                eta_alpha = np.random.beta(self.alpha + 1, total_customers_alpha)
                eta_alpha = np.clip(eta_alpha, 1e-40, 1.0 - 1e-40) # Prevent eta=0 or 1

                log_pi_alpha = np.log(total_tables_alpha + 1e-40) - np.log(total_tables_alpha + self.alpha + 1e-40)
                # Efficient sampling from Bernoulli(pi_alpha)
                shape_alpha = total_tables_alpha + (np.log(np.random.rand() + 1e-40) < log_pi_alpha)
                scale_alpha = 1.0 / (-np.log(eta_alpha)) # eta_alpha already clipped
                self.alpha = np.random.gamma(shape=shape_alpha, scale=scale_alpha)
                self.alpha = max(self.alpha, 1e-6) # Ensure positivity
            except Exception as e:
                warnings.warn(f"Error during alpha sampling: {e}. Keeping previous alpha={self.alpha}")

        # --- Gamma Update ---
        n_customers_gamma = total_tables_alpha # Total tables generated
        n_dishes_gamma = np.sum(self.active_states) # Active states as proxy for K_dot
        n_dishes_gamma = max(n_dishes_gamma, 1) # Ensure at least 1 dish if any tables exist

        if n_customers_gamma > 1e-10 and n_dishes_gamma > 0:
             try:
                eta_gamma = np.random.beta(self.gamma + 1, n_customers_gamma)
                eta_gamma = np.clip(eta_gamma, 1e-40, 1.0 - 1e-40) # Prevent eta=0 or 1

                log_pi_gamma = np.log(n_dishes_gamma + 1e-40) - np.log(n_dishes_gamma + self.gamma + 1e-40)
                 # Efficient sampling from Bernoulli(pi_gamma)
                shape_gamma = n_dishes_gamma + (np.log(np.random.rand() + 1e-40) < log_pi_gamma)
                scale_gamma = 1.0 / (-np.log(eta_gamma)) # eta_gamma already clipped
                self.gamma = np.random.gamma(shape=shape_gamma, scale=scale_gamma)
                self.gamma = max(self.gamma, 1e-6) # Ensure positivity
             except Exception as e:
                 warnings.warn(f"Error during gamma sampling: {e}. Keeping previous gamma={self.gamma}")

        return self.alpha, self.gamma


    def _hdp_posterior_update(self, X, posteriors, framelogprob, fwdlattice, bwdlattice, logprob, lengths=None):
        """Update transition matrix, state weights, startprob, and hypers with sticky HDP posterior,
        using pre-computed E-step results.
        """
        if self.verbose: print("  Running HDP posterior update...")

        # 1. Update state activity based on current posteriors
        self._update_state_activity(posteriors)

        # 2. Compute expected transition counts E[N(i->j)]
        try:
            if self.verbose: print("    Computing expected transitions...")

            # Call with pre-computed E-step results
            # Note: _hmmc functions take raw transmat (not log)
            expected_transitions = self._compute_expected_transitions(
                fwdlattice, bwdlattice, framelogprob, self.transmat_, logprob, lengths
            )
                
            # Check remains the same
            if not np.all(np.isfinite(expected_transitions)):
                 warnings.warn("Non-finite values in expected transitions result. Replacing with zeros.")
                 expected_transitions = np.nan_to_num(expected_transitions, nan=0.0, posinf=0.0, neginf=0.0)
                 
            if self.verbose: print("    Expected transitions computed.")
        except Exception as e:
             warnings.warn(f"Failed to compute expected transitions: {e}. Skipping HDP updates for this iteration.")
             if self.verbose: traceback.print_exc()
             return self.transmat_ # Return unchanged matrix to avoid downstream errors

        # 3. Update Hyperparameters (alpha, gamma) using expected counts (if enabled)
        if self.learn_hyperparameters:
            if self.verbose: print("    Updating hyperparameters (alpha, gamma)...")
            self._update_hyperparameters(expected_transitions)
            if self.verbose: print(f"    Updated alpha={self.alpha:.3f}, gamma={self.gamma:.3f}")
        else:
            if self.verbose: print(f"    Keeping hyperparameters fixed: alpha={self.alpha:.3f}, gamma={self.gamma:.3f}")

        # Ensure expected_transitions are finite, replacing NaN/inf with 0
        if not np.all(np.isfinite(expected_transitions)):
            warnings.warn("Non-finite values detected in expected_transitions. Replacing with zeros.")
            expected_transitions = np.nan_to_num(expected_transitions, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Calculate Effective Counts (Combine observations with prior)
        if self.verbose: print("    Calculating effective counts...")
        effective_counts = np.zeros_like(expected_transitions)
        current_alpha = self.alpha
        current_beta = self.state_weights # Use the latest state_weights

        # Ensure priors are finite before using them
        if not np.all(np.isfinite(current_beta)):
            warnings.warn("Non-finite values in current_beta (state_weights). Resetting to uniform.")
            current_beta = np.ones_like(current_beta) / self.n_components
        if not np.isfinite(current_alpha):
            warnings.warn(f"Non-finite alpha detected ({current_alpha}). Resetting to 1.0.")
            current_alpha = 1.0
        # Add checks for kappa, rho if necessary, although less likely to be non-finite

        for i in range(self.n_components):
            base_counts = current_alpha * current_beta
            sticky_counts = np.zeros(self.n_components)
            sticky_counts[i] = self.kappa * self.rho
            effective_counts[i, :] = expected_transitions[i, :] + base_counts + sticky_counts

        # Clean effective_counts AFTER calculation
        if not np.all(np.isfinite(effective_counts)):
            warnings.warn("Non-finite values generated in effective_counts. Cleaning.")
            effective_counts = np.nan_to_num(effective_counts, nan=0.0, posinf=0.0, neginf=0.0)
        if self.verbose: print("    Effective counts calculated and cleaned.")

        # 5. Update Transition Matrix (self.transmat_)
        if self.verbose: print("    Updating transition matrix...")
        # Ensure effective counts are non-negative before summing
        effective_counts = np.maximum(effective_counts, 0)
        row_sums = effective_counts.sum(axis=1, keepdims=True)

        new_transmat = np.zeros_like(effective_counts)

        # Identify rows with essentially zero probability mass
        zero_sum_mask = (row_sums < 1e-40).flatten()
        # Identify rows with positive probability mass
        positive_sum_mask = ~zero_sum_mask

        # Assign uniform probability to zero-sum rows
        if np.any(zero_sum_mask):
            if self.verbose:
                print(f"    Assigning uniform probability to {np.sum(zero_sum_mask)} rows with zero effective counts.")
            uniform_prob = 1.0 / self.n_components
            new_transmat[zero_sum_mask] = uniform_prob

        # Normalize positive-sum rows
        if np.any(positive_sum_mask):
            # Get the sums for only the positive rows to avoid division by zero
            positive_row_sums = row_sums[positive_sum_mask]
            # Ensure the slice of effective_counts is also finite before dividing
            ec_positive = effective_counts[positive_sum_mask]
            if not np.all(np.isfinite(ec_positive)):
                warnings.warn("Non-finite values in effective_counts slice for positive rows. Cleaning before division.")
                ec_positive = np.nan_to_num(ec_positive, nan=0.0, posinf=0.0, neginf=0.0)
            new_transmat[positive_sum_mask] = ec_positive / positive_row_sums

        # Final precise normalization to correct floating point drift
        # Denominator should theoretically be 1.0 already for all rows
        final_sums = new_transmat.sum(axis=1, keepdims=True)
        # Handle any rows that *still* might sum to zero or NaN (shouldn't happen)
        problematic_rows = ~np.isfinite(final_sums) | (final_sums < 1e-40)
        if np.any(problematic_rows):
            warnings.warn("Problematic rows detected after normalization attempts. Forcing uniform.")
            new_transmat[problematic_rows.flatten()] = 1.0 / self.n_components
            final_sums = new_transmat.sum(axis=1, keepdims=True) # Recalculate
            final_sums[final_sums < 1e-40] = 1.0 # Avoid division by zero

        self.transmat_ = new_transmat / final_sums

        # --- Add a definitive check right after update --- 
        check_sums = self.transmat_.sum(axis=1)
        if not np.allclose(check_sums, 1.0, atol=1e-6):
             warnings.warn(f"transmat_ rows do not sum to 1 after robust update! Sums: {check_sums}")
             # Attempt force normalization again as a last resort
             self.transmat_ /= (self.transmat_.sum(axis=1, keepdims=True) + 1e-40)

        if self.verbose: print("    Transition matrix updated.")

        # 6. Update Global Weights (self.state_weights / beta) using approximation
        if self.verbose: print("    Updating global state weights (beta)...")
        global_effective_counts = effective_counts.sum(axis=0) # Sum columns
        # Posterior params = prior params + effective counts entering state j
        state_posterior_params = np.maximum(self.gamma * current_beta, 0) + np.maximum(global_effective_counts, 0)

        if np.sum(state_posterior_params) > 1e-10:
            # Deterministic posterior mean - consistent with MAP estimation
            # used everywhere else. The previous np.random.dirichlet() draw
            # was a hybrid (stochastic beta in an otherwise deterministic EM)
            # that caused LL oscillations without providing MCMC benefits.
            smoothed = state_posterior_params + 1e-20
            self.state_weights = smoothed / np.sum(smoothed)
        else:
             warnings.warn("All Dirichlet parameters for state_weights are near zero. Resetting to uniform.")
             self.state_weights = np.ones(self.n_components) / self.n_components
        # Ensure sum to 1
        self.state_weights /= (self.state_weights.sum() + 1e-40)
        if self.verbose: print("    Global state weights updated.")

        # 7. Update Initial State Distribution (self.startprob_)
        if self.verbose: print("    Updating start probabilities...")
        # Use the latest linear posteriors passed into this function
        self._update_startprob(posteriors, lengths)
        if self.verbose: print("    Start probabilities updated.")

        if self.verbose: print("  HDP posterior update finished.")
        return self.transmat_

    def fit(self, X, lengths=None):
        """Fit model to data using EM with HDP updates."""
        if self.verbose:
            print("Fitting weak-limit HMM model...")
            print(f"Config: n_components={self.n_components}, alpha={self.alpha:.2f}, gamma={self.gamma:.2f}, "
                  f"kappa={self.kappa:.1f}, rho={self.rho:.2f}, cov={self.covariance_type}, "
                  f"min_iter={self.min_iter}, max_iter={self.n_iter}, tol={self.tol}, "
                  f"n_jobs={self.n_jobs}")

        # Initialize convergence flag
        self.converged_ = False

        # Basic validation of input X
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("Input data X contains non-finite values (NaN or Inf).")

        # Set n_features attribute from input data shape
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Initialize parameters
        try:
              self._init_params(X)
              
              # Validate essential model parameters exist after initialization
              if not hasattr(self, 'means_') or self.means_ is None:
                  raise ValueError("means_ not properly initialized")
              if not hasattr(self, '_covars_') or self._covars_ is None:
                  raise ValueError("_covars_ not properly initialized")
              if not hasattr(self, 'transmat_') or self.transmat_ is None:
                  raise ValueError("transmat_ not properly initialized")
              
              # Check dimensions match the data
              if self.means_.shape[0] != self.n_components:
                  raise ValueError(f"means_ has wrong shape: {self.means_.shape}, expected first dimension: {self.n_components}")
              if self.means_.shape[1] != n_features:
                  raise ValueError(f"means_ has wrong feature dimension: {self.means_.shape[1]}, expected: {n_features}")
              
              # Ensure all parameters contain finite values
              if not np.all(np.isfinite(self.means_)):
                  raise ValueError("means_ contains non-finite values")
              if not np.all(np.isfinite(self._covars_)):
                  raise ValueError("_covars_ contains non-finite values")
              if not np.all(np.isfinite(self.transmat_)):
                  raise ValueError("transmat_ contains non-finite values")
                  
        except Exception as e:
             warnings.warn(f"Parameter initialization failed: {e}")
             traceback.print_exc()
             return self # Cannot proceed

        prev_log_likelihood = -np.inf
        # Clear history at the start of fitting
        self.history = {
            'log_likelihood': [],
            'state_usage': [],      # Full per-state usage array at each iteration
            'active_states': [],    # Count of states above threshold (backward compat)
            'alpha': [],
            'gamma': []
        }

        # --- Main EM Loop ---
        for iter_idx in range(self.n_iter):
            if self.verbose: print(f"\n--- Iteration {iter_idx} ---")

            # --- E-step: Compute Posteriors --- -> Now Unified E-step
            logprob_total = -np.inf
            posteriors = None # Linear probabilities
            framelogprob = None
            fwdlattice = None
            bwdlattice = None
            try:
                if self.verbose: print(f"[{iter_idx}] Running E-step...")
                # Call the unified E-step method with multi-sequence support
                logprob_total, posteriors, framelogprob, fwdlattice, bwdlattice = self._do_e_step(X, lengths)
                if self.verbose: print(f"[{iter_idx}] E-step finished. LL={logprob_total:.2f}")

                # Check results from E-step
                if not np.isfinite(logprob_total) or not np.all(np.isfinite(posteriors)):
                    warnings.warn(f"Iteration {iter_idx}: Non-finite values detected in E-step results. Stopping.")
                    break # Exit loop

            except Exception as e:
                 warnings.warn(f"Error during E-step in iteration {iter_idx}: {type(e).__name__}: {e}")
                 if self.verbose: traceback.print_exc()
                 break # Stop fitting

            # --- M-step for emission parameters --- 
            try:
                if self.verbose: print(f"[{iter_idx}] Running M-step...")
                stats = self._initialize_sufficient_statistics()
                # Pass only posteriors to accumulate stats
                self._accumulate_sufficient_statistics(stats, X, posteriors=posteriors)
                self._do_mstep(stats)
                if self.verbose: print(f"[{iter_idx}] M-step finished.")
            except Exception as e:
                 warnings.warn(f"Error during M-step (emissions) in iteration {iter_idx}: {type(e).__name__}: {e}")
                 if self.verbose: traceback.print_exc()
                 break

            # --- HDP posterior update --- 
            try:
                if self.verbose: print(f"[{iter_idx}] Running HDP Update...")
                # Pass results from the unified E-step
                self._hdp_posterior_update(X, posteriors, framelogprob, fwdlattice, bwdlattice, logprob_total, lengths)
                if self.verbose: print(f"[{iter_idx}] HDP Update finished.")
            except Exception as e:
                 warnings.warn(f"Error during HDP posterior update in iteration {iter_idx}: {type(e).__name__}: {e}")
                 if self.verbose: traceback.print_exc()
                 break

            # --- Log Likelihood for Convergence Check ---
            # Use logprob_total from the E-step (pre M-step parameters).
            # This is the standard EM convention and avoids a redundant
            # forward pass - self.score() would recompute the exact same
            # forward algorithm on the same data, just with post-update
            # parameters. The windowed convergence criterion (100-iter
            # windows) is robust to the pre/post distinction.
            curr_log_likelihood = logprob_total

            # --- Store History ---
            if self.verbose: print(f"[{iter_idx}] Storing history...")
            try:
                self.history['log_likelihood'].append(curr_log_likelihood)

                # Compute per-state usage from posteriors
                post_sum = posteriors.sum()
                if post_sum > 1e-10:
                    state_usage = posteriors.sum(axis=0) / post_sum  # Shape: (n_components,)
                else:
                    state_usage = np.zeros(self.n_components)
                self.history['state_usage'].append(state_usage.copy())

                # Count of states above threshold (backward compatibility)
                n_active = np.sum(state_usage > self.min_state_usage)
                self.history['active_states'].append(n_active)

                self.history['alpha'].append(self.alpha if hasattr(self, 'alpha') else np.nan)
                self.history['gamma'].append(self.gamma if hasattr(self, 'gamma') else np.nan)
                if self.verbose: print(f"[{iter_idx}] History stored. LL={curr_log_likelihood:.2f}, active={n_active}/{self.n_components}")
            except Exception as e:
                warnings.warn(f"Failed to store history at iteration {iter_idx}: {e}")

            # --- Check for Convergence ---
            if iter_idx > 0 and prev_log_likelihood != -np.inf:
                 improvement = curr_log_likelihood - prev_log_likelihood
                 # Calculate relative improvement (more robust than absolute for varying scales)
                 relative_improvement = abs(improvement) / max(abs(curr_log_likelihood), 1e-10)

                 if self.verbose:
                     print(f"[{iter_idx}] Convergence check: LL={curr_log_likelihood:.4f}, improvement={improvement:.4f}, "
                           f"relative={relative_improvement:.6e}, active_states={self.history['active_states'][-1]}, "
                           f"alpha={self.history['alpha'][-1]:.3f}, gamma={self.history['gamma'][-1]:.3f}")

                 if iter_idx >= self.min_iter:
                     # WINDOWED CONVERGENCE (mitigates posterior-update oscillations)
                     # Standard single-step convergence fails because posterior updates cause oscillations.
                     # Solution: Check if MEAN log-likelihood over windows is stable.
                     #
                     # Settings (optimized for Bayesian inference):
                     #   - Window size: 100 iterations (smooth over oscillations)
                     #   - Min iterations: 1000 (allow Bayesian updates to stabilize)
                     #   - Compare two consecutive windows of 100 iterations each

                     window_size = 100
                     # Need at least min_iter + 2*window_size to have two full windows
                     if len(self.history['log_likelihood']) >= self.min_iter + 2 * window_size:
                         ll_history = np.array(self.history['log_likelihood'])

                         # Compare mean LL of two consecutive windows
                         window1_mean = np.mean(ll_history[-(2*window_size):-window_size])  # iterations [N-200:N-100]
                         window2_mean = np.mean(ll_history[-window_size:])                   # iterations [N-100:N]
                         window_improvement = window2_mean - window1_mean
                         window_relative_improvement = abs(window_improvement) / max(abs(window2_mean), 1e-10)

                         # Print progress every 50 iterations to monitor convergence
                         if self.verbose and iter_idx % 50 == 0:
                             print(f"[{iter_idx}] Windowed convergence check:")
                             print(f"    Window 1 [{iter_idx-2*window_size}:{iter_idx-window_size}] mean LL: {window1_mean:.4f}")
                             print(f"    Window 2 [{iter_idx-window_size}:{iter_idx}] mean LL: {window2_mean:.4f}")
                             print(f"    Relative change: {window_relative_improvement:.6e} (tol={self.tol})")

                         # Convergence criterion: windowed means stable within tolerance
                         if window_relative_improvement < self.tol:
                              if self.verbose:
                                  print(f"\nConverged at iteration {iter_idx}:")
                                  print(f"  Windowed relative improvement {window_relative_improvement:.6e} < tol {self.tol}")
                                  print(f"  Window 1 mean: {window1_mean:.6f}")
                                  print(f"  Window 2 mean: {window2_mean:.6f}")
                              self.converged_ = True
                              break

                     # Log oscillations for debugging (not an error)
                     if improvement < 0 and self.verbose and iter_idx % 100 == 0:
                          print(f"[{iter_idx}] Oscillating (expected for HDP): recent change = {improvement:+.6f}")
                 else:
                     if self.verbose:
                         print(f"[{iter_idx}] Not checking tolerance yet. Minimum {self.min_iter} iterations required.")

            elif iter_idx == 0 and self.verbose: # Log initial state
                 print(f"[{iter_idx}] Initial state: LL={curr_log_likelihood:.4f}, "
                       f"active_states={self.history['active_states'][-1]}, alpha={self.history['alpha'][-1]:.3f}, gamma={self.history['gamma'][-1]:.3f}")

            prev_log_likelihood = curr_log_likelihood

            # Check if max iterations reached
            if iter_idx == self.n_iter - 1:
                warnings.warn(f"Maximum number of iterations ({self.n_iter}) reached without convergence (tol={self.tol}).")

        # Final checks and summary
        try:
            if not np.all(np.isfinite(self.startprob_)): warnings.warn("Final startprob_ contains non-finite values.")
            if not np.all(np.isfinite(self.transmat_)): warnings.warn("Final transmat_ contains non-finite values.")
            if not np.all(np.isfinite(self.means_)): warnings.warn("Final means_ contains non-finite values.")
            if not np.all(np.isfinite(self.covars_)): warnings.warn("Final covars_ contains non-finite values.")
        except Exception as final_check_e:
            warnings.warn(f"Error during final parameter checks: {final_check_e}")

        # NOTE: Pruning removed (Dec 2025) - downstream hierarchical clustering handles
        # state organization. Low-usage states will naturally form small clusters.
        # All n_components states are preserved in the final model.

        # Compute final state usage summary for diagnostics
        if self.history['state_usage']:
            final_usage = self.history['state_usage'][-1]
            n_active = np.sum(final_usage > self.min_state_usage)
            if self.verbose:
                print(f"\n--- Final State Usage Summary ---")
                print(f"  Total states: {self.n_components}")
                print(f"  States with usage > {self.min_state_usage:.1%}: {n_active}")
                # Show top 5 states by usage
                top_indices = np.argsort(final_usage)[::-1][:5]
                print(f"  Top 5 states by usage:")
                for i, idx in enumerate(top_indices):
                    print(f"    State {idx}: {final_usage[idx]:.2%}")

        if self.verbose:
            print("\nFitting finished.")
            if self.history['log_likelihood']:
                final_ll = self.history['log_likelihood'][-1]
                print(f"Final log likelihood: {final_ll:.4f} after {len(self.history['log_likelihood'])} iterations.")
                print(f"Model preserved all {self.n_components} states (no pruning).")
            else:
                print("No iterations completed successfully.")
        return self

    def plot_diagnostics(self, figsize=(18, 12)):
        """Plot model diagnostics from training history.

        Includes:
        - Log likelihood convergence
        - Number of active states over iterations
        - State usage heatmap (per-state usage across iterations)
        - Final state usage distribution
        """
        # Make sure history exists and has data
        if not hasattr(self, 'history') or not self.history or not self.history.get('log_likelihood'):
            print("No training history available. This might be because:")
            print("1. The model converged in 1 iteration (or failed early).")
            print("2. History tracking isn't properly implemented or was skipped.")
            return None # Return None if no history

        fig, axs = plt.subplots(2, 2, figsize=figsize)

        n_iterations = len(self.history['log_likelihood'])
        iterations = np.arange(n_iterations)

        # Plot 1: Log likelihood
        axs[0, 0].plot(iterations, self.history['log_likelihood'], marker='.' if n_iterations < 20 else None)
        axs[0, 0].set_title('Log Likelihood')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Log Likelihood')
        axs[0, 0].grid(True, linestyle=':')
        if n_iterations == 1: axs[0, 0].scatter(iterations, self.history['log_likelihood'])

        # Plot 2: Number of active states
        active_states_data = self.history.get('active_states', [])
        if active_states_data and len(active_states_data) == n_iterations:
            axs[0, 1].plot(iterations, active_states_data, marker='.' if n_iterations < 20 else None, color='green')
            if n_iterations == 1: axs[0, 1].scatter(iterations, active_states_data)
            axs[0, 1].set_title(f'Active States (usage > {self.min_state_usage:.1%})')
            axs[0, 1].set_xlabel('Iteration')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].grid(True, linestyle=':')
            axs[0, 1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            axs[0, 1].set_ylim(bottom=0, top=self.n_components + 1)
            axs[0, 1].axhline(y=self.n_components, color='r', linestyle='--', alpha=0.5, label=f'Max ({self.n_components})')
            axs[0, 1].legend()
        else:
            axs[0, 1].set_title('Active States (Data Missing)')

        # Plot 3: State usage heatmap over iterations
        state_usage_data = self.history.get('state_usage', [])
        if state_usage_data and len(state_usage_data) > 0:
            # Convert to array: (n_iterations, n_components)
            usage_matrix = np.array(state_usage_data)

            # Sample iterations for readability if too many
            if n_iterations > 100:
                # Sample every Nth iteration
                sample_rate = max(1, n_iterations // 100)
                sample_indices = np.arange(0, n_iterations, sample_rate)
                usage_matrix_sampled = usage_matrix[sample_indices, :]
                iteration_labels = sample_indices
            else:
                usage_matrix_sampled = usage_matrix
                iteration_labels = iterations

            # Sort states by final usage for better visualization
            final_usage = usage_matrix[-1, :]
            sorted_indices = np.argsort(final_usage)[::-1]  # Descending order

            im = axs[1, 0].imshow(usage_matrix_sampled[:, sorted_indices].T, aspect='auto', cmap='viridis',
                                   extent=[0, len(iteration_labels)-1, self.n_components-0.5, -0.5])
            axs[1, 0].set_title('State Usage Over Training (sorted by final usage)')
            axs[1, 0].set_xlabel('Iteration (sampled)' if n_iterations > 100 else 'Iteration')
            axs[1, 0].set_ylabel('State (sorted)')
            plt.colorbar(im, ax=axs[1, 0], label='Usage proportion')
        else:
            axs[1, 0].set_title('State Usage Heatmap (Data Missing)')

        # Plot 4: Final state usage bar chart
        if state_usage_data and len(state_usage_data) > 0:
            final_usage = state_usage_data[-1]
            state_indices = np.arange(self.n_components)

            # Color bars by whether they're "active"
            colors = ['green' if u > self.min_state_usage else 'lightgray' for u in final_usage]

            axs[1, 1].bar(state_indices, final_usage, color=colors, edgecolor='black', linewidth=0.5)
            axs[1, 1].axhline(y=self.min_state_usage, color='r', linestyle='--', alpha=0.7,
                              label=f'Threshold ({self.min_state_usage:.1%})')
            axs[1, 1].set_title('Final State Usage Distribution')
            axs[1, 1].set_xlabel('State Index')
            axs[1, 1].set_ylabel('Usage Proportion')
            axs[1, 1].legend()
            axs[1, 1].set_xlim(-0.5, self.n_components - 0.5)

            # Add count annotation
            n_active = np.sum(final_usage > self.min_state_usage)
            axs[1, 1].text(0.95, 0.95, f'Active: {n_active}/{self.n_components}',
                          transform=axs[1, 1].transAxes, ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axs[1, 1].set_title('Final State Usage (Data Missing)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'weak-limit HMM Training Diagnostics ({n_iterations} Iterations)', fontsize=14)
        return fig

    def _compute_posteriors(self, X):
        """
        Compute posterior probabilities of each state for each sample.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            
        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in X.
        """
        # Ensure X is properly formatted for our parent class methods
        X = np.asarray(X)
        
        # Use the parent class's methods directly by calling the unbound method
        # This is more stable than trying to reimplement the forward-backward pass
        
        # Get log probability using score method from parent class
        logprob = super(WeakLimitHMM, self).score(X)
        
        # Get posteriors using predict_proba from parent class
        posteriors = super(WeakLimitHMM, self).predict_proba(X)
        
        return logprob, posteriors

    def _accumulate_sufficient_statistics(self, stats, X, posteriors):
        """Accumulate sufficient statistics for M-step using pre-computed posteriors.
        
        Parameters
        ----------
        stats : dict
            Dictionary containing sufficient statistics (will be updated).
        X : array, shape (n_samples, n_features)
            Feature matrix of individual samples.
        posteriors : array, shape (n_samples, n_components)
            State posterior probabilities for each sample (pre-computed).
            
        Returns
        -------
        stats : dict
            Updated statistics dictionary.
        """
        if self.verbose: print("    Accumulating stats...")
        # Ensure inputs are properly formatted
        X = np.asarray(X)
        n_samples, n_features = X.shape
        n_components = self.n_components
        
        # Sum over all samples for each component (state occupancy)
        stats['post'] += posteriors.sum(axis=0)
        
        # Add observation statistics (means and covariances) - vectorized over T
        if 'm' in self.params:
            # posteriors.T: (K, T)  X: (T, D)  → stats['obs']: (K, D)
            stats['obs'] += posteriors.T @ X

        if 'c' in self.params:
            if self.covariance_type == 'full':
                # posteriors.T: (K, T)  X: (T, D)  → (K, D, D) outer products
                for c in range(n_components):
                    Xw = X * posteriors[:, c:c+1]  # (T, D) weighted rows
                    stats['obs*obs.T'][c] += Xw.T @ X  # (D, D)
            else:  # diag, spherical, tied - need weighted sum of squares
                # posteriors.T: (K, T)  X**2: (T, D)  → stats['obs**2']: (K, D)
                stats['obs**2'] += posteriors.T @ (X ** 2)
        
        # NO NEED to calculate transition stats here, as they are handled by HDP update
        # using _compute_expected_transitions.
        # Also, start stats are not needed for 'mc' params update.
        # if 's' in self.params:
        #     stats['start'] += posteriors[0]
        if self.verbose: print("    Stats accumulated.")
        return stats

    def _do_mstep(self, stats):
        """Update emission parameters (means and covars) using the provided stats.
        
        This overrides the parent class method to ensure we only update
        the parameters specified in self.params, and not the transition
        or start probabilities, which are handled by our HDP updates.
        We implement the updates manually based on stats for compatibility.
        """
        # Handle different stat dictionary keys for compatibility
        obs_key = 'obs'
        post_key = 'post'
        obs_sq_key = 'obs**2'
        obs_outer_key = 'obs*obs.T'
        
        # Fallback key detection logic (remains the same)
        if obs_key not in stats:
            for key in stats.keys():
                if key.startswith('obs') and not key.startswith('obs*') and not key.startswith('obs**'):
                    obs_key = key
                    break
        if post_key not in stats:
            for key in stats.keys():
                if key.startswith('post'):
                    post_key = key
                    break
        if obs_sq_key not in stats:
             for key in stats.keys():
                 if key.startswith('obs**'):
                     obs_sq_key = key
                     break
        if obs_outer_key not in stats and self.covariance_type == 'full':
             for key in stats.keys():
                 if key.startswith('obs*'):
                     obs_outer_key = key
                     break

        # --- Update Means --- 
        if 'm' in self.params and obs_key in stats and post_key in stats:
            if self.verbose: print("    Updating means...")
            denom_mean = stats[post_key][:, np.newaxis] + 10 * np.finfo(float).eps
            self.means_ = stats[obs_key] / denom_mean
            if not np.all(np.isfinite(self.means_)):
                warnings.warn("Non-finite values encountered in means_ during M-step update.")
        
        # --- Update Covariances --- 
        if 'c' in self.params and post_key in stats:
            if self.verbose: print(f"    Updating covars ({self.covariance_type})...")
            denom_covar = stats[post_key] # Shape (n_components,) - used differently depending on type
            min_covar_val = self.min_covar # Regularization term
            
            if self.covariance_type == 'full' and obs_outer_key in stats and obs_key in stats:
                # Formula: E[XX^T] - E[X]E[X]^T
                # stats[obs_outer_key] shape: (n_components, n_features, n_features)
                # stats[obs_key] shape: (n_components, n_features)
                # denom_covar shape: (n_components,)
                # self.means_ shape: (n_components, n_features)
                full_covars = np.zeros((self.n_components, self.n_features, self.n_features))
                for c in range(self.n_components):
                    if denom_covar[c] > 1e-10:
                        ExxT = stats[obs_outer_key][c] / denom_covar[c]
                        Ex = self.means_[c][:, np.newaxis] # Shape (n_features, 1)
                        ExExT = Ex @ Ex.T # Outer product
                        cov = ExxT - ExExT
                        # Enforce symmetry before adding jitter
                        cov = 0.5 * (cov + cov.T)
                        cov.flat[::self.n_features + 1] += min_covar_val # Add to diagonal
                        try:
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            eigvals = np.clip(eigvals, min_covar_val, None)
                            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        except Exception as eig_e:
                            warnings.warn(f"Eigenvalue regularization failed for component {c}: {eig_e}. Using diagonal fallback.")
                            cov = np.eye(self.n_features) * min_covar_val
                        full_covars[c] = cov
                    else: # Handle non-occupied states
                        full_covars[c] = np.eye(self.n_features) * min_covar_val
                self.covars_ = full_covars
                        
            elif self.covariance_type == 'diag' and obs_sq_key in stats and obs_key in stats:
                # Formula: E[X^2] - (E[X])^2
                # stats[obs_sq_key] shape: (n_components, n_features)
                # denom_covar shape: (n_components,)
                # self.means_ shape: (n_components, n_features)
                denom_diag = denom_covar[:, np.newaxis] + 10 * np.finfo(float).eps
                Ex2 = stats[obs_sq_key] / denom_diag
                Ex_sq = self.means_ ** 2
                self.covars_ = np.maximum(Ex2 - Ex_sq, min_covar_val)
                
            elif self.covariance_type == 'tied' and obs_outer_key in stats and obs_key in stats:
                # Average full covariance over all components
                # Sum stats across components first
                sum_post = np.sum(denom_covar)
                if sum_post > 1e-10:
                    sum_obs_outer = np.sum(stats[obs_outer_key], axis=0) # Sum over components
                    sum_obs = np.sum(stats[obs_key], axis=0) # Sum over components
                    avg_ExxT = sum_obs_outer / sum_post
                    # Need average E[X] - weighted average of means by posterior
                    avg_Ex = sum_obs / sum_post # Shape (n_features,)
                    avg_Ex = avg_Ex[:, np.newaxis] # Shape (n_features, 1)
                    avg_ExExT = avg_Ex @ avg_Ex.T
                    cov = avg_ExxT - avg_ExExT
                    cov.flat[::self.n_features + 1] += min_covar_val # Add to diagonal
                    self.covars_ = cov
                else:
                     self.covars_ = np.eye(self.n_features) * min_covar_val
                     
            elif self.covariance_type == 'spherical' and obs_sq_key in stats and obs_key in stats:
                # Average diagonal covariance over all components and features
                sum_post = np.sum(denom_covar)
                if sum_post > 1e-10:
                     # Calculate average diagonal variance first
                     denom_diag = denom_covar[:, np.newaxis] + 10 * np.finfo(float).eps
                     Ex2 = stats[obs_sq_key] / denom_diag
                     Ex_sq = self.means_ ** 2
                     diag_vars = Ex2 - Ex_sq # Shape (n_components, n_features)
                     # Weighted average variance across components and features
                     # Weight by posterior probability for each component
                     total_var = np.sum(diag_vars * denom_diag) # Sum of weighted variances
                     total_weight = sum_post * self.n_features # Total posterior mass * n_features
                     avg_var = total_var / total_weight
                     # Apply minimum covariance and broadcast to shape (n_components,)
                     self.covars_ = np.full((self.n_components,), np.maximum(avg_var, min_covar_val))
                else:
                     self.covars_ = np.full((self.n_components,), min_covar_val)
                     
            else:
                warnings.warn(f"Covariance type '{self.covariance_type}' not handled or required stats missing (keys: {list(stats.keys())}). Skipping covar update.")

            # --- Post-update Check and _check() call --- 
            # After updating covars_, call _check() to ensure internal consistency
            try:
                self._check() # This should validate params and update _covars_ if needed
                if self.verbose: print("    self._check() called after covar update.")
            except AttributeError:
                warnings.warn("_check() method not found, internal covariance consistency might be compromised.")
            except Exception as check_e:
                warnings.warn(f"Error during self._check() after M-step: {check_e}")
                
            # Final check on the *internal* representation if it exists
            if hasattr(self, '_covars_') and not np.all(np.isfinite(self._covars_)):
                 warnings.warn("Non-finite values encountered in _covars_ after M-step update and check.")
            elif not np.all(np.isfinite(self.covars_)):
                 warnings.warn("Non-finite values encountered in covars_ after M-step update and check.")
        
        # Don't update 's' (startprob) or 't' (transmat) here - let HDP logic handle it

    def _do_forward_pass(self, framelogprob, lengths=None):
        """Forward pass with multi-sequence support and optional parallelism.

        Uses hmmlearn's Cython _hmmc.forward_log for each sequence.
        When n_jobs > 1 and multiple sequences exist, processes sequences in
        parallel using joblib.
        """
        n_samples, n_components = framelogprob.shape
        startprob = self.startprob_
        transmat = self.transmat_

        if lengths is None:
            lengths = [n_samples]

        n_jobs = getattr(self, 'n_jobs', 1)

        if n_jobs != 1 and len(lengths) > 1:
            # --- Parallel path: Cython forward per sequence via joblib ---
            seq_chunks = []
            pos = 0
            for length in lengths:
                seq_chunks.append(framelogprob[pos:pos + length])
                pos += length

            results = Parallel(n_jobs=n_jobs)(
                delayed(_forward_single_seq)(startprob, transmat, chunk)
                for chunk in seq_chunks if len(chunk) > 0
            )

            fwdlattice = np.zeros((n_samples, n_components))
            total_logprob = 0.0
            pos = 0
            for i, (fwd_chunk, seq_lp) in enumerate(results):
                length = len(fwd_chunk)
                fwdlattice[pos:pos + length] = fwd_chunk
                total_logprob += seq_lp
                pos += length

            return total_logprob, fwdlattice

        # --- Serial path: Cython forward per sequence ---
        fwdlattice = np.zeros((n_samples, n_components))
        current_pos = 0
        total_logprob = 0.0

        for length in lengths:
            if length < 1:
                current_pos += length
                continue

            seq_flp = framelogprob[current_pos:current_pos + length]
            seq_lp, seq_fwd = _hmmc_forward_log(startprob, transmat, seq_flp)
            fwdlattice[current_pos:current_pos + length] = seq_fwd
            total_logprob += seq_lp
            current_pos += length

        return total_logprob, fwdlattice
    
    def _do_backward_pass(self, framelogprob, lengths=None):
        """Backward pass with multi-sequence support and optional parallelism.

        Uses hmmlearn's Cython _hmmc.backward_log for each sequence.
        When n_jobs > 1 and multiple sequences exist, processes sequences in
        parallel using joblib.
        """
        n_samples, n_components = framelogprob.shape
        startprob = self.startprob_
        transmat = self.transmat_

        if lengths is None:
            lengths = [n_samples]

        n_jobs = getattr(self, 'n_jobs', 1)

        if n_jobs != 1 and len(lengths) > 1:
            # --- Parallel path: Cython backward per sequence via joblib ---
            seq_chunks = []
            pos = 0
            for length in lengths:
                seq_chunks.append(framelogprob[pos:pos + length])
                pos += length

            results = Parallel(n_jobs=n_jobs)(
                delayed(_backward_single_seq)(startprob, transmat, chunk)
                for chunk in seq_chunks if len(chunk) > 0
            )

            bwdlattice = np.zeros((n_samples, n_components))
            pos = 0
            for bwd_chunk in results:
                length = len(bwd_chunk)
                bwdlattice[pos:pos + length] = bwd_chunk
                pos += length

            return bwdlattice

        # --- Serial path: Cython backward per sequence ---
        bwdlattice = np.zeros((n_samples, n_components))
        current_pos = 0

        for length in lengths:
            if length < 1:
                current_pos += length
                continue

            seq_flp = framelogprob[current_pos:current_pos + length]
            seq_bwd = _hmmc_backward_log(startprob, transmat, seq_flp)
            bwdlattice[current_pos:current_pos + length] = seq_bwd
            current_pos += length

        return bwdlattice

    def _initialize_sufficient_statistics(self):
        """Initialize sufficient statistics required for M step.
        
        This is a compatibility wrapper to ensure consistent API.
        
        Returns
        -------
        stats : dict
            Dictionary containing sufficient statistics.
        """
        try:
            # Try using parent implementation
            return super(WeakLimitHMM, self)._initialize_sufficient_statistics()
        except AttributeError:
            # If parent implementation doesn't exist, create our own stats dict
            stats = {
                'post': np.zeros((self.n_components,)),
                'obs': np.zeros((self.n_components, self.n_features)),
                'obs**2': np.zeros((self.n_components, self.n_features)),
                'trans': np.zeros((self.n_components, self.n_components)),
                'start': np.zeros(self.n_components),
            }
            if self.covariance_type == 'full':
                stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features, self.n_features))
                
            return stats

    def _do_e_step(self, X, lengths=None):
        """Performs the E-step of the EM algorithm with multi-sequence support.

        Computes log-likelihood, posteriors, and forward/backward lattices.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix (possibly concatenated sequences).
        lengths : array-like of integers, shape (n_sequences,), optional
            Lengths of the individual sequences in X. If None, X is treated as a single sequence.

        Returns
        -------
        logprob : float
            Log likelihood of the data.
        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of states.
        framelogprob : array, shape (n_samples, n_components)
            Log likelihood of observations given states.
        fwdlattice : array, shape (n_samples, n_components)
            Forward probabilities (log domain).
        bwdlattice : array, shape (n_samples, n_components)
            Backward probabilities (log domain).
        """
        if self.verbose: print("    Running unified E-step...")
        X = np.asarray(X)

        try:
            # 1. Compute emission log probabilities P(x_t | z_t=k)
            framelogprob = self._compute_log_likelihood(X)
            if not np.all(np.isfinite(framelogprob)):
                 warnings.warn("Non-finite values detected in framelogprob. Clamping.")
                 framelogprob = np.nan_to_num(framelogprob, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

            # 2. Compute forward pass (alpha) with multi-sequence support
            logprob, fwdlattice = self._do_forward_pass(framelogprob, lengths)
            if not np.all(np.isfinite(fwdlattice)):
                 warnings.warn("Non-finite values detected in fwdlattice. Clamping.")
                 fwdlattice = np.nan_to_num(fwdlattice, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            if not np.isfinite(logprob):
                 warnings.warn(f"Non-finite logprob ({logprob}) from forward pass. Setting to -inf.")
                 logprob = -np.inf

            # 3. Compute backward pass (beta) with multi-sequence support
            bwdlattice = self._do_backward_pass(framelogprob, lengths)
            if not np.all(np.isfinite(bwdlattice)):
                 warnings.warn("Non-finite values detected in bwdlattice. Clamping.")
                 bwdlattice = np.nan_to_num(bwdlattice, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

            # 4. Compute posteriors (gamma) P(z_t=k | X)
            # gamma_t(k) = alpha_t(k) * beta_t(k) / P(X)
            # In log space: log(gamma_t(k)) = log(alpha_t(k)) + log(beta_t(k)) - logP(X)
            with np.errstate(under="ignore", invalid='ignore'):
                log_posteriors = fwdlattice + bwdlattice - logprob # Subtract overall log-likelihood for normalization
                # Normalize again carefully in log space to prevent issues if logprob was bad
                log_posteriors -= special_logsumexp(log_posteriors, axis=1)[:, np.newaxis]

            # Convert back from log space
            with np.errstate(under='ignore'):
                posteriors = np.exp(log_posteriors)
            # Clean final posteriors
            posteriors = np.nan_to_num(posteriors, nan=(1.0/self.n_components), posinf=0.0, neginf=0.0)
            posteriors /= (posteriors.sum(axis=1, keepdims=True) + 1e-40) # Ensure sum to 1

            if self.verbose: print("    Unified E-step finished.")
            return logprob, posteriors, framelogprob, fwdlattice, bwdlattice

        except Exception as e:
             warnings.warn(f"Error during unified E-step: {type(e).__name__}: {e}")
             if self.verbose: traceback.print_exc()
             # Return dummy values indicating failure
             n_samples = X.shape[0]
             dummy_post = np.ones((n_samples, self.n_components)) / self.n_components
             dummy_frame = np.zeros((n_samples, self.n_components)) - np.inf
             dummy_fwd = np.zeros((n_samples, self.n_components)) - np.inf
             dummy_bwd = np.zeros((n_samples, self.n_components)) - np.inf
             return -np.inf, dummy_post, dummy_frame, dummy_fwd, dummy_bwd


# =============================================================================
# Model inspection utilities
# =============================================================================

def infer_n_active_states(model, min_state_usage=0.01):
    """Infer count of active states from model usage history.

    Counts states whose final-iteration usage fraction exceeds
    `min_state_usage`. This threshold (0.01 by default) matches
    FIXED_PARAMS['min_state_usage'] in config/combined_hmm_config.py.

    Falls back to `model.n_components` (conservative upper bound) when usage
    history is unavailable - e.g., for freshly-loaded models or models that
    terminated before the first usage update. This fallback is HMM
    specific: the true active count is unknown without training history, so
    over-counting is safer than under-counting.

    Args:
        model:           Fitted WeakLimitHMM instance.
        min_state_usage: Occupancy fraction threshold (default 0.01).

    Returns:
        int: Number of states with usage > min_state_usage, or n_components.
    """
    if hasattr(model, 'history') and model.history:
        usage_hist = model.history.get('state_usage')
        if usage_hist:
            usage = np.asarray(usage_hist[-1], dtype=float)
            if usage.size > 0 and np.all(np.isfinite(usage)):
                return int(np.sum(usage > min_state_usage))
    return int(model.n_components)


# Backward-compatible alias: models pickled before the 2026-06 rename store the
# class as `utils.hdphmm.StickyHDPHMM`; pickle.load resolves that name here.
StickyHDPHMM = WeakLimitHMM
