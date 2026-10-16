"""
Recurrence computation utilities.

Pure functions for computing fractional occupancy, recurrence scores,
and season-specificity metrics from decoded HMM state sequences.

These functions were extracted from 05a_recurrence_analysis.py to allow
import without module-level side effects (load_dotenv, logging.basicConfig).
"""

import logging

import numpy as np

from utils.common import _get_season

logger = logging.getLogger(__name__)

# A state is "active" in a run when its fractional occupancy in that run
# exceeds this value, and belongs to a participant's active repertoire when
# it is active in at least one run. Every analysis that labels states active
# from decoded sequences (Friends and the held-out stimuli) uses this value.
# It is distinct from K_used, the count of states whose posterior usage in the
# fitted model exceeds config min_state_usage (0.01), which sets the BIC
# parameter count.
ACTIVE_FO_THRESHOLD = 0.02

# Posterior-usage cut that defines K_used. It must equal config
# min_state_usage and the StickyHDPHMM default, which apply the same rule
# during fitting.
USED_STATE_THRESHOLD = 0.01


def check_fo_threshold_matches_05a(recurrence_dir, fo_threshold):
    """Raise if 05a labelled Friends states with a different FO threshold.

    The held-out stages take the Friends-inactive set from 05a's saved
    recurrence scores and test held-out presence with ``fo_threshold``; the
    comparison assumes both sides use the same rule.
    """
    import json
    import os
    path = os.path.join(recurrence_dir, "recurrence_summary.json")
    with open(path) as f:
        used = float(json.load(f)["fo_active_threshold"])
    if abs(used - float(fo_threshold)) > 1e-12:
        raise ValueError(
            f"--fo_threshold {fo_threshold} differs from the {used} that 05a "
            f"used for the Friends active set ({path}); rerun 05a with the "
            "same threshold or pass the same value here.")


def compute_fractional_occupancy(decoded_states, n_states):
    """Compute fractional occupancy per run and per state.

    FO_k_e = count(state_seq == k) / len(state_seq)

    Args:
        decoded_states: dict  run_id -> np.array(n_trs,) of state indices
        n_states:       Total number of model states (n_components)

    Returns:
        fo: dict  run_id -> np.array(n_states,) of fractional occupancies
    """
    fo = {}
    for run_id, state_seq in decoded_states.items():
        if len(state_seq) == 0:
            logger.warning(f"compute_fo: empty state_seq for {run_id}")
            fo[run_id] = np.zeros(n_states)
            continue
        counts = np.bincount(state_seq, minlength=n_states).astype(float)
        fo[run_id] = counts / len(state_seq)
    return fo


def compute_recurrence_scores(fo, n_states, fo_threshold):
    """Compute recurrence score per state.

    ACTIVE_k_e = 1 if FO_k_e > fo_threshold
    recurrence_k = |{e : ACTIVE_k_e}| / |E|

    A score of 1.0 means the state appears in every run (fully
    context-invariant); <0.10 indicates an episode-specific state.

    Args:
        fo:           dict  run_id -> np.array(n_states,)
                      (run-level FO)
        n_states:     Number of states
        fo_threshold: Minimum FO to count as "active"

    Returns:
        recurrence: np.array(n_states,) in [0, 1]
    """
    n_units = len(fo)
    if n_units == 0:
        return np.zeros(n_states)
    active_counts = np.zeros(n_states)
    for fo_vec in fo.values():
        active_counts += (fo_vec > fo_threshold).astype(float)
    return active_counts / n_units


def active_state_indices(decoded_states, n_states,
                         fo_threshold=ACTIVE_FO_THRESHOLD):
    """Indices of states active (FO > fo_threshold) in at least one run."""
    fo = compute_fractional_occupancy(decoded_states, n_states)
    recurrence = compute_recurrence_scores(fo, n_states, fo_threshold)
    return np.where(recurrence > 0)[0]


def used_state_indices(model, min_state_usage=USED_STATE_THRESHOLD):
    """Indices of states the fitted model uses (K_used).

    A state is used when its posterior usage at the last EM iteration exceeds
    ``min_state_usage`` (config ``min_state_usage``, 0.01), the same rule as
    ``utils.hdphmm.infer_n_active_states``. Returns an empty array when the
    model carries no usage history.
    """
    history = getattr(model, "history", None) or {}
    usage_hist = history.get("state_usage")
    if not usage_hist:
        return np.array([], dtype=int)
    usage = np.asarray(usage_hist[-1], dtype=float)
    if usage.size == 0 or not np.all(np.isfinite(usage)):
        return np.array([], dtype=int)
    return np.where(usage > min_state_usage)[0]


def states_present_in_runs(fo_by_run, states, fo_threshold=ACTIVE_FO_THRESHOLD):
    """States among ``states`` with FO > fo_threshold in at least one run.

    Used to ask whether a state inactive in Friends becomes present in a
    held-out stimulus, under the same rule that defines Friends-active states.

    Args:
        fo_by_run: list of np.array(n_states,), one FO vector per run.
        states:    iterable of state indices to check.

    Returns:
        dict state -> {'max_run_fo', 'mean_fo', 'n_runs_active'}.
    """
    if not fo_by_run:
        return {}
    fo = np.vstack(fo_by_run)
    out = {}
    for k in states:
        col = fo[:, k]
        if col.max() > fo_threshold:
            out[int(k)] = {
                'max_run_fo': float(col.max()),
                'mean_fo': float(col.mean()),
                'n_runs_active': int((col > fo_threshold).sum()),
            }
    return out


def compute_pooled_decoded_occupancy(decoded_states, n_states):
    """Compute pooled decoded occupancy across all runs."""
    total_counts = np.zeros(n_states, dtype=float)
    total_trs = 0
    for state_seq in decoded_states.values():
        total_counts += np.bincount(state_seq, minlength=n_states)
        total_trs += len(state_seq)
    pooled = total_counts / total_trs if total_trs > 0 else np.zeros(n_states)
    return pooled, int(total_trs)


def compute_per_season_recurrence(fo, n_states, available_seasons, fo_threshold,
                                   season_override=None):
    """Compute mean FO and recurrence per season per state.

    Args:
        fo:                dict  run_id -> np.array(n_states,)
        n_states:          Number of states
        available_seasons: List of season ints (e.g. [1, 2, 3, 4, 5, 6])
        fo_threshold:      Minimum FO to count as "active"
        season_override:   Optional dict run_id -> season (int). When provided,
                           use this mapping instead of extracting season from
                           run_id strings. Used by permutation test to avoid
                           key collision from re-keying run IDs.

    Returns:
        per_season_mean_fo:  dict  season (int) -> np.array(n_states,) mean FO
        season_recurrence:   dict  season (int) -> np.array(n_states,) recurrence
    """
    season_fo = {s: [] for s in available_seasons}
    for run_id, fo_ep in fo.items():
        if season_override is not None:
            s = season_override.get(run_id)
            if s is None:
                continue
        else:
            try:
                s = _get_season(run_id)
            except ValueError:
                continue
        if s in season_fo:
            season_fo[s].append(fo_ep)

    per_season_mean_fo = {}
    season_recurrence = {}
    for s in available_seasons:
        eps = season_fo[s]
        if not eps:
            per_season_mean_fo[s] = np.zeros(n_states)
            season_recurrence[s] = np.zeros(n_states)
        else:
            stack = np.stack(eps)  # (n_ep, n_states)
            per_season_mean_fo[s] = stack.mean(axis=0)
            season_recurrence[s] = (stack > fo_threshold).mean(axis=0)

    return per_season_mean_fo, season_recurrence


def compute_specificity_index(season_recurrence):
    """Season-specificity index = range (max - min) of per-season recurrence.

    0 = perfectly context-invariant; 1 = completely season-specific.
    Scale-free and directly interpretable.

    Args:
        season_recurrence: dict  season (int) -> np.array(n_states,)

    Returns:
        specificity: np.array(n_states,)
    """
    seasons = sorted(season_recurrence.keys())
    if len(seasons) < 2:
        logger.warning(
            "compute_specificity_index: fewer than 2 seasons - returning zeros"
        )
        return np.zeros(len(next(iter(season_recurrence.values()))))
    stack = np.stack([season_recurrence[s] for s in seasons])  # (n_seasons, n_states)
    return stack.max(axis=0) - stack.min(axis=0)
