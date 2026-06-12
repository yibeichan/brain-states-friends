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
