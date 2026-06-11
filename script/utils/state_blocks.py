#!/usr/bin/env python3
"""
Helpers for extracting contiguous state blocks from decoded state sequences.
"""

import csv
import gzip
import json
import logging
from pathlib import Path

import numpy as np

from utils.common import _get_season


TR_SECONDS = 1.49

BLOCK_FIELDNAMES = [
    'run_id',
    'season',
    'block_index',
    'state',
    'recurrence_score',
    'episode_length_tr',
    'start_tr',
    'end_tr',
    'duration_tr',
    'start_time_s',
    'end_time_s',
    'duration_s',
]

EPISODE_STATE_FIELDNAMES = [
    'run_id',
    'season',
    'state',
    'recurrence_score',
    'episode_length_tr',
    'n_blocks',
    'first_start_tr',
    'last_end_tr',
    'total_duration_tr',
    'mean_block_duration_tr',
    'fractional_occupancy',
    'active_for_recurrence',
    'recurrence_score',
    'first_start_time_s',
    'last_end_time_s',
    'total_duration_s',
    'mean_block_duration_s',
]


def extract_state_block_records(
    decoded_states,
    recurrence_scores,
    include_states=None,
    tr_seconds=TR_SECONDS,
):
    """Return one record per contiguous block in the decoded state sequences.

    Parameters
    ----------
    decoded_states : dict
        run_id -> np.array of state indices.
    recurrence_scores : np.ndarray
        Per-state recurrence scores, shape (n_states,).
    include_states : set[int] | None
        If given, only include blocks for these state IDs.
    tr_seconds : float
        TR duration in seconds.

    Notes
    -----
    ``end_tr`` follows Python slicing semantics and is exclusive.
    """
    include_states = set(include_states) if include_states is not None else None
    records = []

    for run_id, state_seq in decoded_states.items():
        state_seq = np.asarray(state_seq)
        episode_length_tr = int(len(state_seq))
        if episode_length_tr == 0:
            continue

        try:
            season = _get_season(run_id)
        except ValueError:
            season = None

        change_points = np.flatnonzero(state_seq[1:] != state_seq[:-1]) + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [episode_length_tr]))
        states = state_seq[starts].astype(int)

        for block_index, (state_id, start_tr, end_tr) in enumerate(
            zip(states, starts, ends)
        ):
            state_id = int(state_id)
            start_tr = int(start_tr)
            end_tr = int(end_tr)
            if include_states is not None and state_id not in include_states:
                continue

            duration_tr = end_tr - start_tr
            records.append({
                'run_id': run_id,
                'season': season,
                'block_index': block_index,
                'state': state_id,
                'recurrence_score': float(recurrence_scores[state_id]),
                'episode_length_tr': episode_length_tr,
                'start_tr': start_tr,
                'end_tr': end_tr,
                'duration_tr': duration_tr,
                'start_time_s': start_tr * tr_seconds,
                'end_time_s': end_tr * tr_seconds,
                'duration_s': duration_tr * tr_seconds,
            })

    return records


def summarize_block_records(
    block_records,
    recurrence_scores,
    fo_threshold,
    tr_seconds=TR_SECONDS,
):
    """Aggregate block records into one row per (run_id, state)."""
    grouped = {}

    for record in block_records:
        key = (record['run_id'], record['state'])
        if key not in grouped:
            grouped[key] = {
                'run_id': record['run_id'],
                'season': record['season'],
                'state': record['state'],
                'episode_length_tr': record['episode_length_tr'],
                'n_blocks': 0,
                'first_start_tr': record['start_tr'],
                'last_end_tr': record['end_tr'],
                'total_duration_tr': 0,
            }

        grouped[key]['n_blocks'] += 1
        grouped[key]['first_start_tr'] = min(
            grouped[key]['first_start_tr'], record['start_tr']
        )
        grouped[key]['last_end_tr'] = max(
            grouped[key]['last_end_tr'], record['end_tr']
        )
        grouped[key]['total_duration_tr'] += record['duration_tr']

    summary_rows = []
    for key in sorted(grouped):
        row = grouped[key]
        total_duration_tr = int(row['total_duration_tr'])
        n_blocks = int(row['n_blocks'])
        episode_length_tr = int(row['episode_length_tr'])
        fractional_occupancy = (
            float(total_duration_tr / episode_length_tr) if episode_length_tr else 0.0
        )
        mean_block_duration_tr = (
            float(total_duration_tr / n_blocks) if n_blocks else 0.0
        )
        first_start_tr = int(row['first_start_tr'])
        last_end_tr = int(row['last_end_tr'])
        state_id = int(row['state'])

        summary_rows.append({
            'run_id': row['run_id'],
            'season': row['season'],
            'state': state_id,
            'episode_length_tr': episode_length_tr,
            'n_blocks': n_blocks,
            'first_start_tr': first_start_tr,
            'last_end_tr': last_end_tr,
            'total_duration_tr': total_duration_tr,
            'mean_block_duration_tr': mean_block_duration_tr,
            'fractional_occupancy': fractional_occupancy,
            'active_for_recurrence': fractional_occupancy > fo_threshold,
            'recurrence_score': float(recurrence_scores[state_id]),
            'first_start_time_s': first_start_tr * tr_seconds,
            'last_end_time_s': last_end_tr * tr_seconds,
            'total_duration_s': total_duration_tr * tr_seconds,
            'mean_block_duration_s': mean_block_duration_tr * tr_seconds,
        })

    return summary_rows


def write_records_csv(path, fieldnames, records):
    """Write records to CSV or CSV.GZ depending on the path suffix."""
    path = Path(path)
    open_fn = gzip.open if str(path).endswith('.gz') else open
    with open_fn(path, 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_eligible_states(recurrence_dir):
    """Load eligible state IDs from 05a's eligible_states.json.

    Parameters
    ----------
    recurrence_dir : str or Path
        Path to the 05a output directory (e.g.,
        ``{SCRATCH_DIR}/output/05a_recurrence_analysis/{parc}/{sub_id}/``).

    Returns
    -------
    eligible_ids : list[int]
        State IDs that passed the sub-HRF filter.
    excluded_ids : list[int]
        State IDs excluded as sub-HRF.
    metadata : dict
        Full contents of ``eligible_states.json`` (criterion, threshold, note).

    Raises
    ------
    FileNotFoundError
        If ``eligible_states.json`` does not exist in *recurrence_dir*.
    """
    logger = logging.getLogger(__name__)
    path = Path(recurrence_dir) / 'eligible_states.json'
    with open(path) as f:
        data = json.load(f)
    eligible_ids = data['eligible_state_ids']
    excluded_ids = data['excluded_sub_hrf_state_ids']
    logger.info(
        "Loaded eligible states from %s: %d eligible, %d sub-HRF excluded "
        "(criterion=%s, threshold=%.1f TRs)",
        path, len(eligible_ids), len(excluded_ids),
        data.get('criterion', '?'), data.get('threshold_tr', 0),
    )
    return eligible_ids, excluded_ids, data
