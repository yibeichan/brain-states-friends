#!/usr/bin/env python
"""
Common utility functions shared across analysis scripts.
"""
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_n_jobs(n_jobs):
    """Resolve worker count, preferring the SLURM CPU allocation when present.

    ``n_jobs=-1`` is the "auto" sentinel: first check ``SLURM_CPUS_PER_TASK``
    (so SLURM jobs don't oversubscribe), falling back to ``os.cpu_count()``.
    Any other value is clamped to ``max(1, n_jobs)``.
    """
    if n_jobs == -1:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                return max(1, int(slurm_cpus))
            except ValueError:
                logger.warning(
                    "Invalid SLURM_CPUS_PER_TASK=%r; falling back to os.cpu_count()",
                    slurm_cpus,
                )
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def check_checkpoint(output_dir, filenames, label, force=False):
    """Check whether an analysis checkpoint exists (all expected files present).

    Parameters
    ----------
    output_dir : str or Path
        Directory where output files are expected.
    filenames : str or list[str]
        One or more filenames to check. All must exist for the checkpoint
        to be considered valid.
    label : str
        Human-readable label for logging (e.g. ``"Analysis 2"``).
    force : bool
        If ``True``, always return ``False`` (re-run regardless).

    Returns
    -------
    bool
        ``True`` if checkpoint exists and *force* is ``False`` - caller
        should skip.  ``False`` otherwise.
    """
    if force:
        return False
    if isinstance(filenames, str):
        filenames = [filenames]
    missing = [f for f in filenames if not os.path.exists(os.path.join(output_dir, f))]
    if not missing:
        logger.info("Skipping %s (checkpoint: all %d output files exist)", label, len(filenames))
        return True
    return False


def _get_season(run_id: str) -> int:
    """Extract season number from run_id (e.g. 's01e02a' -> 1)."""
    m = re.match(r'^s(\d+)', run_id)
    if not m:
        raise ValueError(f"Cannot parse season from run_id: {run_id!r}")
    return int(m.group(1))


def get_episode_base(run_id: str) -> str:
    """Extract episode base from run_id, stripping the part suffix.

    Example: 's01e02a' -> 's01e02'
    """
    m = re.match(r'^(s\d+e\d+)[a-z]$', run_id)
    if not m:
        raise ValueError(f"Cannot parse episode base from run_id: {run_id!r}")
    return m.group(1)


def group_runs_by_episode(run_ids):
    """Group run IDs by episode base.

    Returns dict: episode_base -> sorted list of run_ids.
    Example: {'s01e02': ['s01e02a', 's01e02b']}
    """
    episodes = {}
    for run_id in run_ids:
        ep_base = get_episode_base(run_id)
        episodes.setdefault(ep_base, []).append(run_id)
    for ep_base in episodes:
        episodes[ep_base].sort()
    return episodes


def group_runs_to_broadcast_episodes(run_ids):
    """Group runs into broadcast episodes, splitting 4-part episodes.

    Standard episodes (a+b only): a+b → one broadcast episode.
    4-part episodes (a+b+c+d):  a+b → episode N,  c+d → episode N+1.

    The 4 four-part episodes in the Friends dataset (s04e23, s05e23, s06e15,
    s06e24) have no adjacent episode number, so N+1 is safe.

    Returns:
        broadcast_episodes: dict broadcast_id -> list of run_ids
            broadcast_id is the episode base for 2-part episodes (e.g. 's01e02'),
            or '<base>_ab' / '<base>_cd' for 4-part episodes.
        broadcast_meta: dict broadcast_id -> {
            'season': int,
            'episode_num': int,  # original for ab, original+1 for cd
            'run_ids': list[str],
        }
    """
    episodes = group_runs_by_episode(run_ids)
    broadcast_episodes = {}
    broadcast_meta = {}

    for ep_base, ep_runs in episodes.items():
        season, ep_num = parse_episode_order_key(ep_base)
        suffixes = [r[-1] for r in ep_runs]

        if len(ep_runs) <= 2 or 'c' not in suffixes:
            # Standard 2-part (or 1-part) episode
            broadcast_episodes[ep_base] = ep_runs
            broadcast_meta[ep_base] = {
                'season': season,
                'episode_num': ep_num,
                'run_ids': ep_runs,
            }
        else:
            # 4-part episode: split a+b / c+d
            ab_runs = [r for r in ep_runs if r[-1] in ('a', 'b')]
            cd_runs = [r for r in ep_runs if r[-1] in ('c', 'd')]

            ab_id = f'{ep_base}_ab'
            cd_id = f'{ep_base}_cd'

            broadcast_episodes[ab_id] = ab_runs
            broadcast_meta[ab_id] = {
                'season': season,
                'episode_num': ep_num,
                'run_ids': ab_runs,
            }

            broadcast_episodes[cd_id] = cd_runs
            broadcast_meta[cd_id] = {
                'season': season,
                'episode_num': ep_num + 1,
                'run_ids': cd_runs,
            }

    return broadcast_episodes, broadcast_meta


def aggregate_fo_broadcast(fo_dict, decoded_states, broadcast_episodes):
    """Aggregate run-level FO to broadcast-episode level using TR-weighted pooling.

    Args:
        fo_dict: dict run_id -> np.array(n_states,)
        decoded_states: dict run_id -> np.array(n_trs,)
        broadcast_episodes: dict broadcast_id -> list of run_ids
            (from group_runs_to_broadcast_episodes)

    Returns:
        episode_fo: dict broadcast_id -> np.array(n_states,)
        episode_n_trs: dict broadcast_id -> int
    """
    import numpy as np

    episode_fo = {}
    episode_n_trs = {}

    for bcast_id, run_ids in broadcast_episodes.items():
        total_trs = 0
        weighted_fo = None
        for run_id in run_ids:
            n_trs = len(decoded_states[run_id])
            fo_vec = fo_dict[run_id]
            if weighted_fo is None:
                weighted_fo = fo_vec * n_trs
            else:
                weighted_fo += fo_vec * n_trs
            total_trs += n_trs

        if total_trs > 0:
            episode_fo[bcast_id] = weighted_fo / total_trs
        else:
            episode_fo[bcast_id] = (
                weighted_fo if weighted_fo is not None
                else np.zeros_like(next(iter(fo_dict.values())))
            )
        episode_n_trs[bcast_id] = total_trs

    return episode_fo, episode_n_trs


def aggregate_fo_to_episode_level(fo_dict, decoded_states):
    """Aggregate run-level fractional occupancy to episode level using TR-weighted pooling.

    For multipart episodes (e.g., s01e01a + s01e01b), the episode-level FO is:
        FO_episode[k] = sum(FO_part[k] * n_TRs_part) / sum(n_TRs_part)

    For single-part episodes, the FO is returned unchanged.

    Args:
        fo_dict: dict run_id -> np.array(n_states,)
        decoded_states: dict run_id -> np.array(n_trs,) used to get TR counts

    Returns:
        episode_fo: dict episode_base -> np.array(n_states,)
        episode_n_trs: dict episode_base -> int total TRs
    """
    import numpy as np

    episodes = group_runs_by_episode(list(fo_dict.keys()))
    episode_fo = {}
    episode_n_trs = {}

    for ep_base, run_ids in episodes.items():
        # TR-weighted pooling
        total_trs = 0
        weighted_fo = None
        for run_id in run_ids:
            n_trs = len(decoded_states[run_id])
            fo_vec = fo_dict[run_id]
            if weighted_fo is None:
                weighted_fo = fo_vec * n_trs
            else:
                weighted_fo += fo_vec * n_trs
            total_trs += n_trs

        if total_trs > 0:
            episode_fo[ep_base] = weighted_fo / total_trs
        else:
            episode_fo[ep_base] = weighted_fo if weighted_fo is not None else np.zeros_like(next(iter(fo_dict.values())))
        episode_n_trs[ep_base] = total_trs

    return episode_fo, episode_n_trs


def aggregate_decoded_to_episode_level(decoded_states):
    """Concatenate decoded state sequences for multipart episodes.

    Args:
        decoded_states: dict run_id -> np.array(n_trs,)

    Returns:
        episode_decoded: dict episode_base -> np.array(total_trs,)
    """
    import numpy as np

    episodes = group_runs_by_episode(list(decoded_states.keys()))
    episode_decoded = {}
    for ep_base, run_ids in episodes.items():
        parts = [decoded_states[rid] for rid in run_ids]
        episode_decoded[ep_base] = np.concatenate(parts)
    return episode_decoded


def parse_episode_order_key(episode_id: str):
    """Parse episode ID into (season, episode_num) for chronological sorting.

    Works for both run_ids ('s01e02a') and episode bases ('s01e02').
    """
    m = re.match(r'^s(\d+)e(\d+)', episode_id)
    if not m:
        raise ValueError(f"Cannot parse episode order from: {episode_id!r}")
    return (int(m.group(1)), int(m.group(2)))


def get_movie_type(run_id: str) -> str | None:
    """Extract movie type from a run ID string.

    Works with full BIDS run IDs or short task-only IDs.

    Args:
        run_id: e.g. 'sub-01_ses-001_task-bourne01_space-fsLR_den-91k' or 'bourne01'

    Returns:
        Movie type string ('bourne', 'wolf', 'figures', 'life') or None.
    """
    for prefix in ('bourne', 'wolf', 'figures', 'life'):
        if f'task-{prefix}' in run_id or run_id.startswith(prefix):
            return prefix
    return None


def normalize_cross_stim_run_id(run_id: str, stimulus: str) -> str:
    """Normalize a cross-stimulus run ID to the canonical 08c-compatible short form.

    The cross-stimulus decode scripts (``m10_04``, ``hp_04``, ``pp_04``)
    inherited their run IDs from the parcellated BOLD filenames, which are
    full BIDS strings like::

        sub-01_ses-001_task-bourne01_space-fsLR_den-91k
        sub-01_task-harrypotter_run-1_space-fsLR_den-91k
        sub-01_ses-001_task-lppFR_run-1_part-mag_space-fsLR_den-91k

    The 08c transformer-feature extraction (and all downstream
    cross-stimulus analyses) instead uses compact stimulus-specific
    identifiers matching the ``{run_id}_raw.npy`` filenames that 08c
    produces::

        bourne01, bourne02, ..., wolf01, ...   (movie10)
        harrypotter_run-01, ..., harrypotter_run-08   (harrypotter, 0-padded)
        lppFR_run-01, ..., lppFR_run-09   (petitprince_fr)
        lppEN_run-01, ..., lppEN_run-09   (petitprince_en)
        s01e01a                                         (friends, already short)
        rest_ses-001, ...   (hcptrt restingstate)

    This helper performs the long → short conversion so decode scripts and
    downstream consumers can key state pickles by the same IDs that 08c
    uses for features. The function is **idempotent**: passing an already-
    short ID returns it unchanged (up to run-number zero-padding
    normalization for HP / PP).

    Parameters
    ----------
    run_id : str
        Either the full BIDS ID or an already-short ID.
    stimulus : str
        One of ``"friends"``, ``"movie10"``, ``"harrypotter"``,
        ``"petitprince_fr"``, ``"petitprince_en"``, ``"restingstate"``.

    Returns
    -------
    str
        Short canonical ID matching 08c conventions.

    Raises
    ------
    ValueError
        If ``stimulus`` is unknown, or if a BIDS-format ``run_id`` lacks
        the expected ``task-`` / ``run-`` tokens for the stimulus.
    """
    tokens = run_id.split("_")
    token_map = {}
    for tok in tokens:
        if "-" in tok:
            key, _, val = tok.partition("-")
            token_map[key] = val

    if stimulus == "friends":
        # Friends is already short in all current pipelines (e.g. 's01e01a').
        # If somehow called with a BIDS form, pull the task token and strip
        # any 'friends_' prefix.
        if "task" in token_map:
            return token_map["task"].removeprefix("friends_")
        return run_id.removeprefix("friends_")

    if stimulus == "movie10":
        # Short form = clip name plus optional '_run-N' suffix when the
        # BIDS source had a run entity. cneuromod's movie10 repeats
        # ``figures`` and ``life`` across two sessions (run-1 / run-2),
        # so the fMRI side has two decoded_states entries per clip for
        # those categories - both must survive normalization as distinct
        # keys. ``bourne`` and ``wolf`` have no run entity and use the
        # bare clip name. (The 08c transformer features are stored at the
        # clip level since the stimulus is identical across viewings;
        # see :func:`feature_key_for_cross_stim_run_id` for the
        # corresponding feature-lookup stripping.)
        if "task" in token_map:
            clip = token_map["task"]
            run_value = token_map.get("run")
            if run_value is None:
                return clip
            try:
                run_n = int(run_value)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse run number from run_id={run_id!r} "
                    f"(got run-{run_value!r}): {exc}"
                ) from exc
            return f"{clip}_run-{run_n}"
        # Already short.
        return run_id

    if stimulus in ("harrypotter", "petitprince_fr", "petitprince_en"):
        # Short form = "{task_prefix}_run-{NN}" with zero-padded run number.
        # HP decoded_states uses 'run-1'; 08c features use 'run-01'.
        task_prefix_map = {
            "harrypotter": "harrypotter",
            "petitprince_fr": "lppFR",
            "petitprince_en": "lppEN",
        }
        expected_task = task_prefix_map[stimulus]

        # Extract run number from either BIDS form or already-short form.
        run_value = token_map.get("run")
        if run_value is None:
            # Already short form (e.g. 'harrypotter_run-01' passed via
            # short-form token_map which only has 'run' key on a rejoined
            # split). Fall through to return unchanged.
            return run_id
        try:
            run_n = int(run_value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse run number from run_id={run_id!r} "
                f"(got run-{run_value!r}): {exc}"
            ) from exc

        # Sanity-check task token if present.
        task_token = token_map.get("task")
        if task_token is not None and task_token != expected_task:
            raise ValueError(
                f"run_id={run_id!r} has task-{task_token!r} but "
                f"stimulus={stimulus!r} expects task-{expected_task!r}"
            )

        return f"{expected_task}_run-{run_n:02d}"

    if stimulus in ("restingstate", "rest"):
        # hcptrt rest is always run-1; the session token is the run identity:
        # sub-01_ses-001_task-restingstate_run-1_space-fsLR_den-91k -> rest_ses-001
        # Idempotent: 'rest_ses-001' re-maps to itself via its ses token.
        if "ses" in token_map:
            return f"rest_ses-{token_map['ses']}"
        return run_id

    raise ValueError(
        f"Unknown stimulus={stimulus!r}; expected one of friends, movie10, "
        f"harrypotter, petitprince_fr, petitprince_en, restingstate"
    )


def feature_key_for_cross_stim_run_id(short_id: str, stimulus: str) -> str:
    """Return the 08c feature-file stem for a short cross-stim run ID.

    For most stimuli (friends, harrypotter, petitprince_*) the short
    decoded_states key matches the 08c feature filename stem 1:1, so this
    is the identity. The exception is **movie10**: the fMRI dataset has
    two viewings per clip for ``figures`` and ``life`` (sessions 9/10/11/12
    re-watch session 6/7/8's clips), which :func:`normalize_cross_stim_run_id`
    disambiguates as ``figures01_run-1`` / ``figures01_run-2``. But 08c
    stores a single feature file ``figures01_raw.npy`` per clip because the
    stimulus is identical across viewings. This helper strips the trailing
    ``_run-N`` suffix so both m10 decoded_states entries look up the same
    08c feature file.

    Parameters
    ----------
    short_id : str
        Short run ID from :func:`normalize_cross_stim_run_id`.
    stimulus : str
        Stimulus name.

    Returns
    -------
    str
        Stem matching ``08c_transformer_features/{stimulus}/{model}/layer_NN/{stem}_raw.npy``.
    """
    if stimulus == "movie10":
        # Strip optional ``_run-N`` tail (present only for figures/life).
        if "_run-" in short_id:
            return short_id.rsplit("_run-", 1)[0]
        return short_id
    # friends, harrypotter, petitprince_* - short_id already matches the
    # 08c feature filename.
    return short_id


def short_run_label(short_id: str) -> str:
    """Compact figure tick label for a canonical short cross-stim run ID.

    Strips the stimulus prefix that :func:`normalize_cross_stim_run_id`
    produces, keeping only the per-run identity::

        harrypotter_run-01 -> run-01
        rest_ses-001       -> ses-001
        lppFR_run-01       -> FR-01   (any lpp language code, e.g. EN, CN)

    Movie10 / friends short IDs (``bourne01``, ``s01e01a``) and anything
    unrecognized pass through unchanged, so this is a total function —
    safe to call on any run ID without a fallback branch at the call site.
    """
    lpp = re.match(r"^lpp([A-Z]{2,3})_run-(.+)$", short_id)
    if lpp:
        return f"{lpp.group(1)}-{lpp.group(2)}"
    for prefix in ("harrypotter_", "rest_"):
        if short_id.startswith(prefix):
            return short_id[len(prefix):]
    return short_id


def canonicalize_and_save_decoded(decoded_states, out_dir, stimulus, n_states,
                                  map_stimulus=None):
    """Canonicalize decoded-state run IDs to short form and save stage-04 outputs.

    Shared tail of the four ``*_04_score_and_decode.py`` scripts: builds the
    long->short map via :func:`normalize_cross_stim_run_id`, aborts loudly on
    short-ID collisions, computes short-keyed fractional occupancy, and writes
    ``decoded_states.pkl``, ``fractional_occupancy.pkl``, and
    ``run_id_map.json`` (a required input of the ``*_05`` validation scripts,
    which join long-keyed run-id JSONs against the short-keyed pickles).

    Parameters
    ----------
    decoded_states : dict
        Long BIDS run_id -> np.array(n_trs,) of decoded state indices.
    out_dir : str
        Output directory (created by the caller).
    stimulus : str or callable
        Stimulus name for :func:`normalize_cross_stim_run_id`, or a callable
        ``long_id -> stimulus`` for per-run resolution (petit-prince infers
        the language from each run's BIDS token).
    n_states : int
        Total number of model states.
    map_stimulus : str, optional
        Value recorded in ``run_id_map.json``'s ``stimulus`` field. Defaults
        to ``stimulus`` when it is a string; required when ``stimulus`` is a
        callable.

    Returns
    -------
    (long_to_short, decoded_states_short, fo_short)
    """
    import json
    import pickle
    # Deferred: recurrence_utils imports from utils.common at module level.
    from utils.recurrence_utils import compute_fractional_occupancy

    stimulus_of = stimulus if callable(stimulus) else (lambda _rid: stimulus)
    if map_stimulus is None:
        if callable(stimulus):
            raise ValueError("map_stimulus is required when stimulus is a callable")
        map_stimulus = stimulus

    long_to_short = {
        long_id: normalize_cross_stim_run_id(long_id, stimulus_of(long_id))
        for long_id in decoded_states.keys()
    }
    if len(set(long_to_short.values())) != len(long_to_short):
        dupes = [v for v in long_to_short.values()
                 if list(long_to_short.values()).count(v) > 1]
        raise RuntimeError(
            f"Short run_ids have duplicates after normalization: {set(dupes)}"
        )
    decoded_states_short = {
        long_to_short[rid]: seq for rid, seq in decoded_states.items()
    }
    fo_short = compute_fractional_occupancy(decoded_states_short, n_states)
    run_id_map = {
        "short_to_long": {short: long for long, short in long_to_short.items()},
        "long_to_short": dict(long_to_short),
        "stimulus": map_stimulus,
    }

    with open(os.path.join(out_dir, 'decoded_states.pkl'), 'wb') as f:
        pickle.dump(decoded_states_short, f, protocol=4)

    with open(os.path.join(out_dir, 'fractional_occupancy.pkl'), 'wb') as f:
        pickle.dump(fo_short, f, protocol=4)

    with open(os.path.join(out_dir, 'run_id_map.json'), 'w') as f:
        json.dump(run_id_map, f, indent=2)

    return long_to_short, decoded_states_short, fo_short


def normalize_parcellation_name(parcellation: str) -> str:
    """
    Normalize parcellation argument to full directory name format.

    Supports both short form (e.g., '4S156', '156') and full form (e.g., 'atlas-4S156Parcels').
    Also supports Schaefer parcellation variants.

    Examples:
        '156' -> 'atlas-4S156Parcels'
        '4S156' -> 'atlas-4S156Parcels'
        'atlas-4S156Parcels' -> 'atlas-4S156Parcels' (unchanged)
        '4S456' -> 'atlas-4S456Parcels'
        'Schaefer400' -> 'Schaefer_400_Tian_S4'

    Args:
        parcellation: Short or full parcellation name

    Returns:
        Full parcellation name in proper format
    """
    # Explicit mapping for known parcellations (including common typos)
    short_to_full = {
        # 4S series (ascending order)
        '156': 'atlas-4S156Parcels',
        '4S156': 'atlas-4S156Parcels',
        'S156': 'atlas-4S156Parcels',  # Common typo: missing '4'
        '456': 'atlas-4S456Parcels',
        '4S456': 'atlas-4S456Parcels',
        'S456': 'atlas-4S456Parcels',
        '556': 'atlas-4S556Parcels',
        '4S556': 'atlas-4S556Parcels',
        'S556': 'atlas-4S556Parcels',
        '656': 'atlas-4S656Parcels',
        '4S656': 'atlas-4S656Parcels',
        'S656': 'atlas-4S656Parcels',
        '756': 'atlas-4S756Parcels',
        '4S756': 'atlas-4S756Parcels',
        'S756': 'atlas-4S756Parcels',
        '856': 'atlas-4S856Parcels',
        '4S856': 'atlas-4S856Parcels',
        'S856': 'atlas-4S856Parcels',
        '956': 'atlas-4S956Parcels',
        '4S956': 'atlas-4S956Parcels',
        'S956': 'atlas-4S956Parcels',
        '1056': 'atlas-4S1056Parcels',
        '4S1056': 'atlas-4S1056Parcels',
        'S1056': 'atlas-4S1056Parcels',
        # Schaefer variants
        'Schaefer400': 'Schaefer_400_Tian_S4',
        'Schaefer_400': 'Schaefer_400_Tian_S4',
    }

    # If already in full format, validate and correct if needed
    if parcellation.startswith('atlas-') and parcellation.endswith('Parcels'):
        # Extract the core part to validate against known mappings (catches typos in full names)
        # e.g., "atlas-S156Parcels" -> "S156" -> "atlas-4S156Parcels"
        core = parcellation.replace('atlas-', '').replace('Parcels', '')
        if core in short_to_full:
            return short_to_full[core]
        # Otherwise return as-is (assume it's correct)
        return parcellation

    if parcellation.startswith('Schaefer'):
        # Check if it matches a known variant
        if parcellation in short_to_full:
            return short_to_full[parcellation]
        # Otherwise return as-is
        return parcellation

    # Check explicit mapping for short names
    if parcellation in short_to_full:
        return short_to_full[parcellation]

    # Fallback: generic string manipulation for unknown parcellations
    # Add 'atlas-' prefix if missing
    if not parcellation.startswith('atlas-'):
        parcellation = f'atlas-{parcellation}'
    # Add 'Parcels' suffix if missing
    if not parcellation.endswith('Parcels'):
        parcellation = f'{parcellation}Parcels'

    logger.warning(f"Using generic normalization for unknown parcellation: {parcellation}")
    return parcellation



def resolve_stage_file(base_dir, filename, label):
    """Resolve a stage output file, preferring non-vt path and falling back to vt*/.

    Many downstream scripts produce both vt-qualified outputs (e.g.
    ``{base}/vt0.95/{filename}``) and unqualified outputs (``{base}/{filename}``).
    This helper prefers the direct path if present; otherwise it looks for
    exactly one ``vt*`` subdirectory containing ``filename``. Multiple candidates
    raise ``FileNotFoundError`` to avoid silent ambiguity.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Directory to search.
    filename : str
        Filename (not path) to resolve.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    str
        Resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If no candidate is found or if multiple ``vt*`` candidates exist.
    """
    import os
    from pathlib import Path

    direct_path = os.path.join(str(base_dir), filename)
    if os.path.exists(direct_path):
        return direct_path
    vt_matches = sorted(Path(base_dir).glob(f"vt*/{filename}"))
    if len(vt_matches) == 1:
        resolved = str(vt_matches[0])
        logger.info("Using %s from legacy VT path: %s", label, resolved)
        return resolved
    if len(vt_matches) > 1:
        raise FileNotFoundError(
            f"Multiple VT candidates found for {label}: "
            f"{', '.join(str(p) for p in vt_matches)}"
        )
    raise FileNotFoundError(f"{label} not found at {direct_path}")


def load_training_split(sub_id, parc, scratch_dir):
    """Load the 03a primary 70/15/15 season-stratified training split.

    Parameters
    ----------
    sub_id : str
        Subject ID (e.g. ``'sub-01'``).
    parc : str
        Parcellation directory name (already normalized via
        :func:`normalize_parcellation_name`).
    scratch_dir : str
        Base scratch directory (typically ``os.getenv('SCRATCH_DIR')``).

    Returns
    -------
    dict
        ``{'train': set[str], 'valid': set[str], 'test': set[str]}`` - run IDs
        per split.

    Raises
    ------
    FileNotFoundError
        If ``03a_pca4combined_hmm/{parc}/{sub_id}/splits/primary.json`` is
        missing.
    """
    import json
    import os

    split_path = os.path.join(
        scratch_dir, "output", "03a_pca4combined_hmm", parc, sub_id,
        "splits", "primary.json",
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"03a training split not found: {split_path}. "
            f"Run 03a_pca4combined_hmm.py first."
        )
    with open(split_path) as f:
        data = json.load(f)
    splits = {
        "train": set(data["train"]),
        "valid": set(data["valid"]),
        "test": set(data["test"]),
    }
    logger.info(
        "Loaded 03a split from %s: %d train, %d valid, %d test runs",
        split_path, len(splits["train"]), len(splits["valid"]), len(splits["test"]),
    )
    return splits
