#!/usr/bin/env python
"""
physio_io.py - I/O utilities for physiological data processing.

Handles loading physprep preprocessed physiological data, discovering physio
files per run (for both Friends and Movie10), and aligning continuous
physiological signals to TR boundaries.

Physprep data format (1 kHz):
  - preproc_physio.tsv.gz: 6 columns (PPG, ECG, RSP, EDA, EDATonic, EDAPhasic)
  - events.tsv: columns (onset, duration, trial_type, channel)
  - quality.json: per-channel {QualityAssessment: Pass/Fail, PercentageValid: float}

Filename conventions differ between Friends and Movie10:
  - Friends events: *_task-{id}_desc-physio_events.tsv
  - Movie10 events: *_task-{id}[_run-NN]_events.tsv
  - Quality files: same format in both; Movie10 quality lacks run-NN entity
"""

import gzip
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
SAMPLES_PER_TR = int(1.49 * 1000)  # 1490 samples per TR at 1 kHz
SR_PHYSIO = 1000  # Hz

FEATURE_COLUMNS = [
    "HR_bpm",
    "HRV_RMSSD",
    "breathing_rate",
    "RVT",
    "EDA_tonic",
    "EDA_phasic",
    "SCR_binary",
]

# Which channels each feature depends on (for NaN masking)
FEATURE_CHANNEL_DEPS = {
    0: "ECG",  # HR
    1: "ECG",  # HRV_RMSSD
    2: "RSP",  # breathing_rate
    3: "RSP",  # RVT
    4: "EDA",  # EDA_tonic
    5: "EDA",  # EDA_phasic
    6: "EDA",  # SCR_binary
}

# ── BIDS Entity Parsing ───────────────────────────────────────────────────

# Regex for BIDS key-value pairs in filenames
_BIDS_ENTITY_RE = re.compile(r"(sub|ses|task|run|space|den|desc)-([^\s_]+)")


def parse_bids_entities(name: str) -> dict:
    """Extract BIDS entities from a filename or key string.

    Args:
        name: BIDS-like string, e.g.
          'sub-01_ses-007_task-s01e17a_desc-preproc_physio.tsv.gz'
          'sub-01_ses-001_task-bourne01_space-fsLR_den-91k'
          's01e17a'  (Friends short key — returns task only)

    Returns:
        dict with keys like 'sub', 'ses', 'task', 'run' (only those present).
    """
    entities = {k: v for k, v in _BIDS_ENTITY_RE.findall(name)}
    # Handle Friends short run_id (e.g. 's01e17a') — no BIDS entities at all
    if not entities and re.match(r"^s\d+e\d+[a-z]$", name):
        entities["task"] = name
    return entities


def _normalize_run(run_str: str | None) -> str | None:
    """Normalize run entity to zero-padded two-digit string.

    Bridges the mismatch between decoded_states keys ('run-1') and
    physprep filenames ('run-01').

    Examples:
        '1' -> '01', '01' -> '01', '2' -> '02', None -> None
    """
    if run_str is None:
        return None
    return run_str.lstrip("0").zfill(2)  # '1' -> '01', '01' -> '01'


def _make_match_key(entities: dict) -> tuple:
    """Create a hashable matching key from BIDS entities.

    Returns (sub, ses, task, normalized_run) tuple for matching physprep
    files to decoded_states keys.
    """
    return (
        entities.get("sub"),
        entities.get("ses"),
        entities.get("task"),
        _normalize_run(entities.get("run")),
    )


# ── Physprep File Discovery ──────────────────────────────────────────────


def build_physio_inventory(
    sub_id: str,
    physprep_dir: str,
    decoded_state_keys: list[str] | None = None,
    stimulus: str = "friends",
) -> dict[str, dict[str, Path | None]]:
    """Build inventory mapping decoded_state run keys to physprep file paths.

    Globs all physprep files for a subject and matches them to decoded_states
    keys using BIDS entity parsing.

    Args:
        sub_id: Subject ID (e.g., 'sub-01')
        physprep_dir: Root physprep directory
            (e.g., '$DATA_DIR/friends.physprep')
        decoded_state_keys: List of run_id keys from decoded_states.pkl.
            If None, returns inventory keyed by physprep-derived task entities.
        stimulus: 'friends' or 'movie10' — controls filename pattern matching.

    Returns:
        dict: {run_key: {'preproc_physio': Path, 'events': Path, 'quality': Path}}
        Missing files have value None.
    """
    base = Path(physprep_dir) / sub_id
    if not base.exists():
        logger.warning("Physprep directory not found: %s", base)
        return {}

    # ── Glob all three file types ─────────────────────────────────────
    # Session-less stimuli (e.g., harrypotter) use sub-XX/func/ directly
    if stimulus == "harrypotter":
        glob_prefix = "func"
    else:
        glob_prefix = "ses-*/func"

    preproc_files = sorted(base.glob(f"{glob_prefix}/*_desc-preproc_physio.tsv.gz"))
    quality_files = sorted(base.glob(f"{glob_prefix}/*_desc-quality.json"))

    # Events files differ between stimuli
    if stimulus == "friends":
        events_files = sorted(base.glob(f"{glob_prefix}/*_desc-physio_events.tsv"))
    else:
        # Movie10 / harrypotter: *_events.tsv but NOT *_desc-*_events.tsv
        all_events = sorted(base.glob(f"{glob_prefix}/*_events.tsv"))
        events_files = [f for f in all_events if "_desc-" not in f.name]

    # ── Index files by BIDS match key ─────────────────────────────────
    def _index_by_key(files, use_run=True):
        idx = {}
        for f in files:
            ent = parse_bids_entities(f.name)
            if use_run:
                key = _make_match_key(ent)
            else:
                # Quality files: match on (sub, ses, task) only
                key = (ent.get("sub"), ent.get("ses"), ent.get("task"), None)
            idx[key] = f
        return idx

    preproc_idx = _index_by_key(preproc_files, use_run=True)
    events_idx = _index_by_key(events_files, use_run=True)
    # Quality: match without run (movie10 quality files lack run-NN)
    quality_idx = _index_by_key(quality_files, use_run=False)

    # Task-only fallback indexes for Friends short keys (e.g. 's01e15a')
    # Friends decoded_states keys have no sub/ses, so full-tuple matching fails.
    # Build task → file mappings as fallback.
    def _index_by_task(files):
        idx = {}
        for f in files:
            ent = parse_bids_entities(f.name)
            task = ent.get("task")
            if task:
                idx[task] = f
        return idx

    preproc_by_task = _index_by_task(preproc_files)
    events_by_task = _index_by_task(events_files)
    quality_by_task = _index_by_task(quality_files)

    # ── Match decoded_state keys to physprep files ────────────────────
    inventory = {}

    if decoded_state_keys is None:
        # Build inventory from physprep files (no decoded_states to match)
        for key, preproc_path in preproc_idx.items():
            ent = parse_bids_entities(preproc_path.name)
            task = ent.get("task", "unknown")
            run = ent.get("run")
            inv_key = f"{task}_run-{run}" if run else task
            quality_key = (key[0], key[1], key[2], None)
            inventory[inv_key] = {
                "preproc_physio": preproc_path,
                "events": events_idx.get(key),
                "quality": quality_idx.get(quality_key),
            }
    else:
        for run_key in decoded_state_keys:
            ent = parse_bids_entities(run_key)
            key = _make_match_key(ent)

            # Determine if this is a short key (Friends: sub=None)
            is_short_key = key[0] is None  # no sub entity

            if is_short_key:
                # Friends short key: match on task entity alone
                task = ent.get("task")
                preproc = preproc_by_task.get(task)
                events = events_by_task.get(task)
                quality = quality_by_task.get(task)
            else:
                # Full BIDS key (Movie10): match on (sub, ses, task, run)
                quality_key_no_run = (key[0], key[1], key[2], None)
                preproc = preproc_idx.get(key)
                events = events_idx.get(key)
                quality = (
                    quality_idx.get(key)
                    or quality_idx.get(quality_key_no_run)
                )

            if preproc is None:
                logger.warning(
                    "No physprep preproc file for run key %s (match key: %s)",
                    run_key,
                    key,
                )
            inventory[run_key] = {
                "preproc_physio": preproc,
                "events": events,
                "quality": quality,
            }

    logger.info(
        "Physio inventory for %s: %d runs matched, %d with preproc, %d with events",
        sub_id,
        len(inventory),
        sum(1 for v in inventory.values() if v["preproc_physio"] is not None),
        sum(1 for v in inventory.values() if v["events"] is not None),
    )
    return inventory


def load_quality(quality_path: Path | str | None) -> tuple[dict[str, bool], str]:
    """Load per-channel QC status from physprep quality JSON.

    Args:
        quality_path: Path to *_desc-quality.json, or None.

    Returns:
        tuple of (channel_qc, quality_source):
            channel_qc: {channel: True if pass, False if fail}.
            quality_source: "verified" if quality file was parsed,
                "assumed" if quality_path is None (all channels assumed pass).
    """
    default = {"ECG": True, "RSP": True, "EDA": True, "PPG": True}
    if quality_path is None:
        logger.warning("No quality file — assuming all channels pass")
        return default, "assumed"

    with open(quality_path) as f:
        data = json.load(f)

    result = {}
    for channel in ["ECG", "RSP", "EDA", "PPG"]:
        if channel in data:
            result[channel] = data[channel].get("QualityAssessment") == "Pass"
        else:
            result[channel] = True  # assume pass if not present
    return result, "verified"


def load_preproc_physio(preproc_path: Path | str) -> pd.DataFrame:
    """Load preprocessed physio TSV (gzipped, 1 kHz, 6 columns).

    Args:
        preproc_path: Path to *_desc-preproc_physio.tsv.gz

    Returns:
        DataFrame with columns: PPG, ECG, RSP, EDA, EDATonic, EDAPhasic
    """
    with gzip.open(preproc_path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    return df


def load_physio_events(events_path: Path | str) -> pd.DataFrame:
    """Load physio events TSV.

    Args:
        events_path: Path to *_events.tsv or *_desc-physio_events.tsv

    Returns:
        DataFrame with columns: onset, duration, trial_type, channel
    """
    df = pd.read_csv(events_path, sep="\t")
    return df


def get_events_by_type(
    events_df: pd.DataFrame, trial_type: str
) -> np.ndarray:
    """Extract onset times for a specific event type.

    Args:
        events_df: Events DataFrame from load_physio_events()
        trial_type: e.g. 'r_peak_corrected', 'inhale_max', 'scr_onset'

    Returns:
        1D array of onset times in seconds, sorted.
    """
    mask = events_df["trial_type"] == trial_type
    return events_df.loc[mask, "onset"].values.astype(float)


# ── Signal Processing Utilities ───────────────────────────────────────────


def align_continuous_to_trs(
    signal_1khz: np.ndarray,
    n_trs: int,
    samples_per_tr: int = SAMPLES_PER_TR,
) -> np.ndarray:
    """Aggregate 1 kHz physiological signal into TR bins via mean.

    Args:
        signal_1khz: 1D array at 1000 Hz.
        n_trs: Number of TRs (authoritative count from decoded_states).
        samples_per_tr: Samples per TR (default 1490 for TR=1.49s).

    Returns:
        np.ndarray shape (n_trs,): mean signal per TR.
    """
    result = np.full(n_trs, np.nan)
    for t in range(n_trs):
        start = t * samples_per_tr
        end = (t + 1) * samples_per_tr
        if start >= len(signal_1khz):
            break
        chunk = signal_1khz[start : min(end, len(signal_1khz))]
        if len(chunk) > 0:
            result[t] = np.nanmean(chunk)
    return result


def events_to_instantaneous_rate(
    event_onsets: np.ndarray,
    duration_s: float,
    sr: int = SR_PHYSIO,
) -> np.ndarray:
    """Convert event times to instantaneous rate via linear interpolation.

    Computes inverse inter-event intervals and interpolates to produce a
    smooth rate signal. Used for HR (from R-peaks) and breathing rate
    (from inhale peaks). Avoids count-quantization artifact at 1.49s bins.

    Args:
        event_onsets: 1D array of event times in seconds, sorted.
        duration_s: Total duration in seconds.
        sr: Output sampling rate (default 1000 Hz).

    Returns:
        np.ndarray shape (int(sr * duration_s),): instantaneous rate in events/min.

    Raises:
        ValueError: If fewer than 2 events (cannot compute intervals).
    """
    if len(event_onsets) < 2:
        raise ValueError(
            f"Need ≥2 events for instantaneous rate, got {len(event_onsets)}"
        )

    event_onsets = np.sort(event_onsets)
    n_samples = int(sr * duration_s)

    # Inter-event intervals → instantaneous rate at midpoints
    intervals = np.diff(event_onsets)
    midpoints = (event_onsets[:-1] + event_onsets[1:]) / 2.0
    inst_rate = 60.0 / intervals  # events per minute

    # Remove physiologically implausible values
    valid = (inst_rate > 0) & np.isfinite(inst_rate)
    midpoints = midpoints[valid]
    inst_rate = inst_rate[valid]

    if len(inst_rate) < 2:
        raise ValueError("Fewer than 2 valid rate estimates after filtering")

    # Linearly interpolate to continuous signal
    time_axis = np.arange(n_samples) / sr
    rate_continuous = np.interp(
        time_axis, midpoints, inst_rate,
        left=inst_rate[0], right=inst_rate[-1],
    )
    return rate_continuous


def compute_rvt(
    rsp_signal: np.ndarray,
    inhale_onsets: np.ndarray,
    exhale_onsets: np.ndarray,
    sr: int = SR_PHYSIO,
) -> np.ndarray:
    """Compute respiratory volume per time (RVT) from RSP signal.

    RVT = (peak amplitude - trough amplitude) / cycle duration for each
    breath cycle, linearly interpolated to the original sampling rate.
    Critical: non-RETROICOR'd BOLD retains RVT effects (Birn et al., 2006).

    Args:
        rsp_signal: 1D RSP signal at sr Hz.
        inhale_onsets: Inhalation peak times in seconds.
        exhale_onsets: Exhalation trough times in seconds.
        sr: Sampling rate (default 1000 Hz).

    Returns:
        np.ndarray shape (len(rsp_signal),): RVT at original sampling rate.
        Returns all-NaN if insufficient peaks.
    """
    n_samples = len(rsp_signal)
    rvt_signal = np.full(n_samples, np.nan)

    if len(inhale_onsets) < 2 or len(exhale_onsets) < 1:
        logger.warning("Insufficient RSP peaks for RVT computation")
        return rvt_signal

    inhale_onsets = np.sort(inhale_onsets)

    # For each breath cycle (between consecutive inhale peaks), find the
    # intervening exhale trough
    rvt_times = []
    rvt_values = []

    for i in range(len(inhale_onsets) - 1):
        t_peak_start = inhale_onsets[i]
        t_peak_end = inhale_onsets[i + 1]
        cycle_duration = t_peak_end - t_peak_start

        if cycle_duration <= 0:
            continue

        # Find exhale troughs between these two inhale peaks
        mask = (exhale_onsets > t_peak_start) & (exhale_onsets < t_peak_end)
        troughs_in_cycle = exhale_onsets[mask]

        if len(troughs_in_cycle) == 0:
            continue

        # Peak amplitude at inhale, trough amplitude at exhale
        peak_idx = min(int(t_peak_start * sr), n_samples - 1)
        trough_idx = min(int(troughs_in_cycle[0] * sr), n_samples - 1)

        peak_amp = rsp_signal[peak_idx]
        trough_amp = rsp_signal[trough_idx]

        rvt_val = (peak_amp - trough_amp) / cycle_duration
        midpoint = (t_peak_start + t_peak_end) / 2.0

        rvt_times.append(midpoint)
        rvt_values.append(rvt_val)

    if len(rvt_values) < 2:
        logger.warning("Fewer than 2 valid RVT estimates")
        return rvt_signal

    # Interpolate to continuous signal
    time_axis = np.arange(n_samples) / sr
    rvt_signal = np.interp(
        time_axis,
        np.array(rvt_times),
        np.array(rvt_values),
        left=rvt_values[0],
        right=rvt_values[-1],
    )
    return rvt_signal


def compute_hrv_rmssd(
    rpeak_onsets: np.ndarray,
    duration_s: float,
    window_s: float = 30.0,
    sr: int = SR_PHYSIO,
) -> np.ndarray:
    """Compute HRV (RMSSD) using a rolling window, output at 1 kHz.

    Uses a 30s rolling window centered on each time point (minimum
    defensible window for ultra-short HRV; Munoz et al., 2015).

    Args:
        rpeak_onsets: R-peak times in seconds, sorted.
        duration_s: Total recording duration in seconds.
        window_s: Rolling window size in seconds (default 30).
        sr: Output sampling rate (default 1000 Hz).

    Returns:
        np.ndarray shape (int(sr * duration_s),): RMSSD in ms at each sample.
        NaN where insufficient R-peaks in window.
    """
    n_samples = int(sr * duration_s)
    rmssd_signal = np.full(n_samples, np.nan)

    if len(rpeak_onsets) < 3:
        return rmssd_signal

    rpeak_onsets = np.sort(rpeak_onsets)
    rr_intervals = np.diff(rpeak_onsets) * 1000.0  # ms
    rr_midpoints = (rpeak_onsets[:-1] + rpeak_onsets[1:]) / 2.0

    half_win = window_s / 2.0

    # Evaluate RMSSD at regular intervals (every 100ms for efficiency)
    eval_step = int(0.1 * sr)  # 100 samples = 100ms
    eval_times = np.arange(0, n_samples, eval_step) / sr
    rmssd_at_eval = np.full(len(eval_times), np.nan)

    for i, t in enumerate(eval_times):
        mask = (rr_midpoints >= t - half_win) & (rr_midpoints <= t + half_win)
        rr_in_window = rr_intervals[mask]

        if len(rr_in_window) < 3:
            continue

        successive_diffs = np.diff(rr_in_window)
        rmssd_at_eval[i] = np.sqrt(np.mean(successive_diffs**2))

    # Interpolate back to full resolution
    valid = np.isfinite(rmssd_at_eval)
    if np.sum(valid) < 2:
        return rmssd_signal

    rmssd_signal = np.interp(
        np.arange(n_samples) / sr,
        eval_times[valid],
        rmssd_at_eval[valid],
        left=np.nan,
        right=np.nan,
    )
    return rmssd_signal


def events_to_tr_binary(
    event_onsets: np.ndarray,
    n_trs: int,
    tr_s: float = 1.49,
) -> np.ndarray:
    """Convert event onsets to binary TR indicator (any event in TR window?).

    Args:
        event_onsets: Event times in seconds.
        n_trs: Number of TRs.
        tr_s: TR duration in seconds (default 1.49).

    Returns:
        np.ndarray shape (n_trs,): 0 or 1 per TR.
    """
    result = np.zeros(n_trs, dtype=np.float64)
    for onset in event_onsets:
        tr_idx = int(onset // tr_s)
        if 0 <= tr_idx < n_trs:
            result[tr_idx] = 1.0
    return result


# ── Feature Extraction (per run) ─────────────────────────────────────────


def extract_physio_features(
    preproc_path: Path | str,
    events_path: Path | str,
    n_trs: int,
    channel_qc: dict[str, bool],
) -> np.ndarray:
    """Extract 7 TR-aligned physiological features for one run.

    Features (columns):
        0: HR (bpm) — from r_peak_corrected, interpolated
        1: HRV_RMSSD (ms) — 30s rolling window
        2: breathing_rate (breaths/min) — from inhale_max, interpolated
        3: RVT — respiratory volume per time
        4: EDA_tonic — mean per TR
        5: EDA_phasic — mean per TR (log-transform before z-score in caller)
        6: SCR_binary — any SCR onset in TR window

    QC: failed channels → NaN for dependent features.
    PPG used as ECG backup when ECG fails but PPG passes.

    Args:
        preproc_path: Path to *_desc-preproc_physio.tsv.gz
        events_path: Path to *_events.tsv or *_desc-physio_events.tsv
        n_trs: Authoritative TR count from decoded_states.
        channel_qc: {channel: bool} from load_quality().

    Returns:
        np.ndarray shape (n_trs, 7): feature matrix.
    """
    features = np.full((n_trs, 7), np.nan)

    # Load data
    physio_df = load_preproc_physio(preproc_path)
    events_df = load_physio_events(events_path)
    duration_s = n_trs * 1.49

    # ── Cardiac features (HR, HRV) ───────────────────────────────────
    cardiac_ok = False
    rpeak_type = None

    if channel_qc.get("ECG", False):
        rpeak_type = "r_peak_corrected"
        cardiac_ok = True
    elif channel_qc.get("PPG", False):
        # PPG backup
        rpeak_type = "systolic_peak_corrected"
        cardiac_ok = True
        logger.info("ECG failed, using PPG backup (systolic_peak_corrected)")

    if cardiac_ok:
        try:
            rpeaks = get_events_by_type(events_df, rpeak_type)
            # HR
            hr_1khz = events_to_instantaneous_rate(rpeaks, duration_s)
            features[:, 0] = align_continuous_to_trs(hr_1khz, n_trs)
            # HRV RMSSD
            hrv_1khz = compute_hrv_rmssd(rpeaks, duration_s)
            features[:, 1] = align_continuous_to_trs(hrv_1khz, n_trs)
        except (ValueError, Exception) as e:
            logger.warning("Cardiac feature extraction failed: %s", e)

    # ── Respiratory features (breathing rate, RVT) ────────────────────
    if channel_qc.get("RSP", False):
        try:
            inhale_peaks = get_events_by_type(events_df, "inhale_max")
            exhale_troughs = get_events_by_type(events_df, "exhale_max")

            # Breathing rate
            br_1khz = events_to_instantaneous_rate(inhale_peaks, duration_s)
            features[:, 2] = align_continuous_to_trs(br_1khz, n_trs)

            # RVT
            rsp_signal = physio_df["RSP"].values
            rvt_1khz = compute_rvt(
                rsp_signal, inhale_peaks, exhale_troughs
            )
            features[:, 3] = align_continuous_to_trs(rvt_1khz, n_trs)
        except (ValueError, Exception) as e:
            logger.warning("Respiratory feature extraction failed: %s", e)

    # ── EDA features (tonic, phasic, SCR) ─────────────────────────────
    if channel_qc.get("EDA", False):
        try:
            # EDA tonic
            eda_tonic = physio_df["EDATonic"].values
            features[:, 4] = align_continuous_to_trs(eda_tonic, n_trs)

            # EDA phasic
            eda_phasic = physio_df["EDAPhasic"].values
            features[:, 5] = align_continuous_to_trs(eda_phasic, n_trs)

            # SCR binary
            scr_onsets = get_events_by_type(events_df, "scr_onset")
            features[:, 6] = events_to_tr_binary(scr_onsets, n_trs)
        except (ValueError, Exception) as e:
            logger.warning("EDA feature extraction failed: %s", e)

    return features


def normalize_features_within_run(features: np.ndarray) -> np.ndarray:
    """Z-score continuous features within run. Skip SCR_binary (col 6).

    EDA phasic (col 5) is log-transformed before z-scoring due to right skew.
    NaN values are preserved.

    Args:
        features: shape (n_trs, 7) from extract_physio_features().

    Returns:
        np.ndarray shape (n_trs, 7): normalized features.
    """
    out = features.copy()

    # Log-transform EDA phasic before z-scoring (add small epsilon for zeros)
    col5 = out[:, 5]
    valid_5 = np.isfinite(col5)
    if np.any(valid_5):
        # Shift to positive before log (min value may be negative after decomp)
        min_val = np.nanmin(col5[valid_5])
        offset = max(0, -min_val) + 1e-10
        out[valid_5, 5] = np.log(col5[valid_5] + offset)

    # Z-score columns 0-5 (not col 6 = SCR_binary)
    for col in range(6):
        valid = np.isfinite(out[:, col])
        if np.sum(valid) < 2:
            continue
        mean = np.nanmean(out[valid, col])
        std = np.nanstd(out[valid, col])
        if std > 0:
            out[valid, col] = (out[valid, col] - mean) / std
        else:
            out[valid, col] = 0.0

    return out
