#!/usr/bin/env python3
"""
content_io.py - I/O utilities for narrative content annotations.

Handles loading te-charnet sentence and scene annotations, mapping annotation
intervals to TR bins, and extracting TR-level content features.

Annotation data from te-charnet:
  - Sentences: per-utterance timing + speaker + row_type (dialogue/reaction)
  - Scenes: narrative scene boundaries with timing and speaker sets

Filename convention:
  - friends_{run_id}_sentence_speaker_table.tsv  (in s{N}/ directory)
  - friends_{run_id}_scene_summary.tsv            (in s{N}/ directory)

ID mapping: te-charnet uses 'friends_s01e01a', pipeline uses 's01e01a'.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.common import _get_season
from utils.env import require_env

logger = logging.getLogger(__name__)

TR_SECONDS = 1.49

# Main 6 Friends characters (case-sensitive, matching te-charnet speaker labels)
MAIN_CHARACTERS = ["Monica", "Ross", "Rachel", "Chandler", "Joey", "Phoebe"]

# Feature column names for the (n_trs, 16) output array
CONTENT_FEATURE_COLUMNS = [
    # Dialogue structure (cols 0-5)
    "speech_presence",       # 0: binary — any utterance overlapping this TR
    "dialogue_rate",         # 1: float — utterances overlapping / TR_s
    "n_speakers",            # 2: int   — unique speakers (dialogue rows only)
    "speaker_change",        # 3: binary — speaker differs between consecutive utterances
    "silence_duration_s",    # 4: float — seconds since last utterance end (0 if speaking)
    "utterance_duration_s",  # 5: float — total speech seconds within TR window
    # Scene features (cols 6-8)
    "scene_boundary",        # 6: binary — scene boundary within ±1 TR
    "n_scene_speakers",      # 7: int   — |scene_speaker_set| for current scene
    "scene_duration_s",      # 8: float — duration of current scene
    # Character features (cols 9-15)
    "monica_speaking",       # 9:  binary — Monica has dialogue overlapping this TR
    "ross_speaking",         # 10: binary — Ross speaking
    "rachel_speaking",       # 11: binary — Rachel speaking
    "chandler_speaking",     # 12: binary — Chandler speaking
    "joey_speaking",         # 13: binary — Joey speaking
    "phoebe_speaking",       # 14: binary — Phoebe speaking
    "n_main_in_scene",       # 15: int   — count of main 6 in current scene
]

N_FEATURES = len(CONTENT_FEATURE_COLUMNS)  # 16


# ── Inventory ────────────────────────────────────────────────────────────────


def build_content_inventory(
    run_ids: list[str],
    annotation_dir: str | None = None,
) -> dict[str, dict[str, Path | None]]:
    """Map decoded_states run keys to annotation file paths.

    Args:
        run_ids: List of run_id keys (e.g., ['s01e01a', 's01e01b', ...]).
        annotation_dir: Root annotation directory containing sentences/ and
            scenes/. Defaults to the ANNOTATION_DIR environment variable.

    Returns:
        dict: {run_id: {'sentences': Path|None, 'scenes': Path|None}}
    """
    ann_dir = Path(annotation_dir or require_env("ANNOTATION_DIR"))
    sentences_dir = ann_dir / "sentences"
    scenes_dir = ann_dir / "scenes"

    inventory = {}
    for run_id in run_ids:
        try:
            season = _get_season(run_id)
        except ValueError:
            logger.warning("Cannot parse season from run_id %s — skipping", run_id)
            inventory[run_id] = {"sentences": None, "scenes": None}
            continue

        season_dir = f"s{season}"
        sent_name = f"friends_{run_id}_sentence_speaker_table.tsv"
        scene_name = f"friends_{run_id}_scene_summary.tsv"

        sent_path = sentences_dir / season_dir / sent_name
        scene_path = scenes_dir / season_dir / scene_name

        inventory[run_id] = {
            "sentences": sent_path if sent_path.exists() else None,
            "scenes": scene_path if scene_path.exists() else None,
        }

        if inventory[run_id]["sentences"] is None:
            logger.warning("No sentence annotation for %s: %s", run_id, sent_path)

    n_with_sent = sum(1 for v in inventory.values() if v["sentences"] is not None)
    n_with_scene = sum(1 for v in inventory.values() if v["scenes"] is not None)
    logger.info(
        "Content inventory: %d runs, %d with sentences, %d with scenes",
        len(inventory), n_with_sent, n_with_scene,
    )
    return inventory


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_sentences(tsv_path: Path | str) -> pd.DataFrame:
    """Load sentence annotation TSV.

    Args:
        tsv_path: Path to *_sentence_speaker_table.tsv

    Returns:
        DataFrame with at least: start, end, speaker, row_type columns.
        Returns empty DataFrame if file is empty or missing required columns.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"start", "end", "speaker", "row_type"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Sentence TSV %s missing columns: %s", tsv_path, missing)
        return pd.DataFrame(columns=list(required))
    # Ensure numeric timing
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    return df


def load_scenes(tsv_path: Path | str) -> pd.DataFrame:
    """Load scene annotation TSV.

    Args:
        tsv_path: Path to *_scene_summary.tsv

    Returns:
        DataFrame with at least: scene_id, start, end columns.
        Returns empty DataFrame if file is empty or missing required columns.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"scene_id", "start", "end"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Scene TSV %s missing columns: %s", tsv_path, missing)
        return pd.DataFrame(columns=list(required))
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    return df


# ── Interval-to-TR mapping ──────────────────────────────────────────────────


def _overlapping_rows(df: pd.DataFrame, tr_start: float, tr_end: float) -> pd.DataFrame:
    """Return rows whose [start, end) interval overlaps [tr_start, tr_end)."""
    return df[(df["start"] < tr_end) & (df["end"] > tr_start)]


def sentences_to_tr_features(
    sent_df: pd.DataFrame,
    n_trs: int,
    tr_s: float = TR_SECONDS,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Convert sentence annotations to TR-level features.

    Features (columns 0-11):
        0: speech_presence     — any utterance overlapping this TR
        1: dialogue_rate       — count of overlapping utterances / tr_s
        2: n_speakers          — unique speakers (dialogue rows only)
        3: speaker_change      — speaker changed between consecutive dialogue utterances
        4: silence_duration_s  — time since last utterance end (0 if speaking)
        5: utterance_duration_s — total speech seconds within TR window
        6-11: character presence — one column per main character (dialogue rows only)

    Args:
        sent_df: DataFrame from load_sentences().
        n_trs: Number of TRs (authoritative count).
        tr_s: TR duration in seconds.
        offset_s: Time offset in seconds to subtract from annotation timestamps
            before mapping to TRs. Use when annotation time 0 ≠ fMRI acquisition
            start. Default 0.0 (no offset).

    Returns:
        np.ndarray shape (n_trs, 12). Cols 0-5: dialogue structure, cols 6-11: character.
    """
    features = np.zeros((n_trs, 12))

    if len(sent_df) == 0:
        return features

    # Apply offset: shift annotation timestamps to fMRI time
    sent_df = sent_df.copy()
    sent_df["start"] = sent_df["start"] - offset_s
    sent_df["end"] = sent_df["end"] - offset_s

    # Drop annotations that fall entirely before fMRI start
    sent_df = sent_df[sent_df["end"] > 0].reset_index(drop=True)

    # Sort by start time
    sent_df = sent_df.sort_values("start").reset_index(drop=True)

    # ── Vectorized overlap computation ────────────────────────────────
    tr_starts = np.arange(n_trs) * tr_s
    tr_ends = tr_starts + tr_s
    s_starts = sent_df["start"].values
    s_ends = sent_df["end"].values

    # Boolean overlap matrix: (n_sentences, n_trs)
    overlaps = (s_starts[:, None] < tr_ends[None, :]) & (s_ends[:, None] > tr_starts[None, :])

    # speech_presence: any utterance overlapping this TR
    features[:, 0] = overlaps.any(axis=0).astype(float)

    # dialogue_rate: count of overlapping utterances / tr_s
    features[:, 1] = overlaps.sum(axis=0) / tr_s

    # utterance_duration_s: total speech seconds within each TR window
    # Clip each utterance's [start, end) to [tr_start, tr_end) then sum
    clipped_starts = np.maximum(s_starts[:, None], tr_starts[None, :])  # (n_sent, n_trs)
    clipped_ends = np.minimum(s_ends[:, None], tr_ends[None, :])
    durations = np.maximum(clipped_ends - clipped_starts, 0.0) * overlaps
    features[:, 5] = durations.sum(axis=0)

    # silence_duration_s: time since last utterance end (0 if speaking)
    last_utt_end = 0.0
    for t in range(n_trs):
        if features[t, 0] > 0:  # speech present
            # Find max end of utterances overlapping this TR
            mask = overlaps[:, t]
            last_utt_end = max(last_utt_end, s_ends[mask].max())
            features[t, 4] = 0.0
        else:
            features[t, 4] = tr_starts[t] - last_utt_end if last_utt_end > 0 else tr_starts[t]

    # ── Speaker features (targeted loop, dialogue rows only) ──────────
    dialogue_df = sent_df[sent_df["row_type"] == "dialogue"]
    if len(dialogue_df) > 0:
        d_starts = dialogue_df["start"].values
        d_ends = dialogue_df["end"].values
        d_speakers = dialogue_df["speaker"].values
        d_overlaps = (d_starts[:, None] < tr_ends[None, :]) & (d_ends[:, None] > tr_starts[None, :])
        d_counts = d_overlaps.sum(axis=0)  # dialogue utterances per TR

        for t in np.where(d_counts > 0)[0]:
            mask = d_overlaps[:, t]
            spk = d_speakers[mask]
            spk = spk[pd.notna(spk)]
            features[t, 2] = len(set(spk))
            if len(spk) >= 2:
                features[t, 3] = 1.0 if len(set(spk)) > 1 else 0.0

        # ── Character presence (cols 6-11, vectorized) ────────────────
        for char_idx, char_name in enumerate(MAIN_CHARACTERS):
            char_mask = d_speakers == char_name  # (n_dialogue,)
            if np.any(char_mask):
                # Any dialogue row from this character overlapping each TR
                char_overlaps = d_overlaps[char_mask]  # (n_char_rows, n_trs)
                features[:, 6 + char_idx] = char_overlaps.any(axis=0).astype(float)

    return features


def scenes_to_tr_features(
    scene_df: pd.DataFrame,
    n_trs: int,
    tr_s: float = TR_SECONDS,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Convert scene annotations to TR-level features.

    Features (columns 0-2):
        0: scene_boundary      — scene boundary within ±1 TR of this TR
        1: n_scene_speakers    — |scene_speaker_set| for current scene
        2: scene_duration_s    — duration of current scene

    Args:
        scene_df: DataFrame from load_scenes().
        n_trs: Number of TRs.
        tr_s: TR duration in seconds.
        offset_s: Time offset in seconds to subtract from annotation timestamps.

    Returns:
        np.ndarray shape (n_trs, 3). NaN if no scene annotations available.
    """
    features = np.full((n_trs, 3), np.nan)

    if len(scene_df) == 0:
        return features

    scene_df = scene_df.copy()
    scene_df["start"] = scene_df["start"] - offset_s
    scene_df["end"] = scene_df["end"] - offset_s
    scene_df = scene_df[scene_df["end"] > 0].reset_index(drop=True)
    scene_df = scene_df.sort_values("start").reset_index(drop=True)

    # Scene boundaries (starts of each scene after the first)
    boundary_times = scene_df["start"].values[1:]  # skip first scene start

    # Vectorized TR centers
    tr_centers = (np.arange(n_trs) + 0.5) * tr_s

    # scene_boundary: any boundary within ±1 TR of this TR's center
    if len(boundary_times) > 0:
        # (n_boundaries, n_trs) distance matrix
        dists = np.abs(boundary_times[:, None] - tr_centers[None, :])
        features[:, 0] = np.where(np.any(dists <= tr_s, axis=0), 1.0, 0.0)
    else:
        features[:, 0] = 0.0

    # scene_duration_s: use searchsorted to find current scene per TR
    scene_starts = scene_df["start"].values
    scene_ends = scene_df["end"].values
    # For each TR center, find the rightmost scene whose start <= tr_center
    scene_idx = np.searchsorted(scene_starts, tr_centers, side="right") - 1
    for t in range(n_trs):
        si = scene_idx[t]
        if 0 <= si < len(scene_df) and scene_starts[si] <= tr_centers[t] < scene_ends[si]:
            features[t, 2] = scene_ends[si] - scene_starts[si]

    # n_scene_speakers requires cross-referencing with sentence data
    # (scene_speaker_set is in sentence TSV). Leave as NaN here;
    # extract_content_features() handles the cross-reference.

    return features


def _compute_scene_speaker_counts(
    sent_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    n_trs: int,
    tr_s: float = TR_SECONDS,
) -> np.ndarray:
    """Compute n_scene_speakers per TR from sentence+scene data.

    For each TR, find the current scene, then count unique speakers
    from dialogue rows within that scene.

    The caller must ensure scene_df timestamps are already offset-adjusted
    (in fMRI time). Speaker counts are matched by scene_id, so sent_df
    timestamps are not used — only sent_df scene_id and speaker columns.

    Returns:
        1D array (n_trs,) with speaker counts. NaN where no scene.
    """
    counts = np.full(n_trs, np.nan)
    if len(scene_df) == 0 or len(sent_df) == 0:
        return counts

    scene_df = scene_df.sort_values("start").reset_index(drop=True)
    dialogue_df = sent_df[sent_df["row_type"] == "dialogue"]

    # Precompute speakers per scene_id
    if "scene_id" in sent_df.columns and "scene_id" in scene_df.columns:
        # Use scene_speaker_set from sentence TSV if available
        if "scene_speaker_set" in sent_df.columns:
            # Parse pipe-separated speaker sets per scene_id
            scene_speakers = {}
            for sid in scene_df["scene_id"].unique():
                rows = sent_df[sent_df["scene_id"] == sid]
                sset = set()
                for val in rows["scene_speaker_set"].dropna().unique():
                    for name in str(val).split("|"):
                        name = name.strip()
                        if name:
                            sset.add(name)
                scene_speakers[sid] = len(sset) if sset else 0
        else:
            # Count unique speakers from dialogue rows per scene
            scene_speakers = {}
            for sid in scene_df["scene_id"].unique():
                rows = dialogue_df[dialogue_df["scene_id"] == sid]
                scene_speakers[sid] = rows["speaker"].nunique()
    else:
        return counts

    for t in range(n_trs):
        tr_center = (t + 0.5) * tr_s
        for _, row in scene_df.iterrows():
            if row["start"] <= tr_center < row["end"]:
                sid = row["scene_id"]
                counts[t] = scene_speakers.get(sid, 0)
                break

    return counts


def _compute_main_chars_in_scene(
    sent_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    n_trs: int,
    tr_s: float = TR_SECONDS,
) -> np.ndarray:
    """Count how many of the main 6 characters are present in each TR's scene.

    Uses scene_speaker_set from the sentence TSV (pipe-separated character names).
    The caller must ensure scene_df timestamps are already offset-adjusted.

    Returns:
        1D array (n_trs,) with counts (0-6). NaN where no scene covers TR center.
    """
    counts = np.full(n_trs, np.nan)
    if len(scene_df) == 0:
        return counts

    scene_df = scene_df.sort_values("start").reset_index(drop=True)
    main_set = set(MAIN_CHARACTERS)

    # Precompute main character count per scene_id
    scene_main_count = {}
    if "scene_id" in sent_df.columns and "scene_speaker_set" in sent_df.columns:
        for sid in scene_df["scene_id"].unique():
            rows = sent_df[sent_df["scene_id"] == sid]
            all_speakers = set()
            for val in rows["scene_speaker_set"].dropna().unique():
                for name in str(val).split("|"):
                    name = name.strip()
                    if name:
                        all_speakers.add(name)
            scene_main_count[sid] = len(all_speakers & main_set)
    elif "scene_id" in sent_df.columns and "speaker" in sent_df.columns:
        dialogue_df = sent_df[sent_df["row_type"] == "dialogue"]
        for sid in scene_df["scene_id"].unique():
            rows = dialogue_df[dialogue_df["scene_id"] == sid]
            speakers = set(rows["speaker"].dropna().unique())
            scene_main_count[sid] = len(speakers & main_set)
    else:
        return counts

    # Map TRs to scenes via searchsorted
    scene_starts = scene_df["start"].values
    scene_ends = scene_df["end"].values
    scene_ids = scene_df["scene_id"].values
    tr_centers = (np.arange(n_trs) + 0.5) * tr_s
    scene_idx = np.searchsorted(scene_starts, tr_centers, side="right") - 1

    for t in range(n_trs):
        si = scene_idx[t]
        if 0 <= si < len(scene_df) and scene_starts[si] <= tr_centers[t] < scene_ends[si]:
            counts[t] = scene_main_count.get(scene_ids[si], 0)

    return counts


# ── Combined feature extraction ──────────────────────────────────────────────


def validate_timestamp_alignment(
    sent_path: Path | str | None,
    n_trs: int,
    tr_s: float = TR_SECONDS,
) -> tuple[float, pd.DataFrame | None]:
    """Check whether annotation timestamps fit within fMRI recording duration.

    Must be called BEFORE feature extraction. Returns the offset and the loaded
    DataFrame (to avoid re-reading the file in extract_content_features).

    Args:
        sent_path: Path to sentence TSV, or None.
        n_trs: Number of TRs.
        tr_s: TR duration in seconds.

    Returns:
        (offset_s, sent_df): Recommended offset and loaded DataFrame.
            offset_s is 0.0 if timestamps already aligned.
            sent_df is None if sent_path is None.

    Raises:
        ValueError: If annotation timestamps extend far beyond fMRI duration
            and no reasonable offset can be determined.
    """
    if sent_path is None:
        return 0.0, None

    sent_df = load_sentences(sent_path)
    if len(sent_df) == 0:
        return 0.0, sent_df

    fmri_duration_s = n_trs * tr_s
    max_annotation_end = sent_df["end"].max()
    min_annotation_start = sent_df["start"].min()

    # Case 1: Annotations fit within fMRI duration → no offset needed
    if max_annotation_end <= fmri_duration_s + 5.0:  # 5s tolerance
        return 0.0, sent_df

    # Case 2: Annotations extend beyond fMRI duration → try offset
    # Heuristic: if shifting by min_annotation_start brings max_end within range,
    # the annotations are probably relative to episode start (not fMRI start)
    candidate_offset = min_annotation_start
    if (max_annotation_end - candidate_offset) <= fmri_duration_s + 5.0:
        logger.warning(
            "Annotations extend to %.1fs but fMRI is %.1fs. "
            "Applying offset of %.1fs (first annotation start).",
            max_annotation_end, fmri_duration_s, candidate_offset,
        )
        return candidate_offset, sent_df

    # Case 3: Even with offset, annotations don't fit → error
    raise ValueError(
        f"Annotation timestamps (max={max_annotation_end:.1f}s) exceed "
        f"fMRI duration ({fmri_duration_s:.1f}s) even after offset "
        f"({candidate_offset:.1f}s). Cannot determine alignment for "
        f"{sent_path}."
    )


def extract_content_features(
    sent_path: Path | str | None,
    scene_path: Path | str | None,
    n_trs: int,
    tr_s: float = TR_SECONDS,
    offset_s: float | None = None,
    sent_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Extract all 9 TR-level content features for one run.

    Args:
        sent_path: Path to sentence TSV, or None.
        scene_path: Path to scene TSV, or None.
        n_trs: Number of TRs (authoritative).
        tr_s: TR duration in seconds.
        offset_s: Time offset to subtract from annotation timestamps before
            mapping to TRs. If None, auto-detected via validate_timestamp_alignment().
        sent_df: Pre-loaded sentence DataFrame (to avoid re-reading the file).
            If None and sent_path is provided, the file is loaded.

    Returns:
        np.ndarray shape (n_trs, 16). See CONTENT_FEATURE_COLUMNS for column order.
    """
    # Auto-detect offset if not provided
    if offset_s is None:
        offset_s, sent_df = validate_timestamp_alignment(sent_path, n_trs, tr_s)

    features = np.full((n_trs, N_FEATURES), np.nan)

    # Sentence features (columns 0-5 dialogue structure, 9-14 character presence)
    if sent_path is not None:
        if sent_df is None:
            sent_df = load_sentences(sent_path)
        if len(sent_df) > 0:
            sent_feats = sentences_to_tr_features(sent_df, n_trs, tr_s, offset_s)
            features[:, :6] = sent_feats[:, :6]     # dialogue structure
            features[:, 9:15] = sent_feats[:, 6:12]  # character presence
    else:
        sent_df = pd.DataFrame()

    # Scene features (columns 6-8)
    if scene_path is not None:
        scene_df = load_scenes(scene_path)
        if len(scene_df) > 0:
            scene_feats = scenes_to_tr_features(scene_df, n_trs, tr_s, offset_s)
            features[:, 6] = scene_feats[:, 0]  # scene_boundary
            features[:, 8] = scene_feats[:, 2]  # scene_duration_s

            # Apply offset to scene_df for correct TR-to-scene mapping
            scene_df_shifted = scene_df.copy()
            scene_df_shifted["start"] = scene_df_shifted["start"] - offset_s
            scene_df_shifted["end"] = scene_df_shifted["end"] - offset_s
            scene_df_shifted = scene_df_shifted[scene_df_shifted["end"] > 0]

            # n_scene_speakers from cross-reference
            if len(sent_df) > 0:
                features[:, 7] = _compute_scene_speaker_counts(
                    sent_df, scene_df_shifted, n_trs, tr_s,
                )

            # n_main_in_scene (col 15): count of main 6 in current scene
            features[:, 15] = _compute_main_chars_in_scene(
                sent_df, scene_df_shifted, n_trs, tr_s,
            )

    return features


def normalize_content_features(features: np.ndarray) -> np.ndarray:
    """Z-score continuous features within run. Skip binary features.

    Binary features are left unchanged. Continuous features are z-scored per column.
    NaN values are preserved.

    Args:
        features: shape (n_trs, N_FEATURES) from extract_content_features().

    Returns:
        np.ndarray shape (n_trs, N_FEATURES): normalized features.
    """
    out = features.copy()
    # Binary: speech_presence(0), speaker_change(3), scene_boundary(6),
    #         character presence(9-14)
    binary_cols = {0, 3, 6, 9, 10, 11, 12, 13, 14}

    for col in range(out.shape[1]):
        if col in binary_cols:
            continue
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
