"""Shared I/O and constants for 05e_a4 state-flag taxonomy.

Provides loading helpers so that downstream scripts (07b, 07c, 08b, …)
can annotate their outputs with the summary_category from state_flags.csv
without duplicating constants or file-discovery logic.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# ── shared constants ─────────────────────────────────────────────────────

TAG_COLUMNS = [
    "sub_hrf", "unused", "run_onset",
    "a_anchored", "b_anchored",
    "session_trend_down", "session_trend_up",
    "season_structured", "global_trend",
]

CATEGORY_PRIORITY = [
    "unused", "low_confidence", "run_onset_anchored",
    "season_temporal",
    "eligible_for_content_analysis", "rare",
]

TAG_COLORS = {
    "sub_hrf": "#F59E0B",
    "unused": "#6B7280",
    "run_onset": "#EF4444",
    "a_anchored": "#F97316",
    "b_anchored": "#EC4899",
    "session_trend_down": "#3B82F6",
    "session_trend_up": "#06B6D4",
    "season_structured": "#8B5CF6",
    "global_trend": "#10B981",
}

CATEGORY_COLORS = {
    "unused": "#6B7280",
    "low_confidence": "#F59E0B",
    "run_onset_anchored": "#EF4444",
    "season_temporal": "#8B5CF6",
    "eligible_for_content_analysis": "#22C55E",
    "rare": "#BDC3C7",
    "unknown": "#999999",
}

CATEGORY_DISPLAY_NAMES = {
    "unused": "Unused",
    "low_confidence": "Low confidence",
    "run_onset_anchored": "Run-onset anchored",
    "season_temporal": "Season/temporal",
    "eligible_for_content_analysis": "Eligible",
    "rare": "Rare",
    "unknown": "Unknown",
}

CATEGORY_MARKERS = {
    "eligible_for_content_analysis": "o",
    "run_onset_anchored": "^",
    "season_temporal": "s",
    "low_confidence": "D",
    "rare": "X",
    "unused": "+",
    "unknown": ".",
}


# ── I/O helpers ──────────────────────────────────────────────────────────

def load_state_flags(
    sub_id: str,
    parcellation: str,
    scratch_dir: str,
    vt: str | None = None,
) -> pd.DataFrame | None:
    """Load state_flags.csv produced by 05e_a4.

    Returns None (with a log warning) if the file does not exist,
    so callers can degrade gracefully.
    """
    base = os.path.join(
        scratch_dir, "output", "05e_temporal_trend_a4", parcellation, sub_id,
    )
    if vt is not None:
        base = os.path.join(base, f"vt{vt}")

    csv_path = os.path.join(base, "state_flags.csv")
    if not os.path.isfile(csv_path):
        logger.warning(
            "state_flags.csv not found at %s - categories will be 'unknown'.",
            csv_path,
        )
        return None

    df = pd.read_csv(csv_path)
    if "summary_category" not in df.columns:
        logger.warning("state_flags.csv missing 'summary_category' column.")
        return None

    logger.info(
        "Loaded state flags (%d states): %s",
        len(df),
        df["summary_category"].value_counts().to_dict(),
    )
    return df


def annotate_dataframe(
    df: pd.DataFrame,
    state_flags: pd.DataFrame | None,
    state_col: str = "state",
) -> pd.DataFrame:
    """Add ``summary_category`` column to *df* by merging on *state_col*.

    If *state_flags* is None every row gets ``"unknown"``.
    """
    if state_flags is None:
        df["summary_category"] = "unknown"
        return df

    mapping = dict(zip(state_flags["state"], state_flags["summary_category"]))
    df["summary_category"] = (
        df[state_col].map(mapping).fillna("unknown")
    )
    return df
