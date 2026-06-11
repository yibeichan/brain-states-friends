"""Physio QC utilities for 07b/07c annotation and sensitivity analyses.

Loads PercentageValid from source physprep quality JSONs, classifies
per-channel confidence, and runs the EDA MNAR diagnostic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.physio_io import build_physio_inventory

logger = logging.getLogger(__name__)

CHANNELS = ["ECG", "RSP", "EDA", "PPG"]


# ── PercentageValid loading ──────────────────────────────────────────────

def load_run_percentage_valid(
    sub_id: str,
    physprep_dir: str,
    run_ids: list[str],
    stimulus: str = "friends",
) -> pd.DataFrame:
    """Extract per-channel PercentageValid from source physprep quality JSONs.

    Returns
    -------
    DataFrame with columns: run_id, ECG_pv, RSP_pv, EDA_pv, PPG_pv.
    Missing quality files or channels get NaN.
    """
    inventory = build_physio_inventory(
        sub_id, physprep_dir, run_ids, stimulus=stimulus,
    )

    rows: list[dict] = []
    for run_id in run_ids:
        row: dict = {"run_id": run_id}
        entry = inventory.get(run_id)
        qpath = entry.get("quality") if entry else None

        if qpath is None or not Path(qpath).is_file():
            for ch in CHANNELS:
                row[f"{ch}_pv"] = np.nan
        else:
            with open(qpath) as f:
                data = json.load(f)
            for ch in CHANNELS:
                ch_data = data.get(ch, {})
                if isinstance(ch_data, dict):
                    row[f"{ch}_pv"] = ch_data.get("PercentageValid", np.nan)
                else:
                    row[f"{ch}_pv"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    n_valid = df[[f"{ch}_pv" for ch in CHANNELS]].notna().all(axis=1).sum()
    logger.info(
        "Loaded PercentageValid for %d/%d runs (stimulus=%s)",
        n_valid, len(run_ids), stimulus,
    )
    return df


# ── Per-channel confidence classification ────────────────────────────────

def classify_channel_confidence(
    pv_df: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Flag per-channel, per-run confidence based on PercentageValid.

    Returns
    -------
    Boolean DataFrame with same index as *pv_df* and columns
    ECG_ok, RSP_ok, EDA_ok, PPG_ok.  True = meets threshold.
    """
    out = pd.DataFrame(index=pv_df.index)
    for ch in CHANNELS:
        col = f"{ch}_pv"
        out[f"{ch}_ok"] = pv_df[col] >= threshold
    return out


# ── QC summary report ────────────────────────────────────────────────────

def compute_qc_report(
    pv_df: pd.DataFrame,
    threshold: float = 0.80,
) -> dict:
    """Produce a JSON-serialisable QC summary.

    Returns dict with per-channel stats (mean, median, min PV; n_valid;
    fraction valid) and the overall number of runs.
    """
    report: dict = {"n_runs_total": len(pv_df), "pv_threshold": threshold}
    per_channel: dict = {}

    for ch in CHANNELS:
        col = f"{ch}_pv"
        vals = pv_df[col].dropna()
        n_above = int((vals >= threshold).sum())
        per_channel[ch] = {
            "n_with_pv": int(len(vals)),
            "n_valid": n_above,
            "frac_valid": round(n_above / max(len(vals), 1), 4),
            "mean_pv": round(float(vals.mean()), 4) if len(vals) else None,
            "median_pv": round(float(vals.median()), 4) if len(vals) else None,
            "min_pv": round(float(vals.min()), 4) if len(vals) else None,
        }

    report["per_channel"] = per_channel
    return report


# ── EDA MNAR diagnostic ─────────────────────────────────────────────────

def run_eda_mnar_diagnostic(
    physio_features: dict[str, np.ndarray],
    pv_df: pd.DataFrame,
) -> dict:
    """Compare mean HR between EDA-pass and EDA-fail runs.

    If mean HR is significantly higher in EDA-fail runs, this suggests
    MNAR (high arousal → sweat → sensor failure).

    Parameters
    ----------
    physio_features : {run_id: ndarray(n_trs, 7)}
        Feature matrices (col 0 = HR).
    pv_df : DataFrame
        Output of :func:`load_run_percentage_valid`.

    Returns
    -------
    dict with keys: n_pass, n_fail, hr_mean_pass, hr_mean_fail,
    mwu_statistic, mwu_pvalue, interpretation.
    """
    pv_lookup = dict(zip(pv_df["run_id"], pv_df["EDA_pv"]))

    hr_pass: list[float] = []
    hr_fail: list[float] = []

    for run_id, feat in physio_features.items():
        # HR is column 0
        hr_vals = feat[:, 0]
        mean_hr = np.nanmean(hr_vals)
        if np.isnan(mean_hr):
            continue

        eda_pv = pv_lookup.get(run_id, np.nan)
        if np.isnan(eda_pv):
            continue

        # Use physprep's own 0.80 boundary for pass/fail
        if eda_pv >= 0.80:
            hr_pass.append(mean_hr)
        else:
            hr_fail.append(mean_hr)

    result: dict = {
        "n_eda_pass": len(hr_pass),
        "n_eda_fail": len(hr_fail),
        "hr_mean_pass": round(float(np.mean(hr_pass)), 4) if hr_pass else None,
        "hr_mean_fail": round(float(np.mean(hr_fail)), 4) if hr_fail else None,
    }

    if len(hr_pass) >= 5 and len(hr_fail) >= 5:
        stat, pval = sp_stats.mannwhitneyu(
            hr_fail, hr_pass, alternative="greater",
        )
        result["mwu_statistic"] = round(float(stat), 4)
        result["mwu_pvalue"] = round(float(pval), 6)
        result["interpretation"] = (
            "EDA failures have significantly higher HR (MNAR likely)"
            if pval < 0.05
            else "No evidence of HR difference between EDA-pass/fail runs"
        )
    else:
        result["mwu_statistic"] = None
        result["mwu_pvalue"] = None
        result["interpretation"] = (
            f"Too few EDA-fail runs ({len(hr_fail)}) for diagnostic test"
        )

    logger.info("EDA MNAR diagnostic: %s", result["interpretation"])
    return result
