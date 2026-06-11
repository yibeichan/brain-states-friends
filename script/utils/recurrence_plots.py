"""
Publication-quality recurrence visualization functions.

All functions accept pre-loaded data (no I/O) and return {"figure": fig}.
"""

import logging

import numpy as np

from utils.plot_style import (
    RECURRENCE_CMAP,
    recurrence_color,
    make_recurrence_colorbar,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Utility helpers (shared across 05d, etc.)
# ===================================================================

def compute_mean_fo_when_active(fo, n_states, fo_threshold):
    """Mean FO per state across episodes where the state was active (FO > threshold).

    Args:
        fo:           dict  run_id -> np.array(n_states,)
        n_states:     int
        fo_threshold: float, minimum FO to count as active

    Returns:
        mean_fo: np.array(n_states,)
            NaN for states that were never active in any episode.
    """
    if not fo:
        logger.warning("FO dictionary is empty; returning all-NaN mean FO vector.")
        return np.full(n_states, np.nan)

    fo_matrix = np.stack(list(fo.values()))    # (n_episodes, n_states)
    mask = fo_matrix > fo_threshold             # (n_episodes, n_states)
    state_counts = mask.sum(axis=0)             # (n_states,) — episodes where active

    # Vectorized: zero out inactive episodes, then divide by active count
    sum_fo = np.where(mask, fo_matrix, 0.0).sum(axis=0)   # (n_states,)

    mean_fo = np.full(n_states, np.nan)
    active = state_counts > 0
    mean_fo[active] = sum_fo[active] / state_counts[active]

    return mean_fo


def plot_recurrence_scatter(summary: dict, dwell_metrics) -> dict:
    """Scatter: recurrence score vs mean dwell time, coloured by recurrence.

    Parameters
    ----------
    summary : dict
        recurrence_summary.json content.
    dwell_metrics : pd.DataFrame
        Must have columns 'state' and 'mean_dwell_s'.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt

    scores = np.array(summary["recurrence_scores"], dtype=float)
    dwell_df = dwell_metrics.copy()
    dwell_by_state = (
        dwell_df.set_index("state")["mean_dwell_s"].reindex(range(summary["n_states"]))
    )
    mean_dwell_s = dwell_by_state.to_numpy(dtype=float)

    # Only plot active states
    active = scores > 0
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        scores[active], mean_dwell_s[active],
        c=scores[active], cmap=RECURRENCE_CMAP, vmin=0, vmax=1,
        s=40, alpha=0.8, edgecolors="white", linewidth=0.5,
    )
    fig.colorbar(sc, ax=ax, label="Recurrence score", shrink=0.8)

    # Annotate top-5 states
    active_idx = np.where(active)[0]
    top5 = sorted(active_idx, key=lambda i: scores[i], reverse=True)[:5]
    for idx in top5:
        ax.annotate(
            f"s{idx}", (scores[idx], mean_dwell_s[idx]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, color=recurrence_color(scores[idx]),
        )

    n_episodes = summary["n_episodes"]
    ax.set_xlabel(f"Recurrence score (n_active / {n_episodes} episodes)")
    ax.set_ylabel("Mean dwell time (s)")
    ax.set_yscale("log")
    dwell_ticks = [1, 2, 5, 10, 20, 30]
    finite_dwell = mean_dwell_s[np.isfinite(mean_dwell_s) & (mean_dwell_s > 0)]
    if finite_dwell.size:
        ymin = max(finite_dwell.min() * 0.85, 0.5)
        ymax = finite_dwell.max() * 1.15
        ax.set_ylim(ymin, ymax)
        valid_ticks = [t for t in dwell_ticks if ymin <= t <= ymax]
        if valid_ticks:
            ax.set_yticks(valid_ticks)
            ax.set_yticklabels([str(t) for t in valid_ticks])
    fig.tight_layout()
    return {"figure": fig}


def plot_recurrence_distribution(summary: dict) -> dict:
    """Histogram of recurrence scores for all active states.

    Parameters
    ----------
    summary : dict
        recurrence_summary.json content.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt

    scores = np.array(summary["recurrence_scores"], dtype=float)
    active_scores = scores[scores > 0]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    n_bins = min(20, max(5, len(active_scores) // 3))
    ax.hist(active_scores, bins=n_bins, color="#4682B4", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Recurrence score")
    ax.set_ylabel("Number of states")
    ax.axvline(np.median(active_scores), color="red", linestyle="--", alpha=0.7,
               label=f"Median = {np.median(active_scores):.2f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return {"figure": fig}


def plot_season_heatmap(per_season_rec: dict, summary: dict) -> dict:
    """Season x state heatmap of within-season recurrence scores.

    Parameters
    ----------
    per_season_rec : dict
        Mapping season_str -> list[float] (per-state within-season recurrence).
    summary : dict
        recurrence_summary.json content.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt

    scores = np.array(summary["recurrence_scores"], dtype=float)
    n_states = summary["n_states"]
    seasons = sorted(per_season_rec.keys(), key=lambda s: int(s))

    mat = np.zeros((n_states, len(seasons)))
    for j, s in enumerate(seasons):
        vals = per_season_rec[s]
        for i in range(min(n_states, len(vals))):
            mat[i, j] = vals[i]

    order = np.argsort(scores)
    mat_sorted = mat[order].T  # (n_seasons, n_states)
    scores_sorted = scores[order]

    fig, (ax_strip, ax_hm) = plt.subplots(
        2, 1, figsize=(9, 2.8),
        gridspec_kw={"height_ratios": [0.22, 2.2]},
    )

    # Recurrence score colour strip (replaces old category strip)
    for i, sc in enumerate(scores_sorted):
        color = recurrence_color(sc) if sc > 0 else "#999999"
        ax_strip.add_patch(
            plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none")
        )
    ax_strip.set_xlim(0, n_states)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    im = ax_hm.imshow(mat_sorted, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax_hm.set_xticks([])
    ax_hm.set_yticks(range(len(seasons)))
    ax_hm.set_yticklabels([f"S{s}" for s in seasons])
    ax_hm.set_xlabel("States (sorted by recurrence score)")
    ax_hm.set_ylabel("Season")
    fig.colorbar(
        im, ax=ax_hm, label="Within-season recurrence score",
        shrink=0.7, orientation="horizontal", pad=0.16,
    )
    fig.tight_layout()
    return {"figure": fig}


def plot_dwell_vs_recurrence(dwell_metrics) -> dict:
    """2-D histogram: recurrence score vs log mean dwell time.

    Parameters
    ----------
    dwell_metrics : pd.DataFrame
        Must have columns 'recurrence_score' and 'mean_dwell_s'.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt

    df = dwell_metrics.copy()
    log_dwell = np.log10(df["mean_dwell_s"].values.clip(min=1e-3))
    rec_scores = df["recurrence_score"].values

    n_xbins, n_ybins = 20, 15
    x_edges = np.linspace(rec_scores.min(), rec_scores.max(), n_xbins + 1)
    y_edges = np.linspace(log_dwell.min(), log_dwell.max(), n_ybins + 1)

    hist, _, _ = np.histogram2d(rec_scores, log_dwell, bins=[x_edges, y_edges])
    hist = hist.T
    hist_display = hist.copy().astype(float)
    hist_display[hist_display == 0] = np.nan

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(
        hist_display, origin="lower", aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="YlOrRd", interpolation="none",
    )

    for ix in range(n_xbins):
        for iy in range(n_ybins):
            count = int(hist[iy, ix])
            if count > 0:
                cx = (x_edges[ix] + x_edges[ix + 1]) / 2
                cy = (y_edges[iy] + y_edges[iy + 1]) / 2
                color = "white" if count > hist[~np.isnan(hist)].max() * 0.6 else "black"
                ax.text(
                    cx, cy, str(count),
                    ha="center", va="center", fontsize=5, color=color,
                )

    ax.set_xlabel("Recurrence score")
    ax.set_ylabel("Mean dwell time (s)")

    ytick_log = [np.log10(v) for v in [1, 2, 5, 10, 20]
                 if y_edges[0] <= np.log10(v) <= y_edges[-1]]
    ytick_labels = [
        str(int(10 ** v)) if 10 ** v == int(10 ** v) else f"{10 ** v:.0f}"
        for v in ytick_log
    ]
    ax.set_yticks(ytick_log)
    ax.set_yticklabels(ytick_labels)

    fig.colorbar(im, ax=ax, label="Number of states", shrink=0.8)
    fig.tight_layout()
    return {"figure": fig}
