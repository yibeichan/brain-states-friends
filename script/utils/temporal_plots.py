"""
Publication-quality temporal dynamics visualization functions.

All functions accept pre-loaded data (no I/O) and return {"figure": fig}.
"""

import numpy as np

from utils.plot_style import CATEGORY_COLORS, CATEGORY_ORDER, make_category_legend_handles
from utils.state_blocks import TR_SECONDS


def plot_barcodes(
    barcode_episodes: list,
    decoded_states: dict,
    categories,
    n_states: int,
    tr_s: float = TR_SECONDS,
) -> dict:
    """State-sequence barcode panel coloured by category.

    Parameters
    ----------
    barcode_episodes : list[str]
        Ordered run IDs to display (one row per episode).
    decoded_states : dict
        run_id -> np.array(n_trs,) of integer state indices.
    categories : np.ndarray
        Per-state category string array, shape (n_states,).
    n_states : int
        Total number of HMM states (including inactive).
    tr_s : float
        TR in seconds (default: TR_SECONDS from state_blocks).

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from itertools import groupby

    n_ep = len(barcode_episodes)

    fig, axes = plt.subplots(
        n_ep, 1, figsize=(18, 0.78 * n_ep),
        gridspec_kw={"hspace": 1.5},
    )
    if n_ep == 1:
        axes = [axes]

    cmap_list = [
        mcolors.to_rgba(
            CATEGORY_COLORS.get(str(categories[i]) if i < len(categories) else "inactive", "#999999")
        )
        for i in range(n_states)
    ]

    for ax, run_id in zip(axes, barcode_episodes):
        state_seq = decoded_states[run_id]
        color_array = np.array(
            [cmap_list[int(s)] for s in state_seq]
        ).reshape(1, -1, 4)

        n_trs = len(state_seq)
        ax.imshow(
            color_array, aspect="auto",
            extent=[0, n_trs * tr_s, 0, 1],
        )

        tr_pos = 0
        for state_val, group in groupby(state_seq):
            block_len = sum(1 for _ in group)
            state_id = int(state_val)
            cat = (
                str(categories[state_id])
                if state_id < len(categories)
                else "inactive"
            )
            if tr_pos > 0:
                ax.axvline(
                    tr_pos * tr_s, color="white",
                    linewidth=0.5, linestyle="--", alpha=0.7,
                )
            if cat != "inactive" and block_len >= 4:
                center_s = (tr_pos + block_len / 2) * tr_s
                color = CATEGORY_COLORS.get(cat, "#999999")
                ax.text(
                    center_s, 1.05, str(state_id),
                    ha="center", va="bottom", fontsize=5,
                    fontweight="bold", color=color, clip_on=False,
                )
            tr_pos += block_len

        ax.set_yticks([])
        ax.set_ylabel(run_id, rotation=0, labelpad=28, va="center")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, n_trs * tr_s)
        ax.tick_params(axis="x", labelsize=7, pad=1)

    axes[-1].set_xlabel("Time (s)", labelpad=2)
    fig.tight_layout()
    return {"figure": fig}


def plot_dwell_raincloud(df_blocks) -> dict:
    """Raincloud (half-violin + boxplot + jitter) of dwell times for recurring/partial states.

    Parameters
    ----------
    df_blocks : pd.DataFrame
        Must have columns 'category' and 'duration_s'.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    cats = ["recurring", "partial"]
    df = df_blocks[df_blocks["category"].isin(cats)].copy()

    fig, ax = plt.subplots(figsize=(5, 4))

    for ci, cat in enumerate(cats):
        subset = df.loc[df["category"] == cat, "duration_s"].values
        color = CATEGORY_COLORS[cat]
        if len(subset) < 2:
            continue

        log_vals = np.log10(subset[subset > 0])

        kde = gaussian_kde(log_vals)
        y_range = np.linspace(log_vals.min(), log_vals.max(), 200)
        density = kde(y_range)
        density = density / density.max() * 0.35
        ax.fill_betweenx(y_range, ci - density, ci, color=color, alpha=0.6)

        ax.boxplot(
            [log_vals], positions=[ci + 0.15], widths=0.08,
            vert=True, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.8),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="", markersize=0),
        )

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.06, 0.06, len(log_vals))
        ax.scatter(ci + 0.28 + jitter, log_vals, s=3, alpha=0.15, color=color)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.capitalize() for c in cats])

    ytick_vals = [0, np.log10(3), 1, np.log10(30), 2]
    ytick_labels = ["1", "3", "10", "30", "100"]
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel("Dwell time (s)")
    fig.tight_layout()
    return {"figure": fig}


def plot_transition_matrix(P: np.ndarray, summary: dict, categories) -> dict:
    """Transition probability heatmap (LogNorm) sorted by category.

    Parameters
    ----------
    P : np.ndarray
        Shape (n_states, n_states) row-normalised transition matrix.
    summary : dict
        recurrence_summary.json content (for n_states and thresholds).
    categories : np.ndarray
        Per-state category string array.

    Returns
    -------
    {"figure": matplotlib.figure.Figure}
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch

    scores = np.array(summary["recurrence_scores"], dtype=float)
    n_states = summary["n_states"]

    order = []
    cat_boundaries = []
    for cat in CATEGORY_ORDER:
        cat_mask = categories == cat
        cat_idx = np.where(cat_mask)[0]
        cat_idx_sorted = cat_idx[np.argsort(-scores[cat_idx])]
        if len(cat_idx_sorted) > 0:
            cat_boundaries.append(
                (len(order), len(order) + len(cat_idx_sorted), cat)
            )
        order.extend(cat_idx_sorted.tolist())

    order = np.array(order)
    P_sorted = P[np.ix_(order, order)]

    P_display = P_sorted.copy()
    P_display[P_display == 0] = np.nan

    nonzero_vals = P_sorted[P_sorted > 0]
    vmin = float(nonzero_vals.min()) if len(nonzero_vals) > 0 else 1e-6
    vmax = float(P_sorted.max())

    strip_w = 1.5

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        P_display, aspect="equal", cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="none",
        extent=[strip_w, strip_w + n_states, strip_w + n_states, strip_w],
    )

    for start, end, _cat in cat_boundaries:
        if start > 0:
            ax.axhline(strip_w + start, color="white", linewidth=1.5, linestyle="--")
            ax.axvline(strip_w + start, color="white", linewidth=1.5, linestyle="--")

    for start, end, cat in cat_boundaries:
        color = CATEGORY_COLORS.get(cat, "#999999")
        ax.add_patch(plt.Rectangle(
            (0, strip_w + start), strip_w, end - start,
            facecolor=color, edgecolor="none", clip_on=False,
        ))
        ax.add_patch(plt.Rectangle(
            (strip_w + start, 0), end - start, strip_w,
            facecolor=color, edgecolor="none", clip_on=False,
        ))

    ax.set_xlim(0, strip_w + n_states)
    ax.set_ylim(strip_w + n_states, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"Source state (sorted by category, recurrence $\downarrow$)")
    ax.set_xlabel("Target state")

    bbox = ax.get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.15, 0.02, bbox.height * 0.5])
    fig.colorbar(im, cax=cbar_ax, label="Transition probability")

    handles = make_category_legend_handles(
        [c for c in CATEGORY_ORDER if (categories == c).any()]
    )
    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
        fontsize=7, frameon=False,
    )

    fig.tight_layout(rect=[0, 0.03, 0.92, 1.0])
    bbox = ax.get_position()
    cbar_ax.set_position([bbox.x1 + 0.015, bbox.y0 + 0.15, 0.015, bbox.height * 0.5])

    return {"figure": fig}
