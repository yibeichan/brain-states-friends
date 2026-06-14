"""Network-participation summaries for parcel-space state maps.

Canonical resting-state networks are used here only as an *annotation frame* for
fitted parcel-space state maps. The discovered unit of analysis is the HMM state
(a parcel-space coactivation profile), not the network. These summaries describe
how a state's magnitude is distributed across canonical network labels; they do
NOT claim the discovery of new functional networks, and they are not measures of
connectivity, coupling, or temporal/transition dynamics.

Because every metric is built from ``abs(state_mean)``, the summaries describe
participation *magnitude* and intentionally do not distinguish activation from
deactivation polarity.

The same metric definitions back both the content-eligible Figure 2C panel and
the supplementary all-category figure; pass ``summary_categories`` to select the
state set. The screening categories (``run_onset_anchored``, ``low_confidence``,
``unused``, ``rare``, ``season_temporal``) are provenance/QC labels and must not
be interpreted as cognitive or biological state classes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ELIGIBLE_CATEGORY = "eligible_for_content_analysis"

# Per-state scalar metrics summarized across states (medians, IQRs, etc.).
METRIC_NAMES = (
    "top1_share",
    "top3_share",
    "n_networks_ge_10pct",
    "normalized_network_entropy",
)


def network_scores_for_state(
    state_mean: np.ndarray,
    parcel_networks: Sequence[str],
    network_order: Sequence[str],
) -> np.ndarray:
    """Return the mean absolute parcel-space loading within each network.

    Uses the mean across a network's parcels (not the sum) so that large
    networks are not mechanically favored. Networks absent from
    ``parcel_networks`` score 0.
    """
    state_mean = np.asarray(state_mean, dtype=float)
    parcel_networks = np.asarray(parcel_networks)
    if state_mean.ndim != 1:
        raise ValueError("state_mean must be one-dimensional")
    if state_mean.shape[0] != parcel_networks.shape[0]:
        raise ValueError(
            "state_mean and parcel_networks must have the same length "
            f"({state_mean.shape[0]} != {parcel_networks.shape[0]})"
        )

    abs_pattern = np.abs(state_mean)
    return np.array(
        [
            float(abs_pattern[parcel_networks == network].mean())
            if np.any(parcel_networks == network)
            else 0.0
            for network in network_order
        ],
        dtype=float,
    )


def network_composition_for_state(
    state_mean: np.ndarray,
    parcel_networks: Sequence[str],
    network_order: Sequence[str],
) -> np.ndarray:
    """Return the normalized network composition (shares summing to 1).

    The composition is ``network_scores_for_state`` divided by its sum. Returns
    a zero vector when every network score is 0 (degenerate state).
    """
    scores = network_scores_for_state(state_mean, parcel_networks, network_order)
    total = float(scores.sum())
    if total <= 0:
        return np.zeros(len(network_order), dtype=float)
    return scores / total


def _state_column(flags: pd.DataFrame) -> str:
    if "state" in flags.columns:
        return "state"
    if "state_id" in flags.columns:
        return "state_id"
    raise KeyError("state_flags must contain either 'state' or 'state_id'")


def _normalized_entropy(comp: np.ndarray, n_networks_total: int) -> float:
    """Shannon entropy of a composition, normalized by ``log(n_networks_total)``.

    The denominator is the log of the *total* canonical-network count (the same
    for every state), so entropies are comparable across states regardless of
    how many networks have positive share. Range: 0 (all mass in one network) to
    ~1 (mass spread evenly across all networks). This is a spread-across-labels
    measure, not temporal or transition entropy.
    """
    nonzero = comp[comp > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(n_networks_total))


def _resolve_categories(
    state_flags: Mapping[str, pd.DataFrame],
    subjects: Sequence[str],
    summary_categories: Sequence[str] | None,
) -> set[str] | None:
    """None means keep every category; otherwise return the requested set."""
    if summary_categories is None:
        return None
    return set(summary_categories)


def compute_network_participation_metrics(
    state_flags: Mapping[str, pd.DataFrame],
    state_means: Mapping[str, np.ndarray],
    subjects: Sequence[str],
    parcel_networks: Sequence[str],
    network_order: Sequence[str],
    summary_categories: Sequence[str] | None = (ELIGIBLE_CATEGORY,),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute per-state canonical-network participation metrics.

    Parameters
    ----------
    summary_categories
        Which ``summary_category`` values to include. The default keeps only
        content-eligible states (Figure 2C). Pass ``None`` to keep every
        category (supplementary all-category figure), or a custom list to
        select specific categories.

    Returns
    -------
    metrics
        One row per state with shares, top-k summaries, and per-network share
        columns (``network_share_<net>``).
    summary
        Summary of the returned set (see ``summarize_network_participation``).
    """
    rows: list[dict[str, object]] = []
    order_index = {network: idx for idx, network in enumerate(network_order)}
    n_networks_total = len(network_order)
    keep = _resolve_categories(state_flags, subjects, summary_categories)

    for subject in subjects:
        flags = state_flags[subject]
        state_col = _state_column(flags)
        means = np.asarray(state_means[subject], dtype=float)
        if keep is None:
            selected = flags
        else:
            selected = flags[flags["summary_category"].isin(keep)]

        for _, row in selected.iterrows():
            state_id = int(row[state_col])
            comp = network_composition_for_state(
                means[state_id], parcel_networks, network_order
            )
            if not np.any(comp > 0):
                # Truly degenerate (all-zero) state map: no composition to
                # summarize. Skip rather than emit a spurious uniform row.
                continue

            entropy = _normalized_entropy(comp, n_networks_total)
            top_idx = np.argsort(comp)[::-1]
            ordered_top3 = tuple(network_order[idx] for idx in top_idx[:3])
            unordered_top3 = tuple(
                sorted(ordered_top3, key=lambda network: order_index[network])
            )

            record: dict[str, object] = {
                "subject": subject,
                "state": state_id,
                "summary_category": str(row["summary_category"]),
                "top1_network": network_order[top_idx[0]],
                "top1_share": float(comp[top_idx[0]]),
                "ordered_top3_networks": "|".join(ordered_top3),
                "unordered_top3_networks": "|".join(unordered_top3),
                "top3_share": float(comp[top_idx[:3]].sum()),
                "n_networks_ge_10pct": int(np.sum(comp >= 0.10)),
                "normalized_network_entropy": entropy,
            }
            # Optional provenance columns, when present in state_flags.
            if "recurrence_score" in flags.columns:
                record["recurrence_score"] = float(row["recurrence_score"])
            if "dominant_network" in flags.columns:
                record["dominant_network"] = str(row["dominant_network"])
            record.update(
                {
                    f"network_share_{network}": float(comp[idx])
                    for idx, network in enumerate(network_order)
                }
            )
            rows.append(record)

    columns = [
        "subject",
        "state",
        "summary_category",
        "top1_network",
        "top1_share",
        "ordered_top3_networks",
        "unordered_top3_networks",
        "top3_share",
        "n_networks_ge_10pct",
        "normalized_network_entropy",
        "recurrence_score",
        "dominant_network",
        *[f"network_share_{network}" for network in network_order],
    ]
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = metrics[[c for c in columns if c in metrics.columns]]
    summary = _summarize_frame(metrics, network_order)
    return metrics, summary


def _metric_iqr(frame: pd.DataFrame) -> dict[str, list[float]]:
    return {
        metric: [
            float(frame[metric].quantile(0.25)),
            float(frame[metric].quantile(0.75)),
        ]
        for metric in METRIC_NAMES
    }


def _counter_records(
    counter: Counter, key_name: str
) -> list[dict[str, object]]:
    records = []
    for key, n_states in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if isinstance(key, tuple):
            records.append({key_name: list(key), "n_states": int(n_states)})
        else:
            records.append({key_name: key, "n_states": int(n_states)})
    return records


def _summarize_frame(
    metrics: pd.DataFrame, network_order: Sequence[str]
) -> dict[str, object]:
    """Build a descriptive summary dict for a metrics (sub)frame."""
    empty = {
        "n_states": 0,
        "n_subjects": 0,
        "per_subject_counts": {},
        "metric_medians": {},
        "metric_iqr": {},
        "per_subject_medians": {},
        "mean_network_share": {},
        "top1_network_counts": [],
        "ordered_top3_rank_counts": [],
        "unordered_top3_combination_counts": [],
    }
    if metrics.empty:
        return empty

    ordered_counter = Counter(
        tuple(value.split("|")) for value in metrics["ordered_top3_networks"]
    )
    unordered_counter = Counter(
        tuple(value.split("|")) for value in metrics["unordered_top3_networks"]
    )
    top1_counter = Counter(metrics["top1_network"])

    per_subject = metrics.groupby("subject", sort=True)
    share_cols = [
        f"network_share_{network}"
        for network in network_order
        if f"network_share_{network}" in metrics.columns
    ]
    return {
        "n_states": int(len(metrics)),
        "n_subjects": int(metrics["subject"].nunique()),
        "per_subject_counts": {
            str(subject): int(count)
            for subject, count in metrics["subject"].value_counts().sort_index().items()
        },
        "metric_medians": {
            metric: float(metrics[metric].median()) for metric in METRIC_NAMES
        },
        "metric_iqr": _metric_iqr(metrics),
        "per_subject_medians": {
            str(subject): {
                metric: float(frame[metric].median()) for metric in METRIC_NAMES
            }
            for subject, frame in per_subject
        },
        "mean_network_share": {
            col.replace("network_share_", ""): float(metrics[col].mean())
            for col in share_cols
        },
        "top1_network_counts": _counter_records(top1_counter, "network"),
        "ordered_top3_rank_counts": _counter_records(ordered_counter, "networks"),
        "unordered_top3_combination_counts": _counter_records(
            unordered_counter, "networks"
        ),
    }


def summarize_network_participation(
    metrics: pd.DataFrame,
    network_order: Sequence[str],
    group_col: str = "summary_category",
) -> dict[str, object]:
    """Summarize a metrics table overall and per ``group_col`` value.

    Returns a dict with an ``"overall"`` summary and a ``"by_category"`` mapping
    from category label to its own summary. Each summary contains state counts,
    per-subject counts, metric medians/IQRs, mean per-network share, and ordered
    and unordered top-3 tallies (ordered = rank profiles; unordered = sets).
    """
    out: dict[str, object] = {
        "overall": _summarize_frame(metrics, network_order),
        "by_category": {},
    }
    if metrics.empty:
        return out
    out["by_category"] = {
        str(category): _summarize_frame(frame, network_order)
        for category, frame in metrics.groupby(group_col, sort=True)
    }
    return out


def save_network_participation_outputs(
    metrics: pd.DataFrame,
    summary: Mapping[str, object],
    output_dir: str | Path,
    prefix: str = "fig2_C_network_participation",
) -> tuple[Path, Path]:
    """Write per-state metrics CSV and summary JSON; return their paths."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    metrics.to_csv(metrics_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return metrics_path, summary_path


def plot_network_participation(
    metrics: pd.DataFrame,
    summary: Mapping[str, object],
    output_stem: str | Path,
    subjects: Sequence[str],
    subject_markers: Mapping[str, str],
) -> None:
    """Render the Figure 2C content-eligible network-participation panel."""
    import matplotlib.pyplot as plt

    if metrics.empty:
        raise ValueError("Cannot plot an empty network-participation table")

    output_stem = Path(output_stem)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(6.7, 2.7), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )

    metric_defs = [
        ("top1_share", "Largest\nnetwork share"),
        ("top3_share", "Top-3\nnetwork share"),
        ("normalized_network_entropy", "Network\nentropy"),
    ]
    rng = np.random.default_rng(1729)
    for x_pos, (metric, label) in enumerate(metric_defs):
        values = metrics[metric].values
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax0.add_patch(
            plt.Rectangle(
                (x_pos - 0.18, q1),
                0.36,
                q3 - q1,
                facecolor="#E8E8E8",
                edgecolor="#777777",
                linewidth=0.7,
                zorder=2,
            )
        )
        ax0.plot(
            [x_pos - 0.22, x_pos + 0.22],
            [median, median],
            color="#111111",
            lw=1.4,
            zorder=4,
        )
        for subject in subjects:
            subject_values = metrics.loc[metrics["subject"] == subject, metric].values
            jitter = rng.uniform(-0.11, 0.11, size=len(subject_values))
            ax0.scatter(
                np.full(len(subject_values), x_pos) + jitter,
                subject_values,
                s=13,
                color="#4A4A4A",
                alpha=0.45,
                marker=subject_markers.get(subject, "o"),
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )

    ax0.set_xticks(range(len(metric_defs)))
    ax0.set_xticklabels([label for _, label in metric_defs], fontsize=6)
    ax0.set_ylim(0, 1.02)
    ax0.set_ylabel("Network-composition value", fontsize=7)
    ax0.tick_params(axis="y", labelsize=6)
    ax0.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#E2E2E2", zorder=0)
    for spine in ("top", "right"):
        ax0.spines[spine].set_visible(False)

    counts = metrics["n_networks_ge_10pct"].value_counts().sort_index()
    x_vals = counts.index.values.astype(int)
    ax1.bar(
        x_vals,
        counts.values,
        color="#BDBDBD",
        edgecolor="#6A6A6A",
        linewidth=0.6,
        width=0.72,
    )
    ax1.axvline(
        float(summary["metric_medians"]["n_networks_ge_10pct"]),
        color="#111111",
        lw=1.0,
        linestyle=":",
    )
    ax1.set_xlabel("Networks >=10%", fontsize=7)
    ax1.set_ylabel("States", fontsize=7)
    ax1.set_xticks(x_vals)
    ax1.tick_params(axis="both", labelsize=6)
    ax1.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#E2E2E2", zorder=0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.95, wspace=0.34)
    for suffix, kwargs in (
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".svg", {}),
    ):
        fig.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def _ordered_categories(
    metrics: pd.DataFrame, category_order: Sequence[str] | None
) -> list[str]:
    present = list(pd.unique(metrics["summary_category"]))
    if category_order is None:
        # Largest category first; stable for ties.
        counts = metrics["summary_category"].value_counts()
        return list(counts.index)
    ordered = [c for c in category_order if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def plot_network_participation_by_category(
    metrics: pd.DataFrame,
    summary: Mapping[str, object],
    output_stem: str | Path,
    subjects: Sequence[str],
    network_order: Sequence[str],
    subject_markers: Mapping[str, str],
    display_network=None,
    category_labels: Mapping[str, str] | None = None,
    category_order: Sequence[str] | None = None,
) -> None:
    """Render the supplementary all-category network-participation figure.

    Four descriptive panels:
      A. state counts by category and subject (stacked bars)
      B. category-by-network heatmap of mean normalized composition
      C. category-wise distributions of top1_share, top3_share, entropy
      D. category-wise counts of n_networks_ge_10pct

    Categories are provenance/screening labels, not cognitive classes; the panels
    are intentionally descriptive (no inferential tests).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if metrics.empty:
        raise ValueError("Cannot plot an empty network-participation table")

    if display_network is None:
        def display_network(name):  # noqa: ANN001
            return name

    def cat_label(cat: str) -> str:
        if category_labels and cat in category_labels:
            return category_labels[cat]
        return cat.replace("_", " ")

    categories = _ordered_categories(metrics, category_order)
    net_present = [
        net for net in network_order if f"network_share_{net}" in metrics.columns
    ]

    fig = plt.figure(figsize=(11.0, 9.0))
    gs = GridSpec(
        2, 2, figure=fig, hspace=0.45, wspace=0.30,
        left=0.10, right=0.97, top=0.95, bottom=0.10,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    cat_x = np.arange(len(categories))

    # ── Panel A: stacked state counts by category and subject ────────────────
    bottoms = np.zeros(len(categories))
    cmapA = plt.get_cmap("tab10")
    for s_idx, subject in enumerate(subjects):
        counts = [
            int(
                ((metrics["summary_category"] == cat) & (metrics["subject"] == subject)).sum()
            )
            for cat in categories
        ]
        axA.bar(
            cat_x, counts, bottom=bottoms, width=0.72,
            color=cmapA(s_idx % 10), edgecolor="white", linewidth=0.4,
            label=subject,
        )
        bottoms += np.array(counts, dtype=float)
    for x, total in zip(cat_x, bottoms):
        axA.text(x, total + 1.0, f"{int(total)}", ha="center", va="bottom", fontsize=7)
    axA.set_xticks(cat_x)
    axA.set_xticklabels([cat_label(c) for c in categories], rotation=35, ha="right", fontsize=7)
    axA.set_ylabel("States", fontsize=8)
    axA.set_title("A  State counts by category and subject", fontsize=9, loc="left")
    axA.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    axA.margins(y=0.12)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)

    # ── Panel B: category x network mean composition heatmap ─────────────────
    heat = np.array(
        [
            [
                float(metrics.loc[metrics["summary_category"] == cat,
                                  f"network_share_{net}"].mean())
                for net in net_present
            ]
            for cat in categories
        ]
    )
    im = axB.imshow(heat, aspect="auto", cmap="magma", vmin=0.0)
    axB.set_xticks(np.arange(len(net_present)))
    axB.set_xticklabels([display_network(n) for n in net_present], rotation=60, ha="right", fontsize=6)
    axB.set_yticks(cat_x)
    axB.set_yticklabels([cat_label(c) for c in categories], fontsize=7)
    axB.set_title("B  Mean network composition by category", fontsize=9, loc="left")
    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.set_label("Mean share", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ── Panel C: distributions of top1/top3/entropy by category ──────────────
    metric_defs = [
        ("top1_share", "#4C72B0"),
        ("top3_share", "#55A868"),
        ("normalized_network_entropy", "#C44E52"),
    ]
    n_m = len(metric_defs)
    group_w = 0.8
    sub_w = group_w / n_m
    rng = np.random.default_rng(2027)
    for m_idx, (metric, color) in enumerate(metric_defs):
        offset = -group_w / 2 + sub_w * (m_idx + 0.5)
        for c_idx, cat in enumerate(categories):
            vals = metrics.loc[metrics["summary_category"] == cat, metric].values
            xc = c_idx + offset
            if len(vals):
                q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                axC.add_patch(
                    plt.Rectangle(
                        (xc - sub_w * 0.4, q1), sub_w * 0.8, max(q3 - q1, 1e-9),
                        facecolor=color, alpha=0.30, edgecolor=color, linewidth=0.6,
                        zorder=2,
                    )
                )
                axC.plot([xc - sub_w * 0.45, xc + sub_w * 0.45], [med, med],
                         color=color, lw=1.3, zorder=4)
                jitter = rng.uniform(-sub_w * 0.32, sub_w * 0.32, size=len(vals))
                axC.scatter(np.full(len(vals), xc) + jitter, vals, s=5,
                            color=color, alpha=0.35, linewidth=0, zorder=3)
    axC.set_xticks(cat_x)
    axC.set_xticklabels([cat_label(c) for c in categories], rotation=35, ha="right", fontsize=7)
    axC.set_ylim(0, 1.02)
    axC.set_ylabel("Composition value", fontsize=8)
    axC.set_title("C  Concentration and spread by category", fontsize=9, loc="left")
    handles = [
        plt.Line2D([0], [0], color=color, lw=3,
                   label={"top1_share": "Largest share",
                          "top3_share": "Top-3 share",
                          "normalized_network_entropy": "Network entropy"}[metric])
        for metric, color in metric_defs
    ]
    axC.legend(handles=handles, fontsize=6, frameon=False, loc="upper right")
    axC.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#E2E2E2", zorder=0)
    for spine in ("top", "right"):
        axC.spines[spine].set_visible(False)

    # ── Panel D: n_networks_ge_10pct proportions by category ─────────────────
    max_k = int(metrics["n_networks_ge_10pct"].max())
    k_vals = list(range(1, max_k + 1))
    cmapD = plt.get_cmap("viridis")
    bottoms = np.zeros(len(categories))
    for k in k_vals:
        props = []
        for cat in categories:
            sub = metrics.loc[metrics["summary_category"] == cat, "n_networks_ge_10pct"]
            props.append(float((sub == k).mean()) if len(sub) else 0.0)
        props = np.array(props)
        axD.bar(cat_x, props, bottom=bottoms, width=0.72,
                color=cmapD((k - 1) / max(max_k - 1, 1)),
                edgecolor="white", linewidth=0.4, label=f"{k}")
        bottoms += props
    axD.set_xticks(cat_x)
    axD.set_xticklabels([cat_label(c) for c in categories], rotation=35, ha="right", fontsize=7)
    axD.set_ylim(0, 1.0)
    axD.set_ylabel("Proportion of states", fontsize=8)
    axD.set_title("D  Networks >=10% share by category", fontsize=9, loc="left")
    axD.legend(title="Networks >=10%", fontsize=6, title_fontsize=6,
               ncol=2, frameon=False, loc="upper right")
    for spine in ("top", "right"):
        axD.spines[spine].set_visible(False)

    output_stem = Path(output_stem)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300}), (".svg", {})):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.05, **kwargs)
    plt.close(fig)
