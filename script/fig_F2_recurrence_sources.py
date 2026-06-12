"""Figure F2 - the sources of recurrence (R2).

Split from the former `fig_F1_recurrence_and_taxonomy.py` (2026-06-10). R1 (the
recurrence gradient) is now `fig_F1_recurrence_gradient.py`; this script is R2
(the heterogeneous sources of recurrence).

Panels B and C are lifted verbatim from the former F1 (Sankey + surface
exemplars). Panel A is the former F1 stacked taxonomy bars WITH the recurrence
dots and dwell marginal removed (recurrence now lives in F1); it shows pure
category composition over all 50 latent states (including unused).

| Panel | Content | Chart family | Source | Output |
|---|---|---|---|---|
| A | Per-subject category-count stacked bars (5 categories, all 50 states) | bar | 05e_a4 state_flags.csv | fig2_A_category_bars.{pdf,png} |
| B | Sankey: 5 categories -> 13 networks, top-3 composition | flow/network | state_flags.csv + 04 state_means_parcel.npy | fig2_B_category_network_sankey.{pdf,png} |
| C | Surface exemplars per category (sub-01) + activity timeseries strips | brain map | 04 state_means_parcel.npy + yabplot + decoded_states.pkl | fig2_C_*.{pdf,png} (multi-file) |

Run:
    marimo edit script/fig_F2_recurrence_sources.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import os
    import pickle
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import (
        NETWORK_COLORS,
        NETWORK_ORDER,
        apply_publication_style,
        display_network,
    )

    apply_publication_style()

    return (
        Path, json, np, os, pd, pickle, plt,
        NETWORK_COLORS, NETWORK_ORDER, display_network,
    )


@app.cell
def config(Path, os):
    """Paths, subject list, output dir."""
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]

    FLAGS_DIR = SCRATCH_DIR / "output" / "05e_temporal_trend_a4" / PARCELLATION
    MODEL_DIR = SCRATCH_DIR / "output" / "04_combined_hdphmm" / PARCELLATION

    OUT_F2 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig2"
    OUT_F2.mkdir(parents=True, exist_ok=True)
    return SCRATCH_DIR, PARCELLATION, VT, SUBJECTS, FLAGS_DIR, MODEL_DIR, OUT_F2


@app.cell
def taxonomy_constants():
    """Display labels + colors for the 5 R2 taxonomy categories.

    Fig 2 shows ALL 50 latent states (including unused), so the catch-all is the
    full "Unused + rare" (Fig 1's active beeswarm uses "Rare" alone).
    """
    TAXONOMY_ORDER = [
        "Content-eligible",
        "Run-onset-anchored",
        "Low-confidence",
        "Drift-anchored",
        "Unused + rare",
    ]
    TAXONOMY_COLORS = {
        "Content-eligible":   "#0C7BDC",
        "Run-onset-anchored": "#FFC20A",
        "Low-confidence":     "#5A5A5A",
        "Drift-anchored":     "#D35FB7",
        "Unused + rare":      "#CCCCCC",
    }
    TAXONOMY_MAP = {
        "eligible_for_content_analysis": "Content-eligible",
        "run_onset_anchored":            "Run-onset-anchored",
        "low_confidence":                "Low-confidence",
        "season_temporal":               "Drift-anchored",
        "unused":                        "Unused + rare",
        "rare":                          "Unused + rare",
    }
    return TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP


@app.cell
def load_state_flags(SUBJECTS, FLAGS_DIR, VT, pd):
    """Load 05e_a4 state_flags.csv per subject (50 rows: recurrence_score,
    summary_category, dominant_network)."""
    state_flags = {}
    for _sub in SUBJECTS:
        state_flags[_sub] = pd.read_csv(FLAGS_DIR / _sub / VT / "state_flags.csv")
    print({_s: _d.shape for _s, _d in state_flags.items()})
    return (state_flags,)


@app.cell
def load_state_means(SUBJECTS, MODEL_DIR, VT, np):
    """Load state mean activation patterns in parcel space per subject."""
    state_means = {}
    for _sub in SUBJECTS:
        _path = MODEL_DIR / _sub / "final" / VT / "state_means_parcel.npy"
        state_means[_sub] = np.load(_path)
    print({_s: _m.shape for _s, _m in state_means.items()})
    return (state_means,)


@app.cell
def load_decoded_states_sub01(MODEL_DIR, VT, pickle):
    """Viterbi-decoded state sequences for sub-01 (all runs) - for Panel C strips."""
    _path = MODEL_DIR / "sub-01" / "final" / VT / "decoded_states.pkl"
    with open(_path, "rb") as _f:
        decoded_states_sub01 = pickle.load(_f)
    print(f"sub-01 decoded_states: {len(decoded_states_sub01)} runs")
    return (decoded_states_sub01,)


@app.cell
def panel_A_category_bars(
    state_flags, SUBJECTS, OUT_F2,
    TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP, plt, np,
):
    """Panel A - per-subject category-count stacked bars (no recurrence dots).

    Horizontal stacked bars, one per subject (sub-01 bottom), counting all 50
    latent states across the 5 source categories. Pure composition by source;
    recurrence is shown in Figure 1. Channels: x = count, y(row) = subject,
    segment color = category.
    """
    # luminance-aware in-segment label color (white on dark fills, dark on light)
    _TEXT_ON = {
        "Content-eligible":   "white",
        "Run-onset-anchored": "#3A2D00",   # amber is bright -> dark text
        "Low-confidence":     "white",
        "Drift-anchored":     "white",
        "Unused + rare":      "#5A5A5A",    # pale gray -> dark text
    }
    _row_h = 0.64
    _fig, _ax = plt.subplots(figsize=(7.4, 3.6))

    for _j, _sub in enumerate(SUBJECTS):
        _df = state_flags[_sub].copy()
        _df["_cat"] = _df["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare")
        _x_cursor = 0
        for _cat in TAXONOMY_ORDER:
            _n = int((_df["_cat"] == _cat).sum())
            if _n == 0:
                continue
            _ax.barh(
                _j, _n, left=_x_cursor, height=_row_h,
                color=TAXONOMY_COLORS[_cat], edgecolor="white", linewidth=1.4,
                zorder=2,
            )
            if _n >= 1:
                _ax.text(_x_cursor + _n / 2, _j, str(_n), ha="center",
                         va="center", fontsize=7.5, color=_TEXT_ON[_cat], zorder=3)
            _x_cursor += _n

    _ax.set_yticks(np.arange(len(SUBJECTS)))
    _ax.set_yticklabels(SUBJECTS, fontsize=8)
    _ax.tick_params(axis="y", length=0)
    _ax.set_ylim(-0.6, len(SUBJECTS) - 0.4)
    _ax.set_xlim(0, 50)
    _ax.set_xticks([0, 10, 20, 30, 40, 50])
    _ax.set_xlabel("Number of latent states", fontsize=10)
    _ax.tick_params(axis="x", labelsize=8)
    _ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#E2E2E2", zorder=0)
    _ax.set_axisbelow(True)
    for _s in ("top", "right", "left"):
        _ax.spines[_s].set_visible(False)

    _LABEL_DARKEN = {"Unused + rare": "#888888"}
    _handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TAXONOMY_COLORS[_c],
                      edgecolor=_LABEL_DARKEN.get(_c, TAXONOMY_COLORS[_c]),
                      linewidth=0.5, label=_c)
        for _c in TAXONOMY_ORDER
    ]
    _ax.legend(handles=_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=5, frameon=False, fontsize=8, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.2)

    _fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88)
    _stem = OUT_F2 / "fig2_A_category_bars"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"saved: {_stem}.pdf")
    plt.close(_fig)
    return


@app.cell
def panel_B_category_network_sankey(
    state_flags, state_means, SUBJECTS, PARCELLATION, OUT_F2,
    TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP,
    NETWORK_ORDER, NETWORK_COLORS, display_network, plt, np,
):
    """Panel B - Sankey: 5 taxonomy categories -> 13 networks (top-3 composition).

    Lifted verbatim from the former F1 Panel B. Per-state composition: for each
    network, mean(|z-activation|) over its parcels; keep top-3 networks per
    state, renormalize to 1 unit of mass; sum across states in each category.
    """
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from utils.plot_style import load_parcel_networks

    _K_TOP = 3
    _parcel_nets = np.array(load_parcel_networks(PARCELLATION))

    _flow = np.zeros((len(TAXONOMY_ORDER), len(NETWORK_ORDER)))
    _cat_counts = np.zeros(len(TAXONOMY_ORDER), dtype=int)
    _CORTICAL = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}

    for _sub in SUBJECTS:
        _df = state_flags[_sub].copy()
        _df["_cat"] = _df["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare")
        _flag_id = "state" if "state" in _df.columns else "state_id"
        _means = state_means[_sub]
        for _state_id, _cat in zip(_df[_flag_id].values, _df["_cat"].values):
            if _cat not in TAXONOMY_ORDER:
                continue
            _cat_idx = TAXONOMY_ORDER.index(_cat)
            _cat_counts[_cat_idx] += 1
            _abs_pat = np.abs(_means[int(_state_id)])
            _net_scores = np.zeros(len(NETWORK_ORDER))
            for _nj, _net in enumerate(NETWORK_ORDER):
                _mask = _parcel_nets == _net
                if _mask.any():
                    _net_scores[_nj] = _abs_pat[_mask].mean()
            if _net_scores.sum() <= 0:
                continue
            _full_comp = _net_scores / _net_scores.sum()
            _top_idx = np.argpartition(_full_comp, -_K_TOP)[-_K_TOP:]
            _topk = np.zeros_like(_full_comp)
            _topk[_top_idx] = _full_comp[_top_idx]
            if _topk.sum() > 0:
                _topk /= _topk.sum()
            _flow[_cat_idx] += _topk

    _total_mass = _flow.sum()

    _cat_totals = _flow.sum(axis=1)
    _net_totals = _flow.sum(axis=0)
    _min_node_h = 0.04 * max(_cat_totals.sum(), _net_totals.sum())
    _node_visual_h = np.maximum(_net_totals, _min_node_h)
    _pad_frac = 0.025
    _cat_pad = _pad_frac * _cat_totals.sum()
    _net_pad = _pad_frac * _net_totals.sum()

    _cat_top = np.zeros(len(TAXONOMY_ORDER))
    _y = 0.0
    for _i in range(len(TAXONOMY_ORDER)):
        _cat_top[_i] = _y
        _y += _cat_totals[_i] + _cat_pad
    _cat_height = _y - _cat_pad

    _net_top = np.zeros(len(NETWORK_ORDER))
    _net_inner_offset = np.zeros(len(NETWORK_ORDER))
    _y = 0.0
    for _j in range(len(NETWORK_ORDER)):
        _net_top[_j] = _y
        _net_inner_offset[_j] = (_node_visual_h[_j] - _net_totals[_j]) / 2
        _y += _node_visual_h[_j] + _net_pad
    _net_height = _y - _net_pad

    _h_max = max(_cat_height, _net_height)
    _cat_y_off = (_h_max - _cat_height) / 2
    _net_y_off = (_h_max - _net_height) / 2

    _fig, _ax = plt.subplots(figsize=(7.0, 5.0))
    _node_w = 0.018
    _x0, _x1 = _node_w, 1 - _node_w
    _xc = (_x0 + _x1) / 2
    _src_used = np.zeros(len(TAXONOMY_ORDER))
    _dst_used = np.zeros(len(NETWORK_ORDER))
    _ribbon_eps = 0.05

    for _i in range(len(TAXONOMY_ORDER)):
        for _j in range(len(NETWORK_ORDER)):
            _w = _flow[_i, _j]
            if _w < _ribbon_eps:
                continue
            _y_src_top = _cat_top[_i] + _cat_y_off + _src_used[_i]
            _y_src_bot = _y_src_top + _w
            _y_dst_top = _net_top[_j] + _net_y_off + _net_inner_offset[_j] + _dst_used[_j]
            _y_dst_bot = _y_dst_top + _w
            _src_used[_i] += _w
            _dst_used[_j] += _w
            _verts = [
                (_x0, _y_src_top),
                (_xc, _y_src_top), (_xc, _y_dst_top), (_x1, _y_dst_top),
                (_x1, _y_dst_bot),
                (_xc, _y_dst_bot), (_xc, _y_src_bot), (_x0, _y_src_bot),
                (_x0, _y_src_top),
            ]
            _codes = [
                MplPath.MOVETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            _ax.add_patch(PathPatch(
                MplPath(_verts, _codes),
                facecolor=TAXONOMY_COLORS[TAXONOMY_ORDER[_i]],
                alpha=0.40, edgecolor="none", zorder=2,
            ))

    _LABEL_DARKEN = {
        "Unused + rare": "#888888",
        "Limbic":        "#8F7700",
        "Cerebellum":    "#7A6A55",
    }

    for _i in range(len(TAXONOMY_ORDER)):
        _ax.add_patch(plt.Rectangle(
            (0, _cat_top[_i] + _cat_y_off), _node_w, _cat_totals[_i],
            color=TAXONOMY_COLORS[TAXONOMY_ORDER[_i]], zorder=4, lw=0,
        ))
        _ax.text(
            -0.015, _cat_top[_i] + _cat_y_off + _cat_totals[_i] / 2,
            f"{TAXONOMY_ORDER[_i]}\n(n={int(_cat_counts[_i])} states)",
            ha="right", va="center", fontsize=8,
            color=_LABEL_DARKEN.get(TAXONOMY_ORDER[_i], TAXONOMY_COLORS[TAXONOMY_ORDER[_i]]),
        )

    for _j in range(len(NETWORK_ORDER)):
        _ax.add_patch(plt.Rectangle(
            (1 - _node_w, _net_top[_j] + _net_y_off), _node_w, _node_visual_h[_j],
            color=NETWORK_COLORS[NETWORK_ORDER[_j]], zorder=4, lw=0,
            alpha=1.0 if _net_totals[_j] > 0 else 0.45,
        ))
        _pct = 100 * _net_totals[_j] / _total_mass
        _ax.text(
            1.015, _net_top[_j] + _net_y_off + _node_visual_h[_j] / 2,
            f"{display_network(NETWORK_ORDER[_j])} ({_pct:.1f}%)",
            ha="left", va="center", fontsize=8,
            color=_LABEL_DARKEN.get(NETWORK_ORDER[_j], NETWORK_COLORS[NETWORK_ORDER[_j]]),
        )

    _ax.set_xlim(-0.25, 1.25)
    _ax.set_ylim(_h_max + _h_max * 0.02, -_h_max * 0.02)
    _ax.axis("off")
    _fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    _stem = OUT_F2 / "fig2_B_category_network_sankey"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"saved: {_stem}.pdf")
    plt.close(_fig)
    return


@app.cell
def panel_C_surface_contrast(state_flags, state_means, PARCELLATION, OUT_F2, plt, np):
    """Panel C - surface exemplars per category (sub-01), cortical + subcortical.

    Lifted verbatim from the former F1 Panel C. One exemplar per category
    (eligible rank-2, run-onset rank-1, low-conf rank-1, drift rank-1), rendered
    cortical + subcortical with shared per-region colorbars.
    """
    from utils.viz_yabplot import (
        setup_yabplot_headless,
        load_parcel_labels,
        pattern_to_cortical_dict,
        pattern_to_subcortical_dict,
        get_subcortical_atlas_dir,
        render_cortical_to_image,
        render_subcortical_to_image,
    )
    setup_yabplot_headless()

    _sub = "sub-01"
    _df = state_flags[_sub]
    _flag_id = "state" if "state" in _df.columns else "state_id"

    _ranked = {
        "eligible": _df[_df["summary_category"] == "eligible_for_content_analysis"]
                       .sort_values("recurrence_score", ascending=False),
        "runonset": _df[_df["summary_category"] == "run_onset_anchored"]
                       .sort_values("recurrence_score", ascending=False),
        "lowconf":  _df[_df["summary_category"] == "low_confidence"]
                       .sort_values("recurrence_score", ascending=False),
        "drift":    _df[_df["summary_category"] == "season_temporal"]
                       .sort_values("recurrence_score", ascending=False),
    }
    _rank_idx = {"eligible": 1, "runonset": 0, "lowconf": 0, "drift": 0}
    _picks = {}
    for _name, _ranked_df in _ranked.items():
        if len(_ranked_df) == 0:
            print(f"  WARNING: {_sub} has no {_name} states - skipping")
            continue
        _row = _ranked_df.iloc[min(_rank_idx[_name], len(_ranked_df) - 1)]
        _picks[_name] = int(_row[_flag_id])
        print(f"  {_sub} {_name:9} : state {int(_row[_flag_id]):3} "
              f"(r={float(_row['recurrence_score']):.3f}, dom={_row['dominant_network']})")

    _patterns = {_name: state_means[_sub][_sid] for _name, _sid in _picks.items()}

    _labels_df = load_parcel_labels(PARCELLATION)
    _cort_mask = (_labels_df["atlas_name"] == "4S156").values
    _subc_mask = ~_cort_mask

    _p98 = {}
    for _name, _pattern in _patterns.items():
        _p98[_name] = {
            "cort": float(np.nanpercentile(np.abs(_pattern[_cort_mask]), 98)),
            "subc": float(np.nanpercentile(np.abs(_pattern[_subc_mask]), 98)),
        }
    _vc_max = max(_p98[_n]["cort"] for _n in _p98)
    _vs_max = max(_p98[_n]["subc"] for _n in _p98)
    _shared_cort = (-_vc_max, _vc_max)
    _shared_subc = (-_vs_max, _vs_max)
    print(f"  shared cortical ±{_vc_max:.3f}, subcortical ±{_vs_max:.4f}")

    for _name, _pattern in _patterns.items():
        _cort_dict = pattern_to_cortical_dict(_pattern, _labels_df, PARCELLATION)
        _cort_img = render_cortical_to_image(_cort_dict, _shared_cort)
        _subc_dict = pattern_to_subcortical_dict(_pattern, _labels_df, PARCELLATION)
        _subc_img = render_subcortical_to_image(
            _subc_dict, _shared_subc, atlas_dir=get_subcortical_atlas_dir(),
        )
        for _region, _img, _width in (("cortical", _cort_img, 4.2),
                                      ("subcortical", _subc_img, 3.2)):
            _h = _img.shape[0] * _width / _img.shape[1]
            _fig = plt.figure(figsize=(_width, _h))
            _ax = _fig.add_axes([0, 0, 1, 1])
            _ax.imshow(_img)
            _ax.axis("off")
            _out = OUT_F2 / f"fig2_C_{_name}_{_region}.png"
            _fig.savefig(_out, bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close(_fig)
            print(f"  saved: {_out.name}")

    for _region, _range in (("cortical", _shared_cort), ("subcortical", _shared_subc)):
        _cfig, _cax = plt.subplots(figsize=(3.0, 0.5))
        _cfig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.90)
        _sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                    norm=plt.Normalize(vmin=_range[0], vmax=_range[1]))
        _sm.set_array([])
        _cb = _cfig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.set_label(f"Mean activation (z), {_region}", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        _stem = OUT_F2 / f"fig2_C_colorbar_{_region}"
        _cfig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        _cfig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_cfig)
        print(f"  saved: fig2_C_colorbar_{_region}.{{pdf,png}}")
    return


@app.cell
def panel_C_state_timeseries(
    state_flags, decoded_states_sub01, OUT_F2, TAXONOMY_COLORS, plt, np,
):
    """Panel C inset - binary-active timeseries strips per category exemplar.

    Lifted verbatim from the former F1. Same 4 picks (sub-01); each strip is a
    600-bin FO trace over all runs, colored by that category's taxonomy color.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _sub = "sub-01"
    _df = state_flags[_sub]
    _flag_id = "state" if "state" in _df.columns else "state_id"
    _ranked = {
        "eligible": _df[_df["summary_category"] == "eligible_for_content_analysis"]
                       .sort_values("recurrence_score", ascending=False),
        "runonset": _df[_df["summary_category"] == "run_onset_anchored"]
                       .sort_values("recurrence_score", ascending=False),
        "lowconf":  _df[_df["summary_category"] == "low_confidence"]
                       .sort_values("recurrence_score", ascending=False),
        "drift":    _df[_df["summary_category"] == "season_temporal"]
                       .sort_values("recurrence_score", ascending=False),
    }
    _rank_idx = {"eligible": 1, "runonset": 0, "lowconf": 0, "drift": 0}
    _category_color = {
        "eligible": TAXONOMY_COLORS["Content-eligible"],
        "runonset": TAXONOMY_COLORS["Run-onset-anchored"],
        "lowconf":  TAXONOMY_COLORS["Low-confidence"],
        "drift":    TAXONOMY_COLORS["Drift-anchored"],
    }

    _concat = np.concatenate(
        [decoded_states_sub01[_k] for _k in decoded_states_sub01.keys()]
    )
    _n_total = len(_concat)
    _n_bins = 600
    _edges = np.linspace(0, _n_total, _n_bins + 1, dtype=int)

    for _name, _ranked_df in _ranked.items():
        if len(_ranked_df) == 0:
            print(f"  WARNING: no {_name} states for {_sub} - skipping")
            continue
        _row = _ranked_df.iloc[min(_rank_idx[_name], len(_ranked_df) - 1)]
        _sid = int(_row[_flag_id])
        _binary = (_concat == _sid).astype(float)
        _binned = np.array([
            _binary[_edges[_i]:_edges[_i + 1]].mean() if _edges[_i + 1] > _edges[_i] else 0.0
            for _i in range(_n_bins)
        ])
        _vmax = max(float(_binned.max()), 0.01)
        _cmap = LinearSegmentedColormap.from_list(f"c_{_name}", ["white", _category_color[_name]])
        _fig, _ax = plt.subplots(figsize=(1.5, 0.18))
        _ax.imshow(_binned[None, :], aspect="auto", cmap=_cmap, vmin=0, vmax=_vmax,
                   interpolation="nearest")
        _ax.set_xticks([]); _ax.set_yticks([])
        for _s in ("top", "right", "bottom", "left"):
            _ax.spines[_s].set_visible(False)
        _fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        _stem = OUT_F2 / f"fig2_C_{_name}_timeseries"
        _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.0)
        _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
        plt.close(_fig)
        print(f"  saved: fig2_C_{_name}_timeseries.{{pdf,png}}")
    return


if __name__ == "__main__":
    app.run()
