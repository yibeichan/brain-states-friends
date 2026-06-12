"""Manuscript figure panels.

Authored as a marimo notebook. Each panel is a separate cell that saves
its own .pdf + .png to disk. The user assembles panels into composite
figures in a layout tool (no panel labels, no titles - user adds those
during assembly).

Run:
    marimo edit script/manuscript_figures.py
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
        RECURRENCE_CMAP,
        SUBJECT_ACCENT,
        SUBJECT_MARKERS,
        SUBJECT_NEUTRAL,
        apply_publication_style,
        load_parcel_networks,
    )
    from utils.recurrence_plots import compute_mean_fo_when_active

    apply_publication_style()

    return (
        Path,
        json,
        pickle,
        pd,
        np,
        plt,
        NETWORK_COLORS,
        NETWORK_ORDER,
        RECURRENCE_CMAP,
        SUBJECT_ACCENT,
        SUBJECT_MARKERS,
        SUBJECT_NEUTRAL,
        load_parcel_networks,
        compute_mean_fo_when_active,
        os,
    )


@app.cell
def config(Path, os):
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]

    RECURRENCE_DIR = SCRATCH_DIR / "output" / "05a_recurrence_analysis" / PARCELLATION
    DWELL_DIR = SCRATCH_DIR / "output" / "06a_state_temp_dynamics" / PARCELLATION
    FC_DIR = SCRATCH_DIR / "output" / "05f_state_fc" / PARCELLATION
    FLAGS_DIR = SCRATCH_DIR / "output" / "05e_temporal_trend_a4" / PARCELLATION
    MODEL_DIR = SCRATCH_DIR / "output" / "04_combined_hdphmm" / PARCELLATION

    TRANSITION_DIR = SCRATCH_DIR / "output" / "06b_transition_structure" / PARCELLATION

    OUT_FIG1 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig1"
    OUT_FIG2 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig2"
    OUT_FIG2_SUPP = OUT_FIG2 / "supplementary"
    OUT_FIG3 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig3"
    OUT_FIG3_SUPP = OUT_FIG3 / "supplementary"
    OUT_FIG1.mkdir(parents=True, exist_ok=True)
    OUT_FIG2.mkdir(parents=True, exist_ok=True)
    OUT_FIG2_SUPP.mkdir(parents=True, exist_ok=True)
    OUT_FIG3.mkdir(parents=True, exist_ok=True)
    OUT_FIG3_SUPP.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR,
        PARCELLATION,
        VT,
        SUBJECTS,
        RECURRENCE_DIR,
        DWELL_DIR,
        FC_DIR,
        FLAGS_DIR,
        MODEL_DIR,
        TRANSITION_DIR,
        OUT_FIG1,
        OUT_FIG2,
        OUT_FIG2_SUPP,
        OUT_FIG3,
        OUT_FIG3_SUPP,
    )


@app.cell
def figure1_panel_plan():
    """
    # Figure 1 - The recurring brain state repertoire (per subject)

    Serves manuscript R1 and talk slide 5.1.

    | Panel | Content | Relative size | Output filename |
    |---|---|---|---|
    | A | Dwell × FO × recurrence - 2×3 small multiples, one dot per state; x=mean dwell (log), y=mean FO when active, color=recurrence | wide | `fig1_A_recurrence_dwell.pdf` |
    | B | Exemplar state mean - cortical surface + subcortical (yabplot); independent cortical/subcortical color ranges | wide | `fig1_B_{cortical,subcortical}.png` + `fig1_B_colorbar_{cortical,subcortical}.{pdf,png}` |
    | C | Exemplar state within-state FC - network-sorted matrix | square | `fig1_C_exemplar_fc.pdf` |
    | D | Per-state network loading - per-subject heatmap, states (recurrence-sorted) × 13 networks | wide | `fig1_D_network_loading.pdf` |

    LOSO moves to supplementary (S3 or S8), freeing a main-figure slot.

    Rules: no titles, no on-figure panel labels, one file per panel.
    Panel letters appear in filenames only - user adds on-figure labels during assembly.
    """
    return


@app.cell
def load_mean_fo(
    RECURRENCE_DIR, SUBJECTS, VT, np, pickle, compute_mean_fo_when_active,
):
    """Load per-subject mean fractional occupancy when active.

    For each state, averages its FO across episodes where it was
    active (FO > 0). Source: 05a `fractional_occupancy.pkl`.
    """
    mean_fo_active = {}
    for _sub in SUBJECTS:
        _fo_path = RECURRENCE_DIR / _sub / VT / "fractional_occupancy.pkl"
        with open(_fo_path, "rb") as _f:
            _fo = pickle.load(_f)
        _n_states = int(np.stack(list(_fo.values())).shape[1])
        mean_fo_active[_sub] = compute_mean_fo_when_active(
            _fo, _n_states, fo_threshold=0.0,
        )
    return (mean_fo_active,)


@app.cell
def panel_recurrence_dwell_fo(
    state_summary, mean_fo_active, SUBJECTS, OUT_FIG1, RECURRENCE_CMAP,
    plt, np,
):
    """Panel A - 2×3 small multiples, one dot per active state.

    For each subject: x = mean dwell (s, log), y = mean FO when active,
    color = recurrence score. Combines three orthogonal properties
    per state, revealing that recurrence is independent of both
    persistence (dwell) and per-episode occupancy (FO-when-active).
    Port of talk slide 5.1a to manuscript sizing/styling.
    """
    _fig, _axes = plt.subplots(
        2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True,
    )
    _norm = plt.matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    _all_d = np.concatenate([
        state_summary[_s].loc[state_summary[_s]["recurrence_score"] > 0,
                              "mean_dwell_s"].values
        for _s in SUBJECTS
    ])
    _x_lo = float(_all_d.min()) * 0.85
    _x_hi = float(_all_d.max()) * 1.15

    _all_fo = []
    for _s in SUBJECTS:
        _df = state_summary[_s]
        _active = _df["recurrence_score"].values > 0
        _sids = _df.loc[_active, "state_id"].values.astype(int)
        _all_fo.append(mean_fo_active[_s][_sids])
    _all_fo = np.concatenate(_all_fo)
    _all_fo = _all_fo[~np.isnan(_all_fo)]
    _y_hi = float(np.nanmax(_all_fo)) * 1.08
    _y_lo = 0.0

    _sc = None
    for _ax, _sub in zip(_axes.flat, SUBJECTS):
        _df = state_summary[_sub].set_index("state_id")
        _active = _df["recurrence_score"] > 0
        _sids = _df.index[_active].values.astype(int)
        _d = _df.loc[_sids, "mean_dwell_s"].values
        _r = _df.loc[_sids, "recurrence_score"].values
        _f = mean_fo_active[_sub][_sids]
        _valid = ~np.isnan(_f) & ~np.isnan(_d)
        _sc = _ax.scatter(
            _d[_valid], _f[_valid], c=_r[_valid],
            cmap=RECURRENCE_CMAP, norm=_norm,
            s=22, edgecolor="white", linewidth=0.3, alpha=0.95,
        )
        _ax.text(
            0.04, 0.96, f"{_sub}  K={int(_active.sum())}",
            transform=_ax.transAxes, ha="left", va="top",
            fontsize=8, color="#333333",
        )
        _ax.grid(True, alpha=0.15, linewidth=0.4)
        for _spine in ("top", "right"):
            _ax.spines[_spine].set_visible(False)
        _ax.tick_params(axis="both", labelsize=7)

    for _ax in _axes.flat:
        _ax.set_xscale("log")
        _ax.set_xlim(_x_lo, _x_hi)
        _ax.set_ylim(_y_lo, _y_hi)
        _ax.set_xticks([2, 3, 5, 10, 20])
        _ax.set_xticklabels(["2", "3", "5", "10", "20"])
        _ax.xaxis.set_minor_locator(plt.matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        _ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())

    _fig.supxlabel("Mean dwell (s, log)", fontsize=10)
    _fig.supylabel("Mean FO when active", fontsize=10)
    _fig.subplots_adjust(
        left=0.09, right=0.90, bottom=0.11, top=0.97,
        wspace=0.08, hspace=0.14,
    )

    _cax = _fig.add_axes([0.92, 0.18, 0.018, 0.65])
    _cbar = _fig.colorbar(_sc, cax=_cax)
    _cbar.set_label("Recurrence score", fontsize=9)
    _cbar.ax.tick_params(labelsize=7)

    _out_pdf = OUT_FIG1 / "fig1_A_recurrence_dwell.pdf"
    _out_png = OUT_FIG1 / "fig1_A_recurrence_dwell.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    return


@app.cell
def load_state_summary(SUBJECTS, DWELL_DIR, VT, pd):
    """Load 06a state_summary_table per subject (recurrence, dwell, network)."""
    state_summary = {}
    for _sub in SUBJECTS:
        _tbl = DWELL_DIR / _sub / VT / "state_summary_table.csv"
        state_summary[_sub] = pd.read_csv(_tbl)
    print({_s: len(_d) for _s, _d in state_summary.items()})
    return (state_summary,)


@app.cell
def panel_exemplar_fc(
    FC_DIR, PARCELLATION, state_summary, OUT_FIG1,
    NETWORK_ORDER, NETWORK_COLORS, load_parcel_networks,
    np, plt,
):
    """Panel 3 - within-state FC of the highest-recurrence state (sub-01).

    Network-sorted 156x156 correlation matrix with network-colored
    side/top bars. Square aspect.
    """
    _exemplar_sub = "sub-01"
    _df = state_summary[_exemplar_sub]
    _exemplar_state = int(_df.loc[_df["recurrence_score"].idxmax(), "state_id"])
    _exemplar_r = float(_df["recurrence_score"].max())

    _corr = np.load(FC_DIR / _exemplar_sub / "vt0.95" / "state_empirical_corr.npy")
    _mat = _corr[_exemplar_state]

    _parcel_nets = load_parcel_networks(PARCELLATION)
    _net_rank = {_n: _i for _i, _n in enumerate(NETWORK_ORDER)}
    _order = sorted(range(len(_parcel_nets)),
                    key=lambda _i: (_net_rank.get(_parcel_nets[_i], 99), _i))
    _mat_sorted = _mat[np.ix_(_order, _order)]
    _nets_sorted = [_parcel_nets[_i] for _i in _order]

    _fig = plt.figure(figsize=(3.4, 3.4))
    _gs = _fig.add_gridspec(
        2, 2,
        width_ratios=[0.04, 1.0],
        height_ratios=[0.04, 1.0],
        wspace=0.02, hspace=0.02,
    )
    _ax_top = _fig.add_subplot(_gs[0, 1])
    _ax_left = _fig.add_subplot(_gs[1, 0])
    _ax_main = _fig.add_subplot(_gs[1, 1])

    _offdiag = _mat_sorted[~np.eye(len(_mat_sorted), dtype=bool)]
    _vmax = float(np.nanpercentile(np.abs(_offdiag), 98))
    _im = _ax_main.imshow(
        _mat_sorted, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax,
        aspect="equal", interpolation="nearest",
    )
    _ax_main.set_xticks([]); _ax_main.set_yticks([])
    for _s in _ax_main.spines.values():
        _s.set_linewidth(0.5)

    _net_rgb = np.array([[list(plt.matplotlib.colors.to_rgb(
        NETWORK_COLORS.get(_n, "#999999"))) for _n in _nets_sorted]])
    _ax_top.imshow(_net_rgb, aspect="auto")
    _ax_top.set_xticks([]); _ax_top.set_yticks([])
    for _s in _ax_top.spines.values():
        _s.set_visible(False)
    _ax_left.imshow(np.transpose(_net_rgb, (1, 0, 2)), aspect="auto")
    _ax_left.set_xticks([]); _ax_left.set_yticks([])
    for _s in _ax_left.spines.values():
        _s.set_visible(False)

    _cax = _fig.add_axes([0.92, 0.15, 0.025, 0.55])
    _cbar = _fig.colorbar(_im, cax=_cax)
    _cbar.set_label("Within-state FC (r)", fontsize=9)
    _cbar.ax.tick_params(labelsize=8)

    print(f"exemplar: {_exemplar_sub} state={_exemplar_state} recurrence={_exemplar_r:.2f}")
    _out_pdf = OUT_FIG1 / "fig1_C_exemplar_fc.pdf"
    _out_png = OUT_FIG1 / "fig1_C_exemplar_fc.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    return


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
def panel_exemplar_surface(
    state_summary, state_means, PARCELLATION, OUT_FIG1, plt, np,
):
    """Panel B - exemplar state mean as cortical surface + subcortical 3D.

    Same exemplar state as Panel C (sub-01, highest recurrence).

    Cortical and subcortical use independent symmetric 98th-percentile
    color ranges. Subcortical z-score magnitudes are typically ~10-30x
    smaller than cortical; a shared range would render subcortical
    panels nearly white. Two colorbar files are emitted so the
    assembled figure shows the magnitude mismatch explicitly. Pattern
    matches fig2 Panel C.

    Outputs:
      - fig1_B_cortical.png
      - fig1_B_subcortical.png
      - fig1_B_colorbar_cortical.{pdf,png}
      - fig1_B_colorbar_subcortical.{pdf,png}
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

    _exemplar_sub = "sub-01"
    _df = state_summary[_exemplar_sub]
    _exemplar_state = int(_df.loc[_df["recurrence_score"].idxmax(), "state_id"])
    _pattern = state_means[_exemplar_sub][_exemplar_state]

    _labels_df = load_parcel_labels(PARCELLATION)
    _cort_mask = (_labels_df["atlas_name"] == "4S156").values
    _subc_mask = ~_cort_mask
    _vmax_cort = float(np.nanpercentile(np.abs(_pattern[_cort_mask]), 98))
    _vmax_subc = float(np.nanpercentile(np.abs(_pattern[_subc_mask]), 98))
    _range_cort = (-_vmax_cort, _vmax_cort)
    _range_subc = (-_vmax_subc, _vmax_subc)

    _cort_dict = pattern_to_cortical_dict(_pattern, _labels_df, PARCELLATION)
    _cort_img = render_cortical_to_image(_cort_dict, _range_cort)
    _subcort_dict = pattern_to_subcortical_dict(_pattern, _labels_df, PARCELLATION)
    _subcort_img = render_subcortical_to_image(
        _subcort_dict, _range_subc, atlas_dir=get_subcortical_atlas_dir(),
    )

    for _name, _img, _width in (
        ("cortical", _cort_img, 4.2),
        ("subcortical", _subcort_img, 3.2),
    ):
        _h = _img.shape[0] * _width / _img.shape[1]
        _fig = plt.figure(figsize=(_width, _h))
        _ax = _fig.add_axes([0, 0, 1, 1])
        _ax.imshow(_img)
        _ax.axis("off")
        _out = OUT_FIG1 / f"fig1_B_{_name}.png"
        _fig.savefig(_out, bbox_inches="tight", pad_inches=0.0, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out.name}")

    for _suffix, _range in (("cortical", _range_cort), ("subcortical", _range_subc)):
        _cfig, _cax = plt.subplots(figsize=(3.0, 0.5))
        _cfig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.90)
        _sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=_range[0], vmax=_range[1]),
        )
        _sm.set_array([])
        _cb = _cfig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.set_label(f"Mean activation (z), {_suffix}", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        _cfig.savefig(OUT_FIG1 / f"fig1_B_colorbar_{_suffix}.pdf",
                      bbox_inches="tight", pad_inches=0.02)
        _cfig.savefig(OUT_FIG1 / f"fig1_B_colorbar_{_suffix}.png",
                      bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_cfig)
    print(f"exemplar: {_exemplar_sub} state={_exemplar_state} "
          f"cortical range: ±{_vmax_cort:.2f}, "
          f"subcortical range: ±{_vmax_subc:.3f}")
    return


@app.cell
def panel_network_loading(
    state_summary, state_means, SUBJECTS, PARCELLATION, OUT_FIG1,
    NETWORK_ORDER, NETWORK_COLORS, load_parcel_networks, plt, np,
):
    """Panel D - per-state network loading, one heatmap per subject.

    Each cell = row-normalized mean |activation| per network per state -
    i.e. the proportion of a state's total |activation| (taken as mean
    across that network's parcels) that falls in each network. This is
    a loading, not a raw composition: sign is dropped, magnitude is
    normalised away, and parcel count is size-normalised via mean.

    Per subject, two stacked strips: cortical (7 networks on top) and
    subcortical (5 networks on bottom). Each strip uses its own 98th-pct
    color range and its own palette (cortical: cividis; subcortical:
    magma) because subcortical proportions cap at ~0.05 while cortical
    reach ~0.33 (~6× mismatch) - shared scale would wash subcortical
    structure out. Different palettes signal the independent scales
    without requiring the reader to remember the caption.
    Rows = active states, sorted by descending recurrence.
    Network labels are horizontal (rotation=0) throughout.
    """
    CORTICAL_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    SUBCORTICAL_NETWORKS = ["BG", "Midbrain-DA", "Midbrain-Diencephalic",
                            "Thalamus", "Hipp/Amyg", "Cerebellum"]
    CORT_DISPLAY = ["Vis", "SMN", "DAN", "VAN", "Lim", "FPN", "DMN"]
    SUBC_DISPLAY = ["BG", "MidDA", "MidDi", "Thal", "Hipp", "Cereb"]
    # Darker swatch for Limbic tick label so it's legible on white (palette
    # Limbic #DCF8A4 is too pale as text).
    NETWORK_LABEL_COLORS = {**NETWORK_COLORS, "Limbic": "#6FA81F"}
    _parcel_nets = np.array(load_parcel_networks(PARCELLATION))

    def _profile_for(_means, _order=NETWORK_ORDER):
        _prof = np.zeros((_means.shape[0], len(_order)))
        for _i, _net in enumerate(_order):
            _mask = _parcel_nets == _net
            if _mask.any():
                _prof[:, _i] = np.abs(_means[:, _mask]).mean(axis=1)
        _rowsum = _prof.sum(axis=1, keepdims=True)
        _rowsum[_rowsum == 0] = 1.0
        return _prof / _rowsum

    # Compute full profiles first (for shared row-normalization), then slice
    # into cortical / subcortical views for plotting.
    _cort_idx = [NETWORK_ORDER.index(_n) for _n in CORTICAL_NETWORKS]
    _sub_idx = [NETWORK_ORDER.index(_n) for _n in SUBCORTICAL_NETWORKS]

    # Separate 98th-pct color ranges for cortical vs subcortical. Cortical
    # proportions reach ~0.3 while subcortical caps around ~0.05 (~6× mismatch)
    # because row normalization is computed over all 13 networks - so a shared
    # range washes subcortical strips to a single shade. Per-subregion ranges
    # restore subcortical visibility; the magnitude mismatch is flagged in
    # the caption and via separate colorbar files (below).
    _prepped = {}
    _cort_vals, _sub_vals = [], []
    for _sub in SUBJECTS:
        _df = state_summary[_sub]
        _mask = (_df["recurrence_score"] > 0).values
        _ids = _df.loc[_mask, "state_id"].values.astype(int)
        _recur = _df.loc[_mask, "recurrence_score"].values
        _order = np.argsort(-_recur)
        _full = _profile_for(state_means[_sub])[_ids[_order]]
        _prepped[_sub] = (_full, _recur[_order])
        _cort_vals.append(_full[:, _cort_idx].ravel())
        _sub_vals.append(_full[:, _sub_idx].ravel())
    _vmax_cort = float(np.nanpercentile(np.concatenate(_cort_vals), 98))
    _vmax_sub = float(np.nanpercentile(np.concatenate(_sub_vals), 98))

    # Layout: 2x3 subjects. Each subject = vertical stack of 2 axes
    # (cortical on top, subcortical below). Both strips share the same
    # row count (K active states), so use equal height_ratios - unequal
    # heights would mislead into suggesting different numbers of states.
    _fig = plt.figure(figsize=(9.2, 6.0))
    _outer = _fig.add_gridspec(
        2, 3, left=0.08, right=0.97, bottom=0.10, top=0.94,
        wspace=0.30, hspace=0.35,
    )
    _im_cort = _im_sub = None
    for _k, _sub in enumerate(SUBJECTS):
        _row, _col = _k // 3, _k % 3
        _inner = _outer[_row, _col].subgridspec(
            2, 1, height_ratios=[1, 1], hspace=0.08,
        )
        _ax_cort = _fig.add_subplot(_inner[0, 0])
        _ax_sub = _fig.add_subplot(_inner[1, 0])
        _full, _recur_ord = _prepped[_sub]
        _cort = _full[:, _cort_idx]
        _subc = _full[:, _sub_idx]

        _im_cort = _ax_cort.imshow(
            _cort, aspect="auto", cmap="cividis",
            vmin=0.0, vmax=_vmax_cort, interpolation="nearest",
        )
        _im_sub = _ax_sub.imshow(
            _subc, aspect="auto", cmap="magma",
            vmin=0.0, vmax=_vmax_sub, interpolation="nearest",
        )

        # cortical x-ticks on top (so they don't collide with subcortical heatmap)
        _ax_cort.set_xticks(range(len(CORTICAL_NETWORKS)))
        _ax_cort.set_xticklabels(CORT_DISPLAY, rotation=0, ha="center", fontsize=6)
        _ax_cort.xaxis.tick_top()
        for _t, _net in zip(_ax_cort.get_xticklabels(), CORTICAL_NETWORKS):
            _t.set_color(NETWORK_LABEL_COLORS.get(_net, "#333"))
        _ax_cort.tick_params(axis="x", length=0, pad=2)

        # subcortical x-ticks at bottom
        _ax_sub.set_xticks(range(len(SUBCORTICAL_NETWORKS)))
        _ax_sub.set_xticklabels(SUBC_DISPLAY, rotation=0, ha="center", fontsize=6)
        for _t, _net in zip(_ax_sub.get_xticklabels(), SUBCORTICAL_NETWORKS):
            _t.set_color(NETWORK_LABEL_COLORS.get(_net, "#333"))
        _ax_sub.tick_params(axis="x", length=0, pad=2)

        # Subcortical magnitudes are low (near-zero on the shared cividis
        # range) - draw light separators between network columns so the
        # reader can see the 5 network bands.
        for _j in range(1, len(SUBCORTICAL_NETWORKS)):
            _ax_sub.axvline(_j - 0.5, color="white", linewidth=0.6, alpha=0.9)

        # Same y-tick recurrence labels on both strips - makes it visually
        # explicit that both strips share the same state ordering / row count.
        _n = len(_recur_ord)
        _yticks = [0, _n // 2, _n - 1] if _n >= 3 else list(range(_n))
        _ytick_labels = [f"{_recur_ord[_i]:.2f}" for _i in _yticks]
        for _ax in (_ax_cort, _ax_sub):
            _ax.set_yticks(_yticks)
            _ax.set_yticklabels(_ytick_labels, fontsize=7)
            _ax.tick_params(axis="y", length=2)

        _ax_cort.text(
            0.02, 1.16, _sub,
            transform=_ax_cort.transAxes, ha="left", va="bottom", fontsize=8,
        )

    _fig.supylabel("State recurrence score", fontsize=9)

    _out_pdf = OUT_FIG1 / "fig1_D_network_loading.pdf"
    _out_png = OUT_FIG1 / "fig1_D_network_loading.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)

    # Separate colorbar mini-files, one per subregion. The magnitude mismatch
    # (cortical vmax {_vmax_cort:.2f} vs subcortical vmax {_vmax_sub:.2f}) is
    # why shared-range rendering washes out subcortical structure.
    for _region, _vmax_r, _tag, _cmap in [
        ("cortical", _vmax_cort, "cortical", "cividis"),
        ("subcortical", _vmax_sub, "subcortical", "magma"),
    ]:
        _cfig, _cax = plt.subplots(figsize=(3.0, 0.5))
        _cfig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.90)
        _sm = plt.cm.ScalarMappable(
            cmap=_cmap,
            norm=plt.matplotlib.colors.Normalize(vmin=0.0, vmax=_vmax_r),
        )
        _sm.set_array([])
        _cb = _cfig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.set_label(
            f"Proportion of |activation| ({_region})", fontsize=8,
        )
        _cb.ax.tick_params(labelsize=7)
        _cfig.savefig(OUT_FIG1 / f"fig1_D_colorbar_{_tag}.pdf",
                      bbox_inches="tight", pad_inches=0.02)
        _cfig.savefig(OUT_FIG1 / f"fig1_D_colorbar_{_tag}.png",
                      bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_cfig)

    print(f"saved: {_out_pdf.name}, {_out_png.name}, "
          f"fig1_D_colorbar_cortical.pdf, fig1_D_colorbar_subcortical.pdf "
          f"(vmax_cort={_vmax_cort:.3f}, vmax_sub={_vmax_sub:.3f})")
    return


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 - Heterogeneous sources of recurrence (R2)
# ──────────────────────────────────────────────────────────────────────────────


@app.cell
def figure2_panel_plan():
    """
    # Figure 2 - Heterogeneous sources of recurrence

    Serves manuscript R2 and talk slide 5.2 (talk uses only Panel A).

    | Panel | Content | Relative size | Output filename(s) |
    |---|---|---|---|
    | A | Recurrence strip plot per subject, dots colored by taxonomy | wide | `fig2_A_taxonomy_recurrence_strip.pdf` |
    | A-supp | Reversed stacked bar per subject (Content-eligible on top, Unused+rare at bottom) | wide | `supplementary/fig2_A_supp_taxonomy_stacked_bar.pdf` |
    | B | Flag-overview heatmap (sub-01): states (recurrence-sorted) × 4 numeric criteria | square | `fig2_B_flag_overview.pdf` |
    | C | Two exemplar states (sub-01, run-onset-anchored + content-eligible); each: cortical surface + subcortical + diagnostic curve (occupancy × TR-within-run) | wide | `fig2_C_{run_onset,content_eligible}_{cortical,subcortical,occupancy}.{pdf,png}` + `fig2_C_colorbar.{pdf,png}` |
    | C-supp | One exemplar per display category (sub-01): content-eligible, run-onset-anchored, low-confidence, drift-anchored, rare (stand-in for "Unused + rare" - unused states have no decoded pattern). Same template as main Panel C (cortical + subcortical + occupancy curve). Independent color ranges from main Panel C - reader inside supp can compare across categories; cross-figure comparison requires reading both colorbars. | wide | `supplementary/fig2_C_supp_{category}_{cortical,subcortical,occupancy}.{pdf,png}` + `supplementary/fig2_C_supp_colorbar_{cortical,subcortical}.{pdf,png}` |
    | D | Low-confidence prevalence per subject (sub-06 accent) | wide | `fig2_D_low_confidence_per_subject.pdf` |

    Language: "Content-eligible," "Run-onset-anchored," "Low-confidence," "Drift-anchored," "Unused + rare".
    Never "noise / artifact / nuisance / design-driven / theme-song states".
    """
    return


@app.cell
def load_state_flags(SUBJECTS, FLAGS_DIR, VT, pd):
    """Load 05e_a4 state_flags.csv per subject."""
    state_flags = {}
    for _sub in SUBJECTS:
        _path = FLAGS_DIR / _sub / VT / "state_flags.csv"
        state_flags[_sub] = pd.read_csv(_path)
    print({_s: len(_d) for _s, _d in state_flags.items()})
    return (state_flags,)


@app.cell
def load_decoded_exemplar(MODEL_DIR, VT, pickle):
    """Load decoded Viterbi states for sub-01 (for Panel C diagnostic curves)."""
    with open(MODEL_DIR / "sub-01" / "final" / VT / "decoded_states.pkl", "rb") as _f:
        decoded_sub01 = pickle.load(_f)
    print(f"episodes: {len(decoded_sub01)}, example len: {len(next(iter(decoded_sub01.values())))}")
    return (decoded_sub01,)


@app.cell
def taxonomy_labels_and_colors():
    """Display labels + colors for the 5 taxonomy categories.

    Collapses raw summary_category values into 5 display categories:
      eligible_for_content_analysis -> "Content-eligible"
      run_onset_anchored            -> "Run-onset-anchored"
      low_confidence                -> "Low-confidence"
      season_temporal               -> "Drift-anchored"
      {unused, rare}                -> "Unused + rare"
    """
    TAXONOMY_ORDER = [
        "Content-eligible",
        "Run-onset-anchored",
        "Low-confidence",
        "Drift-anchored",
        "Unused + rare",
    ]
    TAXONOMY_COLORS = {
        "Content-eligible":    "#377EB8",
        "Run-onset-anchored":  "#FF7F00",
        "Low-confidence":      "#999999",
        "Drift-anchored":      "#984EA3",
        "Unused + rare":       "#CCCCCC",
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
def panel_A_taxonomy(
    state_flags, SUBJECTS, OUT_FIG2,
    TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP, plt, np,
):
    """Panel A - per-subject strip plot: recurrence × taxonomy.

    x = subject (jittered); y = recurrence score; one dot per state,
    colored by taxonomy category. Shows counts (dot density per color)
    AND where in the recurrence gradient each category sits, tying
    Figure 2 back to Figure 1's recurrence distribution.

    Draw order (back → front): Unused+rare, Drift-anchored, Low-confidence,
    Run-onset-anchored, Content-eligible - puts content-eligible dots on top.
    """
    _rng = np.random.default_rng(20260422)
    _jitter_w = 0.32
    _draw_order = [
        "Unused + rare",
        "Drift-anchored",
        "Low-confidence",
        "Run-onset-anchored",
        "Content-eligible",
    ]

    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))

    _counts = {_c: 0 for _c in TAXONOMY_ORDER}
    for _j, _sub in enumerate(SUBJECTS):
        _df = state_flags[_sub].copy()
        _df["_cat"] = _df["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare")
        _jit = _rng.uniform(-_jitter_w, _jitter_w, size=len(_df))
        for _cat in _draw_order:
            _mask = (_df["_cat"] == _cat).values
            if not _mask.any():
                continue
            _ax.scatter(
                _j + _jit[_mask],
                _df.loc[_mask, "recurrence_score"].values,
                s=22, color=TAXONOMY_COLORS[_cat],
                edgecolor="white", linewidth=0.3,
                alpha=0.9, zorder=3,
            )
            _counts[_cat] += int(_mask.sum())

    _ax.set_xticks(np.arange(len(SUBJECTS)))
    _ax.set_xticklabels(SUBJECTS, fontsize=8)
    _ax.set_xlim(-0.55, len(SUBJECTS) - 0.45)
    _ax.set_ylim(-0.03, 1.0)
    _ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _ax.set_ylabel("Recurrence score", fontsize=10)
    _ax.tick_params(axis="y", labelsize=8)
    _ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#BBBBBB", zorder=1)
    _ax.set_axisbelow(True)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)

    _handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=TAXONOMY_COLORS[_c], markeredgecolor="white",
            markeredgewidth=0.3, markersize=6, label=_c,
        )
        for _c in TAXONOMY_ORDER
    ]
    _ax.legend(
        handles=_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.5,
    )
    _fig.subplots_adjust(left=0.10, right=0.72, bottom=0.14, top=0.97)

    _out_pdf = OUT_FIG2 / "fig2_A_taxonomy_recurrence_strip.pdf"
    _out_png = OUT_FIG2 / "fig2_A_taxonomy_recurrence_strip.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    print(f"dots drawn per category: {_counts}")
    return


@app.cell
def panel_A_supp_taxonomy_bar(
    state_flags, SUBJECTS, OUT_FIG2_SUPP,
    TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP, plt, np,
):
    """Panel A (supplementary) - reversed stacked bar.

    x = subject; y = count of states; stacked by category with
    Content-eligible on TOP and Unused+rare on BOTTOM (reverse of the
    default TAXONOMY_ORDER). Keeps the same color mapping as main
    Panel A so the reader can cross-reference.
    """
    _counts = np.zeros((len(TAXONOMY_ORDER), len(SUBJECTS)), dtype=int)
    for _j, _sub in enumerate(SUBJECTS):
        _df = state_flags[_sub]
        _display = _df["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare")
        for _i, _cat in enumerate(TAXONOMY_ORDER):
            _counts[_i, _j] = int((_display == _cat).sum())

    # stack bottom → top = reversed TAXONOMY_ORDER (Unused+rare drawn first)
    _stack_order = list(reversed(TAXONOMY_ORDER))

    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _x = np.arange(len(SUBJECTS))
    _bottom = np.zeros(len(SUBJECTS))
    for _cat in _stack_order:
        _i = TAXONOMY_ORDER.index(_cat)
        _ax.bar(
            _x, _counts[_i], bottom=_bottom, width=0.72,
            color=TAXONOMY_COLORS[_cat], edgecolor="white", linewidth=0.6,
            label=_cat,
        )
        _bottom = _bottom + _counts[_i]

    _ax.set_xticks(_x)
    _ax.set_xticklabels(SUBJECTS, fontsize=8)
    _ax.set_ylabel("# states", fontsize=10)
    _ax.tick_params(axis="y", labelsize=8)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)

    # legend order follows visual stack (top → bottom = TAXONOMY_ORDER)
    _handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TAXONOMY_COLORS[_c],
                      edgecolor="white", linewidth=0.6, label=_c)
        for _c in TAXONOMY_ORDER
    ]
    _ax.legend(
        handles=_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.5,
    )
    _fig.subplots_adjust(left=0.10, right=0.72, bottom=0.14, top=0.97)

    _out_pdf = OUT_FIG2_SUPP / "fig2_A_supp_taxonomy_stacked_bar.pdf"
    _out_png = OUT_FIG2_SUPP / "fig2_A_supp_taxonomy_stacked_bar.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved (supp): {_out_pdf.name}, {_out_png.name}")
    print(f"counts per category (rows=category, cols=subjects):\n{_counts}")
    return


@app.cell
def panel_B_flag_overview(
    state_flags, OUT_FIG2,
    TAXONOMY_COLORS, TAXONOMY_MAP, plt, np,
):
    """Panel B - binary flag overview for sub-01 (mirrors 05e_a4 style).

    Rows = active states (recurrence > 0), sorted by descending recurrence.
    Cols = 3 diagnostic flag groups:
      - Run-onset    = any of {run_onset, a_anchored, b_anchored}
      - Sub-HRF      = sub_hrf
      - Session/drift = any of {session_trend_down, session_trend_up,
                                  season_structured, global_trend}
    Cells are colored (on) / white (off). Each column uses the taxonomy color
    of the category that flag drives, so the reader can trace "this flag
    fires → this category" directly.

    Y-axis recurrence labels are placed on the far-left category-strip axis
    so they sit well clear of the heatmap and colorbar.
    """
    _sub = "sub-01"
    _df = state_flags[_sub].copy()
    _df = _df[_df["recurrence_score"] > 0].reset_index(drop=True)
    _df = _df.sort_values("recurrence_score", ascending=False).reset_index(drop=True)

    _flag_groups = [
        ("Run-\nonset",   ["run_onset", "a_anchored", "b_anchored"],                         "Run-onset-anchored"),
        ("Sub-\nHRF",     ["sub_hrf"],                                                       "Low-confidence"),
        ("Session/\ndrift", ["session_trend_down", "session_trend_up",
                              "season_structured", "global_trend"],                          "Drift-anchored"),
    ]
    _col_labels = [_g[0] for _g in _flag_groups]
    _col_colors = [TAXONOMY_COLORS[_g[2]] for _g in _flag_groups]

    _binary = np.zeros((len(_df), len(_flag_groups)), dtype=bool)
    for _j, (_label, _cols, _cat) in enumerate(_flag_groups):
        _any = np.zeros(len(_df), dtype=bool)
        for _c in _cols:
            if _c in _df.columns:
                _any |= _df[_c].astype(bool).values
        _binary[:, _j] = _any

    _cats = _df["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare").values
    _cat_rgb = np.array([[list(plt.matplotlib.colors.to_rgb(TAXONOMY_COLORS[_c])) for _c in _cats]])
    _cat_rgb = np.transpose(_cat_rgb, (1, 0, 2))

    _fig = plt.figure(figsize=(2.4, 4.4))
    _gs = _fig.add_gridspec(
        1, 2, width_ratios=[0.10, 1.0], wspace=0.10,
    )
    _ax_cat = _fig.add_subplot(_gs[0, 0])
    _ax_main = _fig.add_subplot(_gs[0, 1])

    _ax_cat.imshow(_cat_rgb, aspect="auto", extent=[0, 1, len(_df) - 0.5, -0.5])
    _ax_cat.set_xticks([])
    _n = len(_df)
    _yticks = [0, _n // 2, _n - 1]
    _ax_cat.set_yticks(_yticks)
    _ax_cat.set_yticklabels(
        [f"{_df.iloc[_i]['recurrence_score']:.2f}" for _i in _yticks],
        fontsize=7, rotation=90, va="center", ha="right",
    )
    _ax_cat.tick_params(axis="y", length=2, pad=2)
    _ax_cat.set_ylim(_n - 0.5, -0.5)
    _ax_cat.set_ylabel("State recurrence score", fontsize=8)
    for _s in _ax_cat.spines.values():
        _s.set_visible(False)

    _ax_main.set_facecolor("#F2F2F2")
    _ax_main.set_xlim(-0.5, len(_flag_groups) - 0.5)
    _ax_main.set_ylim(_n - 0.5, -0.5)
    for _j, _color in enumerate(_col_colors):
        for _i in range(_n):
            if _binary[_i, _j]:
                _ax_main.add_patch(plt.matplotlib.patches.Rectangle(
                    (_j - 0.5, _i - 0.5), 1, 1,
                    facecolor=_color, edgecolor="none",
                ))
    # subtle grid lines
    for _i in range(_n + 1):
        _ax_main.axhline(_i - 0.5, color="white", linewidth=0.3)
    for _j in range(len(_flag_groups) + 1):
        _ax_main.axvline(_j - 0.5, color="white", linewidth=0.3)

    _ax_main.set_xticks(range(len(_col_labels)))
    _ax_main.set_xticklabels(_col_labels, rotation=0, ha="center", fontsize=7)
    _ax_main.set_yticks([])
    _ax_main.xaxis.tick_top()
    _ax_main.tick_params(axis="x", length=0, pad=4)
    for _s in _ax_main.spines.values():
        _s.set_visible(False)

    _fig.subplots_adjust(left=0.14, right=0.97, bottom=0.04, top=0.90)

    _out_pdf = OUT_FIG2 / "fig2_B_flag_overview.pdf"
    _out_png = OUT_FIG2 / "fig2_B_flag_overview.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    print(f"exemplar: {_sub}, n active states: {_n}, flag counts: {_counts.tolist()}")
    return


@app.cell
def pick_fig2_exemplars(state_flags, np):
    """Pick exemplar states from sub-01 for Panel C.

    Run-onset exemplar: run_onset_anchored state with highest recurrence.
    Content-eligible exemplar: SECOND-highest recurrence content-eligible
      state (the top one is the widespread-baseline pattern already used
      as Figure 1's exemplar - this panel picks the next one for visual
      variety without conceding recurrence-rank credibility).
    """
    _df = state_flags["sub-01"]
    _run_onset = _df[_df["summary_category"] == "run_onset_anchored"].sort_values(
        "recurrence_score", ascending=False,
    )
    _content = _df[_df["summary_category"] == "eligible_for_content_analysis"].sort_values(
        "recurrence_score", ascending=False,
    )
    run_onset_state = int(_run_onset.iloc[0]["state"])
    content_state = int(_content.iloc[1]["state"])
    print(f"run_onset exemplar: state {run_onset_state}, r={_run_onset.iloc[0]['recurrence_score']:.2f}")
    print(f"content-eligible exemplar: state {content_state}, r={_content.iloc[1]['recurrence_score']:.2f} (2nd highest)")
    return run_onset_state, content_state


@app.cell
def panel_C_surfaces(
    state_means, run_onset_state, content_state,
    PARCELLATION, OUT_FIG2, plt, np,
):
    """Panel C surface mini-pieces - two exemplar states on cortical + subcortical.

    Writes:
      fig2_C_run_onset_cortical.png
      fig2_C_run_onset_subcortical.png
      fig2_C_content_eligible_cortical.png
      fig2_C_content_eligible_subcortical.png
      fig2_C_colorbar_cortical.{pdf,png}
      fig2_C_colorbar_subcortical.{pdf,png}

    Cortical and subcortical use independent symmetric color ranges. Subcortical
    z-score magnitudes in these exemplars are ~10-30x smaller than cortical; a
    shared range would render the subcortical panels nearly white. Two colorbars
    are emitted so the assembled figure shows the magnitude mismatch explicitly.
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

    _means = state_means["sub-01"]
    _p_run = _means[run_onset_state]
    _p_content = _means[content_state]

    _labels_df = load_parcel_labels(PARCELLATION)
    _subdir = get_subcortical_atlas_dir()

    # Split cortical vs subcortical by atlas_name ('4S156' == Schaefer cortical);
    # compute symmetric 98th-percentile vmax independently for each region so the
    # smaller-magnitude subcortical values remain visible.
    _cort_mask = (_labels_df["atlas_name"] == "4S156").values
    _subc_mask = ~_cort_mask
    _vmax_cort = float(max(
        np.nanpercentile(np.abs(_p_run[_cort_mask]), 98),
        np.nanpercentile(np.abs(_p_content[_cort_mask]), 98),
    ))
    _vmax_subc = float(max(
        np.nanpercentile(np.abs(_p_run[_subc_mask]), 98),
        np.nanpercentile(np.abs(_p_content[_subc_mask]), 98),
    ))
    _range_cort = (-_vmax_cort, _vmax_cort)
    _range_subc = (-_vmax_subc, _vmax_subc)

    for _tag, _pattern in (
        ("run_onset", _p_run),
        ("content_eligible", _p_content),
    ):
        _cort = pattern_to_cortical_dict(_pattern, _labels_df, PARCELLATION)
        _sub_d = pattern_to_subcortical_dict(_pattern, _labels_df, PARCELLATION)
        _cort_img = render_cortical_to_image(_cort, _range_cort)
        _sub_img = render_subcortical_to_image(_sub_d, _range_subc, atlas_dir=_subdir)
        for _name, _img, _w in (
            (f"{_tag}_cortical", _cort_img, 4.2),
            (f"{_tag}_subcortical", _sub_img, 3.2),
        ):
            _h = _img.shape[0] * _w / _img.shape[1]
            _fig = plt.figure(figsize=(_w, _h))
            _ax = _fig.add_axes([0, 0, 1, 1])
            _ax.imshow(_img); _ax.axis("off")
            _out = OUT_FIG2 / f"fig2_C_{_name}.png"
            _fig.savefig(_out, bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close(_fig)
            print(f"saved: {_out.name}")

    for _suffix, _range in (("cortical", _range_cort), ("subcortical", _range_subc)):
        _cfig, _cax = plt.subplots(figsize=(3.0, 0.5))
        _cfig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.90)
        _sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=_range[0], vmax=_range[1]),
        )
        _sm.set_array([])
        _cb = _cfig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.set_label(f"Mean activation (z), {_suffix}", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        _cfig.savefig(OUT_FIG2 / f"fig2_C_colorbar_{_suffix}.pdf", bbox_inches="tight", pad_inches=0.02)
        _cfig.savefig(OUT_FIG2 / f"fig2_C_colorbar_{_suffix}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_cfig)
    print(f"cortical range: ±{_vmax_cort:.2f}, subcortical range: ±{_vmax_subc:.3f}")
    return


@app.cell
def panel_C_diagnostic_curves(
    decoded_sub01, run_onset_state, content_state, OUT_FIG2,
    TAXONOMY_COLORS, plt, np,
):
    """Panel C diagnostic curves - occupancy × TR-position-within-run.

    For each exemplar, compute per-run occupancy(t): fraction of runs where
    the state is active at TR index t (0-based, truncated to the shortest run).
    Two separate PDFs (one per exemplar) so user can place them next to their
    matched surfaces.
    """
    _runs = [np.asarray(_v) for _v in decoded_sub01.values()]
    _min_len = min(len(_r) for _r in _runs)
    _trunc = np.stack([_r[:_min_len] for _r in _runs])  # (n_runs, T)

    def _occupancy_curve(_state_id):
        return (_trunc == _state_id).mean(axis=0)

    for _tag, _state_id, _category in (
        ("run_onset", run_onset_state, "Run-onset-anchored"),
        ("content_eligible", content_state, "Content-eligible"),
    ):
        _curve = _occupancy_curve(_state_id)
        _t = np.arange(len(_curve))
        _fig, _ax = plt.subplots(figsize=(3.4, 1.8))
        _ax.fill_between(_t, 0, _curve, color=TAXONOMY_COLORS[_category], alpha=0.35, linewidth=0)
        _ax.plot(_t, _curve, color=TAXONOMY_COLORS[_category], linewidth=1.2)
        _ax.set_xlim(0, len(_curve) - 1)
        _ax.set_ylim(0, max(0.1, float(_curve.max()) * 1.1))
        _ax.set_xlabel("TR-within-run", fontsize=9)
        _ax.set_ylabel("P(state active)", fontsize=9)
        _ax.tick_params(axis="both", labelsize=7)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)
        _fig.subplots_adjust(left=0.16, right=0.97, bottom=0.24, top=0.95)

        _out_pdf = OUT_FIG2 / f"fig2_C_{_tag}_occupancy.pdf"
        _out_png = OUT_FIG2 / f"fig2_C_{_tag}_occupancy.png"
        _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out_pdf.name} (peak={_curve.max():.2f} at t={_curve.argmax()})")
    return


@app.cell
def pick_fig2_supp_exemplars(state_flags, run_onset_state, content_state):
    """Pick one exemplar per display category for supplementary Panel C.

    All from sub-01 (matches main Panel C's subject so exemplars share PCA,
    HMM, and decoded episodes - the occupancy curves are directly comparable).

      Content-eligible      - reuses main Panel C's content_state (2nd highest).
      Run-onset-anchored    - reuses main Panel C's run_onset_state (highest).
      Low-confidence        - highest-recurrence low_confidence state.
      Drift-anchored        - highest-recurrence season_temporal state.
      Unused + rare         - highest-recurrence "rare" state. Pure "unused"
                              states have recurrence = 0 and no decoded
                              occupancy, so they have no meaningful spatial
                              pattern to display; "rare" is the live member
                              of this display category.
    """
    _df = state_flags["sub-01"]

    _low_conf = _df[_df["summary_category"] == "low_confidence"].sort_values(
        "recurrence_score", ascending=False,
    )
    _drift = _df[_df["summary_category"] == "season_temporal"].sort_values(
        "recurrence_score", ascending=False,
    )
    _rare = _df[_df["summary_category"] == "rare"].sort_values(
        "recurrence_score", ascending=False,
    )
    low_conf_state = int(_low_conf.iloc[0]["state"])
    drift_state = int(_drift.iloc[0]["state"])
    rare_state = int(_rare.iloc[0]["state"])

    supp_exemplars = [
        ("content_eligible",   "Content-eligible",      content_state,    float(_df.loc[_df["state"] == content_state, "recurrence_score"].iloc[0])),
        ("run_onset",          "Run-onset-anchored",    run_onset_state,  float(_df.loc[_df["state"] == run_onset_state, "recurrence_score"].iloc[0])),
        ("low_confidence",     "Low-confidence",        low_conf_state,   float(_low_conf.iloc[0]["recurrence_score"])),
        ("drift_anchored",     "Drift-anchored",        drift_state,      float(_drift.iloc[0]["recurrence_score"])),
        ("rare",               "Unused + rare (rare)",  rare_state,       float(_rare.iloc[0]["recurrence_score"])),
    ]
    for _tag, _label, _sid, _r in supp_exemplars:
        print(f"  {_tag:18s} state={_sid:2d}  rec={_r:.3f}  ({_label})")
    return (supp_exemplars,)


@app.cell
def panel_C_supp_surfaces(
    state_means, supp_exemplars, PARCELLATION, OUT_FIG2_SUPP, plt, np,
):
    """Supplementary Panel C surfaces - one exemplar per category.

    Writes per-category cortical/subcortical PNGs plus two colorbars
    (independent symmetric 98th-percentile ranges across the 5 supp
    patterns; NOT shared with main Panel C). Matches main Panel C's
    two-colorbar convention because subcortical magnitudes are ~10-30x
    smaller than cortical regardless of state choice.
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

    _means = state_means["sub-01"]
    _patterns = [(t, lab, _means[sid]) for (t, lab, sid, _r) in supp_exemplars]

    _labels_df = load_parcel_labels(PARCELLATION)
    _subdir = get_subcortical_atlas_dir()
    _cort_mask = (_labels_df["atlas_name"] == "4S156").values
    _subc_mask = ~_cort_mask

    _vmax_cort = float(max(
        np.nanpercentile(np.abs(_p[_cort_mask]), 98) for (_, _, _p) in _patterns
    ))
    _vmax_subc = float(max(
        np.nanpercentile(np.abs(_p[_subc_mask]), 98) for (_, _, _p) in _patterns
    ))
    _range_cort = (-_vmax_cort, _vmax_cort)
    _range_subc = (-_vmax_subc, _vmax_subc)

    for _tag, _label, _pattern in _patterns:
        _cort = pattern_to_cortical_dict(_pattern, _labels_df, PARCELLATION)
        _sub_d = pattern_to_subcortical_dict(_pattern, _labels_df, PARCELLATION)
        _cort_img = render_cortical_to_image(_cort, _range_cort)
        _sub_img = render_subcortical_to_image(_sub_d, _range_subc, atlas_dir=_subdir)
        for _name, _img, _w in (
            (f"{_tag}_cortical", _cort_img, 4.2),
            (f"{_tag}_subcortical", _sub_img, 3.2),
        ):
            _h = _img.shape[0] * _w / _img.shape[1]
            _fig = plt.figure(figsize=(_w, _h))
            _ax = _fig.add_axes([0, 0, 1, 1])
            _ax.imshow(_img); _ax.axis("off")
            _out = OUT_FIG2_SUPP / f"fig2_C_supp_{_name}.png"
            _fig.savefig(_out, bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close(_fig)
            print(f"saved: {_out.name}")

    for _suffix, _range in (("cortical", _range_cort), ("subcortical", _range_subc)):
        _cfig, _cax = plt.subplots(figsize=(3.0, 0.5))
        _cfig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.90)
        _sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=_range[0], vmax=_range[1]),
        )
        _sm.set_array([])
        _cb = _cfig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.set_label(f"Mean activation (z), {_suffix}", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        _cfig.savefig(OUT_FIG2_SUPP / f"fig2_C_supp_colorbar_{_suffix}.pdf", bbox_inches="tight", pad_inches=0.02)
        _cfig.savefig(OUT_FIG2_SUPP / f"fig2_C_supp_colorbar_{_suffix}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_cfig)
    print(f"supp cortical range: ±{_vmax_cort:.2f}, subcortical range: ±{_vmax_subc:.3f}")
    return


@app.cell
def panel_C_supp_curves(
    decoded_sub01, supp_exemplars, OUT_FIG2_SUPP,
    TAXONOMY_COLORS, TAXONOMY_MAP, plt, np,
):
    """Supplementary Panel C diagnostic curves - one per category.

    Occupancy × TR-within-run for each exemplar state. Same template as
    main Panel C's curves; colored by display category so curves visually
    match the strip plot and supplementary bar (Panel A).

    Interpretation varies by category:
      - Run-onset-anchored: peak at t=0 is the defining feature.
      - Content-eligible: roughly flat - the absence of anchoring.
      - Low-confidence / drift-anchored / rare: low-amplitude curves are
        expected given low recurrence; these are shown for completeness,
        not because the curve carries the category-defining signal.
    """
    _runs = [np.asarray(_v) for _v in decoded_sub01.values()]
    _min_len = min(len(_r) for _r in _runs)
    _trunc = np.stack([_r[:_min_len] for _r in _runs])

    _category_to_display = {
        "content_eligible": "Content-eligible",
        "run_onset":        "Run-onset-anchored",
        "low_confidence":   "Low-confidence",
        "drift_anchored":   "Drift-anchored",
        "rare":             "Unused + rare",
    }

    for _tag, _label, _sid, _rec in supp_exemplars:
        _curve = (_trunc == _sid).mean(axis=0)
        _t = np.arange(len(_curve))
        _display = _category_to_display[_tag]
        _color = TAXONOMY_COLORS[_display]

        _fig, _ax = plt.subplots(figsize=(3.4, 1.8))
        _ax.fill_between(_t, 0, _curve, color=_color, alpha=0.35, linewidth=0)
        _ax.plot(_t, _curve, color=_color, linewidth=1.2)
        _ax.set_xlim(0, len(_curve) - 1)
        _ax.set_ylim(0, max(0.1, float(_curve.max()) * 1.1))
        _ax.set_xlabel("TR-within-run", fontsize=9)
        _ax.set_ylabel("P(state active)", fontsize=9)
        _ax.tick_params(axis="both", labelsize=7)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)
        _fig.subplots_adjust(left=0.16, right=0.97, bottom=0.24, top=0.95)

        _out_pdf = OUT_FIG2_SUPP / f"fig2_C_supp_{_tag}_occupancy.pdf"
        _out_png = OUT_FIG2_SUPP / f"fig2_C_supp_{_tag}_occupancy.png"
        _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out_pdf.name}  (peak={_curve.max():.2f} at t={_curve.argmax()})")
    return


@app.cell
def panel_D_low_confidence(
    state_flags, SUBJECTS, OUT_FIG2,
    SUBJECT_NEUTRAL, SUBJECT_ACCENT, plt, np,
):
    """Panel D - low-confidence prevalence per subject.

    Horizontal bar: fraction of states classified as "Low-confidence" (out of
    total K per subject, so the denominator matches Panel A's bar heights).
    Sub-06 drawn in accent color, others in neutral - the outlier is the
    individual-differences finding about state granularity.
    """
    _total = {_sub: int(len(state_flags[_sub])) for _sub in SUBJECTS}
    _low = {
        _sub: int((state_flags[_sub]["summary_category"] == "low_confidence").sum())
        for _sub in SUBJECTS
    }
    _frac = np.array([_low[_s] / max(_total[_s], 1) for _s in SUBJECTS])
    _counts = np.array([_low[_s] for _s in SUBJECTS])
    _totals = np.array([_total[_s] for _s in SUBJECTS])

    _fig, _ax = plt.subplots(figsize=(4.8, 2.8))
    _y = np.arange(len(SUBJECTS))
    _colors = [SUBJECT_ACCENT if _s == "sub-06" else SUBJECT_NEUTRAL for _s in SUBJECTS]
    _ax.barh(_y, _frac, color=_colors, edgecolor="white", linewidth=0.6)
    for _i, (_n, _t) in enumerate(zip(_counts, _totals)):
        _ax.text(
            _frac[_i] + 0.008, _i, f" {_n}/{_t}",
            va="center", ha="left", fontsize=8,
            color=_colors[_i],
        )
    _ax.set_yticks(_y)
    _ax.set_yticklabels(SUBJECTS, fontsize=8)
    _ax.invert_yaxis()
    _ax.set_xlabel("Fraction of active states\nclassified as Low-confidence", fontsize=9)
    _ax.set_xlim(0, float(_frac.max()) * 1.25)
    _ax.tick_params(axis="x", labelsize=8)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.subplots_adjust(left=0.18, right=0.97, bottom=0.24, top=0.95)

    _out_pdf = OUT_FIG2 / "fig2_D_low_confidence_per_subject.pdf"
    _out_png = OUT_FIG2 / "fig2_D_low_confidence_per_subject.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    print(f"fractions: {dict(zip(SUBJECTS, [round(f, 3) for f in _frac]))}")
    return


@app.cell
def figure3_panel_plan():
    """
    # Figure 3 - Transitions connect functionally similar states (per subject)

    Serves manuscript R3 and talk slide 5.3.

    | Panel | Content | Size | Output filename |
    |---|---|---|---|
    | A | Transition probability matrix (sub-01 exemplar), states sorted by recurrence descending; log color for off-diagonal | square | `fig3_A_transition_matrix.pdf` |
    | B | Effect-size summary - four metrics × six subjects (recurrence assortativity ρ, FC–transition ρ, MFPT–FC ρ, network homophily ratio). Subject = marker shape; sub-05 homophily null flagged (open marker) | wide | `fig3_B_effect_summary.pdf` |
    | C | FC–transition binned scatter (sub-01 exemplar) - mirrors talk slide 5.3 | wide | `fig3_C_fc_transition.pdf` |
    | D | MFPT matrix (sub-01 exemplar), states sorted by recurrence | square | `fig3_D_mfpt_landscape.pdf` |
    | E | Network × network empirical transition-probability matrix (sub-01 exemplar), 13×13, LogNorm magma_r; diagonal = within-network homophily, scalar-consistent P<1e-15 filter | square | `fig3_E_network_transition_matrix.pdf` |

    Language: "associated with" not "predicts"; FC / MFPT come from empirical
    Ledoit-Wolf state correlations (05f), one step removed from HMM emissions.

    All-subject versions of A, C, D, E are written to `fig3/supplementary/` -
    user can drop these into the supplement composite. Panel B already
    aggregates six subjects, so no supplement for it. Panel E sibling files
    (`*_counts.npy`, `*_mean.npy`) store per-cell pair counts and mean P
    for reproducibility / caption stats.

    Rules: no titles, no on-figure panel labels, one file per panel. Panel
    letters appear in filenames only.
    """
    return


@app.cell
def load_fig3_data(
    SUBJECTS, DWELL_DIR, TRANSITION_DIR, FC_DIR, RECURRENCE_DIR, VT,
    np, json, pd,
):
    """Load all Figure 3 data: transition / MFPT / RV matrices, state order.

    For each subject we record:
      - tp_full (50x50) transition probabilities
      - rv_full (50x50) FC similarity (RV coefficient, empirical state corr)
      - mfpt (K_active x K_active) - 06b restricts to active already
      - active_ids (K_active,) - original state indices in 0..49 space
      - recurrence (K_active,) - recurrence score aligned with active_ids
      - graph_metrics dataframe - row order = MFPT row order
    """
    fig3_data = {}
    for _sub in SUBJECTS:
        _tp = np.load(DWELL_DIR / _sub / VT / "transition_probabilities.npy")
        _rv = np.load(FC_DIR / _sub / VT / "fc_similarity_corr_rv.npy")
        _mfpt = np.load(TRANSITION_DIR / _sub / VT / "mfpt_matrix.npy")
        _gm = pd.read_csv(TRANSITION_DIR / _sub / VT / "graph_metrics.csv")
        with open(RECURRENCE_DIR / _sub / VT / "recurrence_summary.json") as _f:
            _summary = json.load(_f)
        with open(TRANSITION_DIR / _sub / VT / "transition_structure_summary.json") as _f:
            _tsumm = json.load(_f)

        _active_ids = _gm["state_id"].values.astype(int)
        _recur = _gm["recurrence_score"].values.astype(float)
        assert len(_active_ids) == _mfpt.shape[0], (
            f"{_sub}: graph_metrics rows {len(_active_ids)} ≠ mfpt {_mfpt.shape[0]}"
        )
        fig3_data[_sub] = {
            "tp_full": _tp,
            "rv_full": _rv,
            "mfpt": _mfpt,
            "active_ids": _active_ids,
            "recurrence": _recur,
            "graph_metrics": _gm,
            "summary": _tsumm,
        }
    print({_s: {"K_act": len(_d["active_ids"])} for _s, _d in fig3_data.items()})
    return (fig3_data,)


@app.cell
def panel_A_transition_matrix(fig3_data, OUT_FIG3, OUT_FIG3_SUPP, SUBJECTS, plt, np):
    """Panel A - transition probability matrix, states sorted by recurrence descending.

    Off-diagonal log-color on a shared vmin/vmax per subject. Diagonal drawn
    separately (grey) so the off-diagonal structure is visible.
    Saves sub-01 as Panel A and each subject to supplementary/.
    """
    from matplotlib.colors import LogNorm

    def _render(_sub, _out_path):
        _d = fig3_data[_sub]
        _active = _d["active_ids"]
        _recur = _d["recurrence"]
        _tp = _d["tp_full"][np.ix_(_active, _active)]
        _order_local = np.argsort(-_recur)
        _tp_sorted = _tp[np.ix_(_order_local, _order_local)]
        _recur_sorted = _recur[_order_local]

        _k = _tp_sorted.shape[0]
        _off = _tp_sorted.copy()
        np.fill_diagonal(_off, np.nan)
        _off_valid = _off[np.isfinite(_off) & (_off > 0)]
        _vmin = max(float(np.nanpercentile(_off_valid, 2)), 1e-4)
        _vmax = float(np.nanpercentile(_off_valid, 98))

        _fig = plt.figure(figsize=(3.6, 3.6))
        _gs = _fig.add_gridspec(
            2, 2,
            width_ratios=[0.05, 1.0], height_ratios=[0.05, 1.0],
            wspace=0.02, hspace=0.02,
        )
        _ax_top = _fig.add_subplot(_gs[0, 1])
        _ax_left = _fig.add_subplot(_gs[1, 0])
        _ax_main = _fig.add_subplot(_gs[1, 1])

        _im = _ax_main.imshow(
            _off, cmap="magma_r",
            norm=LogNorm(vmin=_vmin, vmax=_vmax),
            aspect="equal", interpolation="nearest",
        )
        _diag_mask = np.eye(_k, dtype=bool)
        _diag_img = np.where(_diag_mask, 1.0, np.nan)
        _ax_main.imshow(_diag_img, cmap="Greys", vmin=0, vmax=1,
                        alpha=0.35, aspect="equal", interpolation="nearest")
        _ax_main.set_xticks([]); _ax_main.set_yticks([])
        # labelpad clears the recurrence strip on each edge (strip = 5% of axis).
        _ax_main.set_xlabel("To State (high → low)", fontsize=9, labelpad=4)
        _ax_main.set_ylabel("From State (low → high)", fontsize=9, labelpad=14)
        for _s in _ax_main.spines.values():
            _s.set_linewidth(0.5)

        _recur_strip = np.array([_recur_sorted])
        _ax_top.imshow(_recur_strip, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _ax_top.set_xticks([]); _ax_top.set_yticks([])
        for _s in _ax_top.spines.values(): _s.set_visible(False)
        _ax_left.imshow(_recur_strip.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _ax_left.set_xticks([]); _ax_left.set_yticks([])
        for _s in _ax_left.spines.values(): _s.set_visible(False)

        _cax = _fig.add_axes([0.93, 0.20, 0.025, 0.55])
        _cbar = _fig.colorbar(_im, cax=_cax)
        _cbar.set_label("Transition probability (log)", fontsize=8)
        _cbar.ax.tick_params(labelsize=7)

        _fig.savefig(_out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out_path.name}.{{pdf,png}} K={_k} vmin={_vmin:.4f} vmax={_vmax:.3f}")

    _render("sub-01", OUT_FIG3 / "fig3_A_transition_matrix")
    for _sub in SUBJECTS:
        _render(_sub, OUT_FIG3_SUPP / f"fig3_A_transition_matrix_{_sub}")
    return


@app.cell
def panel_B_effect_summary(fig3_data, SUBJECTS, OUT_FIG3, SUBJECT_MARKERS, plt, np):
    """Panel B - four-metric effect-size summary across six subjects.

    Four rows of subpanels (shared marker convention), each on its own
    x-scale appropriate to the metric:
      row 1: recurrence assortativity ρ
      row 2: FC–transition ρ
      row 3: MFPT–FC ρ
      row 4: network homophily ratio (null line at 1.0)

    Subject identity = marker shape. Significance: filled marker if
    perm p < 0.05, open marker otherwise. sub-05 homophily falls out as
    open because p=0.51.
    """
    # Three ρ metrics share x-scale - they are all Spearman correlations on
    # the same conceptual axis ("strength of coupling"), so direct visual
    # comparison is meaningful. Network homophily is a ratio (within /
    # between, null at 1), not a correlation, and stays on its own scale.
    _RHO_XLIM = (-0.05, 0.80)
    _metrics = [
        {
            "key": "A3_assortativity", "field": "point_estimate", "p_field": "perm_p_value",
            "label": "Recurrence assortativity ρ", "xlim": _RHO_XLIM, "null": 0.0,
        },
        {
            "key": "A3_fc_transition", "field": "rho", "p_field": "p_value",
            "label": "FC–transition ρ", "xlim": _RHO_XLIM, "null": 0.0,
        },
        {
            "key": "A4_mfpt_fc", "field": "rho", "p_field": "p_value",
            "label": "MFPT–FC ρ", "xlim": _RHO_XLIM, "null": 0.0,
        },
        {
            "key": "A3_network_homophily", "field": "ratio", "p_field": "p_value",
            "label": "Network homophily ratio (within / between)",
            "xlim": (0.75, 2.20), "null": 1.0,
        },
    ]
    _n_metrics = len(_metrics)

    _fig, _axes = plt.subplots(
        _n_metrics, 1, figsize=(5.4, 4.2), sharey=False,
    )
    for _ax, _m in zip(_axes, _metrics):
        for _sub in SUBJECTS:
            _entry = fig3_data[_sub]["summary"][_m["key"]]
            _x = float(_entry[_m["field"]])
            _p = float(_entry[_m["p_field"]])
            _sig = _p < 0.05
            _marker = SUBJECT_MARKERS[_sub]
            _ax.scatter(
                _x, 0, marker=_marker, s=70,
                facecolor=("#2F4F7F" if _sig else "white"),
                edgecolor="#2F4F7F", linewidth=1.2, zorder=3,
            )
            if not _sig:
                _ax.annotate(
                    _sub, xy=(_x, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color="#B94A48",
                )
        _ax.axvline(_m["null"], color="#B94A48", linewidth=0.9,
                    linestyle="--", alpha=0.8, zorder=1)
        _ax.set_xlim(*_m["xlim"])
        _ax.set_ylim(-0.5, 0.5)
        _ax.set_yticks([])
        _ax.set_xlabel(_m["label"], fontsize=9)
        _ax.tick_params(axis="x", labelsize=8)
        for _s in ("top", "right", "left"):
            _ax.spines[_s].set_visible(False)

    # Subject legend on top axis (marker shapes only)
    from matplotlib.lines import Line2D
    _handles = [
        Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="None",
               markerfacecolor="#2F4F7F", markeredgecolor="#2F4F7F",
               markersize=7, label=_s)
        for _s in SUBJECTS
    ]
    _handles += [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="#2F4F7F",
               markersize=7, label="p ≥ 0.05"),
    ]
    _axes[0].legend(
        handles=_handles, loc="upper center", bbox_to_anchor=(0.5, 1.9),
        ncol=4, fontsize=7, frameon=False, handletextpad=0.3,
        columnspacing=0.9,
    )
    _fig.subplots_adjust(left=0.04, right=0.98, bottom=0.09, top=0.82, hspace=0.95)

    _out_pdf = OUT_FIG3 / "fig3_B_effect_summary.pdf"
    _out_png = OUT_FIG3 / "fig3_B_effect_summary.png"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    # Print the values that went into the panel for verification
    for _m in _metrics:
        _vals = {
            _s: (
                float(fig3_data[_s]["summary"][_m["key"]][_m["field"]]),
                float(fig3_data[_s]["summary"][_m["key"]][_m["p_field"]]),
            )
            for _s in SUBJECTS
        }
        print(_m["label"], _vals)
    print(f"saved: {_out_pdf.name}, {_out_png.name}")
    return


@app.cell
def panel_C_fc_transition(fig3_data, OUT_FIG3, OUT_FIG3_SUPP, SUBJECTS, plt, np):
    """Panel C - FC–transition binned scatter (sub-01 exemplar + supplementary 6).

    For each active-state pair (i,j), i<j:
      x = FC similarity (RV coefficient on empirical within-state correlations, 05f)
      y = transition probability (06a), treated symmetrically: max(P(i→j), P(j→i))

    Pairs with NaN RV or zero transition are dropped. The panel shows raw
    scatter (light alpha) + binned means with SEM error bars. Spearman ρ
    annotated from 06b summary (so the manuscript number matches the JSON).
    """
    from scipy import stats

    def _render(_sub, _out_path, *, include_rho=True):
        _d = fig3_data[_sub]
        _active = _d["active_ids"]
        _rv = _d["rv_full"][np.ix_(_active, _active)]
        _tp = _d["tp_full"][np.ix_(_active, _active)]

        _n = len(_active)
        _ii, _jj = np.triu_indices(_n, k=1)
        _fc_vals = _rv[_ii, _jj]
        _tp_sym = np.maximum(_tp, _tp.T)
        _tp_vals = _tp_sym[_ii, _jj]

        _valid = np.isfinite(_fc_vals) & np.isfinite(_tp_vals) & (_tp_vals > 0)
        _fc = _fc_vals[_valid]
        _tp_v = _tp_vals[_valid]

        _bin_edges = np.linspace(_fc.min(), _fc.max(), 16)
        _bc, _bm, _bs = [], [], []
        for _i in range(len(_bin_edges) - 1):
            _mask = (_fc >= _bin_edges[_i]) & (_fc < _bin_edges[_i + 1])
            if _i == len(_bin_edges) - 2:
                _mask |= (_fc == _bin_edges[_i + 1])
            if _mask.sum() < 3:
                continue
            _bc.append((_bin_edges[_i] + _bin_edges[_i + 1]) / 2)
            _bm.append(_tp_v[_mask].mean())
            _bs.append(_tp_v[_mask].std() / np.sqrt(_mask.sum()))

        _fig, _ax = plt.subplots(figsize=(4.2, 3.2))
        _ax.scatter(_fc, _tp_v, s=5, alpha=0.10, color="#4A90D9",
                    edgecolors="none", rasterized=True)
        _ax.errorbar(_bc, _bm, yerr=_bs, fmt="o", ms=6, color="#E07B39",
                     ecolor="#E07B39", elinewidth=1.2, capsize=2.5,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        _ax.set_yscale("log")
        _ax.set_xlabel("FC similarity (RV coefficient)", fontsize=9)
        _ax.set_ylabel("Transition probability", fontsize=9)
        _ax.tick_params(labelsize=8)
        _ax.grid(True, alpha=0.12)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)

        if include_rho:
            # ρ is 06b's Spearman on RV-valid pairs. The scatter is restricted
            # to tp>0 (log y), but 06b's ρ includes tp=0 pairs - so we do not
            # annotate a pair count that would not match what is drawn.
            _summary_rho = float(_d["summary"]["A3_fc_transition"]["rho"])
            _ax.text(
                0.97, 0.97, f"ρ = {_summary_rho:.2f}",
                transform=_ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )
        _fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.96)
        _fig.savefig(_out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out_path.name}.{{pdf,png}} "
              f"pairs={_valid.sum()} rho={float(_d['summary']['A3_fc_transition']['rho']):.3f}")

    _render("sub-01", OUT_FIG3 / "fig3_C_fc_transition")
    for _sub in SUBJECTS:
        _render(_sub, OUT_FIG3_SUPP / f"fig3_C_fc_transition_{_sub}")
    return


@app.cell
def panel_D_mfpt_landscape(fig3_data, OUT_FIG3, OUT_FIG3_SUPP, SUBJECTS, plt, np):
    """Panel D - MFPT matrix, states sorted by recurrence descending.

    MFPT (mean first-passage time) is the expected number of TRs for a
    random walk in the HMM transition graph starting in state i to first
    reach state j. Short MFPT = easy to reach; long MFPT = rarely reached.

    Color is clipped to the 5–95% percentile range because a handful of
    low-stationary-probability states have MFPTs ~3 orders of magnitude
    larger than the bulk (e.g. sub-01 col 31 = 112,700 TRs vs 95th
    percentile 548 TRs). Without clipping, those outliers squash the
    color range and the rest of the matrix reads uniformly dark. With
    clipping, the within-bulk structure (high-recurrence cols = short
    MFPT from anywhere = lighter) is visible and outliers saturate to
    the colorbar top.
    """
    from matplotlib.colors import LogNorm

    def _render(_sub, _out_path):
        _d = fig3_data[_sub]
        _mfpt = _d["mfpt"]
        _recur = _d["recurrence"]
        _order = np.argsort(-_recur)
        _m_sorted = _mfpt[np.ix_(_order, _order)]
        _recur_sorted = _recur[_order]

        _k = _m_sorted.shape[0]
        _off = _m_sorted.copy().astype(float)
        np.fill_diagonal(_off, np.nan)
        _valid = _off[np.isfinite(_off) & (_off > 0)]
        _vmin = float(np.nanpercentile(_valid, 5))
        _vmax = float(np.nanpercentile(_valid, 95))

        _fig = plt.figure(figsize=(3.6, 3.6))
        _gs = _fig.add_gridspec(
            2, 2,
            width_ratios=[0.05, 1.0], height_ratios=[0.05, 1.0],
            wspace=0.02, hspace=0.02,
        )
        _ax_top = _fig.add_subplot(_gs[0, 1])
        _ax_left = _fig.add_subplot(_gs[1, 0])
        _ax_main = _fig.add_subplot(_gs[1, 1])

        _im = _ax_main.imshow(
            _off, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma_r",
            norm=LogNorm(vmin=_vmin, vmax=_vmax),
            aspect="equal", interpolation="nearest",
        )
        _ax_main.set_xticks([]); _ax_main.set_yticks([])
        _ax_main.set_xlabel("To State (high → low)", fontsize=9, labelpad=4)
        _ax_main.set_ylabel("From State (low → high)", fontsize=9, labelpad=14)
        for _s in _ax_main.spines.values():
            _s.set_linewidth(0.5)

        _recur_strip = np.array([_recur_sorted])
        _ax_top.imshow(_recur_strip, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _ax_top.set_xticks([]); _ax_top.set_yticks([])
        for _s in _ax_top.spines.values(): _s.set_visible(False)
        _ax_left.imshow(_recur_strip.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _ax_left.set_xticks([]); _ax_left.set_yticks([])
        for _s in _ax_left.spines.values(): _s.set_visible(False)

        _cax = _fig.add_axes([0.93, 0.20, 0.025, 0.55])
        _cbar = _fig.colorbar(_im, cax=_cax, extend="max")
        _cbar.set_label("Mean first-passage time\n(TRs, log; clipped 5–95%)", fontsize=8)
        _cbar.ax.tick_params(labelsize=7)

        _rho = float(_d["summary"]["A4_mfpt_fc"]["rho"])
        _ax_main.text(
            0.03, 0.03, f"MFPT–FC ρ = {_rho:.2f}",
            transform=_ax_main.transAxes, ha="left", va="bottom",
            fontsize=8, color="white",
            bbox=dict(facecolor="#2F4F7F", alpha=0.78, edgecolor="none", pad=2),
        )

        _fig.savefig(_out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)
        print(f"saved: {_out_path.name}.{{pdf,png}} K={_k} "
              f"vmin={_vmin:.1f} vmax={_vmax:.0f} rho={_rho:.3f}")

    _render("sub-01", OUT_FIG3 / "fig3_D_mfpt_landscape")
    for _sub in SUBJECTS:
        _render(_sub, OUT_FIG3_SUPP / f"fig3_D_mfpt_landscape_{_sub}")
    return


@app.cell
def panel_E_network_transition_matrix(
    fig3_data, OUT_FIG3, OUT_FIG3_SUPP, SUBJECTS, plt, np,
):
    """Panel E - network × network empirical transition-probability matrix.

    Aggregates per-state empirical transition probabilities (tp_full) by each
    state's dominant network (13 groups in NETWORK_ORDER). Cell (A, B) =
    mean P_empirical[i, j] over state pairs with i≠j, dominant(i)=A,
    dominant(j)=B. Matches the scalar homophily definition in 06b
    (self-loops excluded, P<1e-15 filtered) so diagonal cells and the
    scalar within_mean are computed from the same underlying pairs, just
    aggregated at different levels.

    Color: LogNorm magma_r, 5th/95th-percentile clipping of finite cells,
    extend="max" - matches MFPT panel D's recipe. Empty cells (no pairs)
    render transparent. Sub-01 = main panel; each subject = supp.

    Sample counts saved alongside each figure for reproducibility.
    """
    from matplotlib.colors import LogNorm, to_rgba
    from utils.plot_style import NETWORK_ORDER, NETWORK_COLORS

    _NET_ABBREV = {
        "Vis": "Vis", "SomMot": "SM", "DorsAttn": "DA", "SalVentAttn": "SVA",
        "Limbic": "Li", "Cont": "Co", "Default": "DMN",
        "BG": "BG", "Midbrain-DA": "MidDA", "Midbrain-Diencephalic": "MidDi",
        "Thalamus": "Th", "Hipp/Amyg": "HA", "Cerebellum": "Cb",
    }

    def _render(_sub, _out_path):
        _d = fig3_data[_sub]
        _active = _d["active_ids"]
        _tp_full = _d["tp_full"]
        _gm = _d["graph_metrics"]
        _nets_by_state = dict(
            zip(_gm["state_id"].astype(int),
                _gm["dominant_network"].astype(str))
        )

        _n_nets = len(NETWORK_ORDER)
        _net_idx = {_n: _i for _i, _n in enumerate(NETWORK_ORDER)}
        _sum = np.zeros((_n_nets, _n_nets))
        _cnt = np.zeros((_n_nets, _n_nets), dtype=int)

        for _i in _active:
            for _j in _active:
                if _i == _j:
                    continue
                _p = float(_tp_full[int(_i), int(_j)])
                if _p < 1e-15:
                    continue
                _ni = _nets_by_state.get(int(_i), "Unknown")
                _nj = _nets_by_state.get(int(_j), "Unknown")
                if _ni not in _net_idx or _nj not in _net_idx:
                    continue
                _a, _b = _net_idx[_ni], _net_idx[_nj]
                _sum[_a, _b] += _p
                _cnt[_a, _b] += 1

        _mean = np.full((_n_nets, _n_nets), np.nan)
        _mask = _cnt > 0
        _mean[_mask] = _sum[_mask] / _cnt[_mask]

        _finite = _mean[np.isfinite(_mean) & (_mean > 0)]
        if len(_finite) == 0:
            print(f"  {_sub}: no finite cells - skipping")
            return
        _vmin = float(np.nanpercentile(_finite, 5))
        _vmax = float(np.nanpercentile(_finite, 95))

        _fig = plt.figure(figsize=(4.4, 4.4))
        _gs = _fig.add_gridspec(
            2, 2,
            width_ratios=[0.05, 1.0], height_ratios=[0.05, 1.0],
            wspace=0.02, hspace=0.02,
        )
        _ax_top = _fig.add_subplot(_gs[0, 1])
        _ax_left = _fig.add_subplot(_gs[1, 0])
        _ax_main = _fig.add_subplot(_gs[1, 1])

        _im = _ax_main.imshow(
            _mean, cmap="magma_r",
            norm=LogNorm(vmin=_vmin, vmax=_vmax),
            aspect="equal", interpolation="nearest",
        )
        _abbrev = [_NET_ABBREV[_n] for _n in NETWORK_ORDER]
        _ax_main.set_xticks(range(_n_nets))
        _ax_main.set_xticklabels(_abbrev, fontsize=7, rotation=0)
        _ax_main.set_yticks(range(_n_nets))
        _ax_main.set_yticklabels(_abbrev, fontsize=7)
        _ax_main.set_xlabel("To network", fontsize=9, labelpad=4)
        _ax_main.set_ylabel("From network", fontsize=9, labelpad=4)
        for _s in _ax_main.spines.values():
            _s.set_linewidth(0.5)

        _strip_rgba = np.array(
            [to_rgba(NETWORK_COLORS.get(_n, "#CCCCCC")) for _n in NETWORK_ORDER]
        ).reshape(1, _n_nets, 4)
        _ax_top.imshow(_strip_rgba, aspect="auto", interpolation="nearest")
        _ax_top.set_xticks([]); _ax_top.set_yticks([])
        for _s in _ax_top.spines.values():
            _s.set_visible(False)
        _ax_left.imshow(_strip_rgba.transpose(1, 0, 2),
                        aspect="auto", interpolation="nearest")
        _ax_left.set_xticks([]); _ax_left.set_yticks([])
        for _s in _ax_left.spines.values():
            _s.set_visible(False)

        _cax = _fig.add_axes([0.95, 0.20, 0.025, 0.55])
        _cbar = _fig.colorbar(_im, cax=_cax, extend="max")
        _cbar.set_label("Mean P (log)", fontsize=8)
        _cbar.ax.tick_params(labelsize=7)

        _fig.savefig(_out_path.with_suffix(".pdf"),
                     bbox_inches="tight", pad_inches=0.02)
        _fig.savefig(_out_path.with_suffix(".png"),
                     bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(_fig)

        np.save(_out_path.parent / f"{_out_path.name}_counts.npy", _cnt)
        np.save(_out_path.parent / f"{_out_path.name}_mean.npy", _mean)

        _n_cells = int((_cnt > 0).sum())
        _n_sparse = int(((_cnt > 0) & (_cnt < 3)).sum())
        _diag_filled = int((_cnt[np.arange(_n_nets), np.arange(_n_nets)] > 0).sum())
        print(
            f"saved: {_out_path.name}.{{pdf,png}} "
            f"cells={_n_cells}/{_n_nets ** 2} "
            f"sparse(<3)={_n_sparse} diag_filled={_diag_filled}/{_n_nets} "
            f"vmin={_vmin:.5f} vmax={_vmax:.4f}"
        )

    _render("sub-01", OUT_FIG3 / "fig3_E_network_transition_matrix")
    for _sub in SUBJECTS:
        _render(_sub, OUT_FIG3_SUPP / f"fig3_E_network_transition_matrix_{_sub}")
    return


@app.cell
def figure5_panel_plan():
    """
    # Figure 5 - Cross-stimulus transfer of the Friends recurrence gradient

    Serves manuscript R5 and talk slide 5.5.

    | Panel | Content | Relative size | Output filename |
    |---|---|---|---|
    | A | ρ × stimulus × subject, full repertoire. 3 stimulus groups (M10 / HP / PP); subjects as distinct markers with horizontal jitter; filled = p<0.05 uncorr, open = ns | wide | `fig5_A_rho_full.pdf` |
    | B | Same layout, content-eligible subset (R2 filter) - shows qualitative pattern preserved | wide | `fig5_B_rho_eligible.pdf` |
    | C | Movie10 genre breakdown - 4 genres (bourne, wolf, figures, life) × 6 subjects; same marker scheme. Shows social-narrative (wolf/figures) vs nature-documentary (life) dissociation. | wide | `fig5_C_m10_genre.pdf` |
    | D | Cross-modality per-subject pattern: 6 connecting lines M10→HP→PP (full ρ), sub-02 accented as pattern-breaker, others neutral/low-alpha | square | `fig5_D_crossmodality_lines.pdf` |
    | E | Same layout as A/B, **run-onset-anchored** subset (2–8 states/subject). Lets reader see whether the cross-stimulus gradient is carried by scan-structure-anchored states. | wide | `fig5_E_rho_runonset.pdf` |
    | F | Same layout as A/B, **low-confidence (sub-HRF)** subset (5–18 states/subject). Included for completeness; wider y-range because ρ is noisier here. | wide | `fig5_F_rho_lowconf.pdf` |
    | G (supp) | Run-onset anchoring decomposition - early-TR enrichment across 5 contexts (Friends a-runs, Friends b-runs, M10, HP, PP) split by the 3 anchoring sub-types from 05e_a2 (`ab-common`, `a-anchored`, `b-anchored`). Supports Panel E by showing that cross-stimulus ρ transfer is carried by `ab-common`; `a-anchored` drops to null in M10. | tall | `fig5_G_anchor_decomposition.pdf` |

    Design:
    - Subject identity = SUBJECT_MARKERS (shape) + SUBJECT_NEUTRAL color. Color channel preserved for future overlays.
    - Significance = fill (p<0.05 uncorrected, to match outline prose "flagged in prose"). No asterisks.
    - Shared y-axis (ρ) across A/B/C/D for direct magnitude comparison.
    - HP/PP missing sub-04: simply omitted (no placeholder). Caption notes "HP/PP: n=5 subjects."
    - No titles, no on-figure panel labels. User adds in assembly.
    """
    return


@app.cell
def config_fig5(SCRATCH_DIR, Path):
    OUT_FIG5 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig5"
    OUT_FIG5_SUPP = OUT_FIG5 / "supplementary"
    OUT_FIG5.mkdir(parents=True, exist_ok=True)
    OUT_FIG5_SUPP.mkdir(parents=True, exist_ok=True)

    M10_DIR = SCRATCH_DIR / "output" / "m10_05_cross_validation"
    HP_DIR = SCRATCH_DIR / "output" / "hp_05_cross_validation"
    PP_DIR = SCRATCH_DIR / "output" / "pp_05_cross_validation"

    M10_DEC = SCRATCH_DIR / "output" / "m10_04_decoded"
    HP_DEC = SCRATCH_DIR / "output" / "hp_04_decoded"
    PP_DEC = SCRATCH_DIR / "output" / "pp_04_decoded"
    FRIENDS_DEC = SCRATCH_DIR / "output" / "04_combined_hdphmm"
    FLAGS_A4_DIR = SCRATCH_DIR / "output" / "05e_temporal_trend_a4"
    return (
        OUT_FIG5, OUT_FIG5_SUPP, M10_DIR, HP_DIR, PP_DIR,
        M10_DEC, HP_DEC, PP_DEC, FRIENDS_DEC, FLAGS_A4_DIR,
    )


@app.cell
def load_fig5_data(
    M10_DIR, HP_DIR, PP_DIR, M10_DEC, HP_DEC, PP_DEC, FLAGS_A4_DIR,
    RECURRENCE_DIR, PARCELLATION, VT, SUBJECTS, json, pickle, pd, np,
):
    """Load cross-stimulus ρ per subject × stimulus for full, eligible, and R2 categories.

    fig5_data structure:
      fig5_data[stim][sub] = {
        "rho_full","p_full","n_full",
        "rho_elig","p_elig","n_elig",
        "rho_runonset","p_runonset","n_runonset",
        "rho_lowconf","p_lowconf","n_lowconf",
        "per_type": {genre: {...}}   # M10 only
      }

    Per-category ρ (run-onset-anchored, low-confidence) is computed here from
    state_flags.csv (05e_a4) + fractional_occupancy.pkl (stim_04_decoded) +
    recurrence_scores.npy (05a). Intersection with active states (recurrence > 0)
    is applied per category, matching the A1_eligible convention in m10_05.
    Categories with n_active < 3 yield None (Spearman undefined).
    """
    from scipy import stats as _stats

    _stem_js = {"M10": M10_DIR, "HP": HP_DIR, "PP": PP_DIR}
    _stem_dec = {"M10": M10_DEC, "HP": HP_DEC, "PP": PP_DEC}
    fig5_data = {stim: {} for stim in _stem_js}

    for _stim in _stem_js:
        _js_dir, _dec_dir = _stem_js[_stim], _stem_dec[_stim]
        for _sub in SUBJECTS:
            _j = _js_dir / PARCELLATION / _sub / VT / "cross_stimulus_summary.json"
            if not _j.exists():
                fig5_data[_stim][_sub] = None
                continue
            _d = json.loads(_j.read_text())
            _a1 = _d["A1_recurrence_correlation"]
            _a1e = _d["A1_recurrence_correlation_eligible"]
            _rec_entry = {
                "n_full": _a1["n_active_states"],
                "rho_full": _a1["spearman_rho"],
                "p_full": _a1["spearman_p"],
                "n_elig": _a1e["n"],
                "rho_elig": _a1e["spearman_rho"],
                "p_elig": _a1e["spearman_p"],
            }
            if _stim == "M10":
                _rec_entry["per_type"] = {
                    _g: {"rho": _r["spearman_rho"],
                         "p": _r["spearman_p"],
                         "n_runs": _r.get("n_runs")}
                    for _g, _r in _d["A2_per_type"].items()
                }

            # Per-category ρ - recompute from flags + FO
            _rec_scores = np.load(
                RECURRENCE_DIR / _sub / VT / "recurrence_scores.npy"
            )
            _flags = pd.read_csv(
                FLAGS_A4_DIR / PARCELLATION / _sub / VT / "state_flags.csv"
            ).set_index("state")
            _fo_p = _dec_dir / PARCELLATION / _sub / VT / "fractional_occupancy.pkl"
            with open(_fo_p, "rb") as _f:
                _fo = pickle.load(_f)
            _mean_fo = np.stack(list(_fo.values())).mean(axis=0)

            for _cat, _tag in [
                ("run_onset_anchored", "runonset"),
                ("low_confidence", "lowconf"),
            ]:
                _ids_all = [int(_s) for _s in _flags.index
                            if _flags.loc[_s, "summary_category"] == _cat]
                _ids = [_s for _s in _ids_all if _rec_scores[_s] > 0]
                _n = len(_ids)
                _rec_entry[f"n_{_tag}"] = _n
                if _n >= 3:
                    _rho, _p = _stats.spearmanr(_rec_scores[_ids], _mean_fo[_ids])
                    _rec_entry[f"rho_{_tag}"] = float(_rho)
                    _rec_entry[f"p_{_tag}"] = float(_p)
                else:
                    _rec_entry[f"rho_{_tag}"] = None
                    _rec_entry[f"p_{_tag}"] = None

            fig5_data[_stim][_sub] = _rec_entry

    print("=== fig5_data audit ===")
    for _stim in ("M10", "HP", "PP"):
        _vals = [(_s, _r) for _s, _r in fig5_data[_stim].items() if _r is not None]
        for _tag in ("full", "elig", "runonset", "lowconf"):
            _n_sig = sum(1 for _, _r in _vals
                         if _r[f"rho_{_tag}"] is not None and _r[f"p_{_tag}"] < 0.05)
            _n_n = sum(1 for _, _r in _vals if _r[f"rho_{_tag}"] is not None)
            print(f"  {_stim} {_tag:9}: sig {_n_sig}/{_n_n}")
    return (fig5_data,)


@app.cell
def panel_5A_rho_full(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, OUT_FIG5, plt, np,
):
    """Panel 5A: ρ × subject × stimulus (full repertoire)."""
    _stims = ["M10", "HP", "PP"]
    _stim_labels = {"M10": "Movie10", "HP": "Harry Potter", "PP": "Petit Prince"}
    _stim_x = {s: i for i, s in enumerate(_stims)}
    _jitter = np.linspace(-0.22, 0.22, len(SUBJECTS))

    _fig, _ax = plt.subplots(figsize=(4.2, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    for _si, _sub in enumerate(SUBJECTS):
        _x_off = _jitter[_si]
        _marker = SUBJECT_MARKERS[_sub]
        for _stim in _stims:
            _r = fig5_data[_stim].get(_sub)
            if _r is None:
                continue
            _x = _stim_x[_stim] + _x_off
            _y = _r["rho_full"]
            _sig = _r["p_full"] < 0.05
            _kw = dict(marker=_marker, s=36, linewidths=1.1,
                       edgecolors=SUBJECT_NEUTRAL, zorder=3)
            if _sig:
                _ax.scatter(_x, _y, facecolors=SUBJECT_NEUTRAL, **_kw)
            else:
                _ax.scatter(_x, _y, facecolors="white", **_kw)

    _ax.set_xticks(list(_stim_x.values()))
    _ax.set_xticklabels([_stim_labels[s] for s in _stims])
    _ax.set_xlim(-0.6, len(_stims) - 0.4)
    _ax.set_ylim(-0.35, 0.95)
    _ax.set_yticks(np.arange(-0.2, 1.0, 0.2))
    _ax.set_ylabel("Spearman ρ\n(Friends recurrence → stimulus FO)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    # Subject-shape legend - compact horizontal
    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.2)

    _out = OUT_FIG5 / "fig5_A_rho_full"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5B_rho_eligible(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, OUT_FIG5, plt, np,
):
    """Panel 5B: same layout as 5A but eligible-subset ρ."""
    _stims = ["M10", "HP", "PP"]
    _stim_labels = {"M10": "Movie10", "HP": "Harry Potter", "PP": "Petit Prince"}
    _stim_x = {s: i for i, s in enumerate(_stims)}
    _jitter = np.linspace(-0.22, 0.22, len(SUBJECTS))

    _fig, _ax = plt.subplots(figsize=(4.2, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    for _si, _sub in enumerate(SUBJECTS):
        _x_off = _jitter[_si]
        _marker = SUBJECT_MARKERS[_sub]
        for _stim in _stims:
            _r = fig5_data[_stim].get(_sub)
            if _r is None:
                continue
            _x = _stim_x[_stim] + _x_off
            _y = _r["rho_elig"]
            _sig = _r["p_elig"] < 0.05
            _kw = dict(marker=_marker, s=36, linewidths=1.1,
                       edgecolors=SUBJECT_NEUTRAL, zorder=3)
            if _sig:
                _ax.scatter(_x, _y, facecolors=SUBJECT_NEUTRAL, **_kw)
            else:
                _ax.scatter(_x, _y, facecolors="white", **_kw)

    _ax.set_xticks(list(_stim_x.values()))
    _ax.set_xticklabels([_stim_labels[s] for s in _stims])
    _ax.set_xlim(-0.6, len(_stims) - 0.4)
    _ax.set_ylim(-0.35, 0.95)
    _ax.set_yticks(np.arange(-0.2, 1.0, 0.2))
    _ax.set_ylabel("Spearman ρ, eligible subset\n(content-eligible states only)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.2)

    _out = OUT_FIG5 / "fig5_B_rho_eligible"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5C_m10_genre(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, OUT_FIG5, plt, np,
):
    """Panel 5C: M10 per-type ρ (genre breakdown), 4 genres × 6 subjects.

    Genre order: bourne, wolf, figures (social narratives) | life (nature documentary).
    Visual separator between social and documentary to cue the dissociation.
    """
    _genres = ["bourne", "wolf", "figures", "life"]
    _genre_labels = {
        "bourne": "Bourne\n(action)",
        "wolf": "Wolf of Wall St.\n(drama)",
        "figures": "Hidden Figures\n(drama)",
        "life": "Life\n(nature doc)",
    }
    _gx = {g: i for i, g in enumerate(_genres)}
    _jitter = np.linspace(-0.22, 0.22, len(SUBJECTS))

    _fig, _ax = plt.subplots(figsize=(4.6, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    # Shade 'life' background to mark the out-of-category genre
    _ax.axvspan(_gx["life"] - 0.45, _gx["life"] + 0.45,
                color="#F4F4F4", zorder=0)

    for _si, _sub in enumerate(SUBJECTS):
        _r = fig5_data["M10"].get(_sub)
        if _r is None or "per_type" not in _r:
            continue
        _x_off = _jitter[_si]
        _marker = SUBJECT_MARKERS[_sub]
        for _g in _genres:
            _gr = _r["per_type"].get(_g)
            if _gr is None:
                continue
            _x = _gx[_g] + _x_off
            _y = _gr["rho"]
            _sig = _gr["p"] < 0.05
            _kw = dict(marker=_marker, s=36, linewidths=1.1,
                       edgecolors=SUBJECT_NEUTRAL, zorder=3)
            if _sig:
                _ax.scatter(_x, _y, facecolors=SUBJECT_NEUTRAL, **_kw)
            else:
                _ax.scatter(_x, _y, facecolors="white", **_kw)

    _ax.set_xticks(list(_gx.values()))
    _ax.set_xticklabels([_genre_labels[g] for g in _genres])
    _ax.set_xlim(-0.6, len(_genres) - 0.4)
    _ax.set_ylim(-0.35, 0.95)
    _ax.set_yticks(np.arange(-0.2, 1.0, 0.2))
    _ax.set_ylabel("Spearman ρ\n(Friends recurrence → Movie10 FO)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.2)

    _out = OUT_FIG5 / "fig5_C_m10_genre"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5D_crossmodality_lines(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_ACCENT, SUBJECT_NEUTRAL,
    OUT_FIG5, plt, np,
):
    """Panel 5D: cross-modality pattern per subject - all 6 subjects as lines.

    Each subject gets a connecting line across M10 → HP → PP (full ρ).
    Sub-02 in accent color (pattern-breaker: weak M10, strong HP/PP).
    Others in neutral gray, low alpha, to establish the typical
    "decline across stimuli" baseline.
    Subject markers differentiate shapes at each stimulus.
    """
    _stims = ["M10", "HP", "PP"]
    _stim_labels = {"M10": "Movie10\n(audiovisual)",
                    "HP": "Harry Potter\n(text)",
                    "PP": "Petit Prince\n(audio)"}
    _stim_x = {s: i for i, s in enumerate(_stims)}

    _fig, _ax = plt.subplots(figsize=(3.4, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    for _sub in SUBJECTS:
        _xs, _ys = [], []
        for _stim in _stims:
            _r = fig5_data[_stim].get(_sub)
            if _r is None:
                continue
            _xs.append(_stim_x[_stim])
            _ys.append(_r["rho_full"])
        if not _xs:
            continue
        _is_accent = _sub == "sub-02"
        _color = SUBJECT_ACCENT if _is_accent else SUBJECT_NEUTRAL
        _alpha_line = 0.95 if _is_accent else 0.35
        _alpha_mark = 1.0 if _is_accent else 0.55
        _lw = 1.5 if _is_accent else 0.9
        _ms = 56 if _is_accent else 32
        _zo = 5 if _is_accent else 2

        _ax.plot(_xs, _ys, color=_color, linewidth=_lw, alpha=_alpha_line, zorder=_zo)
        _ax.scatter(_xs, _ys, marker=SUBJECT_MARKERS[_sub], s=_ms,
                    facecolors=_color, edgecolors=_color,
                    linewidths=1.1, alpha=_alpha_mark, zorder=_zo + 1)

    _ax.set_xticks(list(_stim_x.values()))
    _ax.set_xticklabels([_stim_labels[s] for s in _stims])
    _ax.set_xlim(-0.3, len(_stims) - 0.7)
    _ax.set_ylim(-0.35, 0.95)
    _ax.set_yticks(np.arange(-0.2, 1.0, 0.2))
    _ax.set_ylabel("Spearman ρ (full repertoire)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="-",
                   color=(SUBJECT_ACCENT if _s == "sub-02" else SUBJECT_NEUTRAL),
                   markerfacecolor=(SUBJECT_ACCENT if _s == "sub-02" else SUBJECT_NEUTRAL),
                   markeredgecolor=(SUBJECT_ACCENT if _s == "sub-02" else SUBJECT_NEUTRAL),
                   markersize=5.5, linewidth=1.1,
                   label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.25)

    _out = OUT_FIG5 / "fig5_D_crossmodality_lines"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5E_rho_runonset(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, OUT_FIG5, plt, np,
):
    """Panel 5E: ρ × subject × stimulus, run-onset-anchored states only.

    Layout mirrors 5A/5B for direct comparison. Sub-04 typically omitted
    (n<3 run-onset-anchored states for M10; HP/PP sub-04 absent anyway).
    """
    _stims = ["M10", "HP", "PP"]
    _stim_labels = {"M10": "Movie10", "HP": "Harry Potter", "PP": "Petit Prince"}
    _stim_x = {s: i for i, s in enumerate(_stims)}
    _jitter = np.linspace(-0.22, 0.22, len(SUBJECTS))

    _fig, _ax = plt.subplots(figsize=(4.2, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    for _si, _sub in enumerate(SUBJECTS):
        _x_off = _jitter[_si]
        _marker = SUBJECT_MARKERS[_sub]
        for _stim in _stims:
            _r = fig5_data[_stim].get(_sub)
            if _r is None or _r["rho_runonset"] is None:
                continue
            _x = _stim_x[_stim] + _x_off
            _y = _r["rho_runonset"]
            _sig = _r["p_runonset"] < 0.05
            _kw = dict(marker=_marker, s=36, linewidths=1.1,
                       edgecolors=SUBJECT_NEUTRAL, zorder=3)
            if _sig:
                _ax.scatter(_x, _y, facecolors=SUBJECT_NEUTRAL, **_kw)
            else:
                _ax.scatter(_x, _y, facecolors="white", **_kw)

    _ax.set_xticks(list(_stim_x.values()))
    _ax.set_xticklabels([_stim_labels[s] for s in _stims])
    _ax.set_xlim(-0.6, len(_stims) - 0.4)
    _ax.set_ylim(-0.75, 0.95)
    _ax.set_yticks(np.arange(-0.6, 1.0, 0.2))
    _ax.set_ylabel("Spearman ρ, run-onset-anchored\n(~2–8 states/subject)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.2)

    _out = OUT_FIG5 / "fig5_E_rho_runonset"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5F_rho_lowconf(
    fig5_data, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, OUT_FIG5, plt, np,
):
    """Panel 5F: ρ × subject × stimulus, low-confidence (sub-HRF) states only.

    Layout mirrors 5A/5B/5E. Wider y-range because low-confidence ρ is
    noisier and sometimes swings negative (e.g., sub-01 PP, sub-05 PP).
    """
    _stims = ["M10", "HP", "PP"]
    _stim_labels = {"M10": "Movie10", "HP": "Harry Potter", "PP": "Petit Prince"}
    _stim_x = {s: i for i, s in enumerate(_stims)}
    _jitter = np.linspace(-0.22, 0.22, len(SUBJECTS))

    _fig, _ax = plt.subplots(figsize=(4.2, 2.8))
    _ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)

    for _si, _sub in enumerate(SUBJECTS):
        _x_off = _jitter[_si]
        _marker = SUBJECT_MARKERS[_sub]
        for _stim in _stims:
            _r = fig5_data[_stim].get(_sub)
            if _r is None or _r["rho_lowconf"] is None:
                continue
            _x = _stim_x[_stim] + _x_off
            _y = _r["rho_lowconf"]
            _sig = _r["p_lowconf"] < 0.05
            _kw = dict(marker=_marker, s=36, linewidths=1.1,
                       edgecolors=SUBJECT_NEUTRAL, zorder=3)
            if _sig:
                _ax.scatter(_x, _y, facecolors=SUBJECT_NEUTRAL, **_kw)
            else:
                _ax.scatter(_x, _y, facecolors="white", **_kw)

    _ax.set_xticks(list(_stim_x.values()))
    _ax.set_xticklabels([_stim_labels[s] for s in _stims])
    _ax.set_xlim(-0.6, len(_stims) - 0.4)
    _ax.set_ylim(-1.0, 1.0)
    _ax.set_yticks(np.arange(-1.0, 1.05, 0.25))
    _ax.set_ylabel("Spearman ρ, low-confidence\n(~5–18 states/subject)", fontsize=9)
    _ax.tick_params(axis="both", labelsize=8)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _ax.legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
               handletextpad=0.2)

    _out = OUT_FIG5 / "fig5_F_rho_lowconf"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


@app.cell
def panel_5G_anchor_decomposition(
    FRIENDS_DEC, M10_DEC, HP_DEC, PP_DEC, FLAGS_A4_DIR,
    PARCELLATION, VT, SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL,
    OUT_FIG5_SUPP, pickle, pd, plt, np,
):
    """Panel 5G (supplementary): run-onset decomposition.

    Early-TR enrichment (first 20% TRs) for each Friends-defined anchoring
    sub-type, evaluated across 5 contexts: Friends-a runs, Friends-b runs,
    Movie10, Harry Potter, Petit Prince. Null = 1.0 (uniform occupancy).

    Sub-types (from 05e_a2 `anchoring_type`):
      - ab-common (run_onset): locked to both a- and b-run starts in Friends
      - a-anchored: locked to a-run starts only
      - b-anchored: locked to b-run starts only

    3 rows × 1 col, shared y-axis, subjects as distinct markers. Subject-mean
    per (anchor, context) computed as mean-across-states of mean-across-runs
    of P(state|early TR) / P(state|run).
    """
    import re as _re

    _contexts = ["Fa", "Fb", "M10", "HP", "PP"]
    _ctx_labels = {"Fa": "Friends\na-runs", "Fb": "Friends\nb-runs",
                   "M10": "Movie10", "HP": "Harry\nPotter", "PP": "Petit\nPrince"}
    _cx = {c: i for i, c in enumerate(_contexts)}
    _anchors = ["run_onset", "a_anchored", "b_anchored"]
    _anchor_labels = {"run_onset": "ab-common\n(run_onset)",
                      "a_anchored": "a-anchored",
                      "b_anchored": "b-anchored"}

    def _is_a_run(rid):
        return rid.endswith("a") or _re.search(r"e\d+a(_|$)", rid) is not None

    def _is_b_run(rid):
        return rid.endswith("b") or _re.search(r"e\d+b(_|$)", rid) is not None

    def _enrich(decoded, state_id, filter_fn=None, frac=0.2):
        _out = []
        for _rid, _seq in decoded.items():
            if filter_fn is not None and not filter_fn(_rid):
                continue
            _seq = np.asarray(_seq)
            _n = len(_seq)
            if _n < 10:
                continue
            _cut = max(1, int(round(_n * frac)))
            _early = (_seq[:_cut] == state_id).mean()
            _whole = (_seq == state_id).mean()
            if _whole <= 0:
                continue
            _out.append(_early / _whole)
        return np.array(_out)

    # Build table: per (sub, anchor, context) → mean-over-states of mean-over-runs
    _rows = []
    for _sub in SUBJECTS:
        _flags = pd.read_csv(
            FLAGS_A4_DIR / PARCELLATION / _sub / VT / "state_flags.csv"
        ).set_index("state")
        _fdec_p = FRIENDS_DEC / PARCELLATION / _sub / "final" / VT / "decoded_states.pkl"
        with open(_fdec_p, "rb") as _f:
            _fdec = pickle.load(_f)
        _cross = {}
        for _name, _dec_dir in [("M10", M10_DEC), ("HP", HP_DEC), ("PP", PP_DEC)]:
            _p = _dec_dir / PARCELLATION / _sub / VT / "decoded_states.pkl"
            if _p.exists():
                with open(_p, "rb") as _f:
                    _cross[_name] = pickle.load(_f)
            else:
                _cross[_name] = None

        for _anchor in _anchors:
            _ids = [int(_s) for _s in _flags.index if bool(_flags.loc[_s, _anchor])]
            if not _ids:
                continue
            def _avg(_dec, _fn=None):
                if _dec is None:
                    return None
                _means = []
                for _st in _ids:
                    _r = _enrich(_dec, _st, filter_fn=_fn)
                    if len(_r):
                        _means.append(_r.mean())
                return float(np.mean(_means)) if _means else None

            _vals = {
                "Fa": _avg(_fdec, _fn=_is_a_run),
                "Fb": _avg(_fdec, _fn=_is_b_run),
                "M10": _avg(_cross["M10"]),
                "HP": _avg(_cross["HP"]),
                "PP": _avg(_cross["PP"]),
            }
            _rows.append({"sub": _sub, "anchor": _anchor,
                          "n_states": len(_ids), **_vals})

    _df = pd.DataFrame(_rows)
    print("=== Panel G enrichment table (subject-mean across states) ===")
    print(_df.to_string(index=False))

    # Plot - 3 rows × 1 col, one row per anchor sub-type
    _fig, _axes = plt.subplots(
        3, 1, figsize=(4.4, 5.4), sharex=True, sharey=True,
    )
    _jitter = np.linspace(-0.18, 0.18, len(SUBJECTS))

    for _row_i, _anchor in enumerate(_anchors):
        _ax = _axes[_row_i]
        _ax.axhline(1.0, color="#BBBBBB", linewidth=0.7, zorder=0)
        _sub_rows = _df[_df["anchor"] == _anchor]
        for _si, _sub in enumerate(SUBJECTS):
            _r = _sub_rows[_sub_rows["sub"] == _sub]
            if _r.empty:
                continue
            _r = _r.iloc[0]
            for _c in _contexts:
                _y = _r[_c]
                if _y is None or (isinstance(_y, float) and np.isnan(_y)):
                    continue
                _ax.scatter(_cx[_c] + _jitter[_si], _y,
                            marker=SUBJECT_MARKERS[_sub], s=36,
                            facecolors=SUBJECT_NEUTRAL,
                            edgecolors=SUBJECT_NEUTRAL, linewidths=1.1,
                            zorder=3)
        _ax.set_ylabel(_anchor_labels[_anchor], fontsize=9)
        _ax.tick_params(axis="both", labelsize=8)
        _ax.set_xlim(-0.5, len(_contexts) - 0.5)
        for _spine in ("top", "right"):
            _ax.spines[_spine].set_visible(False)

    _axes[-1].set_xticks(list(_cx.values()))
    _axes[-1].set_xticklabels([_ctx_labels[c] for c in _contexts])
    # shared ylim - inspect data range + 10% headroom
    _all_vals = _df[_contexts].to_numpy().flatten()
    _all_vals = _all_vals[~pd.isna(_all_vals)].astype(float)
    _ymax = min(4.0, float(np.nanmax(_all_vals)) * 1.08)
    _ymin = max(0.0, float(np.nanmin(_all_vals)) * 0.92)
    for _ax in _axes:
        _ax.set_ylim(_ymin, _ymax)
    # Figure-level axis title so per-row ylabels can just be the anchor sub-type
    _fig.supylabel(
        "Early-TR enrichment (first 20% TRs / run mean)",
        fontsize=9, x=-0.02,
    )

    # Subject-shape legend at top
    _handles = [
        plt.Line2D([0], [0], marker=SUBJECT_MARKERS[_s], linestyle="",
                   markerfacecolor=SUBJECT_NEUTRAL, markeredgecolor=SUBJECT_NEUTRAL,
                   markersize=5.5, label=_s.replace("sub-0", "S"))
        for _s in SUBJECTS
    ]
    _axes[0].legend(handles=_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                    ncol=6, frameon=False, fontsize=7, columnspacing=0.6,
                    handletextpad=0.2)

    _out = OUT_FIG5_SUPP / "fig5_G_anchor_decomposition"
    _fig.savefig(_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    print(f"saved: {_out.name}.{{pdf,png}}")
    return


if __name__ == "__main__":
    app.run()
