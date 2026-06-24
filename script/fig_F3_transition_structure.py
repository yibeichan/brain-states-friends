"""Figure F3 - Transition structure (R3): graph topology + FC-transition coupling.

One marimo notebook per figure (see `2026-05-24_manuscript_version_scope.md`).
Panels are per-cell, saved as separate .pdf + .png + .svg mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

R3 claim: "Transitions are organized by the functional relationships between
states, not random diffusion." Intro §5 commits to three findings -
recurrence assortativity, FC-transition coupling, and network homophily "in
most individuals."

Panel division of labor: Panel A is DESCRIPTIVE - it shows *what* the state
transition structure looks like (the sparse, directed transition repertoire
per subject). Panel B is EXPLANATORY - it quantifies *why* that structure
arises (assortativity, homophily, FC- and MFPT-coupling, each vs its null). A
poses the structure; B explains it. The organization claims live only in B.

Panel plan (chart families distinct per the manuscript-figure plot-type-variety
rule; F1 already used bar / flow-Sankey / brain-map, so F2 avoids those within
this figure):

Intended composite (current manuscript layout): TWO rows. Top row = Panel A only
(the 1×6 graph strip, full width). Bottom row = Panel B only. Legends ship as
separate files.

| Panel | Content | Chart family | Source files | Output |
|---|---|---|---|---|
| A | Per-subject directed state transition matrices, 2×3 small-multiples (full square; states ordered by dominant network then recurrence; self-transitions excluded; single-hue Greys cells; network margin ribbon). Separate legend files. | heatmap | 06b transition_graph.graphml (×6 subj) | fig3_A_transition_matrices.{png,svg} + fig3_A_legend_networks.{png,svg} + fig3_A_legend_prob.{png,svg} |
| B | Cross-subject consistency forest (4 metrics × 6 subjects; 7in wide to match Panel A; per-subject marker shapes, neutral color; assortativity CI; per-row null line) + standalone subject-marker legend | point-based 1D | 06b transition_structure_summary.json (all subjects) | fig3_B_cross_subject_consistency.{png,svg} + fig3_B_legend_markers.{png,svg} |

Design notes (2026-05-26 revision):
  * Panel A history: MFPT landscape heatmap (told an R1-reachability story) ->
    network-colored force-directed graphs (1×6 strip; hairballs at print size,
    transition structure illegible) -> 2×3 directed state transition matrices
    (current). The matrices show the actual transition repertoire directly and
    read at print size where the graphs did not. Six matrices, not one pooled
    matrix: states are subject-specific and transitions are within-subject, so a
    pooled matrix is undefined. Network palette is colorblind-safe Okabe–Ito
    (plot_style.NETWORK_COLORS); only ≤7 cortical networks appear as dominant
    (Vis/SomMot/DorsAttn/SalVentAttn/Limbic/Cont/Default). Legends are separate
    files per the assembly workflow (the matrix grid carries none). Panel A is
    descriptive (the transition phenomenon); the organization inferences are
    Panel B's.
  * Old Panel B pooled FC-transition density was removed from the main figure:
    it was display-only, while the inferential statistics are the per-subject
    quantities summarized in the consistency forest. Keep this omission aligned
    with docs/manuscript/figure_captions.md.
  * Panel B: subjects now use SUBJECT_MARKERS shapes (consistent with F4) in a
    single neutral color. sub-05's homophily-null (ratio 0.980, p=0.51) is no
    longer accented - it is visible as the one homophily marker at/below the null
    line and noted in the caption. No per-panel legend (marker→subject key is
    shared project-wide; cross-reference F4). Tightened top padding.

Source-truth audit (2026-05-26, reproduced from raw arrays - see session log):
  * Panel B: FC-Mantel ρ 0.326–0.551, MFPT-FC ρ 0.405–0.680, assortativity
    0.111–0.297, homophily ratio 0.980–1.989 (sub-05 null, p=0.51).

Run:
    marimo edit script/fig_F3_transition_structure.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import (
        NETWORK_COLORS,
        NETWORK_ORDER,
        SUBJECT_MARKERS,
        SUBJECT_NEUTRAL,
        apply_publication_style,
        display_network,
    )

    apply_publication_style()

    return (
        Path, json, np, nx, os, plt,
        NETWORK_COLORS, NETWORK_ORDER, SUBJECT_MARKERS, SUBJECT_NEUTRAL,
        display_network,
    )


@app.cell
def config(Path, os):
    """Paths, subject list, output dir. One config cell shared across panels."""
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]

    TRANS_DIR = SCRATCH_DIR / "output" / "06b_transition_structure" / PARCELLATION

    OUT_F3 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig3"
    OUT_F3.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR, PARCELLATION, VT, SUBJECTS,
        TRANS_DIR, OUT_F3,
    )


@app.cell
def load_summaries(SUBJECTS, TRANS_DIR, VT, json):
    """06b transition_structure_summary.json per subject (for Panel B forest)."""
    trans_summary = {}
    for _sub in SUBJECTS:
        with open(TRANS_DIR / _sub / VT / "transition_structure_summary.json") as _f:
            trans_summary[_sub] = json.load(_f)
    return (trans_summary,)


@app.cell
def panel_A_transition_matrices(
    SUBJECTS, TRANS_DIR, VT, OUT_F3, nx, plt, np,
    NETWORK_COLORS, NETWORK_ORDER, display_network,
):
    """Panel A - per-subject state transition matrices, 2×3 small-multiples.

    A is DESCRIPTIVE: it shows *what* the transition structure looks like - the
    actual transition repertoire each subject's sequence uses. Panel B is
    EXPLANATORY: it quantifies *why* the structure is shaped this way (the
    functional-organization metrics). A poses the structure; B explains it.

    One directed state-by-state transition matrix per subject (06b
    transition_graph.graphml: model transmat_, self-loops removed, edges
    thresholded at P >= 0.01). Row = from-state, col = to-state; the matrix is
    asymmetric (only 84% of edges reciprocated), so the full square is shown.
    The matrices show transitions are sparse - each state connects to only a few
    others, far from an all-to-all (random-diffusion) matrix. States are ordered
    by dominant network (canonical NETWORK_ORDER) then recurrence (descending)
    for legibility. Cells use a single-hue (Greys) shared log scale so the only
    categorical color lives on the network ribbon; zero / sub-threshold cells are
    white. Self-transitions are excluded (a sticky-HMM diagonal would swamp the
    repertoire).

    A network-colored ribbon (grouped spans) on the top + left margins labels
    each state's dominant network. No organization claim is made here - the
    assortativity, homophily, and FC/MFPT-coupling inferences live entirely in
    Panel B.

    Six matrices, not one pooled matrix: states are subject-specific (each
    subject's own HMM, no cross-subject alignment) and transitions exist only
    within a subject's sequence, so a pooled transition matrix is undefined.

    Legends are SEPARATE files (fig3_A_legend_networks, fig3_A_legend_prob).
    """
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable

    _net_index = {_n: _i for _i, _n in enumerate(NETWORK_ORDER)}

    # Pass 1: build each subject's network-ordered DIRECTED matrix (row = from,
    # col = to; the matrix is asymmetric, so the full square is shown). Collect
    # off-diagonal weights for a shared log scale + the networks present.
    _data = {}
    _present_nets = set()
    _all_w = []
    for _sub in SUBJECTS:
        _G = nx.read_graphml(TRANS_DIR / _sub / VT / "transition_graph.graphml")
        _nodes = list(_G.nodes())
        _nets = {_n: _G.nodes[_n].get("dominant_network", "Unknown") for _n in _nodes}
        _recs = {_n: float(_G.nodes[_n].get("recurrence_score", 0.0)) for _n in _nodes}
        _order = sorted(_nodes, key=lambda _n: (_net_index.get(_nets[_n], 99), -_recs[_n]))
        _idx = {_n: _i for _i, _n in enumerate(_order)}
        _N = len(_order)
        _M = np.zeros((_N, _N))
        for _u, _v, _d in _G.edges(data=True):
            if _u == _v:
                continue
            _M[_idx[_u], _idx[_v]] = max(_M[_idx[_u], _idx[_v]], float(_d.get("weight", 0.0)))
        _ordered_nets = [_nets[_n] for _n in _order]
        _present_nets.update(_ordered_nets)
        _all_w.append(_M[_M > 0])
        _data[_sub] = (_M, _ordered_nets)
        print(f"  Panel A {_sub}: {_N} states, {int((_M > 0).sum())} directed edges")

    _all_w = np.concatenate(_all_w)
    # Tight log scale (5/95) so the mid-range transitions read; the few
    # strongest cells saturate to black (colorbar marked "extend=both").
    _vmin = max(float(np.percentile(_all_w, 5)), 1e-3)
    _vmax = float(np.percentile(_all_w, 95))
    _cmap = plt.get_cmap("Greys").copy()
    _cmap.set_bad("white")
    _norm = LogNorm(vmin=_vmin, vmax=_vmax)

    def _groups(_seq):
        _g, _s = [], 0
        for _i in range(1, len(_seq) + 1):
            if _i == len(_seq) or _seq[_i] != _seq[_s]:
                _g.append((_seq[_s], _s, _i))
                _s = _i
        return _g

    # Pass 2: render 2×3, full directed matrix, with one network ribbon (grouped
    # spans) on the top + left margins.
    _fig, _axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    for _ax, _sub in zip(_axes.ravel(), SUBJECTS):
        _M, _onets = _data[_sub]
        _N = _M.shape[0]
        _rib = max(1.0, 0.035 * _N)
        _Mm = np.ma.masked_where(_M <= 0, _M)
        _ax.imshow(_Mm, cmap=_cmap, norm=_norm, interpolation="nearest",
                   aspect="equal", extent=[0, _N, _N, 0], zorder=2)
        for _net, _s, _e in _groups(_onets):
            _c = NETWORK_COLORS.get(_net, "#BBBBBB")
            _ax.add_patch(Rectangle((_s, -_rib), _e - _s, _rib, color=_c,
                                    lw=0, zorder=3, clip_on=False))
            _ax.add_patch(Rectangle((-_rib, _s), _rib, _e - _s, color=_c,
                                    lw=0, zorder=3, clip_on=False))
        for _net, _s, _e in _groups(_onets)[1:]:
            _ax.axvline(_s, color="white", lw=0.5, zorder=2.5)
            _ax.axhline(_s, color="white", lw=0.5, zorder=2.5)
        _ax.set_xlim(-_rib, _N)
        _ax.set_ylim(_N, -_rib)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_title(_sub, fontsize=9, pad=2)
        for _spine in _ax.spines.values():
            _spine.set_visible(False)

    _left, _bottom, _top, _right = 0.045, 0.045, 0.94, 0.99
    _fig.subplots_adjust(left=_left, right=_right, bottom=_bottom, top=_top,
                         wspace=0.08, hspace=0.06)
    # Figure-level direction labels, centered over the whole grid and placed
    # tight against it: cell (row i, col j) = P(state i -> state j), so
    # rows = from, columns = to.
    _y_c = 0.5 * (_bottom + _top)
    _x_c = 0.5 * (_left + _right)
    _fig.text(0.006, _y_c, "From state", rotation=90, ha="left", va="center",
              fontsize=10)
    _fig.text(_x_c, 0.006, "To state", ha="center", va="bottom", fontsize=10)
    _stem = OUT_F3 / "fig3_A_transition_matrices"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)

    # ── Separate legend files ──────────────────────────────────────────────
    # (1) Network color key (networks present, canonical order), one column.
    _present = [_net for _net in NETWORK_ORDER if _net in _present_nets]
    _net_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NETWORK_COLORS.get(_net, "#BBBBBB"),
                      edgecolor="#666666", linewidth=0.4, label=display_network(_net))
        for _net in _present
    ]
    _figL, _axL = plt.subplots(figsize=(1.5, max(0.5, 0.26 * len(_present))))
    _axL.set_axis_off()
    _axL.legend(handles=_net_handles, loc="center", frameon=False, fontsize=7,
                ncol=1, handlelength=1.0, handletextpad=0.4)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figL)

    # (2) Shared transition-probability colorbar.
    _sm = ScalarMappable(norm=_norm, cmap=_cmap)
    _sm.set_array([])
    _figC, _axC = plt.subplots(figsize=(2.0, 0.34))
    _cb = _figC.colorbar(_sm, cax=_axC, orientation="horizontal", extend="both")
    _cb.set_label("Transition probability", fontsize=7)
    _cb.ax.tick_params(labelsize=6)
    _figC.savefig(OUT_F3 / "fig3_A_legend_prob.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figC.savefig(OUT_F3 / "fig3_A_legend_prob.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figC)
    print(f"saved: legends (networks={_present})")
    return


@app.cell
def panel_B_cross_subject_consistency(
    SUBJECTS, trans_summary, OUT_F3, SUBJECT_NEUTRAL, SUBJECT_MARKERS, plt, np,
):
    """Panel B - cross-subject consistency forest (4 metrics × 6 subjects).

    Four R3 metrics, one horizontal track each (own x-scale; small-multiple
    rows sharing the subject styling):
      1. FC-transition coupling (Spearman ρ)   null = 0
      2. MFPT-FC coupling (Spearman ρ)          null = 0
      3. Recurrence assortativity (+ bootstrap CI whiskers)  null = 0
      4. Network homophily (within/between ratio)            null = 1

    Each subject is a distinct SUBJECT_MARKERS shape in a single neutral color
    (the project-wide subject key, consistent with Fig 5 - so no per-panel
    legend is needed; the shapes also de-overlap clustered points). Color is NOT
    spent on subject identity here: marker shape carries it, leaving the panel
    uncluttered. sub-05's homophily null (ratio 0.980, p=0.51 - the intro's "in
    most individuals" caveat) is no longer accented; it reads as the one
    homophily marker sitting at/below the dashed null line, and is named in the
    caption. The point estimate per row is the message; which marker is which
    individual is secondary (recoverable via the Fig 5 marker key).
    """
    # (row label, key, null value, per-row x-axis label). Each row has its own
    # x-scale - the metrics are not commensurable - so each carries its own
    # axis label rather than a single shared "effect size" caption.
    # y-label = the variable; x-label = the statistic (rows 3 & 4 are the SAME
    # variable, functional connectivity, differing only in what it is correlated
    # against - transition probability vs MFPT). Spearman ρ is symmetric and
    # non-causal, so no directional ("A→B") naming. MFPT = mean first passage
    # time (defined in caption).
    _metrics = [
        ("Recurrence", "assort", 0.0, "Assortativity coefficient r (transition)"),
        ("Network\nhomophily", "homophily", 1.0, "Within / between ratio (transition)"),
        ("Functional\nconnectivity", "fc", 0.0, "Spearman ρ (transition)"),
        ("Functional\nconnectivity", "mfpt", 0.0, "Spearman ρ (MFPT)"),
    ]

    def _val(_sub, _key):
        _d = trans_summary[_sub]
        if _key == "fc":
            return _d["A3_fc_transition"]["rho"], None
        if _key == "mfpt":
            return _d["A4_mfpt_fc"]["rho"], None
        if _key == "assort":
            _a = _d["A3_assortativity"]
            return _a["point_estimate"], _a["bootstrap_ci"]
        if _key == "homophily":
            return _d["A3_network_homophily"]["ratio"], None
        raise KeyError(_key)

    _n_rows = len(_metrics)
    _fig, _axes = plt.subplots(
        _n_rows, 1, figsize=(7.0, 3.3), sharex=False,   # match Panel A width
    )
    _rng = np.random.default_rng(20260526)

    # Rows 3 & 4 are the SAME statistic (Spearman ρ) on the SAME variable (FC),
    # so they share one x-scale; the structural metrics (rows 1, 2) keep their
    # own incommensurable scales.
    _rho_keys = {"fc", "mfpt"}
    _rho_vals = [_val(_s, _k)[0] for _k in _rho_keys for _s in SUBJECTS]
    _rlo = min(min(_rho_vals), 0.0)
    _rhi = max(max(_rho_vals), 0.0)
    _rpad = 0.10 * (_rhi - _rlo + 1e-9)
    _rho_xlim = (_rlo - _rpad, _rhi + _rpad)

    for _ax, (_label, _key, _null, _xlabel) in zip(_axes, _metrics):
        _vals = []
        for _j, _sub in enumerate(SUBJECTS):
            _v, _ci = _val(_sub, _key)
            _vals.append(_v)
            # NOTE: _yj is RANDOM vertical jitter with NO meaning - the y-axis
            # has no scale. It only de-overlaps the 6 subject markers and the
            # assortativity CI whiskers so they don't pile onto one line.
            _yj = _rng.uniform(-0.18, 0.18)
            if _ci is not None:
                _ax.plot([_ci[0], _ci[1]], [_yj, _yj], color=SUBJECT_NEUTRAL,
                         lw=1.0, alpha=0.5, zorder=2, solid_capstyle="round")
            _ax.scatter(
                _v, _yj, s=42, color=SUBJECT_NEUTRAL,
                marker=SUBJECT_MARKERS[_sub], edgecolor="white", linewidth=0.55,
                zorder=3, alpha=0.95,
            )
        # null reference line
        _ax.axvline(_null, color="#999999", lw=1.0, ls="--", zorder=1)
        _ax.set_ylim(-0.5, 0.5)
        _ax.set_yticks([])
        # The two FC rows share one grouped label drawn below; blank them here.
        _ax.set_ylabel("" if _key in _rho_keys else _label,
                       rotation=0, ha="right", va="center", fontsize=7.5,
                       labelpad=2)
        _ax.tick_params(axis="x", labelsize=6.5)
        _ax.set_xlabel(_xlabel, fontsize=7.5)
        for _s in ("top", "right", "left"):
            _ax.spines[_s].set_visible(False)
        if _key in _rho_keys:
            _ax.set_xlim(*_rho_xlim)          # shared ρ scale (rows 3 & 4)
        else:
            _lo = min(min(_vals), _null); _hi = max(max(_vals), _null)
            _pad = 0.10 * (_hi - _lo + 1e-9)
            _ax.set_xlim(_lo - _pad, _hi + _pad)

    _fig.subplots_adjust(left=0.135, right=0.99, bottom=0.13, top=0.98, hspace=0.95)

    # Group the two functional-connectivity rows (3 & 4) under one label + a
    # left square bracket, since they are the same variable. Positions are in
    # figure fractions from the axes' final positions (robust to tight bbox).
    import matplotlib.lines as _ml
    _ptop = _axes[2].get_position(); _pbot = _axes[3].get_position()
    _y0, _y1, _yc = _pbot.y0, _ptop.y1, 0.5 * (_pbot.y0 + _ptop.y1)
    _xb, _tk = 0.085, 0.007
    for _xs, _ys in (([_xb, _xb], [_y0, _y1]),
                     ([_xb, _xb + _tk], [_y1, _y1]),
                     ([_xb, _xb + _tk], [_y0, _y0])):
        _fig.add_artist(_ml.Line2D(_xs, _ys, color="0.4", lw=1.0))
    _fig.text(_xb - 0.006, _yc, "Functional\nconnectivity",
              rotation=90, ha="right", va="center", fontsize=7.5)

    _stem = OUT_F3 / "fig3_B_cross_subject_consistency"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)

    # Subject -> marker key (shared project-wide; same shapes as Figs 1 & 5),
    # neutral color, one column.
    _mk_handles = [
        plt.Line2D([], [], marker=SUBJECT_MARKERS[_sub], ls="none",
                   color=SUBJECT_NEUTRAL, mec="white", mew=0.5, ms=6, label=_sub)
        for _sub in SUBJECTS
    ]
    _figM, _axM = plt.subplots(figsize=(1.15, 1.9))
    _axM.set_axis_off()
    _axM.legend(handles=_mk_handles, loc="center", frameon=False, fontsize=7,
                ncol=1, handlelength=1.0, handletextpad=0.4)
    _figM.savefig(OUT_F3 / "fig3_B_legend_markers.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figM.savefig(OUT_F3 / "fig3_B_legend_markers.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figM)
    print("saved: fig3_B_legend_markers.png (+ .svg)")
    return


if __name__ == "__main__":
    app.run()
