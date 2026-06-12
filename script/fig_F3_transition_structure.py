"""Figure F3 - Transition structure (R3): graph topology + FC-transition coupling.

One marimo notebook per figure (see `2026-05-24_manuscript_version_scope.md`).
Panels are per-cell, saved as separate .pdf + .png mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

R3 claim: "Transitions are organized by the functional relationships between
states, not random diffusion." Intro §5 commits to three findings -
recurrence assortativity, FC-transition coupling, and network homophily "in
most individuals."

Panel plan (chart families distinct per the manuscript-figure plot-type-variety
rule; F1 already used bar / flow-Sankey / brain-map, so F2 avoids those within
this figure):

Intended composite (user layout, 2026-05-26): TWO rows. Top row = Panel A only
(the 1×6 graph strip, full width). Bottom row = Panel B + Panel C side by side.
Panel A width ≈ B + C; heights need not match. Legends ship as separate files.

| Panel | Content | Chart family | Source files | Output |
|---|---|---|---|---|
| A | Per-subject transition graphs, 1×6 small-multiples row (one mini graph per subject; nodes colored by dominant network, sized by recurrence; force-directed layout). Separate legend files. | network/graph | 06b transition_graph.graphml (×6 subj) | fig3_A_transition_graphs.{pdf,png} + fig3_A_legend_networks.* + fig3_A_legend_recurrence.* |
| B | Pooled FC→transition 2D density (all 6 subjects, 5,723 state pairs; RANK axes; hexbin + binned-median trend). No in-plot text/legend - ρ range, n, trend all in caption. | distribution (2D density) | 06a transition_probabilities.npy + 05f rv (×6 subj) | fig3_B_fc_transition_density.{pdf,png} |
| C | Cross-subject consistency forest (4 metrics × 6 subjects; per-subject marker shapes, neutral color, no accent, no legend; assortativity CI; per-row null line) | point-based 1D | 06b transition_structure_summary.json (all subjects) | fig3_C_cross_subject_consistency.{pdf,png} |

Design notes (2026-05-26 revision):
  * Panel A: the MFPT landscape heatmap was the original design but told an
    R1-reachability story; replaced by network-colored transition graphs (show
    actual transitions + partial network clustering). Now a 1×6 per-subject
    strip (not a single exemplar): states are subject-specific and transitions
    are within-subject, so a single pooled/connected graph is undefined - the
    honest cross-subject object is six independent graphs. Network palette is
    colorblind-safe Okabe–Ito (plot_style.NETWORK_COLORS) - only ≤7 cortical
    networks ever appear as dominant nodes (Vis/SomMot/DorsAttn/SalVentAttn/
    Limbic/Cont/Default), so 7 CB-distinct hues suffice. Legends are separate
    files per the assembly workflow (graph strip carries none).
  * Panel B: was a single-exemplar FC-transition scatter; now a *pooled* 2D
    density over all 6 subjects' pairs on RANK axes (distribution family,
    distinct from A's graph and C's point-1D). Pooling is for-display only - the
    per-individual ρ is the inferential statistic and lives in Panel C. ρ range
    (0.33–0.55), n, and the trend definition are all in the caption, NOT in the
    plot. Rank axes are used because the raw RV/transition values are
    pathologically concentrated (a linear–linear density collapses to one
    corner); 407 of 5,723 pairs have T_sym=0 and form the bottom rank band.
  * Panel C: subjects now use SUBJECT_MARKERS shapes (consistent with F4) in a
    single neutral color. sub-05's homophily-null (ratio 0.980, p=0.51) is no
    longer accented - it is visible as the one homophily marker at/below the null
    line and noted in the caption. No per-panel legend (marker→subject key is
    shared project-wide; cross-reference F4). Tightened top padding.

Source-truth audit (2026-05-26, reproduced from raw arrays - see session log):
  * Panel B pooled: 5,723 pairs total, 407 with T_sym=0; per-subject ρ reproduces
    JSON exactly (0.417/0.483/0.326/0.551/0.393/0.434 → range 0.33–0.55).
  * Panel C: FC-Mantel ρ 0.326–0.551, MFPT-FC ρ 0.405–0.680, assortativity
    0.111–0.297, homophily ratio 0.980–1.989 (sub-05 null, p=0.51).

Run:
    marimo edit script/fig_F2_transition_structure.py
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
    import networkx as nx
    import numpy as np
    from dotenv import load_dotenv
    from scipy import stats as sp_stats

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
        Path, json, np, nx, os, pickle, plt, sp_stats,
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
    EXEMPLAR = "sub-01"  # exemplar subject for the single-subject panels A & B

    RECUR_DIR = SCRATCH_DIR / "output" / "05a_recurrence_analysis" / PARCELLATION
    FC_DIR    = SCRATCH_DIR / "output" / "05f_state_fc"            / PARCELLATION
    DYN_DIR   = SCRATCH_DIR / "output" / "06a_state_temp_dynamics" / PARCELLATION
    TRANS_DIR = SCRATCH_DIR / "output" / "06b_transition_structure" / PARCELLATION

    OUT_F3 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig3"
    OUT_F3.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR, PARCELLATION, VT, SUBJECTS, EXEMPLAR,
        RECUR_DIR, FC_DIR, DYN_DIR, TRANS_DIR, OUT_F3,
    )


@app.cell
def load_active_states(SUBJECTS, RECUR_DIR, VT, json, np):
    """Active states + recurrence scores per subject (active := recurrence>0).

    This reproduces 06b's `active_states` construction exactly so that array
    rows/cols downstream (MFPT, FC, transition) align the same way 06b aligned
    them when it computed the reported statistics.
    """
    active_states = {}
    recurrence_scores = {}
    for _sub in SUBJECTS:
        with open(RECUR_DIR / _sub / VT / "recurrence_summary.json") as _f:
            _d = json.load(_f)
        _rs = np.array(_d["recurrence_scores"])
        recurrence_scores[_sub] = _rs
        active_states[_sub] = [i for i in range(_d["n_states"]) if _rs[i] > 0]
    print({_s: len(_a) for _s, _a in active_states.items()})
    return active_states, recurrence_scores


@app.cell
def load_summaries(SUBJECTS, TRANS_DIR, VT, json):
    """06b transition_structure_summary.json per subject (for Panel C forest)."""
    trans_summary = {}
    for _sub in SUBJECTS:
        with open(TRANS_DIR / _sub / VT / "transition_structure_summary.json") as _f:
            trans_summary[_sub] = json.load(_f)
    return (trans_summary,)


@app.cell
def panel_A_transition_graphs(
    SUBJECTS, TRANS_DIR, VT, OUT_F3, nx, plt, np,
    NETWORK_COLORS, NETWORK_ORDER, display_network,
):
    """Panel A - per-subject transition graphs, 1×6 small-multiples row.

    One mini directed-transition graph per subject (06b transition_graph.graphml:
    model transmat_, self-loops removed, edges thresholded at P >= 0.01), drawn
    as a 1-row × 6-column strip so the cohort spans the full top row of Fig 3
    (width ≈ B + C). Each node is one active state:
      - color = dominant network (CB-safe NETWORK_COLORS)
      - size  = recurrence score (area ∝ recurrence; SAME scaling across all six)
    Edges = transition probability (width + alpha ∝ weight, normalized PER subject
    so each graph is individually legible). Layout = force-directed spring
    (weighted by transition probability, fixed seed), so states the sequence
    moves between most cluster together.

    Why six graphs, not one pooled graph: states are subject-specific (each
    subject has its own HMM, states are NOT aligned across subjects) and
    transitions exist only within a subject's own sequence - there is no
    cross-subject edge and no pooled transition matrix. A single connected graph
    is therefore undefined; the honest cross-subject object is six independent
    graphs. Their shared signature - same-network nodes pulling into partial
    local groups - is the visual counterpart of the network-homophily column in
    Panel C (ratios 1.55–1.99 in five of six subjects).

    Edges are UNDIRECTED: ~84% of directed edges are reciprocated, so arrowheads
    add clutter without information; net directionality is a supplementary result.

    Legends are emitted as SEPARATE files (fig3_A_legend_networks,
    fig3_A_legend_recurrence) per the user's assembly workflow - the graph strip
    itself carries no legend.
    """
    _present_nets = set()
    _fig, _axes = plt.subplots(1, len(SUBJECTS), figsize=(7.8, 1.55))
    for _ax, _sub in zip(_axes, SUBJECTS):
        _G = nx.read_graphml(TRANS_DIR / _sub / VT / "transition_graph.graphml")
        _U = nx.Graph()
        for _u, _v, _d in _G.edges(data=True):
            _w = float(_d.get("weight", 0.0))
            if _U.has_edge(_u, _v):
                _U[_u][_v]["weight"] = max(_U[_u][_v]["weight"], _w)
            else:
                _U.add_edge(_u, _v, weight=_w)
        for _n in _G.nodes():
            _U.add_node(_n)

        _nets = {_n: _G.nodes[_n].get("dominant_network", "Unknown")
                 for _n in _G.nodes()}
        _recs = {_n: float(_G.nodes[_n].get("recurrence_score", 0.0))
                 for _n in _G.nodes()}
        _present_nets.update(_nets.values())

        _pos = nx.spring_layout(_U, weight="weight", k=0.55, iterations=400, seed=7)
        _nodelist = list(_G.nodes())
        _node_colors = [NETWORK_COLORS.get(_nets[_n], "#BBBBBB") for _n in _nodelist]
        _node_sizes = [5 + 90 * _recs[_n] for _n in _nodelist]  # shared scaling

        _edgelist = list(_U.edges(data=True))
        _ew = np.array([_d["weight"] for *_e, _d in _edgelist])
        _wmax = _ew.max() if _ew.size else 1.0
        _edge_widths = 0.11 + 0.85 * (_ew / _wmax)
        _edge_alphas = np.clip(0.12 + 0.6 * (_ew / _wmax), 0.12, 0.70)
        for (_u, _v, _d), _lw, _al in zip(_edgelist, _edge_widths, _edge_alphas):
            _ax.plot([_pos[_u][0], _pos[_v][0]], [_pos[_u][1], _pos[_v][1]],
                     color="#222222", lw=_lw, alpha=_al, zorder=1,
                     solid_capstyle="round")
        nx.draw_networkx_nodes(
            _G, _pos, nodelist=_nodelist, node_color=_node_colors,
            node_size=_node_sizes, edgecolors="white", linewidths=0.4, ax=_ax,
        )
        _ax.set_axis_off()
        _ax.margins(0.08)
        _ax.set_title(_sub.replace("sub-", "S"), fontsize=7.5, pad=2)
        print(f"  Panel A {_sub}: {_G.number_of_nodes()} nodes, "
              f"{_G.number_of_edges()} edges")

    _fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.90, wspace=0.04)
    _stem = OUT_F3 / "fig3_A_transition_graphs"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem}.pdf (+ .png, .svg)")
    plt.close(_fig)

    # ── Separate legend files ──────────────────────────────────────────────
    # (1) Networks present across the six graphs, in canonical order.
    _present = [_net for _net in NETWORK_ORDER if _net in _present_nets]
    _net_handles = [
        plt.Line2D([], [], marker="o", ls="none",
                   color=NETWORK_COLORS.get(_net, "#BBBBBB"),
                   mec="white", mew=0.4, ms=6.5, label=display_network(_net))
        for _net in _present
    ]
    _figL, _axL = plt.subplots(figsize=(5.2, 0.4))
    _axL.set_axis_off()
    _axL.legend(handles=_net_handles, loc="center", frameon=False, fontsize=7,
                ncol=len(_present), handletextpad=0.25, columnspacing=0.8)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.pdf", bbox_inches="tight",
                  pad_inches=0.02)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figL)

    # (2) Recurrence size key (3 reference dots; same area scaling as the graphs).
    _size_handles = [
        plt.Line2D([], [], marker="o", ls="none", color="#BBBBBB",
                   mec="white", mew=0.5, ms=np.sqrt(5 + 90 * _r) * 0.9,
                   label=f"{_r:.2f}")
        for _r in [0.2, 0.5, 0.85]
    ]
    _figR, _axR = plt.subplots(figsize=(2.0, 0.5))
    _axR.set_axis_off()
    _axR.legend(handles=_size_handles, loc="center", frameon=False, fontsize=7,
                ncol=3, title="Recurrence", title_fontsize=7,
                handletextpad=0.4, columnspacing=1.0, borderpad=0.3)
    _figR.savefig(OUT_F3 / "fig3_A_legend_recurrence.pdf", bbox_inches="tight",
                  pad_inches=0.02)
    _figR.savefig(OUT_F3 / "fig3_A_legend_recurrence.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figR.savefig(OUT_F3 / "fig3_A_legend_recurrence.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figR)
    print(f"saved: legends (networks={_present})")
    return


@app.cell
def panel_B_fc_transition_density(
    SUBJECTS, FC_DIR, DYN_DIR, TRANS_DIR, VT, active_states, OUT_F3,
    json, plt, np, sp_stats,
):
    """Panel B - pooled FC→transition 2D density (all 6 subjects).

    One hexbin over every active-state pair from all six subjects (upper
    triangle, masked exactly as 06b's `test_fc_transition_correlation`):
    x = FC similarity (RV coefficient between two states' connectivity profiles),
    y = symmetrized transition probability (P_ij + P_ji)/2. ~5,700 pairs total.

    Pooling is FOR DISPLAY ONLY. The inferential statistic is the per-individual
    Spearman ρ (R3 is a per-subject claim) - reported in Panel C and summarized
    here as a *range*, never as a pooled ρ (no JSON reports one, and pooling
    across subjects with different repertoires would mix individuals). We verify
    each subject's reproduced ρ against its 06b JSON before pooling.

    Distribution family (hexbin density), distinct from Panel A's graph and
    Panel C's point-1D - satisfies the plot-type-variety rule.

    LINEAR y-axis (not log): 407 of the 5,723 pooled pairs have y=0 (no observed
    transition despite shared FC). A log axis would silently drop them, breaking
    the annotated pair count (the skill's annotation-match trap). The hexbin
    count colorscale is log (bins='log') so the dense low-transition band does
    not flatten the rest to one shade.
    """
    _x_all, _y_all, _rhos = [], [], []
    for _sub in SUBJECTS:
        _active = np.array(active_states[_sub])
        _n = len(_active)
        _Pe = np.load(DYN_DIR / _sub / VT / "transition_probabilities.npy")
        _rv = np.load(FC_DIR / _sub / VT / "fc_similarity_corr_rv.npy")
        _Psub = _Pe[np.ix_(_active, _active)].copy()
        _rvsub = _rv[np.ix_(_active, _active)].copy()
        _T = (_Psub + _Psub.T) / 2.0
        np.fill_diagonal(_T, 0.0)
        np.fill_diagonal(_rvsub, 0.0)
        _tri = np.triu_indices(_n, k=1)
        _t_full = _T[_tri]
        _rv_full = _rvsub[_tri]
        _mask = ((_t_full > 1e-10) | (_rv_full > 1e-10)) & ~np.isnan(_rv_full)
        _xs = _rv_full[_mask]
        _ys = _t_full[_mask]
        _rho, _ = sp_stats.spearmanr(_xs, _ys)
        with open(TRANS_DIR / _sub / VT / "fc_transition_correlation.json") as _f:
            _j = json.load(_f)
        assert abs(_rho - _j["rho"]) < 1e-3 and _mask.sum() == _j["n_pairs"], (
            f"{_sub}: reproduced rho/n ({_rho:.4f}/{_mask.sum()}) != JSON "
            f"({_j['rho']:.4f}/{_j['n_pairs']})"
        )
        _x_all.append(_xs)
        _y_all.append(_ys)
        _rhos.append(_rho)

    _x_raw = np.concatenate(_x_all)
    _y_raw = np.concatenate(_y_all)
    _rmin, _rmax = min(_rhos), max(_rhos)
    print(f"Panel B pooled: n_pairs={_x_raw.size}, y=0 count={(_y_raw <= 1e-10).sum()}, "
          f"per-subject rho range {_rmin:.3f}-{_rmax:.3f}")

    # Display on RANK (percentile) axes - Spearman is a rank correlation, and the
    # raw RV/transition values are pathologically concentrated (RV near 1.0,
    # transition prob near 0 with 407 exact zeros) so a linear-linear density
    # collapses into one corner. Percentile ranks spread both axes uniformly, so
    # the monotone relationship the ρ measures reads as a diagonal density band.
    # Ties (the 407 zero-transition pairs) get averaged ranks → they form a
    # visible horizontal stripe at the bottom rather than vanishing.
    _x = sp_stats.rankdata(_x_raw) / _x_raw.size
    _y = sp_stats.rankdata(_y_raw) / _y_raw.size

    _fig, _ax = plt.subplots(figsize=(3.4, 3.1))
    _hb = _ax.hexbin(
        _x, _y, gridsize=30, cmap="Greys", bins="log",
        mincnt=1, linewidths=0.15, edgecolors="white",
    )
    _cb = _fig.colorbar(_hb, ax=_ax, fraction=0.046, pad=0.03)
    _cb.set_label("State pairs (count)", fontsize=7)
    _cb.ax.tick_params(labelsize=6)

    # Monotone trend overlay: median transition-rank within FC-rank deciles.
    _edges = np.linspace(0, 1, 11)
    _bin_idx = np.clip(np.digitize(_x, _edges[1:-1]), 0, len(_edges) - 2)
    _bx, _by = [], []
    for _b in range(len(_edges) - 1):
        _sel = _bin_idx == _b
        if _sel.sum() >= 3:
            _bx.append(np.median(_x[_sel]))
            _by.append(np.median(_y[_sel]))
    # Red trend = median transition-rank per FC-rank bin. No in-plot label or
    # stats box: ρ range (0.33–0.55), n (5,723 pairs), and the trend definition
    # all live in the caption (user directive 2026-05-26).
    _ax.plot(_bx, _by, color="#D62728", lw=1.4, zorder=4,
             marker="o", ms=3, mec="white", mew=0.5)

    _ax.set_xlabel("State-pair FC similarity (percentile rank)", fontsize=7.5)
    _ax.set_ylabel("Transition probability (percentile rank)", fontsize=7.5)
    _ax.tick_params(labelsize=6.5)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)

    _fig.subplots_adjust(left=0.15, right=0.99, bottom=0.13, top=0.97)
    _stem = OUT_F3 / "fig3_B_fc_transition_density"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem}.pdf (+ .png, .svg)")
    plt.close(_fig)
    return


@app.cell
def panel_C_cross_subject_consistency(
    SUBJECTS, trans_summary, OUT_F3, SUBJECT_NEUTRAL, SUBJECT_MARKERS, plt, np,
):
    """Panel C - cross-subject consistency forest (4 metrics × 6 subjects).

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
        _n_rows, 1, figsize=(3.9, 3.6), sharex=False,
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
                         lw=1.2, alpha=0.5, zorder=2, solid_capstyle="round")
            _ax.scatter(
                _v, _yj, s=50, color=SUBJECT_NEUTRAL,
                marker=SUBJECT_MARKERS[_sub], edgecolor="white", linewidth=0.6,
                zorder=3, alpha=0.95,
            )
        # null reference line
        _ax.axvline(_null, color="#999999", lw=1.0, ls="--", zorder=1)
        _ax.set_ylim(-0.5, 0.5)
        _ax.set_yticks([])
        # The two FC rows share one grouped label drawn below; blank them here.
        _ax.set_ylabel("" if _key in _rho_keys else _label,
                       rotation=0, ha="right", va="center", fontsize=7,
                       labelpad=2)
        _ax.tick_params(axis="x", labelsize=6.5)
        _ax.set_xlabel(_xlabel, fontsize=7)
        for _s in ("top", "right", "left"):
            _ax.spines[_s].set_visible(False)
        if _key in _rho_keys:
            _ax.set_xlim(*_rho_xlim)          # shared ρ scale (rows 3 & 4)
        else:
            _lo = min(min(_vals), _null); _hi = max(max(_vals), _null)
            _pad = 0.10 * (_hi - _lo + 1e-9)
            _ax.set_xlim(_lo - _pad, _hi + _pad)

    _fig.subplots_adjust(left=0.24, right=0.975, bottom=0.10, top=0.99, hspace=1.05)

    # Group the two functional-connectivity rows (3 & 4) under one label + a
    # left square bracket, since they are the same variable. Positions are in
    # figure fractions from the axes' final positions (robust to tight bbox).
    import matplotlib.lines as _ml
    _ptop = _axes[2].get_position(); _pbot = _axes[3].get_position()
    _y0, _y1, _yc = _pbot.y0, _ptop.y1, 0.5 * (_pbot.y0 + _ptop.y1)
    _xb, _tk = 0.165, 0.012
    for _xs, _ys in (([_xb, _xb], [_y0, _y1]),
                     ([_xb, _xb + _tk], [_y1, _y1]),
                     ([_xb, _xb + _tk], [_y0, _y0])):
        _fig.add_artist(_ml.Line2D(_xs, _ys, color="0.4", lw=1.0))
    _fig.text(_xb - 0.010, _yc, "Functional\nconnectivity",
              rotation=90, ha="right", va="center", fontsize=7)

    _stem = OUT_F3 / "fig3_C_cross_subject_consistency"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem}.pdf (+ .png, .svg)")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
