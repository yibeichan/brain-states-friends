"""Figure F3 - Transition structure (R3): graph topology + FC-transition coupling.

One marimo notebook per figure (see `2026-05-24_manuscript_version_scope.md`).
Panels are per-cell, saved as separate .pdf + .png + .svg mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

R3 claim: "Transitions are organized by the functional relationships between
states, not random diffusion." Intro §5 commits to three findings -
recurrence assortativity, FC-transition coupling, and network homophily "in
most individuals."

Panel division of labor: Panel A shows *what* the state transition structure
looks like (the sparse, directed transition repertoire per subject), ordered
by its transition modules; its caption carries the 06e modularity test. Panel
B quantifies what linked states have in common (assortativity, homophily, FC-
and MFPT-coupling, each vs its null).

Panel plan (chart families distinct per the manuscript-figure plot-type-variety
rule; F1 already used bar / flow-Sankey / brain-map, so F2 avoids those within
this figure):

Intended composite (current manuscript layout): TWO rows. Top row = Panel A only
(the 1×6 graph strip, full width). Bottom row = Panel B only. Legends ship as
separate files.

| Panel | Content | Chart family | Source files | Output |
|---|---|---|---|---|
| A | Per-subject directed state transition matrices M (model transmat_, self-transitions removed), 2×3 small-multiples (full square; states ordered by the 06e best-Q partition of M: module rank, then recurrence descending; thin mid-grey module separators; single-hue Greys cells, shared log 5-95 scale; per-state network ribbon with flagged states dotted; non-modular subjects named in the title). Separate legend files. Caption numbers (from 06e summary): per subject Q, Q - mean Q_null and p under N2 and N3 on graph E. | heatmap | 06b transition_graph.graphml + 06e partitions.json + 06e transition_modules_summary.json + 05e state_flags.csv (×6 subj) | fig3_A_transition_matrices.{png,svg} + fig3_A_legend_networks.{png,svg} + fig3_A_legend_prob.{png,svg} + fig3_A_module_stats.json |
| SI | FC similarity (A: RV of within-state correlation matrices, as percentile of the subject's state pairs) and activation-map similarity (B: Pearson r, diverging) in Panel A's module order, same ribbon, dots and separators; recomputed within- minus between-module differences checked against 06e | heatmap | 05f fc_similarity_corr_rv.npy + 04 state_means_parcel.npy / pca_model.pkl + 06e partitions.json / transition_modules_summary.json (×6 subj) | ../si_transition_modules/si_transition_modules_{A_fc_similarity,B_map_similarity,legend_fc,legend_map}.{png,svg} + si_transition_modules_similarity_stats.json |
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
  * Panel A ordering (2026-10-22): states were ordered by dominant network,
    then recurrence. 06e found every subject's transition graph modular under
    the decision-rule nulls, so states are now ordered by the transition
    modules instead (the ordering comes from the transitions, not from the
    networks), with module separators replacing the network-group lines and
    the ribbon drawn per state. Module rank = modules sorted by the
    recurrence of their most recurrent state, highest first.
  * Old Panel B pooled FC-transition density was removed from the main figure:
    it was display-only, while the inferential statistics are the per-subject
    quantities summarized in the consistency forest. Keep this omission aligned
    with docs/manuscript/figure_captions.md.
  * Panel B: subjects now use SUBJECT_MARKERS shapes (the project-wide subject
    key) in a single neutral color. sub-05's homophily-null (ratio 0.980, p=0.51)
    is no longer accented - it is visible as the one homophily marker at/below the
    null line and noted in the caption. No per-panel legend (marker→subject key is
    shared project-wide across the subject-marker figures). Tightened top padding.

Source-truth audit (2026-05-26, reproduced from raw arrays - see session log):
  * Panel B: FC-Mantel ρ 0.326–0.551, MFPT-FC ρ 0.405–0.680, assortativity
    0.111–0.297, homophily ratio 0.980–1.989 (sub-05 null, p=0.51).
  * Panel A caption numbers (from 06e summary, 2026-10-22; Q on graph E,
    p at the floor 1/(1 + draws) for every subject; all six modular):
      subject  modules  Q(E)   Q - mean Q_null: N2   N3
      sub-01   4        0.440  0.242                 0.307
      sub-02   4        0.427  0.209                 0.260
      sub-03   6        0.355  0.267                 0.281
      sub-04   4        0.394  0.262                 0.275
      sub-05   5        0.468  0.294                 0.344
      sub-06   4        0.446  0.382                 0.413
    p = 0.001 under N2 (1000 draws) and 0.002 under N3 (500 draws).
    The script writes these to fig3_A_module_stats.json and prints them.

Outputs go to $SCRATCH_DIR/output/manuscript_figures/fig3, and the SI panels to
its sibling si_transition_modules/. If FIG3_OUT_DIR is set (e.g. for a test
render), everything goes there instead, the SI panels in
FIG3_OUT_DIR/si_transition_modules/.

Run:
    marimo edit script/fig_F3_transition_structure.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.function
def module_order(nodes, labels, recurrence):
    """Order states by (module rank, -recurrence), the Panel A display order.

    Module rank: modules sorted by the recurrence of their most recurrent
    member, highest first, so the module holding the most recurrent state
    comes first (ties broken by module label). Within a module, states run
    from most to least recurrent (ties broken by state id). Returns the
    ordered state ids and each one's module rank (0 = first module).
    """
    nodes = [int(_n) for _n in nodes]
    labels = [int(_l) for _l in labels]
    if len(nodes) != len(labels):
        raise ValueError(f"{len(nodes)} nodes but {len(labels)} labels")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate state ids in the partition")
    _missing = [_n for _n in nodes if _n not in recurrence]
    if _missing:
        raise ValueError(f"no recurrence score for states {_missing}")
    _top = {}
    for _n, _l in zip(nodes, labels):
        _top[_l] = max(_top.get(_l, float("-inf")), float(recurrence[_n]))
    _rank = {_l: _r for _r, _l in
             enumerate(sorted(_top, key=lambda _l: (-_top[_l], _l)))}
    _pairs = sorted(zip(nodes, labels),
                    key=lambda _nl: (_rank[_nl[1]], -float(recurrence[_nl[0]]), _nl[0]))
    return [_n for _n, _ in _pairs], [_rank[_l] for _, _l in _pairs]


@app.function
def module_boundaries(module_ranks):
    """Positions i (0 < i < N) where the module changes between i-1 and i."""
    return [_i for _i in range(1, len(module_ranks))
            if module_ranks[_i] != module_ranks[_i - 1]]


@app.function
def load_partition(path, graph_nodes):
    """06e's tested partition of graph M (partitions.json): (nodes, labels).

    Fails if the file is missing or its states differ from the displayed
    graph's states.
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; Panel A orders states by the 06e partition "
            "(run 06e_transition_modules.py --mode aggregate first)")
    with open(path) as _f:
        _part = _json.load(_f)
    for _k in ("nodes", "labels"):
        if _k not in _part:
            raise KeyError(f"{path}: no '{_k}' field")
    _nodes = [int(_n) for _n in _part["nodes"]]
    if sorted(_nodes) != sorted(int(_n) for _n in graph_nodes):
        raise ValueError(f"{path}: partition states differ from the transition graph's")
    return _nodes, [int(_l) for _l in _part["labels"]]


@app.function
def display_order(graphml_path, partition_path):
    """Panel A's state order for one subject, shared with the SI panels.

    Returns the 06b transition graph, the ordered state ids, each one's module
    rank, and the per-state recurrence and dominant network (keyed by state).
    """
    import networkx as _nx

    _G = _nx.read_graphml(graphml_path)
    _gnodes = {int(_n): _n for _n in _G.nodes()}
    _recs = {_s: float(_G.nodes[_n].get("recurrence_score", 0.0))
             for _s, _n in _gnodes.items()}
    _nets = {_s: _G.nodes[_n].get("dominant_network", "Unknown")
             for _s, _n in _gnodes.items()}
    _pnodes, _plabels = load_partition(partition_path, _gnodes)
    _order, _mods = module_order(_pnodes, _plabels, _recs)
    return _G, _order, _mods, _recs, _nets


@app.function
def offdiag_percentile(S):
    """Each off-diagonal entry's percentile (0-100) among the finite off-diagonal
    entries of the symmetric matrix S; the diagonal and undefined entries are NaN.
    Ties get their mean rank.
    """
    import numpy as _np
    from scipy.stats import rankdata as _rankdata

    S = _np.asarray(S, dtype=float)
    _out = _np.full(S.shape, _np.nan)
    _iu = _np.triu_indices(S.shape[0], 1)
    _v = S[_iu]
    _ok = _np.isfinite(_v)
    if _ok.sum() < 2:
        return _out
    _r = _np.full(_v.shape, _np.nan)
    _r[_ok] = 100.0 * (_rankdata(_v[_ok]) - 1) / (_ok.sum() - 1)
    _out[_iu] = _r
    _out.T[_iu] = _r
    return _out


@app.function
def within_minus_between(S, labels):
    """Mean of S over same-module pairs minus the mean over different-module
    pairs (upper triangle), as 06e's univariate characterisation computes it.
    """
    import numpy as _np

    _lab = _np.asarray(labels)
    _iu = _np.triu_indices(len(_lab), 1)
    _co = (_lab[:, None] == _lab[None, :])[_iu]
    _v = _np.asarray(S, dtype=float)[_iu]
    return float(_v[_co].mean() - _v[~_co].mean())


@app.function
def check_against_06e(sub, name, S, labels, states, theirs, atol=1e-9):
    """Recompute S's within- minus between-module difference and compare it to
    06e's value. S is restricted to the states 06e characterised, in display
    order, with `labels` their module labels and `states` their ids. Raises if a
    state has an undefined off-diagonal similarity or the two values differ by
    more than atol; returns the recomputed value.
    """
    import numpy as _np

    S = _np.asarray(S, dtype=float)
    _remedy = ("Either this script now orders or selects states differently, or the "
               "04/05f/06b inputs changed after 06e ran. Check the script first; re-run "
               "06e_transition_modules.py --mode aggregate only if the inputs changed.")
    _off = ~_np.eye(len(states), dtype=bool)
    _bad = [int(states[_i]) for _i in range(len(states))
            if not _np.isfinite(S[_i][_off[_i]]).all()]
    if _bad:
        raise RuntimeError(f"{sub} {name}: undefined similarity for states {_bad}, "
                           f"which 06e characterised. {_remedy}")
    _mine = within_minus_between(S, labels)
    if not abs(_mine - float(theirs)) <= atol:
        raise RuntimeError(
            f"{sub} {name}: within- minus between-module difference {_mine!r} "
            f"differs from 06e's {float(theirs)!r} by {_mine - float(theirs):+.3g} "
            f"(tolerance {atol:g}). {_remedy}")
    return _mine


@app.function
def check_no_fc_states(sub, rv, states, dropped):
    """Raise if a state 06e left out for lacking a defined FC now has one.

    `rv` is the FC RV matrix in display order and `states` the state ids; a
    state counts as defined when its whole off-diagonal row is finite.
    """
    import numpy as _np

    rv = _np.asarray(rv, dtype=float)
    _off = ~_np.eye(len(states), dtype=bool)
    _now = [int(states[_i]) for _i in range(len(states))
            if int(states[_i]) in dropped and _np.isfinite(rv[_i][_off[_i]]).all()]
    if _now:
        raise RuntimeError(
            f"{sub} fc_rv: states {_now} have a defined FC now, but 06e left them out "
            "for lacking one. Either this script now reads a different RV matrix, or "
            "05f was re-run after 06e; in that case re-run "
            "06e_transition_modules.py --mode aggregate, then this script.")


@app.function
def draw_state_matrix(ax, M, cmap, norm, networks, module_ranks, flagged,
                      network_colors, sep_color):
    """Draw one subject's state-by-state matrix in display order (Panel A style).

    M is already ordered (masked cells take the colormap's bad colour). A
    per-state network ribbon runs along the top and left margins, a black dot
    in a ribbon cell marks a flagged state, and lines in `sep_color` separate
    modules. Ticks and spines are removed; the caller sets the title.
    """
    from matplotlib.patches import Rectangle as _Rectangle

    _N = M.shape[0]
    _rib = max(1.0, 0.035 * _N)
    ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest",
              aspect="equal", extent=[0, _N, _N, 0], zorder=2)
    for _i, _net in enumerate(networks):
        _c = network_colors.get(_net, "#BBBBBB")
        ax.add_patch(_Rectangle((_i, -_rib), 1, _rib, color=_c,
                                lw=0, zorder=3, clip_on=False))
        ax.add_patch(_Rectangle((-_rib, _i), _rib, 1, color=_c,
                                lw=0, zorder=3, clip_on=False))
    _fi = [_i + 0.5 for _i, _fl in enumerate(flagged) if _fl]
    if _fi:
        ax.scatter(_fi, [-_rib / 2] * len(_fi), s=1.2, color="black",
                   lw=0, zorder=4, clip_on=False)
        ax.scatter([-_rib / 2] * len(_fi), _fi, s=1.2, color="black",
                   lw=0, zorder=4, clip_on=False)
    for _b in module_boundaries(module_ranks):
        ax.plot([_b, _b], [-_rib, _N], color=sep_color, lw=0.6, zorder=3.5,
                clip_on=False)
        ax.plot([-_rib, _N], [_b, _b], color=sep_color, lw=0.6, zorder=3.5,
                clip_on=False)
    ax.set_xlim(-_rib, _N)
    ax.set_ylim(_N, -_rib)
    ax.set_xticks([])
    ax.set_yticks([])
    for _spine in ax.spines.values():
        _spine.set_visible(False)


@app.function
def rule_null_stats(summary):
    """Per-subject numbers for the Panel A caption, from 06e's summary JSON.

    The decision rule's nulls are N2 (segment-order shuffle) and N3
    (non-modular chain), both on the empirical graph E. Q here is E's
    modularity, not that of the displayed graph M.
    """
    _out = {"modular": bool(summary["modular"])}
    for _v in ("N2_E", "N3_E"):
        _t = summary["tests"][_v]
        _out[_v] = {"q_obs": float(_t["q_obs"]), "q_null_mean": float(_t["q_null_mean"]),
                    "effect": float(_t["effect"]), "p": float(_t["p"]),
                    "n_draws": int(_t["n_draws"])}
    return _out


@app.cell
def imports():
    import csv
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
        Path, csv, json, np, nx, os, plt,
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
    MODULES_DIR = SCRATCH_DIR / "output" / "06e_transition_modules" / PARCELLATION
    FLAGS_DIR = SCRATCH_DIR / "output" / "05e_temporal_trend_a4" / PARCELLATION
    FC_DIR = SCRATCH_DIR / "output" / "05f_state_fc" / PARCELLATION
    HMM_DIR = SCRATCH_DIR / "output" / "04_combined_hdphmm" / PARCELLATION

    # Default: the manuscript figure directory, with the SI panels in a
    # topic-named sibling folder. FIG3_OUT_DIR (e.g. for a test render)
    # redirects all outputs, the SI panels into a subfolder of it.
    if os.environ.get("FIG3_OUT_DIR"):
        OUT_F3 = Path(os.environ["FIG3_OUT_DIR"])
        OUT_SI_MODULES = OUT_F3 / "si_transition_modules"
    else:
        OUT_F3 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig3"
        OUT_SI_MODULES = OUT_F3.parent / "si_transition_modules"
    OUT_F3.mkdir(parents=True, exist_ok=True)
    OUT_SI_MODULES.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR, PARCELLATION, VT, SUBJECTS,
        TRANS_DIR, MODULES_DIR, FLAGS_DIR, FC_DIR, HMM_DIR, OUT_F3, OUT_SI_MODULES,
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
    SUBJECTS, TRANS_DIR, MODULES_DIR, FLAGS_DIR, VT, OUT_F3, csv, json, plt, np,
    NETWORK_COLORS, NETWORK_ORDER, display_network,
):
    """Panel A - per-subject state transition matrices ordered by transition module.

    One directed state-by-state matrix per subject, 2×3 small-multiples. The
    matrix is M: the model transmat_ with self-transitions removed and edges
    thresholded at P >= 0.01 (06b transition_graph.graphml, which 06e checks
    against its own M). Row = from-state, col = to-state; the matrix is
    asymmetric, so the full square is shown.

    State order comes from the transitions themselves: the best-Q partition of
    M from 06e (partitions.json `nodes`/`labels`, the partition 06e tested and
    characterised). Order key = (module rank, -recurrence). Module rank sorts
    modules by the recurrence of their most recurrent member, highest first,
    so the module holding the most recurrent state is first; within a module,
    states run from most to least recurrent. A missing or mismatched partition
    file is an error. Thin mid-grey lines separate modules.

    Colour scale as before: one shared log scale over all subjects' edges,
    5th-95th percentile, single-hue Greys ramp starting at 20% grey so every
    edge >= 0.01 shows against the white (absent) cells; colorbar ticks are
    plain decimals. A per-state network ribbon on the top and left margins
    gives each state's dominant network; a black dot in a ribbon cell marks a
    flagged state (05e summary_category other than
    eligible_for_content_analysis, the same rule 06e uses).

    Caption numbers (written to fig3_A_module_stats.json and printed): per
    subject, Q, Q - mean Q_null and p under the two nulls in the decision
    rule, N2 (segment-order shuffle) and N3 (non-modular chain), from 06e's
    transition_modules_summary.json. Those tests are on the empirical graph E
    (decoded transitions), while the matrix and partition shown are M's; Q of
    M is listed alongside. A subject that 06e does not call modular is drawn
    the same way and named in its title ("not modular").

    Six matrices, not one pooled matrix: states are subject-specific (each
    subject's own HMM, no cross-subject alignment) and transitions exist only
    within a subject's sequence, so a pooled transition matrix is undefined.

    Legends are SEPARATE files (fig3_A_legend_networks, fig3_A_legend_prob).
    """
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable

    # Pass 1: per subject, the directed matrix in module order, the per-state
    # networks and flags, and the caption numbers. Collect off-diagonal
    # weights for the shared log scale.
    _data = {}
    _stats = {}
    _present_nets = set()
    _all_w = []
    for _sub in SUBJECTS:
        _G, _order, _mods, _recs, _nets = display_order(
            TRANS_DIR / _sub / VT / "transition_graph.graphml",
            MODULES_DIR / _sub / VT / "partitions.json")
        with open(FLAGS_DIR / _sub / VT / "state_flags.csv") as _f:
            _cat = {int(_r["state"]): _r["summary_category"] for _r in csv.DictReader(_f)}
        _flagged = [_cat[_s] != "eligible_for_content_analysis" for _s in _order]
        _idx = {_s: _i for _i, _s in enumerate(_order)}
        _N = len(_order)
        _M = np.zeros((_N, _N))
        for _u, _v, _d in _G.edges(data=True):
            _u, _v = int(_u), int(_v)
            if _u == _v:
                continue
            _M[_idx[_u], _idx[_v]] = max(_M[_idx[_u], _idx[_v]], float(_d.get("weight", 0.0)))
        _onets = [_nets[_s] for _s in _order]
        _present_nets.update(_onets)
        _all_w.append(_M[_M > 0])

        with open(MODULES_DIR / _sub / VT / "transition_modules_summary.json") as _f:
            _summ = json.load(_f)
        _st = rule_null_stats(_summ)
        _st.update({"n_states": _N, "n_modules": len(set(_mods)),
                    "n_flagged": int(sum(_flagged)),
                    "q_M_displayed_partition": float(_summ["tests"]["N1_M"]["q_obs"])})
        _stats[_sub] = _st
        _data[_sub] = (_M, _onets, _mods, _flagged, _st["modular"])
        print(f"  Panel A {_sub}: {_N} states, {int((_M > 0).sum())} directed edges, "
              f"{_st['n_modules']} modules, {_st['n_flagged']} flagged, "
              f"modular={_st['modular']}")
        for _v in ("N2_E", "N3_E"):
            _t = _st[_v]
            print(f"    {_v}: Q(E) {_t['q_obs']:.3f}, Q - mean Q_null {_t['effect']:.3f}, "
                  f"p {_t['p']:.4f} ({_t['n_draws']} draws)")

    _not_modular = [_s for _s in SUBJECTS if not _stats[_s]["modular"]]
    _stats_out = {
        "note": ("Q, Q - mean Q_null and p are 06e's tests on the empirical "
                 "transition graph E under the decision-rule nulls N2 "
                 "(segment-order shuffle) and N3 (non-modular chain); p has a "
                 "floor of 1/(1 + n_draws). The panel shows the model graph M "
                 "(self-transitions removed) ordered by M's best-Q partition; "
                 "q_M_displayed_partition is that partition's Q."),
        "module_rank": ("modules sorted by the recurrence of their most "
                        "recurrent state, highest first; states within a "
                        "module by recurrence, highest first"),
        "not_modular_subjects": _not_modular,
        "subjects": _stats,
    }
    with open(OUT_F3 / "fig3_A_module_stats.json", "w") as _f:
        json.dump(_stats_out, _f, indent=1)
    print(f"saved: fig3_A_module_stats.json (not modular: {_not_modular or 'none'})")

    _all_w = np.concatenate(_all_w)
    # Tight log scale (5/95) so the mid-range transitions read; the few
    # strongest cells saturate to black (colorbar marked "extend=both").
    _vmin = max(float(np.percentile(_all_w, 5)), 1e-3)
    _vmax = float(np.percentile(_all_w, 95))
    # Greys starts at pure white, which made the weakest real edges look like
    # absent ones; start the ramp at a light grey so every edge >= 0.01 shows
    # against the white (absent) cells.
    from matplotlib.colors import ListedColormap
    _cmap = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.2, 1.0, 256)),
                           name="Greys_from_light")
    _cmap.set_bad("white")
    _norm = LogNorm(vmin=_vmin, vmax=_vmax)
    # Mid-grey module separators: white ones vanished over the empty cells.
    _sep_color = "#8C8C8C"

    # Pass 2: render 2×3, full directed matrix, per-state network ribbon on
    # the top + left margins, white separators at module boundaries.
    _fig, _axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    for _ax, _sub in zip(_axes.ravel(), SUBJECTS):
        _M, _onets, _mods, _flagged, _modular = _data[_sub]
        draw_state_matrix(_ax, np.ma.masked_where(_M <= 0, _M), _cmap, _norm,
                          _onets, _mods, _flagged, NETWORK_COLORS, _sep_color)
        _ax.set_title(_sub if _modular else f"{_sub} (not modular)", fontsize=9, pad=2)

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
    # (1) Network color key (networks present, canonical order) plus the
    # flagged-state dot, one column.
    _present = [_net for _net in NETWORK_ORDER if _net in _present_nets]
    _net_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NETWORK_COLORS.get(_net, "#BBBBBB"),
                      edgecolor="#666666", linewidth=0.4, label=display_network(_net))
        for _net in _present
    ]
    _net_handles.append(plt.Line2D([], [], marker="o", ls="none", color="black",
                                   ms=2.5, label="Flagged state"))
    _figL, _axL = plt.subplots(figsize=(1.5, max(0.5, 0.26 * len(_net_handles))))
    _axL.set_axis_off()
    _axL.legend(handles=_net_handles, loc="center", frameon=False, fontsize=7,
                ncol=1, handlelength=1.0, handletextpad=0.4)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.png", bbox_inches="tight",
                  pad_inches=0.02, dpi=300)
    _figL.savefig(OUT_F3 / "fig3_A_legend_networks.svg", bbox_inches="tight",
                  pad_inches=0.02)
    plt.close(_figL)

    # (2) Shared transition-probability colorbar. The 5/95 range spans less
    # than one decade, where the default log locator labels every minor tick
    # in scientific notation and the labels collide; use a few fixed ticks
    # inside the range, written as plain decimals.
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    _sm = ScalarMappable(norm=_norm, cmap=_cmap)
    _sm.set_array([])
    _figC, _axC = plt.subplots(figsize=(2.0, 0.34))
    _cb = _figC.colorbar(_sm, cax=_axC, orientation="horizontal", extend="both")
    _ticks = [_t for _t in (0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5)
              if _vmin <= _t <= _vmax]
    _cb.ax.xaxis.set_major_locator(FixedLocator(_ticks))
    _cb.ax.xaxis.set_minor_locator(NullLocator())
    _cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda _x, _p: f"{_x:g}"))
    _cb.set_label("Transition probability", fontsize=7)
    _cb.ax.tick_params(labelsize=6)
    print(f"  probability colorbar: range {_vmin:.4f}-{_vmax:.4f}, ticks {_ticks}")
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
    (the project-wide subject key - so no per-panel
    legend is needed; the shapes also de-overlap clustered points). Color is NOT
    spent on subject identity here: marker shape carries it, leaving the panel
    uncluttered. sub-05's homophily null (ratio 0.980, p=0.51 - the intro's "in
    most individuals" caveat) is no longer accented; it reads as the one
    homophily marker sitting at/below the dashed null line, and is named in the
    caption. The point estimate per row is the message; which marker is which
    individual is secondary (recoverable via the project-wide marker key).
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


@app.cell
def supp_module_similarity(
    SUBJECTS, TRANS_DIR, MODULES_DIR, FLAGS_DIR, FC_DIR, HMM_DIR, VT,
    OUT_SI_MODULES, csv, json, plt, np, NETWORK_COLORS,
):
    """SI - functional-connectivity and activation-map similarity in module order.

    Two 2×3 grids laid out like Panel A, with the same state order, network
    ribbon, flagged-state dots and module separators, so the reader can see
    whether the transition modules are blocks of similar states:
      A  within-state FC similarity: the RV coefficient between the states'
         within-state correlation matrices (05f fc_similarity_corr_rv.npy,
         the measure 06b correlates with transitions and 06e enters in its
         co-membership model). RV is compressed near 1 (per subject, the
         off-diagonal 5th percentile is 0.91-0.98 and the 95th 0.987-0.998),
         so cells show each pair's
         percentile among that subject's state pairs; a shared raw scale would
         be nearly flat. States without a defined FC (05f drops states with
         fewer than 30 TRs) are drawn as light grey rows and columns.
      B  activation-map similarity: Pearson r across parcels between the
         states' mean maps (04 state_means_parcel minus the PCA mean, as in
         06e's map_r), on a diverging scale shared across subjects.
    Diagonals are white. The matrices are symmetric.

    Before drawing, the within- minus between-module difference of each
    matrix is recomputed and checked against 06e's univariate
    characterisation (same states, same partition); a mismatch is an error.
    The caption numbers (that difference with its permutation p, and the
    joint co-membership model's coefficient sign and p) are written to
    si_transition_modules_similarity_stats.json and printed.
    """
    import pickle as _pickle
    import warnings as _warnings
    from matplotlib.cm import ScalarMappable as _ScalarMappable
    from matplotlib.colors import ListedColormap as _ListedColormap
    from matplotlib.colors import Normalize as _Normalize
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm

    _sep_color = "#8C8C8C"
    _data = {}
    _stats = {}
    _map_abs = []
    for _sub in SUBJECTS:
        _G, _order, _mods, _recs, _nets = display_order(
            TRANS_DIR / _sub / VT / "transition_graph.graphml",
            MODULES_DIR / _sub / VT / "partitions.json")
        with open(FLAGS_DIR / _sub / VT / "state_flags.csv") as _f:
            _cat = {int(_r["state"]): _r["summary_category"] for _r in csv.DictReader(_f)}
        _flagged = [_cat[_s] != "eligible_for_content_analysis" for _s in _order]
        _onets = [_nets[_s] for _s in _order]
        _idx = np.asarray(_order)

        _rv = np.load(FC_DIR / _sub / VT / "fc_similarity_corr_rv.npy")[np.ix_(_idx, _idx)]
        _hmm = HMM_DIR / _sub / "final" / VT
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            with open(_hmm / "pca_model.pkl", "rb") as _f:
                _pca = _pickle.load(_f)
        _maps = np.load(_hmm / "state_means_parcel.npy")[_idx] - _pca.mean_
        _Z = _maps - _maps.mean(1, keepdims=True)
        _Z /= np.linalg.norm(_Z, axis=1, keepdims=True)
        _map_r = _Z @ _Z.T

        # Check against 06e: same states (minus the no-FC ones), same labels.
        with open(MODULES_DIR / _sub / VT / "transition_modules_summary.json") as _f:
            _char = json.load(_f)["characterisation"]
        _drop = set(int(_s) for _s in _char["dropped_no_fc"])
        _keep = [_i for _i, _s in enumerate(_order) if _s not in _drop]
        _kix = np.ix_(_keep, _keep)
        _klab = [_mods[_i] for _i in _keep]
        _kstates = [_order[_i] for _i in _keep]
        check_no_fc_states(_sub, _rv, _order, _drop)
        _chk = {}
        for _name, _S in (("fc_rv", _rv), ("map_r", _map_r)):
            _theirs = float(_char["univariate"][_name]["within_minus_between"])
            check_against_06e(_sub, _name, _S[_kix], _klab, _kstates, _theirs)
            _mr = _char["mrqap"][_name]
            _chk[_name] = {
                "within_minus_between": _theirs,
                "p_two_sided": float(_char["univariate"][_name]["p_two_sided"]),
                "mrqap_beta": float(_mr["beta"]), "mrqap_p": float(_mr["p"]),
            }
        _stats[_sub] = {"n_states": len(_order), "n_modules": len(set(_mods)),
                        "states_without_fc": sorted(_drop), **_chk}

        _fc_pct = offdiag_percentile(_rv)
        _undef = ~np.isfinite(_rv)
        np.fill_diagonal(_undef, False)
        _m = _map_r.copy()
        np.fill_diagonal(_m, np.nan)
        _map_abs.append(np.abs(_m[np.isfinite(_m)]))
        _data[_sub] = (_fc_pct, _undef, _m, _onets, _mods, _flagged)
        print(f"  SI modules {_sub}: {len(_order)} states, {len(set(_mods))} modules, "
              f"no FC {sorted(_drop) or 'none'}; within-between "
              f"FC RV {_chk['fc_rv']['within_minus_between']:+.4f} "
              f"(p {_chk['fc_rv']['p_two_sided']:.4f}), "
              f"map r {_chk['map_r']['within_minus_between']:+.3f} "
              f"(p {_chk['map_r']['p_two_sided']:.4f})")

    _lim = float(np.percentile(np.concatenate(_map_abs), 99))
    _lim = min(1.0, np.ceil(_lim * 10) / 10)
    _n_sat = sum(int((_a > _lim).sum()) for _a in _map_abs) // 2
    _stats_out = {
        "note": ("Within- minus between-module mean similarity and its two-sided "
                 "label-permutation p, and the coefficient and p of each measure in "
                 "the joint co-membership model (MRQAP), from 06e's characterisation "
                 "(states without a defined FC excluded). fc_rv is the RV coefficient "
                 "of within-state correlation matrices; map_r is the Pearson r of "
                 "mean activation maps."),
        "panel_A_scale": "percentile of each pair's FC RV among the subject's state pairs",
        "panel_B_scale": f"map r, diverging, limits +/-{_lim:g} (shared)",
        "panel_B_pairs_beyond_limits": _n_sat,
        "subjects": _stats,
    }
    with open(OUT_SI_MODULES / "si_transition_modules_similarity_stats.json", "w") as _f:
        json.dump(_stats_out, _f, indent=1)

    _fc_cmap = _ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 1, 256)), name="fc_pct")
    _fc_cmap.set_bad("white")
    _fc_norm = _Normalize(vmin=0, vmax=100)
    _map_cmap = _ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256)), name="map_r")
    _map_cmap.set_bad("white")
    _map_norm = _TwoSlopeNorm(vcenter=0.0, vmin=-_lim, vmax=_lim)
    _undef_rgba = np.array([0.85, 0.85, 0.85, 1.0])

    for _panel, _stem_tail in (("A", "A_fc_similarity"), ("B", "B_map_similarity")):
        _fig, _axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
        for _ax, _sub in zip(_axes.ravel(), SUBJECTS):
            _fc_pct, _undef, _m, _onets, _mods, _flagged = _data[_sub]
            if _panel == "A":
                draw_state_matrix(_ax, np.ma.masked_invalid(_fc_pct), _fc_cmap, _fc_norm,
                                  _onets, _mods, _flagged, NETWORK_COLORS, _sep_color)
                if _undef.any():
                    _ov = np.zeros(_undef.shape + (4,))
                    _ov[_undef] = _undef_rgba
                    _N = _undef.shape[0]
                    _ax.imshow(_ov, interpolation="nearest", aspect="equal",
                               extent=[0, _N, _N, 0], zorder=2.5)
            else:
                draw_state_matrix(_ax, np.ma.masked_invalid(_m), _map_cmap, _map_norm,
                                  _onets, _mods, _flagged, NETWORK_COLORS, _sep_color)
            _ax.set_title(_sub, fontsize=9, pad=2)
        _fig.subplots_adjust(left=0.045, right=0.99, bottom=0.045, top=0.94,
                             wspace=0.08, hspace=0.06)
        _fig.text(0.006, 0.4925, "State", rotation=90, ha="left", va="center",
                  fontsize=10)
        _fig.text(0.5175, 0.006, "State", ha="center", va="bottom", fontsize=10)
        _stem = OUT_SI_MODULES / f"si_transition_modules_{_stem_tail}"
        _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
        plt.close(_fig)
        print(f"saved: {_stem.name}.png (+ .svg)")

    # Colorbar legends, one per grid (the network key is fig3_A_legend_networks).
    # Pairs beyond the map scale's limits take its end colours (extend="both").
    for _name, _cmap, _norm, _label, _ticks, _extend in (
        ("fc", _fc_cmap, _fc_norm, "FC similarity (percentile of state pairs)",
         [0, 25, 50, 75, 100], "neither"),
        ("map", _map_cmap, _map_norm, "Activation-map similarity (r)",
         [-_lim, -_lim / 2, 0, _lim / 2, _lim], "both"),
    ):
        _sm = _ScalarMappable(norm=_norm, cmap=_cmap)
        _sm.set_array([])
        _figC, _axC = plt.subplots(figsize=(2.4, 0.34))
        _cb = _figC.colorbar(_sm, cax=_axC, orientation="horizontal", extend=_extend)
        _cb.set_ticks(_ticks)
        _cb.ax.set_xticklabels([f"{_t:g}" for _t in _ticks])
        _cb.set_label(_label, fontsize=7)
        _cb.ax.tick_params(labelsize=6)
        _stem = OUT_SI_MODULES / f"si_transition_modules_legend_{_name}"
        _figC.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        _figC.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
        plt.close(_figC)
    print(f"saved: SI module similarity legends (map r limits +/-{_lim:g}; "
          f"{_n_sat} state pairs beyond them)")
    return


if __name__ == "__main__":
    app.run()
