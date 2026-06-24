"""Figure F1 - graded recurrence distribution (R1).

Split from the former `fig_F1_recurrence_and_taxonomy.py` (2026-06-10): R1 (the
graded recurrence distribution) and R2 (the recurrence-screening taxonomy) are
two figures. This script is R1.

R1/R2 separation (2026-06-10): R1 shows ONLY the recurrence distribution - dots
are a single neutral color, no taxonomy coloring. The source taxonomy
(content-eligible / run-onset / low-confidence / drift / unused) is entirely an
R2 construct and lives in Figure 2. The two figures describe the same ~45 active
states (logically connected) but never mix content: Figure 1 = the recurrence
axis, Figure 2 = recurrence-screening categories and network summaries.

R1's message is descriptive: each participant's fitted model instantiates ~45
active fitted states whose recurrence scores span a graded distribution, and the
rank ordering of recurrence reproduces under independent within-subject re-fits
(split-half reliability). The recurrence x dwell-independence panel was moved out
of R1 (off-message; belongs with 06a temporal dynamics).

Panels A and B of Figure 1 are the brain-state element schematics (mean
activation, functional connectivity, dwell/occupancy, transition probability)
composed manually during assembly; this script produces only panels C and D.

| Panel | Content | Chart family | Source | Output |
|---|---|---|---|---|
| A, B | Brain-state element schematics, added manually during assembly | schematic | n/a | n/a |
| C | Per-subject recurrence beeswarm, decomposed into one strip file per subject (active states, density-stacked, dot color = mean dwell). x-ticks + axis label on sub-01 only. | point-based 1D | 06a state_summary_table.csv (recurrence_score, mean_dwell_s) | fig1_C_<sub>_recurrence.{png,svg} (6 files) + fig1_C_legend_dwell.{png,svg} |
| D | Per-subject split-half reliability scatter (recurrence half A vs half B, matched states) + standalone subject->marker key | scatter | 04rb_split_half (hungarian_matching + half_invariants + split_half_reliability) | fig1_D_split_half_reliability.{png,svg} + fig1_D_legend_markers.{png,svg} |

All panels save PNG + SVG only (no PDF).

Key data facts (audited 2026-06-10):
  * Active states per subject (= state_summary_table.csv row count): sub-01 46,
    sub-02 46, sub-03 44, sub-04 44, sub-05 47, sub-06 42.
  * Split-half matched recurrence Spearman rho per subject: sub-01 0.65, sub-02
    0.68, sub-03 0.76, sub-04 0.82, sub-05 0.60, sub-06 0.76 (all p<1e-4) -> the
    recurrence ordering reproduces under independent re-fits on interleaved
    episode halves. Matching is within-subject only (no cross-subject alignment).

Run:
    marimo edit script/fig_F1_recurrence_gradient.py
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
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import apply_publication_style, SUBJECT_MARKERS

    apply_publication_style()

    return Path, json, np, os, pd, plt, SUBJECT_MARKERS


@app.cell
def config(Path, os):
    """Paths, subject list, output dir, neutral dot color."""
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]

    BLOCKS_DIR = SCRATCH_DIR / "output" / "06a_state_temp_dynamics" / PARCELLATION
    SPLITHALF_DIR = SCRATCH_DIR / "output" / "04rb_split_half" / PARCELLATION

    OUT_F1 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig1"
    OUT_F1.mkdir(parents=True, exist_ok=True)

    # Single neutral dot color - R1 carries no taxonomy, so dots are uncolored
    # by category. Deliberately not one of the R2 category colors.
    DOT_COLOR = "#44546A"

    # Shared legibility scheme (consistent across all F1 panels).
    FS_TICK = 7
    FS_LABEL = 8
    FS_ANNOT = 8
    DOT_S = 26
    FIG_W = 7.0
    return (SUBJECTS, BLOCKS_DIR, SPLITHALF_DIR, OUT_F1, VT, DOT_COLOR,
            FS_TICK, FS_LABEL, FS_ANNOT, DOT_S, FIG_W)


@app.cell
def load_active_states(SUBJECTS, BLOCKS_DIR, VT, pd, np):
    """Per-subject active-state table from 06a state_summary_table.csv.

    This file is already the K_active subset (42-47 states/subject, dwell
    defined). R1 needs only recurrence_score and mean_dwell_s; no taxonomy.
    """
    active = {}
    for _sub in SUBJECTS:
        active[_sub] = pd.read_csv(BLOCKS_DIR / _sub / VT / "state_summary_table.csv")
    _allrec = np.concatenate(
        [_d["recurrence_score"].values for _d in active.values()]
    )
    _allrec = _allrec[np.isfinite(_allrec)]
    print("active n per subject:", {s: len(d) for s, d in active.items()})
    print(f"cohort recurrence [{_allrec.min():.3f}, {_allrec.max():.3f}], "
          f"median {np.median(_allrec):.3f}")
    return (active,)


@app.cell
def panel_C_beeswarm(
    active, SUBJECTS, OUT_F1, SUBJECT_MARKERS, FS_TICK, FS_LABEL, DOT_S,
    FIG_W, plt, np
):
    """Panel C - per-subject recurrence beeswarm, decomposed into one strip per
    subject plus a single shared dwell colorbar (the legend).

    Each strip: x = recurrence (0 -> ~0.93), drawn with that subject's marker
    shape (the project-wide subject key, shared with Figures 3 and 5). Dots are
    stacked upward within fine x-bins (beeswarm) so height reflects local
    density across the graded recurrence distribution. Dot color = the state's
    mean dwell time (shared viridis scale across all strips): dwell is a state
    property orthogonal to the source taxonomy (Figure 2), and its scatter
    across the recurrence axis shows recurrence and dwell are independent.

    All strips share x-limits, density scaling, dwell norm, and width so they
    align when stacked in assembly. Only sub-01 carries x-tick labels and the
    axis label; the others keep the vertical gridlines for alignment but hide
    tick text. No taxonomy encoding (that is Figure 2).
    """
    from matplotlib.colors import Normalize

    _all_dwell = np.concatenate(
        [active[_s]["mean_dwell_s"].values.astype(float) for _s in SUBJECTS]
    )
    _all_dwell = _all_dwell[np.isfinite(_all_dwell)]
    _cmap = plt.get_cmap("viridis")
    _norm = Normalize(vmin=float(np.nanpercentile(_all_dwell, 2)),
                      vmax=float(np.nanpercentile(_all_dwell, 95)))

    _nbins = 70

    def _stack(_rvals):
        _c = np.zeros(_nbins, dtype=int)
        _y = np.empty(len(_rvals))
        for _i in np.argsort(_rvals, kind="stable"):
            _b = min(int(_rvals[_i] * _nbins), _nbins - 1)
            _y[_i] = _c[_b]
            _c[_b] += 1
        return _y, int(_c.max()) if len(_c) else 1

    _prep = {}
    _gmax = 1
    for _sub in SUBJECTS:
        _df = active[_sub]
        _rec = _df["recurrence_score"].values.astype(float)
        _dwell = _df["mean_dwell_s"].values.astype(float)
        _m = np.isfinite(_rec) & np.isfinite(_dwell)
        _rec, _dwell = _rec[_m], _dwell[_m]
        _yoff, _mx = _stack(_rec)
        _prep[_sub] = (_rec, _dwell, _yoff)
        _gmax = max(_gmax, _mx)

    # One strip file per subject. Height scales with the shared density max so
    # dots are not squashed; sub-01 gets extra room for the x-axis label.
    _row_h = 0.32 + 0.085 * _gmax
    for _sub in SUBJECTS:
        _rec, _dwell, _yoff = _prep[_sub]
        _is_bottom = _sub == "sub-01"
        _h = _row_h + (0.42 if _is_bottom else 0.0)
        _fig, _ax = plt.subplots(figsize=(FIG_W, _h))
        _ax.scatter(
            _rec, _yoff, s=DOT_S, c=_dwell, cmap=_cmap, norm=_norm,
            marker=SUBJECT_MARKERS[_sub], edgecolor="white", linewidth=0.3,
            alpha=0.95, zorder=3,
        )
        _ax.set_ylim(-0.6, _gmax + 0.6)
        _ax.set_yticks([])
        _ax.set_ylabel(_sub, rotation=0, ha="right", va="center",
                       fontsize=FS_LABEL)
        _ax.set_xlim(-0.01, 1.0)
        _ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        _ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#E6E6E6",
                       zorder=0)
        _ax.set_axisbelow(True)
        for _s in ("top", "right", "left"):
            _ax.spines[_s].set_visible(False)
        if _is_bottom:
            _ax.set_xlabel("Recurrence (fraction of episodes active)",
                           fontsize=FS_LABEL)
            _ax.tick_params(axis="x", labelsize=FS_TICK)
        else:
            _ax.tick_params(axis="x", labelbottom=False, length=0)

        _stem = OUT_F1 / f"fig1_C_{_sub}_recurrence"
        _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
        plt.close(_fig)
        print(f"saved: {_stem.name}.png (+ .svg)")

    # Single shared legend: standalone vertical dwell colorbar.
    import matplotlib.cm as cm
    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array([])
    _figc = plt.figure(figsize=(0.95, 2.2))
    _cax = _figc.add_axes([0.08, 0.06, 0.30, 0.88])
    _cbar = _figc.colorbar(_sm, cax=_cax)
    _cbar.set_label("Mean dwell time (s)", fontsize=FS_LABEL)
    _cbar.ax.tick_params(labelsize=FS_TICK)
    _stemc = OUT_F1 / "fig1_C_legend_dwell"
    _figc.savefig(f"{_stemc}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _figc.savefig(f"{_stemc}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(_figc)
    print(f"saved: {_stemc.name}.png (+ .svg)")
    return


@app.cell
def load_split_half(SUBJECTS, SPLITHALF_DIR, json, np):
    """Per-subject split-half matched recurrence (half A vs half B).

    Two HMMs are fit independently on interleaved odd/even episode halves of the
    SAME subject (04rb). Their states are matched post-hoc by the Hungarian
    algorithm on parcel-space mean correlations (r > 0.3). For each matched pair
    we read that state's recurrence in half A and half B from the per-half
    recurrence vectors; rho is the reported raw Spearman (verified to match a
    direct recompute). Matching is strictly within-subject - no states are
    aligned across participants.
    """
    splithalf = {}
    for _sub in SUBJECTS:
        _hm = json.load(open(SPLITHALF_DIR / _sub / "hungarian_matching.json"))
        _hi = json.load(open(SPLITHALF_DIR / _sub / "half_invariants.json"))
        _rel = json.load(open(SPLITHALF_DIR / _sub / "split_half_reliability.json"))
        _recA = np.array(_hi["A"]["recurrence_scores"], dtype=float)
        _recB = np.array(_hi["B"]["recurrence_scores"], dtype=float)
        _pairs = [_p for _p in _hm["matching"]["pairs"] if _p.get("above_threshold", True)]
        splithalf[_sub] = {
            "a": np.array([_recA[_p["state_A"]] for _p in _pairs]),
            "b": np.array([_recB[_p["state_B"]] for _p in _pairs]),
            "rho": float(_rel["matched_recurrence_correlation"]["raw_spearman"]),
            "n": len(_pairs),
        }
    print("split-half rho:", {_s: round(_d["rho"], 2) for _s, _d in splithalf.items()})
    return (splithalf,)


@app.cell
def panel_D_split_half(
    splithalf, SUBJECTS, OUT_F1, DOT_COLOR, SUBJECT_MARKERS,
    FS_TICK, FS_LABEL, FS_ANNOT, DOT_S, FIG_W, plt, np
):
    """Panel D - split-half reliability of recurrence, per subject.

    2x3 grid; x = recurrence in half A, y = recurrence in half B, one dot per
    Hungarian-matched state (independent HMM fits on interleaved episode halves
    of the SAME subject), drawn with that subject's marker (shared key, Figs
    3 and 5). Identity line + per-subject Spearman rho. The correlation reports
    whether matched-state recurrence ordering reproduces under independent
    within-subject re-fits. Dots are a
    single neutral color (per-matched-state dwell is not defined in the split
    fits, so the dwell colormap applies to Panel C only).
    """
    # Panel D sits a little larger than the strips, so bump its fonts +1.
    _fs_tick = FS_TICK + 1
    _fs_label = FS_LABEL + 1
    _fs_annot = FS_ANNOT + 1

    _fig, _axes = plt.subplots(2, 3, figsize=(FIG_W, 4.9), sharex=True, sharey=True)
    _axes = _axes.ravel()

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes[_i]
        _d = splithalf[_sub]
        _ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--", zorder=1)
        _ax.scatter(_d["a"], _d["b"], s=DOT_S, c=DOT_COLOR,
                    marker=SUBJECT_MARKERS[_sub], edgecolor="white",
                    linewidth=0.4, alpha=0.95, zorder=3)
        # Annotation in the top-left corner (above the identity line, where
        # reliability points are sparse), two well-separated lines so the
        # enlarged text never overlaps itself or the markers.
        _ax.text(0.05, 0.95, _sub, transform=_ax.transAxes, fontsize=_fs_annot,
                 va="top", ha="left")
        _ax.text(0.05, 0.78, f"$\\rho$ = {_d['rho']:.2f}",
                 transform=_ax.transAxes, fontsize=_fs_annot, va="top", ha="left")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.set_xticks([0, 0.5, 1.0])
        _ax.set_yticks([0, 0.5, 1.0])
        _ax.set_aspect("equal")
        _ax.tick_params(axis="both", labelsize=_fs_tick)
        _ax.grid(True, linestyle=":", linewidth=0.5, color="#D8D8D8")
        _ax.set_axisbelow(True)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)

    for _i in (0, 3):
        _axes[_i].set_ylabel("Recurrence, half B", fontsize=_fs_label)
    for _i in (3, 4, 5):
        _axes[_i].set_xlabel("Recurrence, half A", fontsize=_fs_label)

    _fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.97,
                         wspace=0.18, hspace=0.22)
    _stem = OUT_F1 / "fig1_D_split_half_reliability"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)

    # Standalone subject -> marker key (shared with Figures 3 and 5). Neutral
    # color matches Panel D dots; one row per subject.
    _figm, _axm = plt.subplots(figsize=(1.15, 1.9))
    for _j, _sub in enumerate(SUBJECTS):
        _yj = len(SUBJECTS) - 1 - _j
        _axm.scatter([0.15], [_yj], s=DOT_S, c=DOT_COLOR,
                     marker=SUBJECT_MARKERS[_sub], edgecolor="white",
                     linewidth=0.4, alpha=0.95)
        _axm.text(0.38, _yj, _sub, va="center", ha="left", fontsize=_fs_annot)
    _axm.set_xlim(0, 1.2)
    _axm.set_ylim(-0.6, len(SUBJECTS) - 0.4)
    _axm.axis("off")
    _stemm = OUT_F1 / "fig1_D_legend_markers"
    _figm.savefig(f"{_stemm}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _figm.savefig(f"{_stemm}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stemm.name}.png (+ .svg)")
    plt.close(_figm)
    return


if __name__ == "__main__":
    app.run()
