"""Figure F1 — the recurrence gradient (R1).

Split from the former `fig_F1_recurrence_and_taxonomy.py` (2026-06-10): R1 (the
continuous recurrence gradient) and R2 (the sources of recurrence) are two
figures. This script is R1.

R1/R2 separation (2026-06-10): R1 shows ONLY the recurrence distribution — dots
are a single neutral color, no taxonomy coloring. The source taxonomy
(content-eligible / run-onset / low-confidence / drift / unused) is entirely an
R2 construct and lives in Figure 2. The two figures describe the same ~45 active
states (logically connected) but never mix content: Figure 1 = the recurrence
axis, Figure 2 = decomposing that repertoire by why each state recurs.

R1's message is "we identify robust recurring brain states across subjects":
Panel A shows that states recur (continuous recurrence gradient, every subject);
Panel B shows the recurring repertoire is reproducible under independent re-fits
(split-half reliability). The recurrence x dwell-independence panel was moved out
of R1 (off-message; belongs with 06a temporal dynamics).

| Panel | Content | Chart family | Source | Output |
|---|---|---|---|---|
| A | Per-subject recurrence beeswarm (active states, neutral, density-stacked) + bounded half-KDE | point-based 1D + distribution | 06a state_summary_table.csv (recurrence_score) | fig1_A_recurrence_beeswarm.{pdf,png} |
| B | Per-subject split-half reliability scatter (recurrence half A vs half B, matched states) | scatter | 04rb_split_half (hungarian_matching + half_invariants + split_half_reliability) | fig1_B_split_half_reliability.{pdf,png} |

Key data facts (audited 2026-06-10):
  * Active states per subject (= state_summary_table.csv row count): sub-01 46,
    sub-02 46, sub-03 44, sub-04 44, sub-05 47, sub-06 42.
  * Split-half matched recurrence Spearman rho per subject: sub-01 0.65, sub-02
    0.68, sub-03 0.76, sub-04 0.82, sub-05 0.60, sub-06 0.76 (all p<1e-4) -> the
    recurring repertoire reproduces under independent re-fits on interleaved
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

    # Single neutral dot color — R1 carries no taxonomy, so dots are uncolored
    # by category. Deliberately not one of the R2 category colors.
    DOT_COLOR = "#44546A"
    return SUBJECTS, BLOCKS_DIR, SPLITHALF_DIR, OUT_F1, VT, DOT_COLOR


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
def panel_A_beeswarm(active, SUBJECTS, OUT_F1, SUBJECT_MARKERS, plt, np):
    """Panel A — per-subject recurrence beeswarm, colored by mean dwell time.

    x = recurrence (0 -> ~0.93); one row per subject (sub-01 bottom), each row
    drawn with that subject's marker shape (the project-wide subject key, shared
    with Figures 3C/5). Dots are stacked upward within fine x-bins (beeswarm) so
    vertical height reflects local density; continuous spread with no empty band
    on the recurrence axis = "gradient, not categories." Dot color = the state's
    mean dwell time (shared viridis scale, colorbar at right): dwell is a state
    property orthogonal to the source taxonomy (Figure 2), and its scatter across
    the recurrence axis shows recurrence and dwell are independent.

    Channels: x = recurrence, y(row)+marker = subject, stack height = density,
    color = mean dwell. No taxonomy encoding (that is Figure 2).
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    _all_dwell = np.concatenate(
        [active[_s]["mean_dwell_s"].values.astype(float) for _s in SUBJECTS]
    )
    _all_dwell = _all_dwell[np.isfinite(_all_dwell)]
    _cmap = cm.get_cmap("viridis")
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

    # 6 stacked long subplots (one per subject), sub-01 bottom, shared x-axis.
    _order = list(reversed(SUBJECTS))
    _fig, _axes = plt.subplots(
        len(SUBJECTS), 1, figsize=(7.8, 6.0), sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    for _ax, _sub in zip(_axes, _order):
        _rec, _dwell, _yoff = _prep[_sub]
        _ax.scatter(
            _rec, _yoff, s=22, c=_dwell, cmap=_cmap, norm=_norm,
            marker=SUBJECT_MARKERS[_sub], edgecolor="white", linewidth=0.3,
            alpha=0.95, zorder=3,
        )
        _ax.set_ylim(-0.6, _gmax + 0.6)
        _ax.set_yticks([])
        _ax.set_ylabel(_sub, rotation=0, ha="right", va="center", fontsize=8)
        _ax.set_xlim(-0.01, 1.0)
        _ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#E6E6E6", zorder=0)
        _ax.set_axisbelow(True)
        for _s in ("top", "right", "left"):
            _ax.spines[_s].set_visible(False)

    _axes[-1].set_xlabel("Recurrence (fraction of episodes active)", fontsize=10)
    _axes[-1].tick_params(axis="x", labelsize=8)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array([])
    _cbar = _fig.colorbar(_sm, ax=_axes, fraction=0.022, pad=0.02)
    _cbar.set_label("Mean dwell time (s)", fontsize=9)
    _cbar.ax.tick_params(labelsize=7)

    _stem = OUT_F1 / "fig1_A_recurrence_beeswarm"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"saved: {_stem}.pdf")
    plt.close(_fig)
    return


@app.cell
def load_split_half(SUBJECTS, SPLITHALF_DIR, json, np):
    """Per-subject split-half matched recurrence (half A vs half B).

    Two HMMs are fit independently on interleaved odd/even episode halves of the
    SAME subject (04rb). Their states are matched post-hoc by the Hungarian
    algorithm on parcel-space mean correlations (r > 0.3). For each matched pair
    we read that state's recurrence in half A and half B from the per-half
    recurrence vectors; rho is the reported raw Spearman (verified to match a
    direct recompute). Matching is strictly within-subject — no states are
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
def panel_B_split_half(splithalf, SUBJECTS, OUT_F1, DOT_COLOR, SUBJECT_MARKERS, plt, np):
    """Panel B — split-half reliability of recurrence, per subject.

    2x3 grid; x = recurrence in half A, y = recurrence in half B, one dot per
    Hungarian-matched state (independent HMM fits on interleaved episode halves
    of the SAME subject), drawn with that subject's marker (shared key, Figs
    3C/5). Identity line + per-subject Spearman rho. Points hugging the diagonal
    = the recurring repertoire reproduces under independent re-fits; this is the
    within-subject reliability behind "robust recurring states." Dots are a
    single neutral color (per-matched-state dwell is not defined in the split
    fits, so the dwell colormap applies to Panel A only).
    """
    _fig, _axes = plt.subplots(2, 3, figsize=(8.0, 5.6), sharex=True, sharey=True)
    _axes = _axes.ravel()

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes[_i]
        _d = splithalf[_sub]
        _ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--", zorder=1)
        _ax.scatter(_d["a"], _d["b"], s=26, c=DOT_COLOR,
                    marker=SUBJECT_MARKERS[_sub], edgecolor="white",
                    linewidth=0.4, alpha=0.95, zorder=3)
        _ax.text(0.05, 0.96, _sub, transform=_ax.transAxes, fontsize=8,
                 va="top", ha="left")
        _ax.text(0.05, 0.83, f"$\\rho$ = {_d['rho']:.2f}",
                 transform=_ax.transAxes, fontsize=8, va="top", ha="left")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.set_xticks([0, 0.5, 1.0])
        _ax.set_yticks([0, 0.5, 1.0])
        _ax.set_aspect("equal")
        _ax.tick_params(axis="both", labelsize=8)
        _ax.grid(True, linestyle=":", linewidth=0.5, color="#D8D8D8")
        _ax.set_axisbelow(True)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)

    for _i in (0, 3):
        _axes[_i].set_ylabel("Recurrence, half B", fontsize=9)
    for _i in (3, 4, 5):
        _axes[_i].set_xlabel("Recurrence, half A", fontsize=9)

    _fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.97,
                         wspace=0.18, hspace=0.22)
    _stem = OUT_F1 / "fig1_B_split_half_reliability"
    _fig.savefig(f"{_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"saved: {_stem}.pdf")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
