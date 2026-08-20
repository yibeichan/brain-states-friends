"""Figure S8 (supplementary) - Cross-stimulus validity & repertoire presence.

Supports the reframed R5 (transfer is clearest for audiovisual film and
weaker/variable for reduced-modality stimuli; the cross-modality ordering does
not isolate narrative content). Three validity/robustness views that are NOT main
findings and that explicitly disclose the fit confound behind R5:

| Panel | Content | Chart family |
|---|---|---|
| A | PCA subspace transfer: per (subject, condition) gap = Friends R² − condition R², including Rest. Near-zero and uniform (~0.95 R² everywhere, rest gaps −0.011 to +0.007) → the Friends low-dimensional subspace generalizes to all conditions including rest; the transfer differences are NOT a subspace-fit artifact. | point-1D strip |
| B | HMM model-fit vs transfer: per-condition transfer ρ against the HMM log-likelihood gap (held-out-Friends − condition LL/sample). Within Movie10, ρ declines as fit worsens (the confound); Harry Potter is an OUTLIER (best fit, modest ρ), so transfer is not a single function of fit. Rest shows the temporal-fit dissociation: spatial subspace transfers (Panel A) but LL gaps span the widest range of any condition (sub-01 extreme at 16.0, negative ρ). Disclosure panel for the "within-film grading is fit-confounded" caveat. | scatter |
| C | Repertoire presence, gapped radial-gauge grid: 4 rows (M10/HP/PP/Rest) × 6 subjects. Per category, arc width = subject's Friends count, outlined to full extent, filled for the present fraction (active at FO>0.01 in ≥1 run). Caveat: Viterbi forces every TR onto a state, so presence is biased upward - this is a descriptive existence view, not a clean transfer test. | pie / donut |

Rest is SUPPLEMENTARY-ONLY (placement decided 2026-08-17): it appears in every
panel here and NOT in main Figure 4. No surrogate null was run for rest, so all
rest values are descriptive.

Source-truth audit for rest (rest_05 A1, Friends-active semantics, audited
2026-08-16): ρ = −0.289/0.324/0.138/0.099/0.274/0.445 (sub-01..06; 2/6
uncorrected p<0.05: sub-02 p=0.028, sub-06 p=0.003); all six subjects have
rest data (unlike HP/PP, which lack sub-04). PCA transfer gaps −0.011 to
+0.007 (rest R² 0.943–0.962). LL gaps 0.49–15.98/sample; sub-01 is the fit
outlier (gap 15.98, negative ρ); only sub-02 sits above the heuristic
uniform-assignment LL baseline. Rest mean ρ (0.165) is not below PP's (0.101):
raw ρ carries a stimulus-independent covariance/stationarity floor (R5
phase-null finding), so orderings among the weak conditions are
fit/language/modality-confounded and uninterpretable.

Run:
    marimo edit script/fig_S08_cross_stimulus_validity.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import glob
    import json
    import os
    import pickle
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from scipy import stats as sp_stats

    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import (
        SUBJECT_MARKERS,
        SUBJECT_NEUTRAL,
        apply_publication_style,
    )

    apply_publication_style()
    return (
        Path, glob, json, np, os, pd, pickle, plt, sp_stats,
        SUBJECT_MARKERS, SUBJECT_NEUTRAL,
    )


@app.cell
def config(Path, os):
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
    OUT = SCRATCH_DIR / "output" / "manuscript_figures" / "figS08"
    OUT.mkdir(parents=True, exist_ok=True)

    out_root = SCRATCH_DIR / "output"
    REC_DIR = out_root / "05a_recurrence_analysis" / PARCELLATION
    FLAGS_DIR = out_root / "05e_temporal_trend_a4" / PARCELLATION
    DEC = {"M10": out_root / "m10_04_decoded" / PARCELLATION,
           "HP": out_root / "hp_04_decoded" / PARCELLATION,
           "PP": out_root / "pp_04_decoded" / PARCELLATION,
           "REST": out_root / "rest_04_decoded" / PARCELLATION}
    PROJ = {"M10": out_root / "m10_03_projected" / PARCELLATION,
            "HP": out_root / "hp_03_projected" / PARCELLATION,
            "PP": out_root / "pp_03_projected" / PARCELLATION,
            "REST": out_root / "rest_03_projected" / PARCELLATION}
    XVAL = {"M10": out_root / "m10_05_cross_validation" / PARCELLATION,
            "HP": out_root / "hp_05_cross_validation" / PARCELLATION,
            "PP": out_root / "pp_05_cross_validation" / PARCELLATION,
            "REST": out_root / "rest_05_cross_validation" / PARCELLATION}
    LL_FILE = {"M10": "movie_ll_summary.json", "HP": "hp_ll_summary.json",
               "PP": "pp_ll_summary.json", "REST": "rest_ll_summary.json"}
    PRESENCE_FO = 0.01

    TAXONOMY_ORDER = ["Content-eligible", "Run-onset-anchored", "Low-confidence",
                      "Drift-anchored", "Unused + rare"]
    TAXONOMY_COLORS = {"Content-eligible": "#0C7BDC", "Run-onset-anchored": "#FFC20A",
                       "Low-confidence": "#5A5A5A", "Drift-anchored": "#D35FB7",
                       "Unused + rare": "#CCCCCC"}
    TAXONOMY_MAP = {"eligible_for_content_analysis": "Content-eligible",
                    "run_onset_anchored": "Run-onset-anchored",
                    "low_confidence": "Low-confidence", "season_temporal": "Drift-anchored",
                    "unused": "Unused + rare", "rare": "Unused + rare"}
    FILMS = ["wolf", "figures", "bourne", "life"]
    return (SCRATCH_DIR, PARCELLATION, VT, SUBJECTS, OUT, REC_DIR, FLAGS_DIR,
            DEC, PROJ, XVAL, LL_FILE, PRESENCE_FO,
            TAXONOMY_ORDER, TAXONOMY_COLORS, TAXONOMY_MAP, FILMS)


@app.cell
def load_data(SUBJECTS, REC_DIR, FLAGS_DIR, DEC, VT, PRESENCE_FO,
              TAXONOMY_MAP, np, pd, pickle):
    """Per-subject recurrence, taxonomy, and presence masks for the donut grid."""
    presence = {}
    for _sub in SUBJECTS:
        _rec = np.load(REC_DIR / _sub / VT / "recurrence_scores.npy")
        _flags = pd.read_csv(FLAGS_DIR / _sub / VT / "state_flags.csv")
        _tax = _flags["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare").values
        _e = {"rec": _rec, "tax": _tax, "active": _rec > 0}
        for _stim, _d in DEC.items():
            _p = _d / _sub / VT / "fractional_occupancy.pkl"
            if _p.exists():
                _fo = np.array(list(pickle.load(open(_p, "rb")).values()))
                _e[f"{_stim}_present"] = (_fo > PRESENCE_FO).any(axis=0)
            else:
                _e[f"{_stim}_present"] = None
        presence[_sub] = _e
    return (presence,)


@app.cell
def panel_A_pca_transfer(SUBJECTS, PROJ, VT, FILMS, OUT, glob, json, np, plt,
                         SUBJECT_MARKERS, SUBJECT_NEUTRAL):
    """Panel A - PCA subspace transfer gap per (subject, condition), per Movie10
    film + HP + PP (parallel to Panel B). Near-zero everywhere ⇒ the Friends
    subspace captures every condition comparably; the transfer-ρ gradient is NOT
    a subspace-fit artifact (even the documentary and PP, which transfer least,
    keep ~0.95 subspace R²)."""
    _xlabels = ["Wolf of\nWall St.", "Hidden\nFigures", "Bourne", "Life",
                "Harry\nPotter", "Petit\nPrince", "Rest"]
    _groups = [(0, 3, "Audiovisual film\n(Movie10)"), (4, 4, "Visual\nreading"),
               (5, 5, "Audio\nlistening"), (6, 6, "No\nstimulus")]
    _xstims = ["HP", "PP", "REST"]
    _n_x = len(_xlabels)
    _rng = np.random.default_rng(7)

    def _film_gap(_sub, _film):
        _f = glob.glob(str(PROJ["M10"] / _sub / VT / "pca_transfer_diagnostic.json"))
        if not _f:
            return None
        _d = json.load(open(_f[0]))
        _fr = _d.get("friends_r2_n_pcs")
        _fm = _d.get("r2_by_movie_type", {}).get(_film, {}).get("r2_n_pcs")
        return None if (_fr is None or _fm is None) else _fr - _fm

    def _stim_gap(_sub, _stim):
        _f = glob.glob(str(PROJ[_stim] / _sub / VT / "pca_transfer_diagnostic.json"))
        if not _f:
            return None
        _d = json.load(open(_f[0]))
        return _d.get("transfer_gap")

    _fig, _ax = plt.subplots(figsize=(4.8, 3.3))
    for _xi in range(_n_x):
        for _sub in SUBJECTS:
            if _xi < len(FILMS):
                _gap = _film_gap(_sub, FILMS[_xi])
            else:
                _gap = _stim_gap(_sub, _xstims[_xi - len(FILMS)])
            if _gap is None:
                continue
            _ax.scatter(_xi + _rng.uniform(-0.12, 0.12), _gap, s=28,
                        color=SUBJECT_NEUTRAL, alpha=0.6,
                        marker=SUBJECT_MARKERS[_sub], edgecolor="white", linewidth=0.4)
    _ax.axhline(0, color="#999999", lw=1.0, ls="--", zorder=1)
    for _d in (len(FILMS) - 0.5, len(FILMS) + 0.5, len(FILMS) + 1.5):
        _ax.axvline(_d, color="#BBBBBB", lw=1.0, ls="--", zorder=1)
    _tr = _ax.get_xaxis_transform()
    for _lo, _hi, _lab in _groups:
        _ax.text((_lo + _hi) / 2.0, -0.22, _lab, transform=_tr, ha="center",
                 va="top", fontsize=6.5, color="#333333")
    _ax.set_xticks(range(_n_x))
    _ax.set_xticklabels(_xlabels, fontsize=6.5)
    _ax.set_ylabel("PCA subspace transfer gap\n(Friends R² − condition R²)", fontsize=7)
    _ax.set_ylim(-0.03, 0.10)
    _ax.set_xlim(-0.5, _n_x - 0.5)
    _ax.tick_params(axis="y", labelsize=6)
    _ax.tick_params(axis="x", length=0)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.subplots_adjust(left=0.20, right=0.97, bottom=0.24, top=0.97)
    _fig.savefig(OUT / "figS08_A_pca_transfer.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT / "figS08_A_pca_transfer.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT / "figS08_A_pca_transfer.svg", bbox_inches="tight", pad_inches=0.02)
    print("saved figS08_A_pca_transfer")
    plt.close(_fig)
    return


@app.cell
def panel_B_fit_vs_transfer(SUBJECTS, XVAL, DEC, VT, LL_FILE, FILMS, OUT,
                            glob, json, np, plt):
    """Panel B - transfer ρ vs HMM log-likelihood gap, per condition.

    Within Movie10 ρ declines as the LL gap grows (the fit confound); Harry
    Potter is an outlier (best fit, modest ρ), so transfer is not a single
    function of fit. Condition means labeled; per-subject points drawn faint.
    """
    _cond_color = {"wolf": "#1B7837", "figures": "#5AAE61", "bourne": "#F1A340",
                   "life": "#B2182B", "HP": "#542788", "PP": "#8073AC",
                   "REST": "#4D4D4D"}
    _cond_label = {"wolf": "Wolf", "figures": "Figures", "bourne": "Bourne",
                   "life": "Life", "HP": "Harry Potter", "PP": "Petit Prince",
                   "REST": "Rest"}

    def _rho(_stim, _sub):
        _f = glob.glob(str(XVAL[_stim if _stim in ("HP", "PP", "REST") else "M10"]
                           / _sub / VT / "cross_stimulus_summary.json"))
        if not _f:
            return None
        _d = json.load(open(_f[0]))
        if _stim in FILMS:
            return _d["A2_per_type"].get(_stim, {}).get("spearman_rho")
        return _d["A1_recurrence_correlation"]["spearman_rho"]

    def _llgap(_stim, _sub):
        _key = "M10" if _stim in FILMS else _stim
        _f = glob.glob(str(DEC[_key] / _sub / VT / LL_FILE[_key]))
        if not _f:
            return None
        _d = json.load(open(_f[0]))
        _ft = _d.get("friends_test_ll_per_sample")
        if _stim in FILMS:
            _pt = _d.get("per_type", {}).get(_stim, {}).get("ll_per_sample")
            return None if (_ft is None or _pt is None) else _ft - _pt
        for _k in ("ll_gap_friends_minus_movie", "ll_gap_friends_minus_hp",
                   "ll_gap_friends_minus_pp", "ll_gap_friends_minus_rest",
                   "ll_gap"):
            if _d.get(_k) is not None:
                return _d[_k]
        return None

    _conds = FILMS + ["HP", "PP", "REST"]
    _fig, _ax = plt.subplots(figsize=(4.2, 3.4))
    for _c in _conds:
        _xs, _ys = [], []
        for _sub in SUBJECTS:
            _r, _g = _rho(_c, _sub), _llgap(_c, _sub)
            if _r is None or _g is None:
                continue
            _xs.append(_g); _ys.append(_r)
            _ax.scatter(_g, _r, s=14, color=_cond_color[_c], alpha=0.30,
                        edgecolor="none", zorder=2)
        if _xs:
            _mx, _my = float(np.mean(_xs)), float(np.mean(_ys))
            _ax.scatter(_mx, _my, s=80, color=_cond_color[_c], edgecolor="white",
                        linewidth=0.8, zorder=4)
            # Rest's mean nearly coincides with Life's; label it to the lower left.
            _off, _ha = ((-6, -12), "right") if _c == "REST" else ((6, 4), "left")
            _ax.annotate(_cond_label[_c], (_mx, _my), textcoords="offset points",
                         xytext=_off, ha=_ha, fontsize=6.5, color=_cond_color[_c])
            print(f"  {_c}: mean gap={_mx:.2f} rho={_my:.3f}")
    _ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    _ax.set_xlabel("HMM model-fit gap\n(held-out Friends − stimulus LL/sample)", fontsize=7)
    _ax.set_ylabel("Recurrence → occupancy transfer ρ", fontsize=7)
    _ax.tick_params(labelsize=6)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.96)
    _fig.savefig(OUT / "figS08_B_fit_vs_transfer.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT / "figS08_B_fit_vs_transfer.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT / "figS08_B_fit_vs_transfer.svg", bbox_inches="tight", pad_inches=0.02)
    print("saved figS08_B_fit_vs_transfer")
    plt.close(_fig)
    return


@app.cell
def panel_C_presence_donut(SUBJECTS, presence, OUT, plt,
                           TAXONOMY_ORDER, TAXONOMY_COLORS):
    """Panel C - repertoire presence, gapped radial-gauge grid (4 conditions × 6 subjects).

    Arc width = subject's Friends count for the category; outlined to full
    extent, filled for the present fraction. Caveat (caption): Viterbi forces
    every TR onto a state, so presence is biased upward - a descriptive existence
    view, not a clean transfer test.
    """
    from matplotlib.patches import Wedge as _Wedge

    _stims = [("M10", "Movie10"), ("HP", "Harry Potter"), ("PP", "Petit Prince"),
              ("REST", "Rest")]
    _gap_deg, _radius, _width = 5.0, 1.0, 0.42

    def _draw(_ax, _fr, _pr):
        _drawn = [c for c in TAXONOMY_ORDER if _fr[c] > 0]
        _total = sum(_fr[c] for c in _drawn)
        _avail = 360.0 - _gap_deg * len(_drawn)
        _ang = 90.0
        for _cat in _drawn:
            _span = _fr[_cat] / _total * _avail
            _t2, _t1 = _ang, _ang - _span
            _col = TAXONOMY_COLORS[_cat]
            _ax.add_patch(_Wedge((0, 0), _radius, _t1, _t2, width=_width,
                                 facecolor="none", edgecolor=_col, linewidth=1.1))
            _frac = _pr[_cat] / _fr[_cat]
            if _frac > 0:
                _ax.add_patch(_Wedge((0, 0), _radius, _t2 - _span * _frac, _t2,
                                     width=_width, facecolor=_col, edgecolor="none"))
            _ang = _t1 - _gap_deg
        _ax.set_xlim(-1.15, 1.15)
        _ax.set_ylim(-1.15, 1.15)
        _ax.set_aspect("equal")
        _ax.axis("off")

    _fig, _axes = plt.subplots(len(_stims), len(SUBJECTS), figsize=(6.7, 4.9))
    for _ri, (_stim, _row_lab) in enumerate(_stims):
        for _ci, _sub in enumerate(SUBJECTS):
            _ax = _axes[_ri, _ci]
            _e = presence[_sub]
            _pmask = _e[f"{_stim}_present"]
            if _pmask is None:
                _ax.text(0.5, 0.5, "no data", transform=_ax.transAxes,
                         ha="center", va="center", fontsize=6.5, color="#AAAAAA")
                _ax.axis("off")
                continue
            _act = _e["active"]
            _fr = {c: int((_act & (_e["tax"] == c)).sum()) for c in TAXONOMY_ORDER}
            _pr = {c: int((_act & _pmask & (_e["tax"] == c)).sum()) for c in TAXONOMY_ORDER}
            _draw(_ax, _fr, _pr)
            _ax.text(0, -1.30, f"{int((_act & _pmask).sum())}/{int(_act.sum())}",
                     ha="center", va="top", fontsize=6.5, color="#444444")
            if _ri == 0:
                _ax.set_title(_sub, fontsize=6.5, pad=3)
            if _ci == 0:
                _ax.text(-1.62, 0, _row_lab, rotation=90, ha="center", va="center",
                         fontsize=6.5)
    _fig.subplots_adjust(left=0.07, right=0.99, bottom=0.04, top=0.92,
                         wspace=0.28, hspace=0.40)
    _fig.savefig(OUT / "figS08_C_presence_donut.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT / "figS08_C_presence_donut.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT / "figS08_C_presence_donut.svg", bbox_inches="tight", pad_inches=0.02)
    print("saved figS08_C_presence_donut")
    plt.close(_fig)
    return


@app.cell
def donut_legend(OUT, plt, TAXONOMY_ORDER, TAXONOMY_COLORS):
    """Legend for the presence donut (Panel C): 5 categories + present/absent arcs."""
    import matplotlib.patches as _mpatches
    _handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                           markerfacecolor=TAXONOMY_COLORS[_c], markeredgecolor="white",
                           label=_c) for _c in TAXONOMY_ORDER]
    _handles.append(_mpatches.Patch(facecolor="#5A5A5A", edgecolor="#5A5A5A",
                                    linewidth=1.0, label="Present (filled arc)"))
    _handles.append(_mpatches.Patch(facecolor="none", edgecolor="#5A5A5A",
                                    linewidth=1.0, label="Absent (outline only)"))
    _fig = plt.figure(figsize=(2.25, 1.75))
    _fig.legend(handles=_handles, loc="center", frameon=False, fontsize=6.5,
                handletextpad=0.5, labelspacing=0.6)
    _fig.savefig(OUT / "figS08_donut_legend.pdf", bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT / "figS08_donut_legend.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT / "figS08_donut_legend.svg", bbox_inches="tight", pad_inches=0.02)
    print("saved figS08_donut_legend")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
