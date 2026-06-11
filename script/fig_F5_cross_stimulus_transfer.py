"""Figure F5 — Cross-stimulus recurrence transfer (R5).

One marimo notebook per figure (see `2026-05-24_manuscript_version_scope.md`).
Panels are per-cell, saved as separate .pdf + .png mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

R5 claim (reframed 2026-06-10 — see decision memory #611): a state's recurrence
rank in Friends predicts how much the brain occupies it during other stimuli,
and the transfer is GRADED BY SIMILARITY TO FRIENDS — strongest for the most
Friends-like stimulus (audiovisual narrative film, Movie10), weaker and variable
for reduced-modality reading (Harry Potter) and listening (Petit Prince). The
earlier "tracks social-narrative content, not modality" framing was DROPPED: the
cross-modality ρ ordering contradicts a content account (HP, the least narrative
stimulus, transfers higher on average than PP, a coherent narrative), and the
within-Movie10 spread co-varies with model fit + low-level audiovisual features,
so it is reported descriptively and not attributed to content. The presence and
the fit diagnostics live in the supplementary cross-stimulus validity figure.

Panel plan (chart families distinct within the figure). Filenames carry the
panel letter; the user arranges the composite.

| Panel | Content | Chart family | Source |
|---|---|---|---|
| A | Per-subject recurrence→FO scatter, 2×3 small multiples (Movie10; x=Friends recurrence, y=mean M10 FO; subject = marker shape + OLS line + per-subject ρ). Dots colored by R2 taxonomy category (Fig 2 colors), showing content-eligible states anchor the relationship. Movie10 = the strong (most Friends-like) end of the gradient in B. | scatter (small multiples) | m10_04 fractional_occupancy.pkl + 05a recurrence + 05e_a4 state_flags |
| B | Transfer-ρ by condition, grouped by stimulus type — Audiovisual film (4 Movie10 films) | Visual reading (Harry Potter) | Audio listening (Petit Prince); per-subject markers (shape = subject), dark cohort-mean line over films + ticks for HP/PP. The generalization gradient; within-film spread shown descriptively (fit/AV-confounded, see supplement), NOT as a content axis. | point-1D strip | m10 A2_per_type + hp/pp A1 |

A `fig5_taxonomy_legend` file gives the 5-category color key for Panel A.

Modalities (verified against Methods draft): Movie10 = audiovisual films; Harry
Potter = word-by-word RSVP reading (visual-only, no audio); Petit Prince =
audiobook (audio-only, no visual). sub-04 has no HP/PP data, so HP/PP markers are
absent for sub-04 in B.

The presence donut grid and the validity diagnostics (PCA transfer R² uniform
~0.95; HMM LL gap uneven, ρ↔fit relationship with the HP outlier) move to the
supplementary `fig_S_cross_stimulus_validity` figure.

Source-truth audit (reproduced from raw arrays):
  * Panel A: reproduced sub-01 M10 ρ = 0.5486 (n_active=46) = summary JSON exactly.
  * Full-repertoire ρ: M10 0.264–0.773 (5/6 sig), HP 0.232–0.444 (2/5 sig),
    PP −0.018–0.332 (1/5 sig); sub-04 has no HP/PP data.
  * Per-condition mean ρ: wolf 0.69, figures 0.63, bourne 0.40, life 0.16,
    HP 0.32, PP 0.10. Cross-modality ordering (HP > Life > PP) contradicts a
    content account; within-M10 spread tracks LL-fit gap (0.92/1.54/2.43/4.98).

Run:
    marimo edit script/fig_F4_cross_stimulus_transfer.py
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
    from scipy import stats as sp_stats

    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import (
        SUBJECT_ACCENT,
        SUBJECT_MARKERS,
        SUBJECT_NEUTRAL,
        apply_publication_style,
    )

    apply_publication_style()

    return (
        Path, json, np, os, pd, pickle, plt, sp_stats,
        SUBJECT_ACCENT, SUBJECT_MARKERS, SUBJECT_NEUTRAL,
    )


@app.cell
def config(Path, os):
    """Paths, subject list, stimulus labels, output dir."""
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
    EXEMPLAR = "sub-01"

    # internal stim key -> (05 output dir prefix, display label)
    STIMULI = [
        ("m10", "Movie10"),
        ("hp", "Harry Potter"),
        ("pp", "Petit Prince"),
    ]

    RECUR_DIR = SCRATCH_DIR / "output" / "05a_recurrence_analysis" / PARCELLATION
    FLAGS_DIR = SCRATCH_DIR / "output" / "05e_temporal_trend_a4" / PARCELLATION
    M10_DEC_DIR = SCRATCH_DIR / "output" / "m10_04_decoded" / PARCELLATION
    HP_DEC_DIR = SCRATCH_DIR / "output" / "hp_04_decoded" / PARCELLATION
    PP_DEC_DIR = SCRATCH_DIR / "output" / "pp_04_decoded" / PARCELLATION

    # Per-run FO above which a state counts as "present" in that run (matches
    # the m10_05 / cross-stimulus active-state convention).
    PRESENCE_FO = 0.01

    def xstim_dir(stim):
        return SCRATCH_DIR / "output" / f"{stim}_05_cross_validation" / PARCELLATION

    OUT_F5 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig5"
    OUT_F5.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR, PARCELLATION, VT, SUBJECTS, EXEMPLAR, STIMULI,
        RECUR_DIR, FLAGS_DIR, M10_DEC_DIR, HP_DEC_DIR, PP_DEC_DIR,
        PRESENCE_FO, xstim_dir, OUT_F5,
    )


@app.cell
def load_summaries(SUBJECTS, STIMULI, xstim_dir, VT, json):
    """Load cross_stimulus_summary.json for every (stimulus, subject).

    Stored as summaries[stim][sub] = dict, or None when the subject has no data
    for that stimulus (sub-04 lacks HP/PP).
    """
    summaries = {}
    for _stim, _ in STIMULI:
        summaries[_stim] = {}
        for _sub in SUBJECTS:
            _p = xstim_dir(_stim) / _sub / VT / "cross_stimulus_summary.json"
            if _p.exists():
                with open(_p) as _f:
                    summaries[_stim][_sub] = json.load(_f)
            else:
                summaries[_stim][_sub] = None
    for _stim, _ in STIMULI:
        _have = [s for s in SUBJECTS if summaries[_stim][s] is not None]
        print(f"{_stim}: {len(_have)} subjects with data")
    return (summaries,)


@app.cell
def taxonomy_constants():
    """R2 taxonomy categories — labels + colors, shared with Figure 2.

    Identical to fig_F1's taxonomy_constants cell; kept in sync by hand. Maps the
    raw 05e_a4 `summary_category` values onto the 5 display categories so Panel A
    dots and the cross-stimulus presence panel use the same color key as Fig 2.
    """
    TAXONOMY_ORDER = [
        "Content-eligible",
        "Run-onset-anchored",
        "Low-confidence",
        "Drift-anchored",
        "Unused + rare",
    ]
    TAXONOMY_COLORS = {
        "Content-eligible":    "#0C7BDC",
        "Run-onset-anchored":  "#FFC20A",
        "Low-confidence":      "#5A5A5A",
        "Drift-anchored":      "#D35FB7",
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
def load_presence(
    SUBJECTS, RECUR_DIR, FLAGS_DIR, M10_DEC_DIR, HP_DEC_DIR, PP_DEC_DIR,
    VT, PRESENCE_FO, TAXONOMY_MAP, np, pd, pickle,
):
    """Per-subject recurrence, taxonomy category, and cross-stimulus presence.

    For each subject collects: Friends recurrence scores, the per-state taxonomy
    category (05e_a4 state_flags), the mean Movie10 FO per state, and a boolean
    "present" mask per stimulus (active at FO > PRESENCE_FO in ≥1 run of that
    stimulus). sub-04 lacks Harry Potter and Petit Prince data, so those masks
    are None for sub-04.
    """
    _dec_dirs = {"M10": M10_DEC_DIR, "HP": HP_DEC_DIR, "PP": PP_DEC_DIR}

    def _present_mask(_dec_dir, _sub):
        _p = _dec_dir / _sub / VT / "fractional_occupancy.pkl"
        if not _p.exists():
            return None, None
        with open(_p, "rb") as _f:
            _fo = np.array(list(pickle.load(_f).values()))  # runs × states
        return (_fo > PRESENCE_FO).any(axis=0), _fo.mean(axis=0)

    presence = {}
    for _sub in SUBJECTS:
        _rec = np.load(RECUR_DIR / _sub / VT / "recurrence_scores.npy")
        _flags = pd.read_csv(FLAGS_DIR / _sub / VT / "state_flags.csv")
        _tax = _flags["summary_category"].map(TAXONOMY_MAP).fillna("Unused + rare").values
        _entry = {"rec": _rec, "tax": _tax, "active": _rec > 0}
        for _stim, _d in _dec_dirs.items():
            _mask, _mean_fo = _present_mask(_d, _sub)
            _entry[f"{_stim}_present"] = _mask
            if _stim == "M10":
                _entry["M10_mean_fo"] = _mean_fo
        presence[_sub] = _entry

    for _sub in SUBJECTS:
        _e = presence[_sub]
        _na = int(_e["active"].sum())
        _np = int((_e["active"] & _e["M10_present"]).sum())
        print(f"{_sub}: Friends-active {_na}, present in M10 {_np}")
    return (presence,)


@app.cell
def panel_A_per_subject_scatter(
    SUBJECTS, VT, summaries, presence, OUT_F5,
    np, plt, sp_stats, SUBJECT_MARKERS, TAXONOMY_COLORS,
):
    """Panel A — per-subject recurrence→FO scatter, 2×3 small multiples (Movie10).

    One subplot per subject (shared x and y axes for comparability): x = Friends
    recurrence score (05a), y = mean Movie10 fractional occupancy across runs
    (m10_04), one point per Friends-active state. Each subject keeps its own
    marker shape (SUBJECT_MARKERS — consistent with the transfer-ρ panel) and a
    red OLS guide line; the per-subject Spearman ρ (asserted against the summary
    JSON) is annotated in-panel with a significance star.

    Dot fill = R2 taxonomy category (same colors as Figure 2), showing that the
    content-eligible states anchor the positive relationship. Movie10 is the
    most Friends-like (audiovisual narrative) stimulus, so this panel is the
    strong end of the generalization gradient quantified in panel B; the
    repertoire-presence view and the validity diagnostics live in the
    supplementary cross-stimulus figure.

    Recurrence and occupancy are both within-subject quantities, so each panel's
    ρ is the correct per-subject statistic; no pooled correlation is drawn.
    Width matches the transfer-ρ panel so the two stack as one figure column.
    """
    _fig, _axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True, sharey=True)
    _axes = _axes.ravel()

    _rhos = []
    for _k, _sub in enumerate(SUBJECTS):
        _ax = _axes[_k]
        _e = presence[_sub]
        _active = _e["active"]
        _x = _e["rec"][_active]
        _y = _e["M10_mean_fo"][_active]
        _colors = [TAXONOMY_COLORS[_c] for _c in _e["tax"][_active]]

        _rho, _p = sp_stats.spearmanr(_x, _y)
        _j = summaries["m10"][_sub]["A1_recurrence_correlation"]
        assert abs(_rho - _j["spearman_rho"]) < 1e-3, (
            f"{_sub}: reproduced ρ {_rho:.4f} != summary {_j['spearman_rho']:.4f}"
        )
        _rhos.append(_rho)

        _mk = SUBJECT_MARKERS[_sub]
        _ax.scatter(_x, _y, s=20, c=_colors, alpha=0.75, marker=_mk,
                    edgecolor="white", linewidth=0.3, zorder=3)

        _b, _a = np.polyfit(_x, _y, 1)
        _xx = np.linspace(_x.min(), _x.max(), 40)
        # Fade the guide line for the non-significant subject (sub-02, p≈0.08)
        _line_alpha = 1.0 if _p < 0.05 else 0.35
        _ax.plot(_xx, _b * _xx + _a, color="#D62728", lw=1.4, alpha=_line_alpha,
                 zorder=2)

        _star = "*" if _p < 0.05 else ""
        _ax.text(0.05, 0.95, f"{_sub.replace('sub-', 'S')}\nρ = {_rho:.2f}{_star}",
                 transform=_ax.transAxes, ha="left", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#DDDDDD", lw=0.5))
        _ax.tick_params(labelsize=7)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)

    print(f"Panel A per-subject M10 ρ: {[f'{r:.2f}' for r in _rhos]}")
    _fig.supxlabel("Friends recurrence score", fontsize=10)
    _fig.supylabel("Mean Movie10 fractional occupancy", fontsize=10)
    _fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.98,
                         wspace=0.10, hspace=0.16)
    _out_pdf = OUT_F5 / "fig5_A_recurrence_fo_scatter.pdf"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT_F5 / "fig5_A_recurrence_fo_scatter.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT_F5 / "fig5_A_recurrence_fo_scatter.svg",
                 bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_out_pdf}")
    plt.close(_fig)
    return


@app.cell
def panel_B_transfer_by_condition(
    SUBJECTS, summaries, OUT_F5, np, plt,
    SUBJECT_NEUTRAL, SUBJECT_MARKERS,
):
    """Panel B — recurrence→occupancy transfer ρ by condition, grouped by stimulus type.

    The generalization gradient: a frozen Friends-trained state model transfers
    most strongly to the most Friends-like stimulus (audiovisual narrative film,
    Movie10) and weakly / variably to reduced-modality stimuli — visual-only
    word-by-word reading (Harry Potter) and audio-only listening (Petit Prince).

    Conditions are grouped by stimulus type, not ordered as a content axis. The
    spread across the four Movie10 films is shown descriptively; it co-varies
    with HMM model fit and with low-level audiovisual features (faces/speech/cut
    rate) — see the supplementary cross-stimulus validity figure — so it is NOT
    attributed to narrative content. Note that the reduced-modality conditions do
    not follow narrative content either: Harry Potter (least narrative) transfers
    higher on average than Petit Prince (a coherent narrative), inconsistent with
    a content account and consistent with similarity-graded generalization.

    Per-subject markers (shape = subject, as in panel A; single neutral color).
    Cohort mean: dark line over the Movie10 films, dark ticks for HP and PP.
    Per-condition cohort significance: Movie10 5/6 subjects, HP 2/5, PP 1/5;
    sub-04 lacks HP/PP, so only 5 markers appear in those groups.
    """
    _films = [("wolf", "Wolf of\nWall St."),
              ("figures", "Hidden\nFigures"),
              ("bourne", "Bourne"),
              ("life", "Life")]
    _xlabels = [lab for _, lab in _films] + ["Harry\nPotter", "Petit\nPrince"]
    _groups = [(0, len(_films) - 1, "Audiovisual film (Movie10)"),
               (len(_films), len(_films), "Visual\nreading"),
               (len(_films) + 1, len(_films) + 1, "Audio\nlistening")]
    _n_x = len(_xlabels)
    _rng = np.random.default_rng(20260526)

    def _cond_rho(_sub, _xi):
        if _xi < len(_films):
            return summaries["m10"][_sub]["A2_per_type"][_films[_xi][0]]["spearman_rho"]
        _stim = "hp" if _xi == len(_films) else "pp"
        _d = summaries[_stim][_sub]
        return None if _d is None else _d["A1_recurrence_correlation"]["spearman_rho"]

    _fig, _ax = plt.subplots(figsize=(7.4, 4.4))
    _mean_color = "#333333"

    # Dividers between the three stimulus-type groups (no causal continuum).
    for _d in (len(_films) - 0.5, len(_films) + 0.5):
        _ax.axvline(_d, color="#BBBBBB", lw=1.0, ls="--", zorder=1)

    # Per-subject markers (neutral; subject = shape).
    for _sub in SUBJECTS:
        for _xi in range(_n_x):
            _v = _cond_rho(_sub, _xi)
            if _v is None:
                continue
            _jit = _rng.uniform(-0.12, 0.12)
            _ax.scatter(
                _xi + _jit, _v, s=30, color=SUBJECT_NEUTRAL, alpha=0.55,
                marker=SUBJECT_MARKERS[_sub], edgecolor="white",
                linewidth=0.4, zorder=3,
            )

    # Cohort means: dark line over M10 films, dark ticks for HP/PP
    _m10_means = [float(np.mean([_cond_rho(_s, _xi) for _s in SUBJECTS]))
                  for _xi in range(len(_films))]
    _ax.plot(range(len(_films)), _m10_means, color=_mean_color, lw=2.0, zorder=5,
             label="Cohort mean")
    for _xi in (len(_films), len(_films) + 1):
        _vals = [_cond_rho(_s, _xi) for _s in SUBJECTS if _cond_rho(_s, _xi) is not None]
        _mean = float(np.mean(_vals))
        _ax.plot([_xi - 0.22, _xi + 0.22], [_mean, _mean], color=_mean_color,
                 lw=2.0, zorder=5)
        print(f"  cond {_xlabels[_xi].split(chr(10))[0]}: mean ρ={_mean:.3f} (n={len(_vals)})")
    for _xi in range(len(_films)):
        print(f"  film {_films[_xi][0]}: mean ρ={_m10_means[_xi]:.3f}")

    _ax.axhline(0, color="#999999", lw=1.0, ls="--", zorder=1)

    # Stimulus-type group labels beneath the per-condition ticks.
    _tr = _ax.get_xaxis_transform()
    for _lo, _hi, _lab in _groups:
        _ax.text((_lo + _hi) / 2.0, -0.18, _lab, transform=_tr, ha="center",
                 va="top", fontsize=8, color="#333333")

    _ax.set_xticks(range(_n_x))
    _ax.set_xticklabels(_xlabels, fontsize=8)
    _ax.set_ylabel("Recurrence → occupancy transfer ρ", fontsize=10)
    _ax.set_xlim(-0.5, _n_x - 0.5)
    _ax.tick_params(axis="y", labelsize=8)
    _ax.tick_params(axis="x", length=0)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _h_mean = plt.Line2D([], [], color=_mean_color, lw=2.0, label="Cohort mean")
    _ax.legend(handles=[_h_mean], loc="upper right",
               frameon=False, fontsize=8, handletextpad=0.4)

    _fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.95)
    _out_pdf = OUT_F5 / "fig5_B_transfer_by_condition.pdf"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT_F5 / "fig5_B_transfer_by_condition.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT_F5 / "fig5_B_transfer_by_condition.svg",
                 bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_out_pdf}")
    plt.close(_fig)
    return


@app.cell
def taxonomy_legend(OUT_F5, plt, TAXONOMY_ORDER, TAXONOMY_COLORS):
    """Taxonomy color legend for Panel A (separate mini-file).

    Five R2 taxonomy-category swatches, matching Figure 2's colors and Panel A's
    dot fills. (The cross-stimulus presence donut and its present/absent glyphs
    now live in the supplementary cross-stimulus validity figure.)
    """
    _handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                           markerfacecolor=TAXONOMY_COLORS[_c], markeredgecolor="white",
                           label=_c) for _c in TAXONOMY_ORDER]
    _fig = plt.figure(figsize=(2.4, 1.6))
    _fig.legend(handles=_handles, loc="center", frameon=False, fontsize=8,
                handletextpad=0.5, labelspacing=0.6)
    _out_pdf = OUT_F5 / "fig5_taxonomy_legend.pdf"
    _fig.savefig(_out_pdf, bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(OUT_F5 / "fig5_taxonomy_legend.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(OUT_F5 / "fig5_taxonomy_legend.svg",
                 bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_out_pdf}")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
