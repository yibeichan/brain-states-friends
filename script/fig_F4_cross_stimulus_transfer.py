"""Figure F4 - Cross-stimulus recurrence transfer (R5).

One marimo notebook per figure (see `2026-05-24_manuscript_version_scope.md`).
Panels are per-cell, saved as separate .png + .svg mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

R5 claim: a state's recurrence rank in Friends predicts its mean occupancy
during audiovisual film (Movie10) in every participant, above a phase-randomized
null that preserves each PCA component's power spectrum and variance and the
cross-component covariance. The null reproduces most of the raw correlation, so
the claim is the margin above the null, not the raw ρ. The same null run on
Harry Potter (7 runs) and Petit Prince (16-18 runs) is wider and lower-powered;
there the observed ρ clears it in 1 and 2 of 5 participants.

Rest placement (decided 2026-08-17): supplementary ONLY — rest appears in
fig_S08, not in this figure. Rationale: rest transfer is descriptive only (no
surrogate null was run), and raw ρ carries a stimulus-independent floor from
covariance/stationary structure (the R5 phase-null finding), so rest's mean ρ
(0.165) is NOT below PP's (0.101) and a main-figure Rest column would invite a
monotonic-richness-gradient reading the data don't support. Audited rest
values live in fig_S08's docstring. The earlier "tracks social-narrative content, not
modality" framing was DROPPED: the cross-modality rho ordering does not isolate
narrative content from modality, timing, language, run structure, or model fit.
The within-Movie10 spread is reported descriptively and not attributed to
content. The presence and fit diagnostics live in the supplementary
cross-stimulus validity figure.

Panel plan (chart families distinct within the figure, and B's family is used
in no other main figure: point-based 1D is already in Figures 1 and 3). Filenames
carry the panel letter; the user arranges the composite.

| Panel | Content | Chart family | Source |
|---|---|---|---|
| A | Per-subject recurrence→FO scatter, 2×3 small multiples (Movie10; x=Friends recurrence, y=mean M10 FO; subject = marker shape + OLS line + per-subject ρ). Dots colored by R2 taxonomy category (Fig 2 colors). | scatter (small multiples) | m10_04 fractional_occupancy.pkl + 05a recurrence + 05e_a4 state_flags |
| B | Observed vs phase-randomized null, by stimulus (Movie10, Harry Potter, Petit Prince; one ridge per subject, all active states). Grey ridge = histogram of 10,000 null draws on shared bins, each scaled to the same height; dark tick = observed ρ (for Movie10, the value panel A annotates); Δρ (observed minus null mean) annotated per row. Wide, same width as A. | distribution (ridge) | m10/hp/pp A1 + sm_rel_r5_phase_null summaries + null_draws.npy |

A `fig4_taxonomy_legend` file gives the 5-category color key for Panel A, and
`fig4_B_legend` the observed / null key for Panel B.

Panel B keeps the stimuli of the raw-ρ-by-condition strip it replaced, with
each set against its own null in place of zero. The four Movie10 films are
pooled, since the null is computed over all 61 Movie10 runs; fig_S08 B plots
the per-film ρ against the HMM fit gap.

Modalities (verified against Methods draft): Movie10 = audiovisual films; Harry
Potter = word-by-word RSVP reading (visual-only, no audio); Petit Prince =
audiobook (audio-only, no visual). sub-04 has no HP/PP data, so the HP and PP
groups in B have five rows.

The presence donut grid and the validity diagnostics (PCA transfer R² uniform
~0.95; HMM LL gap uneven, ρ↔fit relationship with the HP outlier) move to the
supplementary `fig_S_cross_stimulus_validity` figure.

Source-truth audit (reproduced from raw arrays):
  * Panel A: reproduced sub-01 M10 ρ = 0.5486 (n_active=46) = summary JSON exactly.
  * Full-repertoire ρ: M10 0.264–0.773 (5/6 uncorrected per-subject
    Spearman p<0.05), HP 0.232–0.444 (2/5), PP −0.018–0.332 (1/5);
    sub-04 has no HP/PP data. These p-values are descriptive per-subject
    flags, not cohort-level or multiplicity-corrected inference.
  * Per-condition mean ρ (fig_S08 B): wolf 0.69, figures 0.63, bourne 0.40,
    life 0.16, HP 0.32, PP 0.10. Cross-modality ordering (HP > Life > PP)
    contradicts a content account; within-M10 spread tracks LL-fit gap
    (0.92/1.54/2.43/4.98).
  * Panel B (2026-10-28): each null summary's observed ρ equals the
    cross-stimulus summary's A1 ρ for all 16 subject x stimulus cells; Δρ
    M10 +0.07 to +0.22 (p 0.0001, the 10,000-draw floor, in 6/6), HP -0.08 to
    +0.09 (p < 0.05 in 1/5, sub-05), PP +0.003 to +0.19 (p < 0.05 in 2/5,
    sub-01 and sub-05).

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

    # Per-run FO above which a state counts as "present" in that run:
    # ACTIVE_FO_THRESHOLD in utils/recurrence_utils.py, the same cut that
    # defines Friends-active states.
    PRESENCE_FO = 0.02

    # Phase-randomized nulls (sm_rel_r5_phase_null, supplements branch): Movie10
    # at the top level, the other stimuli in a subfolder named for each.
    NULL_ROOT = SCRATCH_DIR / "output" / "sm_rel_r5_phase_null"
    NULL_SUBDIR = {"m10": "", "hp": "harrypotter", "pp": "petitprince"}

    def null_dir(stim, sub):
        return NULL_ROOT / NULL_SUBDIR[stim] / PARCELLATION / sub / VT

    def xstim_dir(stim):
        return SCRATCH_DIR / "output" / f"{stim}_05_cross_validation" / PARCELLATION

    OUT_F4 = SCRATCH_DIR / "output" / "manuscript_figures" / "fig4"
    OUT_F4.mkdir(parents=True, exist_ok=True)

    return (
        SCRATCH_DIR, PARCELLATION, VT, SUBJECTS, EXEMPLAR, STIMULI,
        RECUR_DIR, FLAGS_DIR, M10_DEC_DIR, HP_DEC_DIR, PP_DEC_DIR,
        PRESENCE_FO, null_dir, xstim_dir, OUT_F4,
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
    """R2 taxonomy categories - labels + colors, shared with Figure 2.

    Identical to fig_F1's taxonomy_constants cell; kept in sync by hand. Maps the
    raw 05e_a4 `summary_category` values onto the 5 display categories so Panel A
    dots and the cross-stimulus presence panel use the same color key as Fig 2.
    """
    TAXONOMY_ORDER = [
        "Content-eligible",
        "Run-onset-anchored",
        "Low-confidence",
        "Drift-anchored",
        "Rare",
    ]
    TAXONOMY_COLORS = {
        "Content-eligible":    "#0C7BDC",
        "Run-onset-anchored":  "#FFC20A",
        "Low-confidence":      "#5A5A5A",
        "Drift-anchored":      "#D35FB7",
        "Rare":       "#CCCCCC",
    }
    TAXONOMY_MAP = {
        "eligible_for_content_analysis": "Content-eligible",
        "run_onset_anchored":            "Run-onset-anchored",
        "low_confidence":                "Low-confidence",
        "season_temporal":               "Drift-anchored",
        "unused":                        "Rare",
        "rare":                          "Rare",
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
        _tax = _flags["summary_category"].map(TAXONOMY_MAP).fillna("Rare").values
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
    SUBJECTS, VT, summaries, presence, OUT_F4,
    np, plt, sp_stats, SUBJECT_MARKERS, TAXONOMY_COLORS,
):
    """Panel A - per-subject recurrence→FO scatter, 2×3 small multiples (Movie10).

    One subplot per subject (shared x and y axes for comparability): x = Friends
    recurrence score (05a), y = mean Movie10 fractional occupancy across runs
    (m10_04), one point per Friends-active state. Each subject keeps its own
    marker shape (SUBJECT_MARKERS, as in Figures 1 and 3) and a
    red OLS guide line; the per-subject Spearman ρ (asserted against the summary
    JSON) is annotated in-panel without a significance marker.

    Dot fill = R2 taxonomy category (same colors as Figure 2). Panel B sets
    each ρ against its phase-randomized null; the repertoire-presence view and
    the validity diagnostics live in the supplementary cross-stimulus figure.

    Recurrence and occupancy are both within-subject quantities, so each panel's
    ρ is the correct per-subject statistic; no pooled correlation is drawn.
    Width matches panel B so the two stack as one figure column.
    """
    _fig, _axes = plt.subplots(2, 3, figsize=(6.7, 4.25), sharex=True, sharey=True)
    _axes = _axes.ravel()

    _rhos = []
    for _k, _sub in enumerate(SUBJECTS):
        _ax = _axes[_k]
        _e = presence[_sub]
        _active = _e["active"]
        _x = _e["rec"][_active]
        _y = _e["M10_mean_fo"][_active]
        _colors = [TAXONOMY_COLORS[_c] for _c in _e["tax"][_active]]

        _rho, _ = sp_stats.spearmanr(_x, _y)
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
        _ax.plot(_xx, _b * _xx + _a, color="#D62728", lw=1.4, alpha=0.85,
                 zorder=2)

        # Tag in the bottom-right (sparse corner) so it does not cover the
        # top-left dots; no box.
        _ax.text(0.96, 0.05, f"{_sub}\nρ = {_rho:.2f}",
                 transform=_ax.transAxes, ha="right", va="bottom", fontsize=6.5)
        _ax.tick_params(labelsize=6)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)

    print(f"Panel A per-subject M10 ρ: {[f'{r:.2f}' for r in _rhos]}")
    _fig.supxlabel("Friends recurrence score", fontsize=7)
    _fig.supylabel("Mean Movie10 fractional occupancy", fontsize=7)
    _fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.98,
                         wspace=0.10, hspace=0.16)
    _stem = OUT_F4 / "fig4_A_recurrence_fo_scatter"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)
    return


@app.cell
def panel_B_observed_vs_null(
    SUBJECTS, STIMULI, null_dir, summaries, OUT_F4, json, np, plt, SUBJECT_NEUTRAL,
):
    """Panel B - observed ρ against its phase-randomized null, by stimulus.

    Three groups of rows on one shared ρ axis: Movie10 (6 subjects), Harry
    Potter and Petit Prince (5 each; sub-04 has no data for either), subjects in
    order within a group. Each ridge is the histogram of the subject's 10,000
    phase-randomized surrogate ρ values for that stimulus on one shared set of
    bins, scaled so every ridge has the same peak height: only the location and
    spread of the null carry information, as in the supplementary
    null-distribution figure. The dark tick is the observed ρ (for Movie10, the
    value panel A annotates), and each row is tagged with its subject,
    Δρ = observed - null mean (three decimals, so near-zero values stay
    readable) and the empirical p, the share of null draws at or above the
    observed ρ (floored at 1/10,001, shown as p < 0.001). For Petit Prince
    sub-05 the null sits well below zero and the observed ρ near zero, so its
    margin is not a positive correlation. Subject identity is carried by the row and its
    tag, so the tick does not repeat it with a marker shape.

    The null randomizes the phases of the stimulus's PC scores: each run gets one
    random phase vector, shared across components. It keeps each component's
    power spectrum and variance, the cross-component covariance, and the run
    count and lengths, and destroys stimulus-locked timing and higher-order
    temporal structure. Harry Potter (7 runs) and Petit Prince (16-18) give wider
    nulls than Movie10 (61), so those tests have less power.

    Checks, per subject and stimulus: the null summary's observed ρ equals the
    cross-stimulus summary's; the draw file holds the 10,000 draws the caption
    states, with the summary's mean and 2.5/97.5 percentiles; a subject is
    missing only where the cross-stimulus summary is missing too; and every draw
    and observed value lies inside the plotted range.
    """
    _x_min, _x_max = -0.30, 0.88
    _rows = []                                  # (stim label, sub, rho, null mean, draws)
    for _stim, _label in STIMULI:
        for _sub in SUBJECTS:
            _dir = null_dir(_stim, _sub)
            if summaries[_stim][_sub] is None:
                assert (_dir / "skipped.json").exists(), f"{_stim} {_sub}: no data, no skip record"
                continue
            with open(_dir / "r5_phase_null_summary.json") as _f:
                _n = json.load(_f)
            _draws = np.load(_dir / "null_draws.npy")
            _rho = _n["observed"]["rho"]
            _pub = summaries[_stim][_sub]["A1_recurrence_correlation"]["spearman_rho"]
            assert abs(_rho - _pub) < 1e-3, (
                f"{_stim} {_sub}: null-summary ρ {_rho:.4f} != cross-stimulus summary {_pub:.4f}"
            )
            assert _draws.shape == (10000,) == (_n["null"]["n_draws"],), (
                f"{_stim} {_sub}: {_draws.shape} draws, summary says {_n['null']['n_draws']}"
            )
            for _key, _val in (("mean", _draws.mean()),
                               ("pct2.5", np.percentile(_draws, 2.5)),
                               ("pct97.5", np.percentile(_draws, 97.5))):
                assert abs(_val - _n["null"][_key]) < 1e-9, (
                    f"{_stim} {_sub}: draws {_key} {_val:.6f} != summary {_n['null'][_key]:.6f}"
                )
            _lo, _hi = min(_draws.min(), _rho), max(_draws.max(), _rho)
            assert _x_min <= _lo and _hi <= _x_max, (
                f"{_stim} {_sub}: values [{_lo:.3f}, {_hi:.3f}] outside the plotted range"
            )
            _rows.append((_label, _sub, _rho, _n["null"]["mean"], _draws,
                          _n["p_empirical"]))
            print(f"  {_label:12s} {_sub}: observed {_rho:+.3f}  null mean "
                  f"{_n['null']['mean']:+.3f} [{_n['null']['pct2.5']:+.3f}, "
                  f"{_n['null']['pct97.5']:+.3f}]  Δρ {_rho - _n['null']['mean']:+.3f}  "
                  f"p {_n['p_empirical']:.4f}")

    _bins = np.linspace(_x_min, _x_max, 237)    # 0.005-wide, shared by all rows
    _height = 0.8                               # ridge peak, in row units
    _gap = 0.7                                  # extra space between stimulus groups
    _ys, _y, _prev = [], 0.0, None
    for _label, *_rest in _rows:
        if _prev is not None:
            _y -= 1.0 + (_gap if _label != _prev else 0.0)
        _ys.append(_y)
        _prev = _label

    _fig, _ax = plt.subplots(figsize=(6.7, 4.6))
    def _p_text(_p):
        if _p < 0.001:
            return "p < 0.001"
        if _p > 0.99:
            return "p > 0.99"
        return f"p = {_p:.3f}" if _p < 0.1 else f"p = {_p:.2f}"

    for (_label, _sub, _rho, _mu, _draws, _p), _y0 in zip(_rows, _ys):
        _counts, _ = np.histogram(_draws, bins=_bins)
        _ax.stairs(_y0 + _height * _counts / _counts.max(), _bins, baseline=_y0,
                   fill=True, color="#BBBBBB", lw=0, zorder=2)
        _ax.plot([_x_min, _x_max], [_y0, _y0], color="#DDDDDD", lw=0.6, zorder=1)
        _ax.plot([_rho, _rho], [_y0, _y0 + 0.55], color=SUBJECT_NEUTRAL, lw=1.6,
                 solid_capstyle="butt", zorder=4)
        _d = f"{_rho - _mu:+.3f}".replace("-", "\u2212")
        _ax.text(_x_max + 0.005, _y0 + 0.2, f"{_sub}   Δρ = {_d}   {_p_text(_p)}",
                 ha="left", va="center", fontsize=6.5)
    for _label in dict.fromkeys(_r[0] for _r in _rows):
        _yy = [_y0 for _r, _y0 in zip(_rows, _ys) if _r[0] == _label]
        _ax.text(_x_min - 0.02, (max(_yy) + min(_yy) + _height) / 2, _label,
                 ha="right", va="center", fontsize=7)
    _ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=1)

    _ax.set_xlim(_x_min, _x_max)
    _ax.set_ylim(min(_ys) - 0.15, max(_ys) + _height + 0.1)
    _ax.set_yticks([])
    _ax.spines["left"].set_visible(False)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _ax.tick_params(axis="x", labelsize=6)
    _ax.set_xlabel("Spearman ρ, Friends recurrence vs. occupancy", fontsize=7)

    _fig.subplots_adjust(left=0.13, right=0.84, bottom=0.09, top=0.99)
    _stem = OUT_F4 / "fig4_B_observed_vs_null"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)
    return


@app.cell
def panel_B_legend(OUT_F4, plt, SUBJECT_NEUTRAL):
    """Observed / null key for Panel B (separate mini-file, one row)."""
    from matplotlib.patches import Patch

    _handles = [
        plt.Line2D([], [], color=SUBJECT_NEUTRAL, lw=1.6, marker="|", markersize=9,
                   markeredgewidth=1.6, linestyle="none", label="Observed ρ"),
        Patch(facecolor="#BBBBBB", edgecolor="none",
              label="Phase-randomized null, 10,000 draws"),
    ]
    _fig = plt.figure(figsize=(6.7, 0.35))
    _fig.legend(handles=_handles, loc="center", frameon=False, fontsize=6.5,
                ncol=2, handletextpad=0.5, columnspacing=1.6)
    _stem = OUT_F4 / "fig4_B_legend"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)
    return


@app.cell
def taxonomy_legend(OUT_F4, plt, TAXONOMY_ORDER, TAXONOMY_COLORS):
    """Taxonomy color legend for Panel A (separate mini-file).

    Five R2 taxonomy-category swatches, matching Figure 2's colors and Panel A's
    dot fills. (The cross-stimulus presence donut and its present/absent glyphs
    now live in the supplementary cross-stimulus validity figure.)
    """
    _handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                           markerfacecolor=TAXONOMY_COLORS[_c], markeredgecolor="white",
                           label=_c) for _c in TAXONOMY_ORDER]
    # Horizontal legend: one row of 5 categories (matches Panel A width).
    _fig = plt.figure(figsize=(6.7, 0.35))
    _fig.legend(handles=_handles, loc="center", frameon=False, fontsize=6.5,
                ncol=5, handletextpad=0.4, columnspacing=1.2)
    _stem = OUT_F4 / "fig4_taxonomy_legend"
    _fig.savefig(f"{_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{_stem}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {_stem.name}.png (+ .svg)")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
