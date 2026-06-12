"""Figure SM ICA matching: ICA/HMM spatial convergence diagnostics.

Supplement figure for the ICA convergent-validity analysis (sm_ica_states
pipeline).

- Panel A (headline): K-sweep convergence heatmap. Spatial agreement between
  ICA components and HDP-HMM state-mean maps is granularity-dependent (null at
  coarse, HMM-independent K; emerging only as ICA dimensionality approaches each
  subject's K_active) and strongly subject-variable.
- Panel B: per-state matched spatial correlation at K_active, one dot per HMM
  state (subject on x, category by colour). Convergence is taxonomy-agnostic
  (categories intermix across the |r| range within each subject) while the
  per-subject median shifts (subject is a dominant axis).

Source of truth for numbers: docs/findings/sm_ica.md. Values are read live from
each subject's ica_match_summary.json (FDR-surviving matched pairs vs the
subspace-rotation null, BH-FDR within the subject's eligible family) and the
05e_a4 state_flags.csv (per-state taxonomy label).

Panels are per-cell, saved as separate .pdf + .png mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

| Panel | Content | Source files |
|---|---|---|
| A | Convergence heatmap: 6 subjects x ICA K; colour = FDR-surviving fraction of eligible pairs, cell text = surviving/total counts | sm_ica_states/*/ica_match_summary.json |
| B | Per-state matched spatial |r| at K_active; x = subject, colour = taxonomy category, black tick = per-subject median | sm_ica_states/*/ica_match_summary.json + 05e_temporal_trend_a4/*/state_flags.csv |

Run:
    marimo edit script/fig_sm_ica_matching.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import apply_publication_style

    load_dotenv()
    apply_publication_style()
    return json, np, os, plt


@app.cell
def config(json, os):
    SCRATCH = os.getenv("SCRATCH_DIR")
    PARC = "atlas-4S156Parcels"
    SUBS = [f"sub-0{i}" for i in range(1, 7)]
    # K columns: primary HMM-independent low-dim set | near-K_active sensitivity grid
    KS = ["15", "25", "35", "41", "42", "43", "44", "45", "46", "47"]
    SEP_AFTER = 2  # divider between column index 2 (K=35) and 3 (K=41)
    OUT = os.path.join(SCRATCH, "output", "manuscript_figures", "fig_sm_ica")
    os.makedirs(OUT, exist_ok=True)

    def _load(sub):
        p = os.path.join(SCRATCH, "output", "sm_ica_states", PARC, sub,
                         "ica_match_summary.json")
        return json.load(open(p))

    ica_summaries = {s: _load(s) for s in SUBS}

    # per-state taxonomy label from 05e_a4 state_flags.csv (summary_category)
    import csv as _csv
    VT = "0.95"
    state_cats = {}
    for _s in SUBS:
        _fp = os.path.join(SCRATCH, "output", "05e_temporal_trend_a4", PARC,
                           _s, f"vt{VT}", "state_flags.csv")
        with open(_fp) as _fh:
            state_cats[_s] = {int(_r["state"]): _r["summary_category"]
                              for _r in _csv.DictReader(_fh)}

    # manuscript 5-category taxonomy palette (CB-safe set, mirrors
    # fig_F2_recurrence_sources.py TAXONOMY_COLORS on manuscript-prep)
    TAX_ORDER = ["Content-eligible", "Run-onset-anchored", "Low-confidence",
                 "Drift-anchored", "Unused + rare"]
    TAX_COLORS = {
        "Content-eligible":   "#0C7BDC",
        "Run-onset-anchored": "#FFC20A",
        "Low-confidence":     "#5A5A5A",
        "Drift-anchored":     "#D35FB7",
        "Unused + rare":      "#CCCCCC",
    }
    # raw summary_category -> manuscript display category (collapses unused+rare)
    RAW2DISP = {
        "eligible_for_content_analysis": "Content-eligible",
        "run_onset_anchored":            "Run-onset-anchored",
        "low_confidence":                "Low-confidence",
        "season_temporal":               "Drift-anchored",
        "unused":                        "Unused + rare",
        "rare":                          "Unused + rare",
    }
    return (KS, OUT, RAW2DISP, SEP_AFTER, SUBS, TAX_COLORS, TAX_ORDER,
            ica_summaries, state_cats)


@app.cell
def panel_A_convergence(KS, OUT, SEP_AFTER, SUBS, ica_summaries, np, plt):
    # FDR-surviving content-eligible matched pairs vs subspace-rotation null,
    # per subject (rows) x ICA dimensionality K (columns).
    _frac = np.full((len(SUBS), len(KS)), np.nan)
    _surv = np.zeros((len(SUBS), len(KS)), dtype=int)
    _tot = np.zeros((len(SUBS), len(KS)), dtype=int)
    _kact = {}
    for _i, _s in enumerate(SUBS):
        _d = ica_summaries[_s]
        _kact[_s] = _d["K_active"]
        for _j, _k in enumerate(KS):
            if _k not in _d["by_K"]:
                continue
            _e = _d["by_K"][_k]["state_sets"]["eligible"]
            _q = _e["spatial_q"]
            _n = len(_q)
            _ns = sum(1 for x in _q if x is not None and x < 0.05)
            _tot[_i, _j] = _n
            _surv[_i, _j] = _ns
            _frac[_i, _j] = _ns / _n if _n else np.nan

    _fig, _ax = plt.subplots(figsize=(6.6, 3.0))
    _im = _ax.imshow(_frac, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    # cell text: surviving/total, contrast-aware colour
    for _i in range(len(SUBS)):
        for _j in range(len(KS)):
            _f = _frac[_i, _j]
            _txt_c = "white" if (np.isfinite(_f) and _f >= 0.55) else "0.15"
            _ax.text(_j, _i, f"{_surv[_i, _j]}/{_tot[_i, _j]}",
                     ha="center", va="center", fontsize=6, color=_txt_c)

    # divider between the primary low-dim set {15,25,35} and the near-K_active
    # sensitivity grid {41-47}; grey so it stays visible over the light cells
    _ax.axvline(SEP_AFTER + 0.5, color="0.35", lw=1.2)

    _ax.set_xticks(range(len(KS)))
    _ax.set_xticklabels(KS)
    _ax.set_xlabel("ICA dimensionality $K$")
    _ax.set_yticks(range(len(SUBS)))
    _ax.set_yticklabels([f"{s}\n$K_a$={_kact[s]}" for s in SUBS])
    _ax.tick_params(length=0)
    for _sp in ("top", "right", "left", "bottom"):
        _ax.spines[_sp].set_visible(False)

    _cb = _fig.colorbar(_im, ax=_ax, fraction=0.025, pad=0.02)
    _cb.set_label("FDR-surviving fraction\n(eligible matched pairs)", fontsize=8)
    _cb.outline.set_visible(False)

    _fig.savefig(f"{OUT}/fig_sm_ica_A_convergence_heatmap.pdf",
                 bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{OUT}/fig_sm_ica_A_convergence_heatmap.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    return


@app.cell
def panel_B_per_state(OUT, RAW2DISP, SUBS, TAX_COLORS, TAX_ORDER,
                      ica_summaries, state_cats, np, plt):
    # One dot per matched HMM state at K_active. Subject = x-position,
    # category = colour (CB-safe manuscript palette), matched spatial |r| = y.
    # Shows the full per-subject distribution and category composition, rather
    # than a per-category mean. Median |r| per subject drawn as a short tick.
    _rng = np.random.default_rng(0)
    _edge = {"Unused + rare": "#888888"}  # darken pale catch-all for visibility
    _fig, _ax = plt.subplots(figsize=(5.4, 2.8))

    for _si, _s in enumerate(SUBS):
        _d = ica_summaries[_s]
        _a = _d["by_K"][str(_d["K_active"])]["state_sets"]["all"]
        _ids = [int(x) for x in _a["hmm_state_ids"]]
        _rs = np.array(_a["matched_r"], dtype=float)
        _cats = [RAW2DISP[state_cats[_s][_i]] for _i in _ids]
        _cols = [TAX_COLORS[_c] for _c in _cats]
        _x = _si + _rng.uniform(-0.22, 0.22, size=len(_rs))
        _ax.scatter(_x, _rs, s=20, c=_cols, edgecolor="white", linewidth=0.3,
                    alpha=0.9, zorder=2)
        _med = float(np.median(_rs))
        _ax.plot([_si - 0.3, _si + 0.3], [_med, _med], color="0.15", lw=1.3,
                 zorder=3)

    _ax.set_xticks(range(len(SUBS)))
    _ax.set_xticklabels(SUBS)
    _ax.set_xlim(-0.5, len(SUBS) - 0.5)
    _ax.set_ylabel("Matched spatial correlation (|$r$|)")
    _ax.set_ylim(0, None)
    for _sp in ("top", "right"):
        _ax.spines[_sp].set_visible(False)

    # category legend (colour is the encoded variable here); darken the pale
    # catch-all edge so its swatch stays visible (mirrors fig_F2)
    _handles = [plt.Line2D([0], [0], marker="o", ls="none", ms=6,
                           mfc=TAX_COLORS[_c],
                           mec=_edge.get(_c, TAX_COLORS[_c]), mew=0.6, label=_c)
                for _c in TAX_ORDER]
    _ax.legend(handles=_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=False, fontsize=7, handletextpad=0.3,
               title="State category", title_fontsize=7)

    _fig.savefig(f"{OUT}/fig_sm_ica_B_per_category.pdf",
                 bbox_inches="tight", pad_inches=0.02)
    _fig.savefig(f"{OUT}/fig_sm_ica_B_per_category.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
