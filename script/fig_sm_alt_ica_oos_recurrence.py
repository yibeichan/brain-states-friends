"""Figure SM ICA OOS recurrence: ICA out-of-stimulus recurrence (R5 analogue).

Supplement figure for the ICA out-of-stimulus recurrence analysis
(sm_alt_ica_oos_recurrence pipeline).

Rendered per out-of-stimulus dataset (movie10, harrypotter, petitprince):
- Panel A (primary, WTA): 2x3 small-multiples scatter, one cell per subject.
  x = Friends WTA recurrence score, y = mean OOS WTA fractional occupancy.
  Per-subject OLS line + Spearman rho annotation (+ run count) as in-axes text.
- Panel B (robustness, continuous): same layout but y = continuous occupancy,
  showing the recurrence-to-occupancy ordering is not a discretization artifact.

Subjects whose summary JSON is absent are silently skipped (e.g. sub-04 has no
HP/PP). The run-count annotation makes the per-stimulus precision asymmetry
(Movie10 ~61 runs vs HP 7 vs PP 18) visible; the stimuli are NOT compared.

Panels are per-cell, saved as separate .png + .svg mini-figures for manual
assembly, one set per stimulus
(fig_sm_ica_oos_recurrence_{stimulus}_{A_wta,B_continuous}). No on-figure panel
labels, no titles, no subject-ID tick labels.

| Panel | Content | Source files |
|---|---|---|
| A | WTA scatter: friends_recurrence vs movie_occupancy_wta | {sub}/{stimulus}/oos_recurrence_summary.json |
| B | Continuous scatter: friends_recurrence vs movie_occupancy_continuous | {sub}/{stimulus}/oos_recurrence_summary.json |

Run:
    marimo edit script/fig_sm_alt_ica_oos_recurrence.py
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
    STIM_ORDER = ["movie10", "harrypotter", "petitprince"]
    STIM_LABELS = {"movie10": "Movie10", "harrypotter": "Harry Potter",
                   "petitprince": "Petit Prince"}
    OUT = os.path.join(SCRATCH, "output", "manuscript_figures", "fig_sm_ica_oos_recurrence")
    os.makedirs(OUT, exist_ok=True)

    def _load(sub, stimulus):
        p = os.path.join(
            SCRATCH, "output", "sm_ica_oos_recurrence", PARC, sub,
            stimulus, "oos_recurrence_summary.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    # summaries[stimulus][sub] -> summary dict; available_subs[stimulus] -> list
    summaries = {st: {} for st in STIM_ORDER}
    for _st in STIM_ORDER:
        for _s in SUBS:
            _d = _load(_s, _st)
            if _d is not None:
                summaries[_st][_s] = _d
    available_subs = {st: [s for s in SUBS if s in summaries[st]]
                      for st in STIM_ORDER}
    return OUT, STIM_LABELS, STIM_ORDER, SUBS, available_subs, summaries


@app.cell
def render_panels(OUT, STIM_LABELS, STIM_ORDER, SUBS, available_subs,
                  summaries, np, plt):
    """Render WTA (A) + continuous (B) scatter grids, one set per stimulus.

    Single DRY renderer over (stimulus x arm); the two arms differ only in the
    occupancy key, y-label, in-axes annotation, and output tag. Each cell's
    annotation carries the run count so the per-stimulus precision asymmetry
    (Movie10 ~61 vs HP 7 vs PP 18 runs) is visible; stimuli are NOT compared.
    """
    _NCOLS, _NROWS = 3, 2

    def _annotate_wta(_d):
        _rho = _d["overall"]["wta"]["rho"]
        _null = _d["overall"]["wta"].get("null")
        _base = f"ρ = {_rho:.2f}" if _rho is not None else "ρ = n/a"
        if _null is not None and _null.get("z") is not None:
            _base += f"\nz = {_null['z']:+.1f} (vs null {_null['mean']:.2f})"
        return f"{_base}\nn = {_d.get('n_movie_runs')} runs"

    def _annotate_cont(_d):
        _rho = _d["overall"]["continuous"]["rho"]
        _null = _d["overall"]["continuous"].get("null")
        _base = f"ρ = {_rho:.2f}" if _rho is not None else "ρ = n/a"
        if _null is not None and _null.get("residual") is not None:
            _base += f"\nΔ = {_null['residual']:+.2f}"
        return f"{_base}\nn = {_d.get('n_movie_runs')} runs"

    # (arm, occupancy key, y-label metric, output tag, annotation fn)
    _ARMS = [
        ("wta", "movie_occupancy_wta", "WTA fractional occupancy", "A_wta",
         _annotate_wta),
        ("continuous", "movie_occupancy_continuous", "continuous occupancy",
         "B_continuous", _annotate_cont),
    ]

    def _render(stimulus, occ_key, ylabel_metric, tag, ann_fn):
        _avail = available_subs[stimulus]
        if not _avail:
            return
        _label = STIM_LABELS[stimulus]
        _fig, _axes = plt.subplots(_NROWS, _NCOLS, figsize=(6.6, 4.4),
                                   sharex=False, sharey=False)
        _axes_flat = _axes.flatten()
        for _idx, _sub in enumerate(SUBS):
            _ax = _axes_flat[_idx]
            if _sub not in _avail:
                _ax.set_visible(False)
                continue
            _d = summaries[stimulus][_sub]
            _x = np.array(_d["friends_recurrence"])
            _y = np.array(_d[occ_key])
            _ax.scatter(_x, _y, s=14, color="#4A6FA5", alpha=0.75,
                        linewidths=0, zorder=2)
            _coef = np.polyfit(_x, _y, 1)
            _xfit = np.linspace(_x.min(), _x.max(), 100)
            _ax.plot(_xfit, np.polyval(_coef, _xfit), color="#C44E52",
                     lw=1.2, zorder=3)
            _ax.text(0.95, 0.07, ann_fn(_d), transform=_ax.transAxes,
                     ha="right", va="bottom", fontsize=7.5, color="0.2")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.tick_params(labelsize=7)
            _ax.set_xticks([0.0, 0.5, 1.0])
            _ax.set_xticklabels([])
            _ax.set_yticklabels([])

        for _r in range(_NROWS):
            _ax0 = _axes_flat[_r * _NCOLS]
            _ax0.set_ylabel(f"{_label} {ylabel_metric}", fontsize=8)
            _ax0.yaxis.set_tick_params(labelleft=True)
            _ax0.set_yticklabels([f"{v:.2f}" for v in _ax0.get_yticks()])
        for _c in range(_NCOLS):
            _axb = _axes_flat[(_NROWS - 1) * _NCOLS + _c]
            _axb.set_xlabel("Friends WTA recurrence", fontsize=8)
            _axb.set_xticklabels(["0", "0.5", "1"])

        _fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)
        _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_{stimulus}_{tag}.png",
                     bbox_inches="tight", pad_inches=0.02, dpi=300)
        _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_{stimulus}_{tag}.svg",
                     bbox_inches="tight", pad_inches=0.02)
        plt.close(_fig)

    for _stim in STIM_ORDER:
        for _arm, _occ, _ylm, _tag, _annfn in _ARMS:
            _render(_stim, _occ, _ylm, _tag, _annfn)
    return


if __name__ == "__main__":
    app.run()
