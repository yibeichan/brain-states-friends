"""Figure SM ICA OOS recurrence: ICA out-of-stimulus recurrence (R5 analogue).

Supplement figure for the ICA out-of-stimulus recurrence analysis
(sm_alt_ica_oos_recurrence pipeline).

- Panel A (primary, WTA): 2x3 small-multiples scatter, one cell per subject.
  x = Friends WTA recurrence score, y = mean Movie10 WTA fractional occupancy.
  Per-subject OLS line + Spearman rho annotation as in-axes text.
- Panel B (robustness, continuous): same layout but y = continuous occupancy,
  showing the recurrence-to-occupancy ordering is not a discretization artifact.

Subjects whose summary JSON is absent are silently skipped.

Panels are per-cell, saved as separate .png + .svg mini-figures for manual
assembly. No on-figure panel labels, no titles, no subject-ID tick labels.

| Panel | Content | Source files |
|---|---|---|
| A | WTA scatter: friends_recurrence vs movie_occupancy_wta | oos_recurrence_summary.json |
| B | Continuous scatter: friends_recurrence vs movie_occupancy_continuous | oos_recurrence_summary.json |

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
    OUT = os.path.join(SCRATCH, "output", "manuscript_figures", "fig_sm_ica_oos_recurrence")
    os.makedirs(OUT, exist_ok=True)

    def _load(sub):
        p = os.path.join(
            SCRATCH, "output", "sm_ica_oos_recurrence", PARC, sub,
            "oos_recurrence_summary.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    summaries = {}
    for _s in SUBS:
        _d = _load(_s)
        if _d is not None:
            summaries[_s] = _d

    available_subs = [s for s in SUBS if s in summaries]
    return OUT, PARC, SUBS, available_subs, summaries


@app.cell
def panel_A_wta(OUT, SUBS, available_subs, summaries, np, plt):
    """Panel A: WTA occupancy scatter (primary)."""
    _NCOLS = 3
    _NROWS = 2
    _fig, _axes = plt.subplots(_NROWS, _NCOLS, figsize=(6.6, 4.4),
                               sharex=False, sharey=False)
    _axes_flat = _axes.flatten()

    for _idx, _sub in enumerate(SUBS):
        _ax = _axes_flat[_idx]
        if _sub not in available_subs:
            _ax.set_visible(False)
            continue

        _d = summaries[_sub]
        _x = np.array(_d["friends_recurrence"])
        _y = np.array(_d["movie_occupancy_wta"])
        _rho = _d["overall"]["wta"]["rho"]
        _null_wta = _d["overall"]["wta"].get("null")

        _ax.scatter(_x, _y, s=14, color="#4A6FA5", alpha=0.75, linewidths=0,
                    zorder=2)
        _coef = np.polyfit(_x, _y, 1)
        _xfit = np.linspace(_x.min(), _x.max(), 100)
        _ax.plot(_xfit, np.polyval(_coef, _xfit), color="#C44E52", lw=1.2, zorder=3)
        if _null_wta is not None:
            _ann = f"ρ = {_rho:.2f}\nz = {_null_wta['z']:+.1f} (vs null {_null_wta['mean']:.2f})"
        else:
            _ann = f"ρ = {_rho:.2f}"
        _ax.text(0.95, 0.07, _ann,
                 transform=_ax.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="0.2")
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=7)
        _ax.set_xticks([0.0, 0.5, 1.0])
        _ax.set_xticklabels([])
        _ax.set_yticklabels([])

    # outer-panel axis labels
    for _r in range(_NROWS):
        _ax0 = _axes_flat[_r * _NCOLS]
        _ax0.set_ylabel("Movie10 WTA fractional occupancy", fontsize=8)
        _ax0.yaxis.set_tick_params(labelleft=True)
        _ax0.set_yticklabels([f"{v:.2f}" for v in _ax0.get_yticks()])
    for _c in range(_NCOLS):
        _axb = _axes_flat[(_NROWS - 1) * _NCOLS + _c]
        _axb.set_xlabel("Friends WTA recurrence", fontsize=8)
        _axb.set_xticklabels(["0", "0.5", "1"])

    _fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)
    _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_A_wta.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_A_wta.svg",
                 bbox_inches="tight", pad_inches=0.02)
    plt.close(_fig)
    return


@app.cell
def panel_B_continuous(OUT, SUBS, available_subs, summaries, np, plt):
    """Panel B: continuous occupancy scatter (robustness)."""
    _NCOLS = 3
    _NROWS = 2
    _fig, _axes = plt.subplots(_NROWS, _NCOLS, figsize=(6.6, 4.4),
                               sharex=False, sharey=False)
    _axes_flat = _axes.flatten()

    for _idx, _sub in enumerate(SUBS):
        _ax = _axes_flat[_idx]
        if _sub not in available_subs:
            _ax.set_visible(False)
            continue

        _d = summaries[_sub]
        _x = np.array(_d["friends_recurrence"])
        _y = np.array(_d["movie_occupancy_continuous"])
        _rho = _d["overall"]["continuous"]["rho"]
        _null_cont = _d["overall"]["continuous"].get("null")

        _ax.scatter(_x, _y, s=14, color="#4A6FA5", alpha=0.75, linewidths=0,
                    zorder=2)
        _coef = np.polyfit(_x, _y, 1)
        _xfit = np.linspace(_x.min(), _x.max(), 100)
        _ax.plot(_xfit, np.polyval(_coef, _xfit), color="#C44E52", lw=1.2, zorder=3)
        if _null_cont is not None:
            _ann = f"ρ = {_rho:.2f}\nΔ = {_null_cont['residual']:+.2f}"
        else:
            _ann = f"ρ = {_rho:.2f}"
        _ax.text(0.95, 0.07, _ann,
                 transform=_ax.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="0.2")
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=7)
        _ax.set_xticks([0.0, 0.5, 1.0])
        _ax.set_xticklabels([])
        _ax.set_yticklabels([])

    # outer-panel axis labels
    for _r in range(_NROWS):
        _ax0 = _axes_flat[_r * _NCOLS]
        _ax0.set_ylabel("Movie10 continuous occupancy", fontsize=8)
        _ax0.yaxis.set_tick_params(labelleft=True)
        _ax0.set_yticklabels([f"{v:.2f}" for v in _ax0.get_yticks()])
    for _c in range(_NCOLS):
        _axb = _axes_flat[(_NROWS - 1) * _NCOLS + _c]
        _axb.set_xlabel("Friends WTA recurrence", fontsize=8)
        _axb.set_xticklabels(["0", "0.5", "1"])

    _fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)
    _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_B_continuous.png",
                 bbox_inches="tight", pad_inches=0.02, dpi=300)
    _fig.savefig(f"{OUT}/fig_sm_ica_oos_recurrence_B_continuous.svg",
                 bbox_inches="tight", pad_inches=0.02)
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
