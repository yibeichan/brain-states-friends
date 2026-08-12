"""Figure SM R5 phase null: per-subject null distributions vs observed rho.

Supplement figure for the R5 phase-randomized null (sm_rel_r5_phase_null
pipeline). One panel: 2x3 small multiples, one cell per subject in row-major
order (sub-01 .. sub-06). Each cell shows the histogram of the 10,000
surrogate-draw Spearman rho values (shared-phase Prichard-Theiler null) with a
vertical accent line at the observed Friends-recurrence vs Movie10-occupancy
rho, annotated with delta rho and z. A shared x-range across cells keeps the
observed-null distance visually comparable across subjects.

Per project assembly workflow: no on-figure panel labels, no titles, no
subject-ID tick labels; the null histogram is neutral gray and the observed
line is the single accent color (subject identity is carried by cell position
alone).

| Panel | Content | Source files |
|---|---|---|
| A | Null-draw histogram + observed-rho line, per subject | {sub}/vt0.95/r5_phase_null_summary.json + null_draws.npy |

Run:
    marimo edit script/fig_sm_rel_r5_phase_null.py
or headless:
    uv run python script/fig_sm_rel_r5_phase_null.py
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
    from utils.plot_style import (
        SUBJECT_ACCENT,
        SUBJECT_NEUTRAL,
        apply_publication_style,
    )

    load_dotenv()
    apply_publication_style()
    return SUBJECT_ACCENT, SUBJECT_NEUTRAL, json, np, os, plt


@app.cell
def config(os):
    SCRATCH = os.getenv("SCRATCH_DIR")
    PARC = "atlas-4S156Parcels"
    VT = "vt0.95"
    SUBS = [f"sub-0{i}" for i in range(1, 7)]

    NULL_DIR = os.path.join(SCRATCH, "output", "sm_rel_r5_phase_null", PARC)
    OUT = os.path.join(SCRATCH, "output", "manuscript_figures",
                       "fig_sm_r5_phase_null")
    os.makedirs(OUT, exist_ok=True)
    return NULL_DIR, OUT, SUBS, VT


@app.cell
def load_null_results(NULL_DIR, SUBS, VT, json, np, os):
    null_results = {}
    for _sub in SUBS:
        _root = os.path.join(NULL_DIR, _sub, VT)
        with open(os.path.join(_root, "r5_phase_null_summary.json")) as _f:
            _summary = json.load(_f)
        _draws = np.load(os.path.join(_root, "null_draws.npy"))
        # Guard: draws must reproduce the summary's null moments, and the
        # faithfulness gate must have passed when the null was generated.
        assert len(_draws) == _summary["null"]["n_draws"]
        assert abs(float(np.mean(_draws)) - _summary["null"]["mean"]) < 1e-9
        assert _summary["gate"]["abs_delta"] <= _summary["gate"]["tolerance"]
        null_results[_sub] = {
            "draws": _draws,
            "observed": _summary["observed"]["rho"],
            "delta_rho": _summary["delta_rho"],
            "z": _summary["z"],
            "p": _summary["p_empirical"],
        }
    return (null_results,)


@app.cell
def panel_null_vs_observed(
    OUT, SUBJECT_ACCENT, SUBJECT_NEUTRAL, SUBS, np, null_results, plt
):
    _NROWS, _NCOLS = 2, 3
    _fig, _axes = plt.subplots(_NROWS, _NCOLS, figsize=(7.0, 4.2),
                               sharex=True, sharey=False)
    _axes_flat = _axes.ravel()

    _lo = min(min(null_results[_s]["draws"].min(),
                  null_results[_s]["observed"]) for _s in SUBS)
    _hi = max(max(null_results[_s]["draws"].max(),
                  null_results[_s]["observed"]) for _s in SUBS)
    _pad = 0.04 * (_hi - _lo)
    _xlim = (_lo - _pad, _hi + _pad)
    _bins = np.linspace(_xlim[0], _xlim[1], 120)

    for _i, _sub in enumerate(SUBS):
        _ax = _axes_flat[_i]
        _res = null_results[_sub]
        _ax.hist(_res["draws"], bins=_bins, density=True,
                 color=SUBJECT_NEUTRAL, alpha=0.55, linewidth=0)
        _ax.axvline(_res["observed"], color=SUBJECT_ACCENT, linewidth=1.4)
        _ax.annotate(
            f"$\\Delta\\rho$=+{_res['delta_rho']:.2f}\nz=+{_res['z']:.1f}",
            xy=(0.03, 0.95), xycoords="axes fraction",
            ha="left", va="top", fontsize=7.5,
        )
        _ax.set_xlim(_xlim)
        _ax.set_yticks([])
        _ax.spines[["top", "right", "left"]].set_visible(False)
        _ax.tick_params(axis="x", labelsize=8)

    for _c in range(_NCOLS):
        _axes_flat[(_NROWS - 1) * _NCOLS + _c].set_xlabel(
            "Spearman rho", fontsize=8)
    for _r in range(_NROWS):
        _axes_flat[_r * _NCOLS].set_ylabel("Null density", fontsize=8)

    _fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)
    for _ext, _kw in (("png", {"dpi": 300}), ("svg", {})):
        _fig.savefig(
            f"{OUT}/fig_sm_r5_phase_null_A_null_vs_observed.{_ext}",
            bbox_inches="tight", pad_inches=0.02, **_kw)
    plt.close(_fig)
    print(f"saved -> {OUT}/fig_sm_r5_phase_null_A_null_vs_observed.[png|svg]")
    return


if __name__ == "__main__":
    app.run()
