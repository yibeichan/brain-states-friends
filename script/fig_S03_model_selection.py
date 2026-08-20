"""Figure S3 (supplementary) - HMM model selection sweep.

Documents the configuration choice described in Methods (truncation capacity
K_max = 50, concentration gamma = 1, sticky bias kappa = 10, row concentration
alpha = 1, sticky scale rho = 1). Source-of-truth for the selected config is
``final_results.json -> selected_config`` (verified 2026-08-19: all six subjects
carry nc=50, gamma=1, kappa=10, alpha=1.0, rho=1, min_state_usage=0.01).

| Panel | Content | Chart family |
|---|---|---|
| A | Validation LL/sample vs occupied states, one small multiple per subject. Points are the vt0.95 configuration sweep (17-18 per subject), coloured by gamma; the selected config is ringed; the Pareto frontier (no other config has both more LL and fewer states) is drawn. | scatter |
| B | Occupied states vs truncation capacity, one line per gamma. At gamma=1 the occupied count saturates near 35-44 across nc 40-100; at gamma=5/10 it tracks capacity up to 62. | line |
| C | Train-to-validation LL gap vs truncation capacity, one line per gamma. | line |

Occupied-state definition: ``n_active_states`` from the sweep summaries, i.e.
states whose final-iteration usage fraction exceeds ``min_state_usage`` = 0.01
(``utils/hdphmm.infer_n_active_states``).

CAVEAT on panel A's y-axis: validation LL/sample differs by ~10 nats across
subjects (sub-01 approx -3.5, sub-03 approx -13.7), so each small multiple uses
its own y-range. The panel supports within-subject comparison of configurations,
not cross-subject comparison of fit quality.

CAVEAT on panel C: the selected config's gap is smaller than every gamma=5/10
config of higher capacity, but NOT smaller than the gamma=1 configs at nc80 and
nc100 for sub-01, sub-02 and sub-03. Read the gap claim as within-gamma.

Sweep numbers are the stage-04 configuration sweep and are NOT the production
model: the production model is a 10-seed refit whose active counts are 42, 42,
42, 41, 41, 37 (sub-01..06), read from ``final_refit.n_active_states``.

Run:
    marimo edit script/fig_S03_model_selection.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import apply_publication_style

    load_dotenv()
    apply_publication_style()
    return Path, apply_publication_style, json, np, os, pd, plt


@app.cell
def config(Path, os):
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"
    VT = "0.95"

    SEL_DIR = (
        SCRATCH_DIR / "output" / "diagnostics" / PARCELLATION / "selection_metrics"
    )
    HMM_DIR = SCRATCH_DIR / "output" / "04_combined_hdphmm" / PARCELLATION
    OUT = SCRATCH_DIR / "output" / "manuscript_figures" / "figS03"
    OUT.mkdir(parents=True, exist_ok=True)

    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 7)]

    # gamma is the variable that separates the two regimes, so it gets colour.
    # Subject identity is carried by subplot position, not colour.
    GAMMA_COLORS = {1: "#1B4F72", 5: "#E67E22", 10: "#7D3C98"}
    SELECTED_RING = "#D62728"
    return (
        GAMMA_COLORS,
        HMM_DIR,
        OUT,
        PARCELLATION,
        SELECTED_RING,
        SEL_DIR,
        SUBJECTS,
        VT,
    )


@app.cell
def load_data(HMM_DIR, SEL_DIR, SUBJECTS, VT, json, pd):
    sweep = {}
    selected = {}
    final_active = {}

    for _sub in SUBJECTS:
        _csv = SEL_DIR / _sub / f"selection_metrics_vt{VT}.csv"
        sweep[_sub] = pd.read_csv(_csv)

        _fj = HMM_DIR / _sub / "final" / f"vt{VT}" / "final_results.json"
        with open(_fj) as _fh:
            _d = json.load(_fh)
        _cfg = _d["selected_config"]
        selected[_sub] = {
            "nc": int(_cfg["n_components"]),
            "gamma": int(_cfg["gamma"]),
            "name": _d["selected_config_name"],
        }
        final_active[_sub] = int(_d["final_refit"]["n_active_states"])

    # --- audit print: every number that will appear in a panel -----------
    print("=== S3 audit ===")
    for _sub in SUBJECTS:
        _df = sweep[_sub]
        _sel = selected[_sub]
        _row = _df[(_df.nc == _sel["nc"]) & (_df.gamma == _sel["gamma"])]
        print(
            f"{_sub}: n_configs={len(_df)} "
            f"K range={_df.K_best_valid.min()}-{_df.K_best_valid.max()} "
            f"selected={_sel['name']} "
            f"sweep_K@selected={int(_row.K_best_valid.iloc[0])} "
            f"final_refit_K={final_active[_sub]}"
        )
    print(
        "final_refit K range:",
        min(final_active.values()),
        "-",
        max(final_active.values()),
        "(methods.md claims 37-42)",
    )
    return final_active, selected, sweep


@app.cell
def panel_A_pareto(
    GAMMA_COLORS,
    OUT,
    SELECTED_RING,
    SUBJECTS,
    plt,
    selected,
    sweep,
):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.0))

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        _df = sweep[_sub].copy()

        for _g, _c in GAMMA_COLORS.items():
            _m = _df.gamma == _g
            if not _m.any():
                continue
            _ax.scatter(
                _df.loc[_m, "K_best_valid"],
                _df.loc[_m, "valid_ll"],
                s=18,
                facecolor=_c,
                edgecolor="white",
                linewidth=0.4,
                label=f"$\\gamma$ = {_g}",
                zorder=3,
            )

        # Pareto frontier: no other config has >= LL with <= states.
        _pts = _df[["K_best_valid", "valid_ll"]].to_numpy()
        _keep = []
        for _j, (_k, _ll) in enumerate(_pts):
            _dominated = (
                (_pts[:, 0] <= _k)
                & (_pts[:, 1] >= _ll)
                & ((_pts[:, 0] < _k) | (_pts[:, 1] > _ll))
            ).any()
            if not _dominated:
                _keep.append(_j)
        _front = _df.iloc[_keep].sort_values("K_best_valid")
        _ax.plot(
            _front.K_best_valid,
            _front.valid_ll,
            color="#4A4A4A",
            linewidth=0.8,
            linestyle="--",
            zorder=2,
        )

        _sel = selected[_sub]
        _srow = _df[(_df.nc == _sel["nc"]) & (_df.gamma == _sel["gamma"])]
        _ax.scatter(
            _srow.K_best_valid,
            _srow.valid_ll,
            s=90,
            facecolor="none",
            edgecolor=SELECTED_RING,
            linewidth=1.3,
            zorder=4,
        )

        _ax.set_xlabel("Occupied states" if _i >= 3 else "")
        _ax.set_ylabel("Validation log-likelihood\nper sample" if _i % 3 == 0 else "")
        _ax.text(
            0.03,
            0.96,
            _sub,
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=6,
            color="#4A4A4A",
        )
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)

    _axes.flat[0].legend(
        frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.3
    )
    _fig.subplots_adjust(hspace=0.32, wspace=0.34)
    for _ext in ("pdf", "png"):
        _fig.savefig(
            OUT / f"figS03_A_pareto_ll_vs_states.{_ext}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("saved figS03_A_pareto_ll_vs_states")
    plt.close(_fig)
    return


@app.cell
def panel_B_states_vs_capacity(
    GAMMA_COLORS, OUT, SELECTED_RING, SUBJECTS, plt, selected, sweep
):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True, sharey=True)

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        _df = sweep[_sub]

        for _g, _c in GAMMA_COLORS.items():
            _sd = _df[_df.gamma == _g].sort_values("nc")
            if _sd.empty:
                continue
            _ax.plot(
                _sd.nc,
                _sd.K_best_valid,
                color=_c,
                marker="o",
                markersize=2.6,
                linewidth=1.2,
                label=f"$\\gamma$ = {_g}",
            )

        _sel = selected[_sub]
        _ax.axvline(_sel["nc"], color=SELECTED_RING, linewidth=0.8, linestyle=":")

        _ax.set_xlabel("Truncation capacity $K_{max}$" if _i >= 3 else "")
        _ax.set_ylabel("Occupied states" if _i % 3 == 0 else "")
        _ax.text(
            0.03,
            0.96,
            _sub,
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=6,
            color="#4A4A4A",
        )
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)

    _axes.flat[0].legend(
        frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.88),
        handletextpad=0.3,
    )
    _fig.subplots_adjust(hspace=0.22, wspace=0.14)
    for _ext in ("pdf", "png"):
        _fig.savefig(
            OUT / f"figS03_B_states_vs_capacity.{_ext}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("saved figS03_B_states_vs_capacity")
    plt.close(_fig)
    return


@app.cell
def panel_C_overfit_gap(
    GAMMA_COLORS, OUT, SELECTED_RING, SUBJECTS, plt, selected, sweep
):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        _df = sweep[_sub]

        for _g, _c in GAMMA_COLORS.items():
            _sd = _df[_df.gamma == _g].sort_values("nc")
            if _sd.empty:
                continue
            _ax.plot(
                _sd.nc,
                _sd.overfit_gap,
                color=_c,
                marker="o",
                markersize=2.6,
                linewidth=1.2,
                label=f"$\\gamma$ = {_g}",
            )

        _ax.axhline(0.0, color="#B0B0B0", linewidth=0.6, zorder=1)
        _sel = selected[_sub]
        _ax.axvline(_sel["nc"], color=SELECTED_RING, linewidth=0.8, linestyle=":")

        _ax.set_xlabel("Truncation capacity $K_{max}$" if _i >= 3 else "")
        _ax.set_ylabel("Train minus validation\nlog-likelihood" if _i % 3 == 0 else "")
        _ax.text(
            0.03,
            0.96,
            _sub,
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=6,
            color="#4A4A4A",
        )
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)

    _axes.flat[0].legend(
        frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.3
    )
    _fig.subplots_adjust(hspace=0.22, wspace=0.34)
    for _ext in ("pdf", "png"):
        _fig.savefig(
            OUT / f"figS03_C_overfit_gap.{_ext}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("saved figS03_C_overfit_gap")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
