"""Figure 2E (+ supplement): every participant's states in one embedding.

All six participants' state maps are embedded together, with the inputs and
metric that 05h_participant_state_matching matches on: contrast maps
(state_means_parcel - pca.mean_), row z-scored over the 156 parcels,
correlation distance. The UMAP is a display only. Distances and clusters on it
are not evidence; the matching statistics from 05h are.

Colour encodes participant. The project default puts participant identity on
marker shape, but here colour carries no other variable (no network colours on
these panels), so it encodes participant, with a colourblind-safe palette.

Chart-family note: Fig 1D and Fig 5A already use scatter. Panel E is kept as a
documented exception to the at-most-two-main-figures rule, because its axes are
an embedding of one similarity structure, not two measured variables (user
decision 2026-10-22).

| Panel | Content | Chart family | Output |
|---|---|---|---|
| 2E | Content-eligible states (16-31 per participant), pooled UMAP, colour = participant; a line for every matched pair whose full-map correlation exceeds the 95th percentile of the max-statistic configuration null; dark lines also co-occur in time (within-run circular-shift null) | scatter + segments | fig2/fig2_E_participant_umap |
| 2E (no lines) | The same embedding without lines (for talks) | scatter | fig2/fig2_E_participant_umap_nolines |
| S A | As 2E, but lines for matched pairs that survive removal of the three largest components of the other four participants' state maps | scatter + segments | fig2/supp_participant_umap/figS_participant_umap_A_residual_matches |
| S B | All active states, colour = participant, marker = 05e category (content-eligible, low-confidence, run-onset-anchored, drift-anchored, unused/rare), area = fractional occupancy | scatter | .../figS_participant_umap_B_active_categories |
| S C | Content-eligible states, colour = recurrence score | scatter | .../figS_participant_umap_C_recurrence |
| S D | Robustness: UMAP at 3 seeds x n_neighbors 5/15/30, PCA and classical MDS | scatter small multiples | .../figS_participant_umap_D_robustness |
| legends | participant colours; category markers; recurrence colorbar | legend | fig2_E_legend_participants, .../figS_participant_umap_legend_* |

The supplementary panels share one chart family because they show one
embedding with different encodings; the parallel layout is the point.

Also written: fig2/supp_participant_umap/participant_mixing.json, the share of
each state's 5 nearest neighbours (correlation distance, full 156-parcel space,
not the UMAP) that belong to other participants, against the share expected
from the participants' state counts if participant identity did not matter.
Descriptive, no test.

Run:
    marimo edit script/fig_F2E_participant_umap.py
or headless:
    uv run python script/fig_F2E_participant_umap.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import os
    import pickle
    import sys
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import umap
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.plot_style import RECURRENCE_CMAP, apply_publication_style

    apply_publication_style()
    return (Path, RECURRENCE_CMAP, json, np, os, pd, pickle, plt, umap, warnings)


@app.cell
def config(Path, os):
    SCRATCH = Path(os.environ["SCRATCH_DIR"])
    PARC, VT = "atlas-4S156Parcels", "0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
    # Paul Tol "bright" (colourblind-safe)
    SUBJ_COLORS = dict(zip(SUBJECTS, ["#4477AA", "#EE6677", "#228833",
                                      "#CCBB44", "#66CCEE", "#AA3377"]))
    # 05e_a4 summary categories in the manuscript's names
    CATEGORY_NAMES = {"eligible_for_content_analysis": "content-eligible",
                      "low_confidence": "low-confidence",
                      "run_onset_anchored": "run-onset-anchored",
                      "season_temporal": "drift-anchored",
                      "rare": "unused/rare", "unused": "unused/rare"}
    CATEGORY_MARKERS = {"content-eligible": "o", "low-confidence": "^",
                        "run-onset-anchored": "X", "drift-anchored": "s",
                        "unused/rare": "v"}
    UMAP_KW = dict(n_neighbors=15, min_dist=0.3, metric="correlation", random_state=0)
    OUT = SCRATCH / "output" / "manuscript_figures" / "fig2"
    SUPP = OUT / "supp_participant_umap"
    SUPP.mkdir(parents=True, exist_ok=True)
    return (CATEGORY_MARKERS, CATEGORY_NAMES, OUT, PARC, SCRATCH, SUBJECTS,
            SUBJ_COLORS, SUPP, UMAP_KW, VT)


@app.cell
def load_data(CATEGORY_NAMES, PARC, SCRATCH, SUBJECTS, VT, np, pd, pickle, warnings):
    _rows, _maps = [], []
    for _sub in SUBJECTS:
        _hmm = SCRATCH / "output" / "04_combined_hdphmm" / PARC / _sub / "final" / f"vt{VT}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(_hmm / "pca_model.pkl", "rb") as _f:
                _pca = pickle.load(_f)
        _m = np.load(_hmm / "state_means_parcel.npy") - _pca.mean_
        with open(_hmm / "decoded_states.pkl", "rb") as _f:
            _dec = {k: np.asarray(v, dtype=int) for k, v in pickle.load(_f).items()}
        _K = _m.shape[0]
        _fo_run = np.vstack([np.bincount(v, minlength=_K) / v.size for v in _dec.values()])
        _active = np.flatnonzero((_fo_run > 0.02).any(axis=0))
        _trs = np.bincount(np.concatenate(list(_dec.values())), minlength=_K)
        _flags = pd.read_csv(SCRATCH / "output" / "05e_temporal_trend_a4" / PARC / _sub
                             / f"vt{VT}" / "state_flags.csv").set_index("state")
        for _s in _active:
            _cat = CATEGORY_NAMES[_flags.loc[_s, "summary_category"]]
            _rows.append({"sub": _sub, "state": int(_s),
                          "eligible": _cat == "content-eligible", "category": _cat,
                          "recurrence": float(_flags.loc[_s, "recurrence_score"]),
                          "fo": float(_trs[_s] / _trs.sum())})
            _maps.append(_m[_s])
    state_table = pd.DataFrame(_rows)
    _X = np.vstack(_maps)
    state_maps = (_X - _X.mean(1, keepdims=True)) / _X.std(1, keepdims=True)
    print(state_table.groupby("sub")["eligible"].agg(["size", "sum"]))
    print("categories:", state_table["category"].value_counts().to_dict())
    return (state_maps, state_table)


@app.cell
def embeddings(UMAP_KW, state_maps, state_table, umap):
    _elig = state_table["eligible"].to_numpy()
    emb_elig = umap.UMAP(**UMAP_KW).fit_transform(state_maps[_elig])
    emb_active = umap.UMAP(**UMAP_KW).fit_transform(state_maps)
    return (emb_active, emb_elig)


@app.cell
def mixing(SUBJECTS, SUPP, json, np, state_maps, state_table):
    def _mix(mask):
        X = state_maps[mask]
        subj = state_table["sub"].to_numpy()[mask]
        n = X.shape[0]
        D = 1.0 - (X @ X.T) / X.shape[1]
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :5]
        other = (subj[nn] != subj[:, None]).mean(axis=1)
        counts = {s: int(np.sum(subj == s)) for s in SUBJECTS}
        # chance share of other-participant neighbours for a state of participant
        # s is (n - n_s) / (n - 1); average over states
        expected = sum(c * (n - c) for c in counts.values()) / (n * (n - 1))
        return {"mean_share_other_participants": float(other.mean()),
                "by_participant": {s: float(other[subj == s].mean()) for s in SUBJECTS},
                "expected_from_state_counts": float(expected),
                "state_counts": counts, "n_states": int(n)}

    mixing_summary = {"eligible": _mix(state_table["eligible"].to_numpy()),
                      "active": _mix(np.ones(len(state_table), dtype=bool)),
                      "note": "5 nearest neighbours by correlation distance in the "
                              "full 156-parcel space; descriptive, no test"}
    with open(SUPP / "participant_mixing.json", "w") as _f:
        json.dump(mixing_summary, _f, indent=1)
    print(json.dumps(mixing_summary, indent=1))
    return (mixing_summary,)


@app.cell
def load_matches(PARC, SCRATCH, json, pd):
    _root = SCRATCH / "output" / "05h_participant_state_matching" / PARC / "pairs"
    _rows, _full_rows = [], []
    for _d in sorted(_root.iterdir()):
        _sa, _sb = _d.name.split("_")
        _v = json.loads((_d / "summary.json").read_text())["variants"]["eligible|full|raw"]
        _locked = {(m["a"], m["b"]): t for m, t in
                   zip(_v["matches_full"], _v["validation"]["per_pair_time_locked"])}
        for _m in _v["shared_removed"]["3"]["matches"]:
            if _m["beyond_max_null95"]:
                _rows.append({"sub_a": _sa, "a": _m["a"], "sub_b": _sb, "b": _m["b"],
                              "r_residual": _m["r"],
                              "time_locked": bool(_locked.get((_m["a"], _m["b"]), False))})
        for _m in _v["matches_full"]:
            if _m["beyond_max_null95"]:
                _full_rows.append({"sub_a": _sa, "a": _m["a"], "sub_b": _sb, "b": _m["b"],
                                   "r": _m["r"], "time_locked": bool(_locked[(_m["a"], _m["b"])])})
    residual_match_table = pd.DataFrame(_rows)
    full_match_table = pd.DataFrame(_full_rows)
    print(f"{len(full_match_table)} full-map matches pass; "
          f"{int(full_match_table['time_locked'].sum())} of them also time-locked")
    print(f"{len(residual_match_table)} residual-map matches pass; "
          f"{int(residual_match_table['time_locked'].sum())} of them also time-locked")
    return (full_match_table, residual_match_table)


@app.cell
def style_helpers(plt):
    def finish(ax, stem, fig):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)
    return (finish,)


@app.cell
def line_panel(SUBJECTS, SUBJ_COLORS, emb_elig, finish, plt, state_table):
    def draw_match_lines(table, stem):
        _t = state_table[state_table["eligible"]].reset_index(drop=True)
        _pos = {(r.sub, r.state): i for i, r in enumerate(_t.itertuples())}
        _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
        if table is not None:
            for _locked, _col, _lw, _z in ((False, "#BBBBBB", 0.4, 1), (True, "#4A4A4A", 0.6, 2)):
                for _r in table[table["time_locked"] == _locked].itertuples():
                    _i, _j = _pos[(_r.sub_a, _r.a)], _pos[(_r.sub_b, _r.b)]
                    _ax.plot([emb_elig[_i, 0], emb_elig[_j, 0]],
                             [emb_elig[_i, 1], emb_elig[_j, 1]], color=_col, lw=_lw, zorder=_z)
        for _s in SUBJECTS:
            _m = (_t["sub"] == _s).to_numpy()
            _ax.scatter(emb_elig[_m, 0], emb_elig[_m, 1], s=22, c=SUBJ_COLORS[_s],
                        edgecolors="white", linewidths=0.4, alpha=0.95, zorder=3)
        finish(_ax, stem, _fig)
    return (draw_match_lines,)


@app.cell
def panel_2E(OUT, SUPP, draw_match_lines, full_match_table, residual_match_table):
    draw_match_lines(full_match_table, OUT / "fig2_E_participant_umap")
    draw_match_lines(None, OUT / "fig2_E_participant_umap_nolines")
    draw_match_lines(residual_match_table, SUPP / "figS_participant_umap_A_residual_matches")
    return


@app.cell
def panel_S_B(CATEGORY_MARKERS, SUBJECTS, SUBJ_COLORS, SUPP, emb_active, finish, np, plt,
              state_table):
    _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
    _size = 4 + 120 * np.sqrt(state_table["fo"].to_numpy())
    for _cat, _mk in CATEGORY_MARKERS.items():
        for _s in SUBJECTS:
            _m = ((state_table["category"] == _cat) & (state_table["sub"] == _s)).to_numpy()
            if _m.any():
                _ax.scatter(emb_active[_m, 0], emb_active[_m, 1], s=_size[_m], marker=_mk,
                            c=SUBJ_COLORS[_s], edgecolors="white", linewidths=0.4, alpha=0.85)
    finish(_ax, SUPP / "figS_participant_umap_B_active_categories", _fig)
    return


@app.cell
def panel_S_C(RECURRENCE_CMAP, SUPP, emb_elig, finish, plt, state_table):
    _t = state_table[state_table["eligible"]].reset_index(drop=True)
    _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
    _ax.scatter(emb_elig[:, 0], emb_elig[:, 1], s=22, c=_t["recurrence"], cmap=RECURRENCE_CMAP,
                vmin=0, vmax=1, edgecolors="white", linewidths=0.4)
    finish(_ax, SUPP / "figS_participant_umap_C_recurrence", _fig)
    return


@app.cell
def panel_S_D(SUBJECTS, SUBJ_COLORS, SUPP, UMAP_KW, np, plt, state_maps, state_table, umap):
    _elig = state_table["eligible"].to_numpy()
    _X = state_maps[_elig]
    _subj = state_table["sub"].to_numpy()[_elig]
    _layouts = []
    for _nn in (5, 15, 30):
        for _seed in (0, 1, 2):
            _layouts.append((f"n_neighbors {_nn}, seed {_seed}",
                             umap.UMAP(**{**UMAP_KW, "n_neighbors": _nn,
                                          "random_state": _seed}).fit_transform(_X)))
    _Xc = _X - _X.mean(axis=0)
    _U, _S, _ = np.linalg.svd(_Xc, full_matrices=False)
    _layouts.append(("PCA", _U[:, :2] * _S[:2]))
    _D = 1.0 - (_X @ _X.T) / _X.shape[1]
    _n = _D.shape[0]
    _J = np.eye(_n) - np.ones((_n, _n)) / _n
    _B = -0.5 * _J @ (_D ** 2) @ _J
    _w, _v = np.linalg.eigh(_B)
    _layouts.append(("classical MDS", _v[:, -2:][:, ::-1] * np.sqrt(np.maximum(_w[-2:][::-1], 0))))
    _fig, _axes = plt.subplots(2, 6, figsize=(10.5, 3.8))
    for _ax in _axes.ravel():
        _ax.set_axis_off()
    for _ax, (_name, _Y) in zip(_axes.ravel(), _layouts):
        _ax.set_axis_on()
        for _s in SUBJECTS:
            _m = _subj == _s
            _ax.scatter(_Y[_m, 0], _Y[_m, 1], s=7, c=SUBJ_COLORS[_s], linewidths=0)
        _ax.set_xticks([]); _ax.set_yticks([])
        _ax.set_xlabel(_name, fontsize=7)
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
    for _ext in ("png", "svg", "pdf"):
        _fig.savefig(SUPP / f"figS_participant_umap_D_robustness.{_ext}",
                     bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    return


@app.cell
def legends(CATEGORY_MARKERS, OUT, RECURRENCE_CMAP, SUBJECTS, SUBJ_COLORS, SUPP, plt):
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    def _save(fig, stem):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
        plt.close(fig)

    _h = [Line2D([], [], marker="o", ls="", color=SUBJ_COLORS[s], label=s) for s in SUBJECTS]
    _fig, _ax = plt.subplots(figsize=(1.2, 1.6)); _ax.set_axis_off()
    _ax.legend(handles=_h, loc="center", frameon=False, fontsize=7)
    _save(_fig, OUT / "fig2_E_legend_participants")
    _h = [Line2D([], [], marker=_mk, ls="", color="#4A4A4A", label=_cat)
          for _cat, _mk in CATEGORY_MARKERS.items()]
    _fig, _ax = plt.subplots(figsize=(1.6, 1.4)); _ax.set_axis_off()
    _ax.legend(handles=_h, loc="center", frameon=False, fontsize=7)
    _save(_fig, SUPP / "figS_participant_umap_legend_categories")
    _fig, _ax = plt.subplots(figsize=(2.0, 0.3))
    _cb = _fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=RECURRENCE_CMAP),
                        cax=_ax, orientation="horizontal")
    _cb.set_label("Recurrence score", fontsize=7); _cb.ax.tick_params(labelsize=6)
    _save(_fig, SUPP / "figS_participant_umap_legend_recurrence")
    return


if __name__ == "__main__":
    app.run()
