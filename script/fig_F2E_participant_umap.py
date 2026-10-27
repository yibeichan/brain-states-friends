"""Figure 2E (+ supplement): every participant's states in one embedding.

All six participants' state maps are embedded together, with the inputs and
metric that 05h_participant_state_matching matches on: contrast maps
(state_means_parcel - pca.mean_), row z-scored over the 156 parcels,
correlation distance. The UMAP is a display only. Distances and clusters on it
are not evidence; the matching statistics from 05h are.

In 2E (and S A) colour encodes each state's dominant network, with the
network colours of Fig 2C, and marker shape encodes participant, with the
markers of Fig 1C. A line between two states of the same colour is a match
between states with the same dominant network. Marker shape is the participant
in every panel; S B colours states by category with the colours of Fig 2A.

Chart-family note: Fig 1D and Fig 5A already use scatter. Panel E is kept as a
documented exception to the at-most-two-main-figures rule, because its axes are
an embedding of one similarity structure, not two measured variables (user
decision 2026-10-22).

| Panel | Content | Chart family | Output |
|---|---|---|---|
| 2E | 5 in wide, one scale on both UMAP axes. Content-eligible states (16-31 per participant), pooled UMAP, colour = dominant network (05e state_flags, Fig 2C colours), marker = participant (Fig 1C markers); a line for every Hungarian match between two participants (descriptive; width and opacity grow with the matched r); dark lines are time-locked (family-wise threshold on the within-run circular-shift null) | scatter + segments | fig2/fig2_E_participant_umap |
| 2E (no lines) | The same embedding without lines (for talks) | scatter | fig2/fig2_E_participant_umap_nolines |
| S A | As 2E, but for the matches after removing the three largest components of the other four participants' state maps | scatter + segments | si_participant_umap/figS_participant_umap_A_residual_matches |
| S B | All active states, colour = 05e summary category (Fig 2A colours: content-eligible, low-confidence, run-onset-anchored, drift-anchored, rare; a sub-HRF state that is also run-onset-anchored counts as low-confidence, as in Fig 2A), marker = participant (Fig 1C markers), marker area grows with the square root of fractional occupancy (size 4 + 120 sqrt(FO) pt^2) | scatter | .../figS_participant_umap_B_active_categories |
| S C | Content-eligible states, colour = recurrence score | scatter | .../figS_participant_umap_C_recurrence |
| S D | Robustness: UMAP at 3 seeds x n_neighbors 5/15/30, PCA and classical MDS; marker = participant | scatter small multiples | .../figS_participant_umap_D_robustness |
| S E | Occupancy correlation of every map match (all 353, not only time-locked) at lags of -20..20 TRs of the second participant within each run; thin lines = mean per participant pair, thick line = mean over matches; lag 0 equals 05h's locking r | line | .../figS_participant_umap_E_lagged_locking |
| legends | 2E participant markers (also for S A, S B, S D) and dominant-network colours (also for S A); category colours (S B); recurrence colorbar (S C) | legend | fig2_E_legend_participants, fig2_E_legend_networks, .../figS_participant_umap_legend_* |

Supplementary panels A-D share one chart family because they show one
embedding with different encodings; the parallel layout is the point. E shows
the timing behind the time-locked flag, which the embedding cannot show. E's
value at lag 0 is checked against 05h's per-match locking r.

Also written: si_participant_umap/participant_mixing.json, the share of
each state's 5 nearest neighbours (correlation distance, full 156-parcel space,
not the UMAP) that belong to other participants, against the share expected
from the participants' state counts if participant identity did not matter.
Descriptive, no test.

umap-learn is an optional extra (it pins numba, which has no wheels for every
platform). Install it with:
    uv sync --extra umap

Run:
    uv run --extra umap marimo edit script/fig_F2E_participant_umap.py
or headless:
    uv run --extra umap python script/fig_F2E_participant_umap.py
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
    from utils.plot_style import (NETWORK_COLORS, RECURRENCE_CMAP, SUBJECT_MARKERS,
                                  SUBJECT_NEUTRAL, apply_publication_style)
    from utils.recurrence_utils import active_state_indices

    apply_publication_style()
    return (NETWORK_COLORS, Path, RECURRENCE_CMAP, SUBJECT_MARKERS, SUBJECT_NEUTRAL,
            active_state_indices,
            json, np, os, pd, pickle, plt, umap, warnings)


@app.cell
def config(Path, os):
    SCRATCH = Path(os.environ["SCRATCH_DIR"])
    PARC, VT = "atlas-4S156Parcels", "0.95"
    SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
    # 05e_a4 summary categories in the manuscript's names
    CATEGORY_NAMES = {"eligible_for_content_analysis": "content-eligible",
                      "low_confidence": "low-confidence",
                      "run_onset_anchored": "run-onset-anchored",
                      "season_temporal": "drift-anchored",
                      "rare": "rare"}  # active states are never "unused"
    # the category colours of Fig 2A (fig_F2_recurrence_sources.TAXONOMY_COLORS)
    CATEGORY_COLORS = {"content-eligible": "#0C7BDC", "low-confidence": "#5A5A5A",
                       "run-onset-anchored": "#FFC20A", "drift-anchored": "#D35FB7",
                       "rare": "#CCCCCC"}
    UMAP_KW = dict(n_neighbors=15, min_dist=0.3, metric="correlation", random_state=0)
    OUT = SCRATCH / "output" / "manuscript_figures" / "fig2"
    SUPP = SCRATCH / "output" / "manuscript_figures" / "si_participant_umap"
    SUPP.mkdir(parents=True, exist_ok=True)
    return (CATEGORY_COLORS, CATEGORY_NAMES, OUT, PARC, SCRATCH, SUBJECTS,
            SUPP, UMAP_KW, VT)


@app.cell
def load_data(CATEGORY_NAMES, PARC, SCRATCH, SUBJECTS, VT, active_state_indices, np, pd,
              pickle, warnings):
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
        _active = active_state_indices(_dec, _K)
        _trs = np.bincount(np.concatenate(list(_dec.values())), minlength=_K)
        _flags = pd.read_csv(SCRATCH / "output" / "05e_temporal_trend_a4" / PARC / _sub
                             / f"vt{VT}" / "state_flags.csv").set_index("state")
        for _s in _active:
            _cat = CATEGORY_NAMES[_flags.loc[_s, "summary_category"]]
            _rows.append({"sub": _sub, "state": int(_s),
                          "eligible": _cat == "content-eligible", "category": _cat,
                          "recurrence": float(_flags.loc[_s, "recurrence_score"]),
                          "network": _flags.loc[_s, "dominant_network"],
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
def load_matches(PARC, SCRATCH, SUBJECTS, VT, json, pd):
    from itertools import combinations

    _base = SCRATCH / "output" / "05h_participant_state_matching" / PARC
    _expected = {f"{a}_{b}" for a, b in combinations(SUBJECTS, 2)}
    _found = {f.parent.name for f in (_base / "pairs").glob("*/summary.json")}
    if _found != _expected:
        raise RuntimeError(f"05h pair summaries under {_base / 'pairs'}: missing "
                           f"{sorted(_expected - _found)}, unexpected {sorted(_found - _expected)}")

    def _provenance(d):
        # run parameters, code commit, and any content hashes 05h records
        # (top level or under "provenance"); run-specific fields such as the
        # time stamp are left out
        prov = {k: d.get(k) for k in ("params", "code_commit")}
        prov.update({k: v for k, v in d.items() if "sha256" in k})
        prov.update({f"provenance.{k}": v for k, v in (d.get("provenance") or {}).items()
                     if "sha256" in k})
        return prov

    _summaries = {k: json.loads((_base / "pairs" / k / "summary.json").read_text())
                  for k in sorted(_expected)}
    _cross_path = _base / "cross_subject_summary.json"
    if not _cross_path.exists():
        raise RuntimeError(f"{_cross_path} is missing: 05h writes it last, so the pair "
                           "summaries may come from an unfinished run")
    _ref = _provenance(json.loads(_cross_path.read_text()))
    if "code_sha256" not in _ref or "inputs_sha256" not in _ref:
        raise RuntimeError("05h output has no code or input hashes (written by an older "
                           "05h); rerun 05h")
    # the 04 maps and 05e flags this figure reads must be the ones 05h matched
    import importlib
    _m05h = importlib.import_module("05h_participant_state_matching")
    _now = {s: _m05h.sha256_files(_m05h.subject_input_files(s, PARC, VT)) for s in SUBJECTS}
    _now["dseg"] = _m05h.sha256_files([_m05h.dseg_path(PARC)])
    _changed = sorted(k for k, v in _now.items() if _ref["inputs_sha256"].get(k) != v)
    if _changed:
        raise RuntimeError(f"inputs changed since 05h ran for {_changed}; rerun 05h")
    _params = _ref["params"] or {}
    if _params.get("parcellation") != PARC or str(_params.get("vt")) != VT:
        raise RuntimeError(f"05h was run with parcellation {_params.get('parcellation')}, "
                           f"vt {_params.get('vt')}; this figure uses {PARC}, vt {VT}")
    for _k, _d in _summaries.items():
        if _d.get("pair") != _k:
            raise RuntimeError(f"pairs/{_k}/summary.json records pair {_d.get('pair')}")
        if _provenance(_d) != _ref:
            raise RuntimeError(f"pairs/{_k}/summary.json provenance {_provenance(_d)} differs "
                               f"from cross_subject_summary.json {_ref}; rerun 05h")

    _rows, _full_rows = [], []
    for _k, _d in _summaries.items():
        _sa, _sb = _k.split("_")
        _v = _d["variants"]["eligible|full|raw"]
        if _m05h.NULL not in _v:
            raise RuntimeError(f"pairs/{_k}: no {_m05h.NULL} result; rerun 05h")
        _val = _v["validation"]
        # time-locked = family-wise flag (max over matched pairs of the shift null);
        # 05h stores the flags in the order of the matches
        # every Hungarian match is drawn: 05h tests pairs, not single matches
        for _m, _t, _lr in zip(_v["matches_full"], _val["per_pair_time_locked"],
                               _val["per_pair_locking_r"], strict=True):
            _full_rows.append({"sub_a": _sa, "a": _m["a"], "sub_b": _sb, "b": _m["b"],
                               "r": _m["r"], "time_locked": bool(_t), "lock_r": _lr})
        for _m, _t in zip(_v["shared_removed"]["3"]["matches"],
                          _val["residual_matches_time_locked"], strict=True):
            _rows.append({"sub_a": _sa, "a": _m["a"], "sub_b": _sb, "b": _m["b"],
                          "r": _m["r"], "time_locked": bool(_t)})
    residual_match_table = pd.DataFrame(_rows)
    full_match_table = pd.DataFrame(_full_rows)
    print(f"{len(full_match_table)} full-map matches; "
          f"{int(full_match_table['time_locked'].sum())} of them time-locked")
    print(f"{len(residual_match_table)} residual-map matches; "
          f"{int(residual_match_table['time_locked'].sum())} of them time-locked")
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
def line_panel(NETWORK_COLORS, SUBJECTS, SUBJECT_MARKERS, emb_elig, finish, np, plt,
               state_table):
    def draw_match_lines(table, stem, width=None, dot_size=22):
        # line width and opacity grow with the matched r (clipped at 0).
        # width (inches, saved panel) draws both UMAP axes on one scale, so the
        # height follows the embedding's extent; None keeps the 3.4 in square.
        _t = state_table[state_table["eligible"]].reset_index(drop=True)
        _pos = {(r.sub, r.state): i for i, r in enumerate(_t.itertuples())}
        if width is None:
            _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
        else:
            _lo, _hi = emb_elig.min(axis=0), emb_elig.max(axis=0)
            _pad = 0.03 * float((_hi - _lo).max())
            _lo, _hi = _lo - _pad, _hi + _pad
            _aw = width - 0.2  # the y-axis label takes about 0.2 in
            _ah = _aw * float(_hi[1] - _lo[1]) / float(_hi[0] - _lo[0])
            _m = 0.4  # margin for the axis labels; bbox_inches="tight" trims it
            _fig = plt.figure(figsize=(_aw + _m, _ah + _m))
            _ax = _fig.add_axes([_m / (_aw + _m), _m / (_ah + _m),
                                 _aw / (_aw + _m), _ah / (_ah + _m)])
            _ax.set_xlim(_lo[0], _hi[0]); _ax.set_ylim(_lo[1], _hi[1])
        if table is not None:
            for _locked, _col, _z in ((False, "#BBBBBB", 1), (True, "#4A4A4A", 2)):
                for _r in table[table["time_locked"] == _locked].itertuples():
                    _i, _j = _pos[(_r.sub_a, _r.a)], _pos[(_r.sub_b, _r.b)]
                    _w = float(np.clip(_r.r, 0.0, 1.0))
                    _ax.plot([emb_elig[_i, 0], emb_elig[_j, 0]],
                             [emb_elig[_i, 1], emb_elig[_j, 1]], color=_col,
                             lw=0.15 + 0.6 * _w, alpha=0.25 + 0.7 * _w, zorder=_z)
        for _s in SUBJECTS:
            _m = (_t["sub"] == _s).to_numpy()
            _ax.scatter(emb_elig[_m, 0], emb_elig[_m, 1], s=dot_size,
                        c=[NETWORK_COLORS[n] for n in _t.loc[_m, "network"]],
                        marker=SUBJECT_MARKERS[_s], edgecolors="white", linewidths=0.4,
                        alpha=0.95, zorder=3)
        finish(_ax, stem, _fig)
    return (draw_match_lines,)


@app.cell
def panel_2E(OUT, SUPP, draw_match_lines, full_match_table, residual_match_table):
    # the main panel fills the bottom row of Fig 2 next to its legend
    draw_match_lines(full_match_table, OUT / "fig2_E_participant_umap", width=5.0, dot_size=50)
    draw_match_lines(None, OUT / "fig2_E_participant_umap_nolines", width=5.0, dot_size=50)
    draw_match_lines(residual_match_table, SUPP / "figS_participant_umap_A_residual_matches")
    return


@app.cell
def panel_S_B(CATEGORY_COLORS, SUBJECTS, SUBJECT_MARKERS, SUPP, emb_active, finish, np,
              plt, state_table):
    _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
    _size = 4 + 120 * np.sqrt(state_table["fo"].to_numpy())
    for _s in SUBJECTS:
        _m = (state_table["sub"] == _s).to_numpy()
        _ax.scatter(emb_active[_m, 0], emb_active[_m, 1], s=_size[_m],
                    marker=SUBJECT_MARKERS[_s],
                    c=[CATEGORY_COLORS[c] for c in state_table.loc[_m, "category"]],
                    edgecolors="white", linewidths=0.4, alpha=0.85)
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
def panel_S_D(SUBJECTS, SUBJECT_MARKERS, SUBJECT_NEUTRAL, SUPP, UMAP_KW, np, plt, state_maps,
              state_table, umap):
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
            _ax.scatter(_Y[_m, 0], _Y[_m, 1], s=9, marker=SUBJECT_MARKERS[_s],
                        c=SUBJECT_NEUTRAL, linewidths=0, alpha=0.8)
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
def panel_S_E(PARC, SCRATCH, SUBJECTS, SUPP, VT, full_match_table, json, np, pickle, plt):
    # lagged occupancy correlation for every map match: Pearson r between one
    # state's on/off sequence and its match's, with the second participant
    # shifted by -20..20 TRs within each run (no wrap), pooled over the runs both
    # watched (truncated to the shorter run, as in 05h). All 353 matches, not only
    # the time-locked ones, so a peak at lag 0 is not built in by the selection.
    TR_S = 1.49
    LAGS = np.arange(-20, 21)

    def _dec(sub):
        _p = (SCRATCH / "output" / "04_combined_hdphmm" / PARC / sub / "final"
              / f"vt{VT}" / "decoded_states.pkl")
        with open(_p, "rb") as _f:
            return {k: np.asarray(v, dtype=int) for k, v in pickle.load(_f).items()}

    _decs = {s: _dec(s) for s in SUBJECTS}
    _curves, _pair_of, _locked = [], [], []
    for (_sa, _sb), _g in full_match_table.groupby(["sub_a", "sub_b"], sort=True):
        _A, _B = _decs[_sa], _decs[_sb]
        _ia, _ib = _g["a"].to_numpy(), _g["b"].to_numpy()
        _N = np.zeros((LAGS.size, _ia.size)); _na = np.zeros_like(_N)
        _nb = np.zeros_like(_N); _T = np.zeros(LAGS.size)
        for _r in sorted(set(_A) & set(_B)):
            _L = min(_A[_r].size, _B[_r].size)
            _xa = (_A[_r][:_L, None] == _ia[None, :]).astype(float)
            _xb = (_B[_r][:_L, None] == _ib[None, :]).astype(float)
            for _k, _lag in enumerate(LAGS):
                # b at t + lag against a at t
                _a = _xa[max(0, -_lag):_L - max(0, _lag)]
                _b = _xb[max(0, _lag):_L - max(0, -_lag)]
                if not len(_a):
                    continue
                _N[_k] += (_a * _b).sum(0); _na[_k] += _a.sum(0); _nb[_k] += _b.sum(0)
                _T[_k] += len(_a)
        _T = _T[:, None]
        _den = np.sqrt(_na * (_T - _na) * _nb * (_T - _nb))
        _rr = np.where(_den > 0, (_N * _T - _na * _nb) / np.where(_den > 0, _den, 1), np.nan)
        _curves.append(_rr.T); _pair_of += [f"{_sa}_{_sb}"] * _ia.size
        _locked += list(_g["time_locked"])
    _C = np.vstack(_curves)
    _pairs = sorted(set(_pair_of)); _po = np.array(_pair_of)
    _pair_mean = np.array([np.nanmean(_C[_po == p], axis=0) for p in _pairs])
    _lag0 = int(np.where(LAGS == 0)[0][0])
    # check against 05h: lag-0 r per match equals 05h's per-match locking r
    _diff = float(np.nanmax(np.abs(_C[:, _lag0] - np.concatenate(
        [g["lock_r"].to_numpy() for _, g in full_match_table.groupby(["sub_a", "sub_b"], sort=True)]))))
    if _diff > 1e-9:
        raise RuntimeError(f"lag-0 occupancy r differs from 05h locking r by {_diff}")

    _fig, _ax = plt.subplots(figsize=(3.4, 2.4))
    _x = LAGS * TR_S
    for _pm in _pair_mean:
        _ax.plot(_x, _pm, color="#BBBBBB", lw=0.6, zorder=1)
    _ax.plot(_x, np.nanmean(_C, axis=0), color="#4A4A4A", lw=1.6, zorder=2)
    _ax.axhline(0, color="black", lw=0.5, ls=":")
    _ax.axvline(0, color="black", lw=0.5, ls=":")
    _ax.set_xlabel("Lag of second participant (s)", fontsize=7)
    _ax.set_ylabel("Occupancy correlation", fontsize=7)
    _ax.tick_params(labelsize=6)
    for _sp in ("top", "right"):
        _ax.spines[_sp].set_visible(False)
    for _ext in ("png", "svg", "pdf"):
        _fig.savefig(SUPP / f"figS_participant_umap_E_lagged_locking.{_ext}",
                     bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(_fig)
    _summary = {"lags_tr": LAGS.tolist(), "tr_s": TR_S, "n_matches": int(_C.shape[0]),
                "mean_all": np.nanmean(_C, axis=0).tolist(),
                "mean_time_locked": np.nanmean(_C[np.array(_locked)], axis=0).tolist(),
                "mean_not_time_locked": np.nanmean(_C[~np.array(_locked)], axis=0).tolist(),
                "pair_means": dict(zip(_pairs, _pair_mean.tolist())),
                "note": "descriptive; lag 0 equals 05h's locking r per match"}
    with open(SUPP / "lagged_locking.json", "w") as _f:
        json.dump(_summary, _f, indent=1)
    _m = np.nanmean(_C, axis=0)
    print(f"S E: mean r at lag 0 = {_m[_lag0]:.4f}, at +-20 TR = {_m[0]:.4f} / {_m[-1]:.4f}; "
          f"peak lag {LAGS[np.nanargmax(_m)]} TR; pairs peaking at 0: "
          f"{int(sum(LAGS[np.nanargmax(p)] == 0 for p in _pair_mean))}/{len(_pairs)}")
    return


@app.cell
def legends(CATEGORY_COLORS, NETWORK_COLORS, OUT, RECURRENCE_CMAP, SUBJECTS,
            SUBJECT_MARKERS, SUPP, plt, state_table):
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    def _save(fig, stem):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # 2E: participant = marker shape (as in Fig 1C), dominant network = colour
    # (as in Fig 2C), listing only the networks that dominate a content-eligible state
    _h = [Line2D([], [], marker=SUBJECT_MARKERS[s], ls="", color="#4A4A4A", label=s)
          for s in SUBJECTS]
    _fig, _ax = plt.subplots(figsize=(1.2, 1.6)); _ax.set_axis_off()
    _ax.legend(handles=_h, loc="center", frameon=False, fontsize=7)
    _save(_fig, OUT / "fig2_E_legend_participants")
    _nets = set(state_table.loc[state_table["eligible"], "network"])
    _h = [Line2D([], [], marker="o", ls="", color=c, label=n)
          for n, c in NETWORK_COLORS.items() if n in _nets]
    _fig, _ax = plt.subplots(figsize=(1.2, 1.6)); _ax.set_axis_off()
    _ax.legend(handles=_h, loc="center", frameon=False, fontsize=7)
    _save(_fig, OUT / "fig2_E_legend_networks")
    # S B: state category = colour (as in Fig 2A); participant markers as in 2E
    _h = [Line2D([], [], marker="o", ls="", color=_c, label=_cat)
          for _cat, _c in CATEGORY_COLORS.items()]
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
