"""Figure S4 (supplementary) - within-Friends reliability (LOSO + split-half).

Home of the matched-pair spatial correlation distribution that Methods cites as
the reproducibility evidence behind the permissive r > 0.3 Hungarian screen.
NOTE (2026-08-19): Methods previously attributed that distribution to main
Figure 1D, but Figure 1D is the split-half *recurrence* scatter (recurrence in
half A vs half B, Spearman rho 0.60-0.82). The spatial matched-pair r
distribution was never in the manuscript; it lives here.

| Panel | Content | Chart family |
|---|---|---|
| A | Hungarian-matched parcel-space Pearson r, LOSO folds pooled vs split-half, one small multiple per subject. Violin + strip, mean marked, r = 0.3 screen drawn. | distribution |
| B | Structural invariants across LOSO folds (k_active, transition entropy, self-transition probability, median Viterbi dwell), one small multiple per invariant, subjects on x. Shaded band is the 10-seed initialization range of the primary model, a relative-scale anchor only, NOT a confidence interval. | point-based 1D |
| C | Raw within-state FC RV for matched pairs and a mismatched-pair null, LOSO and split-half. Ceiling panel: shows that raw RV does not discriminate. | point-based 1D |
| D | Mean-removed within-state FC cosine, same four groups. The arm that actually discriminates. | point-based 1D |

Audited values (2026-08-19, against 04ra/04rb JSONs):
  * Matched-pair r MEAN (not median) per subject.
    LOSO:       0.912 0.890 0.910 0.910 0.883 0.884  (sub-01..06)
    Split-half: 0.823 0.811 0.867 0.815 0.850 0.882
    These reproduce the annotations on the legacy 04rv fig1. The medians are
    higher (LOSO 0.952-0.985, split-half 0.838-0.958); the distribution is
    left-skewed, so mean and median differ by design. Panel A marks the mean and
    labels it as such.
  * LOSO pooled pair counts: 232 226 234 145 236 221 (sub-04 has 4 folds, not 6).
  * Fraction of matched pairs above the r = 0.3 screen: 0.946-0.991 (LOSO),
    0.949-1.000 (split-half).

Panels C and D add the within-state FC arm that Methods describes. Correction to
an earlier note in this file: the legacy ``04rv`` fig7a-d outputs are NOT blank.
They are git-annex symlinks whose 132-byte ``ls`` size is the symlink target
path length; the annex objects are present and 196-444 kB. ``04rc`` has also
already run, writing into the fold directories rather than into an ``04rc_*``
output tree. What was missing was promotion into the SI, not the computation.

Two FC quantities, because the obvious one does not work:
  * raw FC RV (panel C) is at ceiling and carries almost no state-specific
    information. Matched pairs average 0.973-0.995 per subject, but
    DELIBERATELY MISMATCHED pairs average 0.949-0.992. The gap is 0.002-0.024.
    Every within-state FC matrix is dominated by a component common to all
    states, so RV cannot distinguish a matched pair from an unmatched one and
    cannot flag the "mean-only match" Methods wants it to flag.
  * mean-removed FC cosine (panel D) does work. After subtracting each fit's
    across-state mean FC, matched pairs average 0.229-0.504 while the mismatched
    null centres near zero (-0.098 to +0.046), a difference of 0.275-0.571.

Panel D uses an UNCLIPPED cosine, not ``compute_rv_coefficient``. That helper
ends in ``np.clip(rv, 0.0, 1.0)``, which is right for the raw matrices (positive
semi-definite, so the quantity cannot be negative) but wrong once the mean is
removed: 2.8-23.7% of matched pairs are genuinely negative, and clipping them to
zero both hides the sign and inflates the null to 0.076-0.216.

Run:
    marimo edit script/fig_S04_reliability.py
"""

import marimo

__generated_with = "0.9.0"
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
    return Path, json, np, os, plt


@app.cell
def config(Path, os):
    SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
    PARCELLATION = "atlas-4S156Parcels"

    PARCELLATION_FC = PARCELLATION
    LOSO_DIR = SCRATCH_DIR / "output" / "04ra_loso_struct_comp" / PARCELLATION
    SH_DIR = SCRATCH_DIR / "output" / "04rb_split_half" / PARCELLATION
    OUT = SCRATCH_DIR / "output" / "manuscript_figures" / "figS04"
    OUT.mkdir(parents=True, exist_ok=True)

    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 7)]

    LOSO_COLOR = "#1B4F72"
    SH_COLOR = "#E67E22"
    MEAN_MARK = "#D62728"
    BAND_COLOR = "#B0B0B0"

    INVARIANTS = [
        ("k_active", "Active states"),
        ("transition_entropy", "Transition entropy"),
        ("self_transition_prob", "Self-transition\nprobability"),
        ("dwell_median_tr", "Median dwell (TR)"),
    ]
    return (
        BAND_COLOR,
        INVARIANTS,
        LOSO_COLOR,
        LOSO_DIR,
        MEAN_MARK,
        OUT,
        PARCELLATION_FC,
        SH_COLOR,
        SH_DIR,
        SUBJECTS,
    )


@app.cell
def load_data(LOSO_DIR, SH_DIR, SUBJECTS, json, np):
    matched_r = {}
    invariants = {}
    noise_floor = {}

    for _sub in SUBJECTS:
        with open(LOSO_DIR / _sub / "hungarian_matching.json") as _fh:
            _lo = json.load(_fh)
        _lc = []
        for _season in _lo["per_fold"]:
            _lc.extend(_lo["per_fold"][_season]["matched_correlations"])

        with open(SH_DIR / _sub / "hungarian_matching.json") as _fh:
            _sh = json.load(_fh)
        _sc = [_p["correlation"] for _p in _sh["matching"]["pairs"]]

        matched_r[_sub] = {
            "loso": np.asarray(_lc, dtype=float),
            "split_half": np.asarray(_sc, dtype=float),
            "n_folds": len(_lo["per_fold"]),
        }

        with open(LOSO_DIR / _sub / "cross_fold_consistency.json") as _fh:
            invariants[_sub] = json.load(_fh)["scalar_invariants"]
        with open(LOSO_DIR / _sub / "noise_floor.json") as _fh:
            noise_floor[_sub] = json.load(_fh)

    # --- audit print -----------------------------------------------------
    print("=== S4 audit: matched-pair r ===")
    for _sub in SUBJECTS:
        _d = matched_r[_sub]
        print(
            f"{_sub}: LOSO n={_d['loso'].size} folds={_d['n_folds']} "
            f"mean={_d['loso'].mean():.3f} median={np.median(_d['loso']):.3f} "
            f"frac>0.3={np.mean(_d['loso'] > 0.3):.3f} | "
            f"SH n={_d['split_half'].size} mean={_d['split_half'].mean():.3f} "
            f"median={np.median(_d['split_half']):.3f} "
            f"frac>0.3={np.mean(_d['split_half'] > 0.3):.3f}"
        )
    print("=== S4 audit: noise-floor keys ===")
    print(sorted(k for k, v in noise_floor["sub-01"].items() if isinstance(v, dict)))
    return invariants, matched_r, noise_floor


@app.cell
def panel_A_matched_r(
    LOSO_COLOR, MEAN_MARK, OUT, SH_COLOR, SUBJECTS, matched_r, np, plt
):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
    _rng = np.random.default_rng(0)

    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        _d = matched_r[_sub]

        for _pos, (_key, _col) in enumerate(
            [("loso", LOSO_COLOR), ("split_half", SH_COLOR)]
        ):
            _v = _d[_key]
            _parts = _ax.violinplot(
                _v, positions=[_pos], widths=0.7, showextrema=False, showmeans=False
            )
            for _b in _parts["bodies"]:
                _b.set_facecolor(_col)
                _b.set_alpha(0.22)
                _b.set_edgecolor(_col)
                _b.set_linewidth(0.6)

            _jit = _rng.uniform(-0.11, 0.11, size=_v.size)
            _ax.scatter(
                np.full(_v.size, _pos) + _jit,
                _v,
                s=3.5,
                color=_col,
                alpha=0.5,
                linewidth=0,
                zorder=3,
            )
            _ax.scatter(
                [_pos],
                [_v.mean()],
                marker="D",
                s=26,
                facecolor=MEAN_MARK,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            _ax.annotate(
                f"mean {_v.mean():.2f}",
                xy=(_pos, _v.mean()),
                xytext=(6, -1),
                textcoords="offset points",
                fontsize=5.2,
                color=MEAN_MARK,
                va="center",
            )

        _ax.axhline(0.3, color="#4A4A4A", linewidth=0.7, linestyle="--", zorder=1)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(
            [f"LOSO\n({_d['n_folds']} folds)", "Split-half"], fontsize=6
        )
        _ax.set_xlim(-0.6, 1.75)
        _ax.set_ylabel("Matched-pair Pearson $r$" if _i % 3 == 0 else "")
        _ax.text(
            0.03,
            0.06,
            _sub,
            transform=_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=6,
            color="#4A4A4A",
        )
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)

    _fig.subplots_adjust(hspace=0.30, wspace=0.12)
    for _ext in ("pdf", "png"):
        _fig.savefig(
            OUT / f"figS04_A_matched_pair_r.{_ext}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("saved figS04_A_matched_pair_r")
    plt.close(_fig)
    return


@app.cell
def panel_B_invariants(
    BAND_COLOR,
    INVARIANTS,
    LOSO_COLOR,
    OUT,
    SUBJECTS,
    invariants,
    noise_floor,
    np,
    plt,
):
    _fig, _axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    _rng = np.random.default_rng(1)

    for _j, (_key, _label) in enumerate(INVARIANTS):
        _ax = _axes[_j]

        for _i, _sub in enumerate(SUBJECTS):
            _inv = invariants[_sub].get(_key)
            if _inv is None:
                continue
            _vals = np.asarray(list(_inv["per_fold"].values()), dtype=float)

            # 10-seed initialization range: relative-scale anchor only.
            _nf = noise_floor[_sub].get(_key)
            if isinstance(_nf, dict) and "range" in _nf:
                _lo, _hi = _nf["range"]
                _ax.add_patch(
                    plt.Rectangle(
                        (_i - 0.32, _lo),
                        0.64,
                        max(_hi - _lo, 1e-9),
                        facecolor=BAND_COLOR,
                        alpha=0.30,
                        edgecolor="none",
                        zorder=1,
                    )
                )

            _jit = _rng.uniform(-0.13, 0.13, size=_vals.size)
            _ax.scatter(
                np.full(_vals.size, _i) + _jit,
                _vals,
                s=11,
                facecolor=LOSO_COLOR,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )

        _ax.set_xticks(range(len(SUBJECTS)))
        _ax.set_xticklabels([_s.replace("sub-", "") for _s in SUBJECTS], fontsize=6)
        _ax.set_xlabel("Participant")
        _ax.set_ylabel(_label)
        _ax.set_xlim(-0.6, len(SUBJECTS) - 0.4)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)

    _fig.subplots_adjust(wspace=0.60)
    for _ext in ("pdf", "png"):
        _fig.savefig(
            OUT / f"figS04_B_structural_invariants.{_ext}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    print("saved figS04_B_structural_invariants")
    plt.close(_fig)
    return


@app.cell
def load_fc_rv(LOSO_DIR, PARCELLATION_FC, SH_DIR, SUBJECTS, json, np):
    """Within-state FC RV for matched pairs, with a mismatched-pair null.

    Two FC variants per comparison:
      raw   - the Ledoit-Wolf within-state correlation matrices as stored.
      delta - the same matrices minus the across-state mean FC of their own fit.

    The null deranges the target indices, so a "mismatched" pair is a real state
    from fit A against a different real state from fit B. Comparing matched
    against mismatched is the only way to tell whether RV carries state-specific
    information or is dominated by structure common to every state.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from utils.stats import compute_rv_coefficient

    _HMM = LOSO_DIR.parent.parent / "04_combined_hdphmm" / PARCELLATION_FC
    _F05 = LOSO_DIR.parent.parent / "05f_state_fc" / PARCELLATION_FC
    _rng = np.random.default_rng(0)

    def _rv(a, b):
        """RV for the raw (positive semi-definite) FC matrices."""
        return float(compute_rv_coefficient(np.stack([a, b]))[0, 1])

    def _cos(a, b):
        """Unclipped matrix cosine: tr(AB) / sqrt(tr(AA) tr(BB)).

        Identical to RV in form, but WITHOUT the [0, 1] clip that
        ``compute_rv_coefficient`` applies. The clip is correct for the raw FC
        matrices, which are positive semi-definite so the quantity cannot be
        negative. Mean-removed FC is NOT positive semi-definite: two states can
        deviate from the common mean in opposite directions, which is a genuine
        negative similarity. Clipping it to zero would pile those pairs onto an
        artificial floor and hide the sign.
        """
        _num = float(np.sum(a * b))
        _den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return _num / _den if _den > 0 else float("nan")

    def _derange(idx):
        _out = list(idx)
        for _ in range(500):
            _rng.shuffle(_out)
            if all(_x != _y for _x, _y in zip(idx, _out)):
                return _out
        return _out

    fc_rv = {}
    for _sub in SUBJECTS:
        _prim = np.load(_F05 / _sub / "vt0.95" / "state_empirical_corr.npy")

        # ---- LOSO: each fold against the primary model ----
        with open(LOSO_DIR / _sub / "hungarian_matching.json") as _fh:
            _lo = json.load(_fh)
        _lr = {"raw": [], "delta": []}
        _ln = {"raw": [], "delta": []}
        for _fk, _fd in _lo["per_fold"].items():
            _fc = np.load(_HMM / _sub / "loso" / f"season_{_fk}" / "state_empirical_corr.npy")
            _fs = [_m["fold_state"] for _m in _fd["matches"]]
            _ps = [_m["primary_state"] for _m in _fd["matches"]]
            _dfc = _fc - _fc[np.unique(_fs)].mean(0)
            _dpr = _prim - _prim[np.unique(_ps)].mean(0)
            _psh = _derange(_ps)
            for _a, _b, _bn in zip(_fs, _ps, _psh):
                _lr["raw"].append(_rv(_fc[_a], _prim[_b]))
                _lr["delta"].append(_cos(_dfc[_a], _dpr[_b]))
                _ln["raw"].append(_rv(_fc[_a], _prim[_bn]))
                _ln["delta"].append(_cos(_dfc[_a], _dpr[_bn]))

        # ---- Split-half: half A against half B ----
        _A = np.load(_HMM / _sub / "split_half" / "A" / "state_empirical_corr.npy")
        _B = np.load(_HMM / _sub / "split_half" / "B" / "state_empirical_corr.npy")
        with open(SH_DIR / _sub / "hungarian_matching.json") as _fh:
            _sh = json.load(_fh)
        _pairs = _sh["matching"]["pairs"]
        _as = [_p["state_A"] for _p in _pairs]
        _bs = [_p["state_B"] for _p in _pairs]
        _dA = _A - _A[np.unique(_as)].mean(0)
        _dB = _B - _B[np.unique(_bs)].mean(0)
        _bsh = _derange(_bs)
        _sr = {"raw": [], "delta": []}
        _sn = {"raw": [], "delta": []}
        for _a, _b, _bn in zip(_as, _bs, _bsh):
            _sr["raw"].append(_rv(_A[_a], _B[_b]))
            _sr["delta"].append(_cos(_dA[_a], _dB[_b]))
            _sn["raw"].append(_rv(_A[_a], _B[_bn]))
            _sn["delta"].append(_cos(_dA[_a], _dB[_bn]))

        fc_rv[_sub] = {
            ("loso", "matched"): {k: np.asarray(v) for k, v in _lr.items()},
            ("loso", "mismatched"): {k: np.asarray(v) for k, v in _ln.items()},
            ("split_half", "matched"): {k: np.asarray(v) for k, v in _sr.items()},
            ("split_half", "mismatched"): {k: np.asarray(v) for k, v in _sn.items()},
        }

    print("=== S4 audit: FC RV (mean) ===")
    print(f"{'sub':7}{'raw m':>8}{'raw mm':>8}{'del m':>8}{'del mm':>8}{'diff':>7}  (split-half)")
    for _sub in SUBJECTS:
        _d = fc_rv[_sub]
        _rm = _d[("split_half", "matched")]["raw"].mean()
        _rn = _d[("split_half", "mismatched")]["raw"].mean()
        _dm = _d[("split_half", "matched")]["delta"].mean()
        _dn = _d[("split_half", "mismatched")]["delta"].mean()
        _neg = np.mean(_d[("split_half", "matched")]["delta"] < 0)
        print(f"{_sub:7}{_rm:>8.3f}{_rn:>8.3f}{_dm:>8.3f}{_dn:>8.3f}{_dm - _dn:>7.3f}"
              f"   frac_neg_matched={_neg:.3f}")
    return fc_rv


@app.cell
def panel_C_fc_rv_raw(LOSO_COLOR, OUT, SH_COLOR, SUBJECTS, fc_rv, np, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
    _spec = [
        (0, ("loso", "matched"), LOSO_COLOR, "o"),
        (1, ("loso", "mismatched"), LOSO_COLOR, "x"),
        (2, ("split_half", "matched"), SH_COLOR, "o"),
        (3, ("split_half", "mismatched"), SH_COLOR, "x"),
    ]
    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        for _pos, _key, _col, _mk in _spec:
            _v = fc_rv[_sub][_key]["raw"]
            _ax.scatter(
                np.full(_v.size, _pos)
                + np.random.default_rng(_pos).uniform(-0.13, 0.13, _v.size),
                _v,
                s=3.0,
                color=_col,
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
            _ax.scatter(
                [_pos], [_v.mean()], marker="D", s=24, facecolor="#D62728",
                edgecolor="white", linewidth=0.5, zorder=5,
            )
        _ax.set_xticks([0, 1, 2, 3])
        _ax.set_xticklabels(["LOSO\nmatch", "LOSO\nnull", "Split\nmatch", "Split\nnull"], fontsize=5.5)
        _ax.set_ylabel("FC RV, raw" if _i % 3 == 0 else "")
        _ax.set_ylim(0.55, 1.02)
        _ax.text(0.03, 0.06, _sub, transform=_ax.transAxes, va="bottom",
                 ha="left", fontsize=6, color="#4A4A4A")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)
    _fig.subplots_adjust(hspace=0.34, wspace=0.12)
    for _ext in ("pdf", "png"):
        _fig.savefig(OUT / f"figS04_C_fc_rv_raw.{_ext}", bbox_inches="tight", pad_inches=0.02)
    print("saved figS04_C_fc_rv_raw")
    plt.close(_fig)
    return


@app.cell
def panel_D_fc_rv_delta(LOSO_COLOR, OUT, SH_COLOR, SUBJECTS, fc_rv, np, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
    _spec = [
        (0, ("loso", "matched"), LOSO_COLOR),
        (1, ("loso", "mismatched"), LOSO_COLOR),
        (2, ("split_half", "matched"), SH_COLOR),
        (3, ("split_half", "mismatched"), SH_COLOR),
    ]
    for _i, _sub in enumerate(SUBJECTS):
        _ax = _axes.flat[_i]
        for _pos, _key, _col in _spec:
            _v = fc_rv[_sub][_key]["delta"]
            _ax.scatter(
                np.full(_v.size, _pos)
                + np.random.default_rng(_pos).uniform(-0.13, 0.13, _v.size),
                _v,
                s=3.0,
                color=_col,
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
            _ax.scatter(
                [_pos], [_v.mean()], marker="D", s=24, facecolor="#D62728",
                edgecolor="white", linewidth=0.5, zorder=5,
            )
        _ax.set_xticks([0, 1, 2, 3])
        _ax.set_xticklabels(["LOSO\nmatch", "LOSO\nnull", "Split\nmatch", "Split\nnull"], fontsize=5.5)
        _ax.set_ylabel("FC cosine, mean removed" if _i % 3 == 0 else "")
        _ax.text(0.03, 0.94, _sub, transform=_ax.transAxes, va="top",
                 ha="left", fontsize=6, color="#4A4A4A")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(length=2)
    _fig.subplots_adjust(hspace=0.34, wspace=0.12)
    for _ext in ("pdf", "png"):
        _fig.savefig(OUT / f"figS04_D_fc_rv_delta.{_ext}", bbox_inches="tight", pad_inches=0.02)
    print("saved figS04_D_fc_rv_delta")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
