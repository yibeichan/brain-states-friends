"""fig_F4_within_friends.py - Figure F4 within-Friends representational depth.

Leads manuscript Figure 4 (§R4b) with the WITHIN-Friends primary result that the
old "R4b slim" figure (``08e_plots.py``, transfer only) was missing. Three
per-modality panels keep video / audio / text separate; a fourth panel condenses
the cross-stimulus transfer to a peak-depth grid.

Design:

  A video  (DINOv2-large,   24L)  ┐
  B audio  (Wav2Vec-BERT,    24L)  ├ within-Friends "peak-locator strip":
  C text   (LLaMA-3.2-3B W=1,28L)  ┘ 6 subject rows × relative depth (0→1).
  D transfer peak-depth grid (stimulus × modality), separate chart family.

Per-subject row (A–C): thin filled profile of decoding strength in
**Δ-above-confound-floor** units along relative network depth; the argmax is
marked with the subject's conventional shape and bracketed by a **peak band**
(contiguous layers with Δ-above-floor ≥ 0.90 × peak Δ-above-floor). Shared
light "mid ~0.5 / deep ~0.9" depth zones are the categorical readout - the
x-position is a *within-model* normalization, not a cross-architecture ruler.
Per-row best lag annotated; a floor-exceedance tick marks peak > timing floor.

Confound floor: dinov2 / w2v use their own ``D1_confound_baseline.json``. LLaMA
W=1 has none, so the dinov2 lag-3 floor is **reused** - valid because the floor
depends only on (subject, lag): the 6 timing regressors carry no model features.
Reuse is guarded by full-key equivalence asserts (lag, chance_level, n_eligible,
eligibility_source); any mismatch hard-stops (recompute is out of scope).

No on-figure panel labels / titles (added in assembly). Saves PDF + PNG + SVG.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import (  # noqa: E402
    MODEL_DISPLAY,
    SUBJECT_MARKERS,
    SUBJECT_NEUTRAL,
    apply_publication_style,
    modality_color,
)

apply_publication_style()

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
PARCELLATION = "atlas-4S156Parcels"
OUT_DIR = SCRATCH_DIR / "output" / "manuscript_figures" / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
D1_ROOT = SCRATCH_DIR / "output" / "08d_transformer_depth" / PARCELLATION

SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
PEAK_LAG_EXCLUDE = {0}            # lag-0 is a synchrony diagnostic, not a peak
PEAK_BAND_FRAC = 0.90             # contiguous Δ-above-floor ≥ 0.90 × peak (Res A)

# Within-Friends panels. A = video, B = audio, C = text (Res D mapping).
WITHIN_PANELS = [
    {"name": "A_video", "model": "dinov2-large", "n_layers": 24, "modality": "video"},
    {"name": "B_audio", "model": "w2v-bert-2.0", "n_layers": 24, "modality": "audio"},
    {"name": "C_text",  "model": "llama-3.2-3b", "n_layers": 28, "modality": "text"},
]

# Confound-floor donor for models lacking their own D1_confound_baseline.json.
# Only LLaMA W=1 is missing one; dinov2 lag-3 is the donor (Res E).
CONFOUND_DONOR = "dinov2-large"

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_d1_profile(sub, model):
    """Load D1_depth_profile.json → (profile dict, meta dict).

    profile: {lag_int: {layer_int: {balanced_accuracy, chance_level, p_fdr, ...}}}
    meta:    {n_eligible_states, eligibility_source, lags}
    """
    path = D1_ROOT / sub / f"friends_{model}" / "D1_depth_profile.json"
    d = json.loads(path.read_text())
    profile = {}
    for lag_key, layers in d["results"].items():
        lag = int(lag_key.split("_")[1])
        profile[lag] = {int(li): v for li, v in layers.items()}
    meta = {
        "n_eligible_states": d.get("n_eligible_states"),
        "eligibility_source": d.get("eligibility_source"),
        "lags": d.get("lags"),
    }
    return profile, meta


def best_lag_layer(profile, n_layers):
    """Argmax balanced_accuracy over lag × layer (excluding lag-0).

    Returns (best_lag, peak_layer, curve_acc, curve_chance, curve_pfdr) where
    the curves are length-n_layers arrays at best_lag (NaN for absent layers).
    """
    best = (-np.inf, None, None)
    for lag, layers in profile.items():
        if lag in PEAK_LAG_EXCLUDE:
            continue
        for li, v in layers.items():
            if v["balanced_accuracy"] > best[0]:
                best = (v["balanced_accuracy"], lag, li)
    _, best_lag, peak_layer = best
    layers = profile[best_lag]
    acc = np.full(n_layers, np.nan)
    chance = np.full(n_layers, np.nan)
    pfdr = np.full(n_layers, np.nan)
    for li, v in layers.items():
        acc[li] = v["balanced_accuracy"]
        chance[li] = v["chance_level"]
        pfdr[li] = v.get("p_fdr", np.nan)
    return best_lag, peak_layer, acc, chance, pfdr


def load_confound_floor(sub, model, best_lag, chance_at_peak, n_eligible,
                        eligibility_source):
    """Return the timing-only floor balanced accuracy for (sub, model).

    dinov2 / w2v read their own D1_confound_baseline.json. LLaMA reuses the
    dinov2 donor floor, guarded by full-key equivalence asserts (Res E): the
    floor depends only on (subject, lag) because the confound classifier uses 6
    timing regressors as features - no transformer features, no PCA - against
    the identical content-eligible labels. Mismatch hard-stops.
    """
    own = D1_ROOT / sub / f"friends_{model}" / "D1_confound_baseline.json"
    if own.exists():
        return float(json.loads(own.read_text())["balanced_accuracy"])

    donor_path = D1_ROOT / sub / f"friends_{CONFOUND_DONOR}" / "D1_confound_baseline.json"
    if not donor_path.exists():
        raise FileNotFoundError(
            f"{model} has no confound baseline and donor {CONFOUND_DONOR} "
            f"baseline is missing for {sub}: {donor_path}"
        )
    donor = json.loads(donor_path.read_text())
    _, donor_meta = load_d1_profile(sub, CONFOUND_DONOR)

    # Full-key equivalence guard - reuse ONLY if every key matches (Res E).
    if donor["best_lag"] != best_lag:
        raise AssertionError(
            f"{sub} {model}: confound reuse blocked - donor lag "
            f"{donor['best_lag']} != {model} peak lag {best_lag}."
        )
    if not np.isclose(donor["chance_level"], chance_at_peak, atol=1e-4):
        raise AssertionError(
            f"{sub} {model}: confound reuse blocked - donor chance "
            f"{donor['chance_level']} != {model} chance {chance_at_peak} "
            "(label sets differ)."
        )
    if donor_meta["n_eligible_states"] != n_eligible:
        raise AssertionError(
            f"{sub} {model}: confound reuse blocked - donor n_eligible "
            f"{donor_meta['n_eligible_states']} != {model} {n_eligible}."
        )
    if donor_meta["eligibility_source"] != eligibility_source:
        raise AssertionError(
            f"{sub} {model}: confound reuse blocked - eligibility source "
            f"{donor_meta['eligibility_source']} != {eligibility_source}."
        )
    return float(donor["balanced_accuracy"])


def peak_band(delta_above_floor):
    """Contiguous layer band around the argmax with Δ ≥ 0.90 × peak Δ (Res A).

    Returns (lo, hi, peak_layer) layer indices. Distinguishes a sharp spike
    (narrow band) from a broad plateau (wide band) honestly.
    """
    finite = np.where(np.isfinite(delta_above_floor))[0]
    peak_layer = int(finite[np.nanargmax(delta_above_floor[finite])])
    thresh = PEAK_BAND_FRAC * delta_above_floor[peak_layer]
    lo = hi = peak_layer
    while lo - 1 >= 0 and np.isfinite(delta_above_floor[lo - 1]) \
            and delta_above_floor[lo - 1] >= thresh:
        lo -= 1
    while hi + 1 < len(delta_above_floor) and np.isfinite(delta_above_floor[hi + 1]) \
            and delta_above_floor[hi + 1] >= thresh:
        hi += 1
    return lo, hi, peak_layer


# ── Per-subject row collection ─────────────────────────────────────────────────

def collect_within(panel):
    """Load every subject's within-Friends depth row for one modality panel."""
    rows = []
    for sub in SUBJECTS:
        profile, meta = load_d1_profile(sub, panel["model"])
        best_lag, _, acc, chance, pfdr = best_lag_layer(profile, panel["n_layers"])
        floor = load_confound_floor(
            sub, panel["model"], best_lag, chance[np.isfinite(chance)][0],
            meta["n_eligible_states"], meta["eligibility_source"],
        )
        delta = acc - floor                       # Δ-above-floor units (Res F)
        lo, hi, peak_layer = peak_band(delta)
        denom = panel["n_layers"] - 1
        rows.append({
            "sub": sub,
            "best_lag": best_lag,
            "delta": delta,
            "acc": acc,                            # absolute balanced accuracy curve
            "chance": float(chance[np.isfinite(chance)][0]),
            "floor": floor,                        # scalar confound floor (Res F)
            "rel_depth": np.arange(panel["n_layers"]) / denom,
            "peak_layer": peak_layer,
            "peak_rel": peak_layer / denom,
            "band_rel": (lo / denom, hi / denom),
            "peak_delta": float(delta[peak_layer]),
            "exceeds_floor": bool(delta[peak_layer] > 0),
            "n_layers": panel["n_layers"],
        })
    return rows


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_within_panel(panel, rows, ylim):
    """One within-Friends modality panel: all 6 subjects' absolute decoding-
    accuracy curves overlaid on a shared y-axis (balanced accuracy vs relative
    depth), one colour per subject, with each subject's peak-depth marker. The
    shared ``ylim`` (passed from main) is identical across A/B/C so absolute
    magnitudes are directly comparable between modalities."""
    color = modality_color(panel["modality"])
    fig, ax = plt.subplots(figsize=(2.0, 1.9))
    for r in rows:
        xs, acc, pk = r["rel_depth"], r["acc"], r["peak_layer"]
        # One colour per modality; subjects distinguished by marker at the peak.
        # Solid markers (filled modality colour + white edge) to match Fig 3.
        ax.plot(xs, acc, color=color, linewidth=1.0, alpha=0.6, zorder=2)
        ax.scatter([xs[pk]], [acc[pk]], marker=SUBJECT_MARKERS[r["sub"]], s=28,
                   color=color, edgecolor="white", linewidth=0.6, zorder=4)
    # Cohort-mean chance reference (per-subject chance differs only trivially
    # via n_classes; the spread is < marker size at this scale).
    ch = float(np.mean([r["chance"] for r in rows]))
    ax.axhline(ch, color="0.6", linewidth=0.7, linestyle=":", zorder=1)
    ax.text(0.985, ch, "chance", fontsize=5.5, color="0.5", va="bottom", ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("relative depth", fontsize=6.5)
    ax.set_ylabel("balanced accuracy", fontsize=6.5)
    ax.set_title(MODEL_DISPLAY[panel["model"]], fontsize=7)
    ax.tick_params(labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _save(fig, f"fig4_{panel['name']}_within")
    print(f"  {panel['name']} ({MODEL_DISPLAY[panel['model']]}): "
          f"peak rel-depth {min(r['peak_rel'] for r in rows):.2f}–"
          f"{max(r['peak_rel'] for r in rows):.2f}, "
          f"floor-exceed {sum(r['exceeds_floor'] for r in rows)}/{len(rows)}")


def make_subject_legend():
    """Standalone 6-subject legend (marker → subject) for the within-Friends
    panels A/B/C. Markers are neutral grey because the lines are coloured by
    modality (one colour per panel), not by subject."""
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=SUBJECT_NEUTRAL, linewidth=0,
                             marker=SUBJECT_MARKERS[s],
                             markerfacecolor=SUBJECT_NEUTRAL,
                             markeredgecolor="white", markersize=6, label=s)
               for s in SUBJECTS]
    fig, ax = plt.subplots(figsize=(1.0, 1.5))
    ax.legend(handles=handles, frameon=False, loc="center", fontsize=7,
              handlelength=1.0)
    ax.axis("off")
    _save(fig, "fig4_ABC_subject_legend")
    print("  ABC_subject_legend (separate, 6 subjects, marker-only) rendered.")


def _save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}",
                    bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


# ── Panel D: cross-stimulus transfer peak-depth grid ───────────────────────────

TRANSFER_ROOT = (SCRATCH_DIR / "output"
                 / "08e_transformer_cross_stim_aggregate" / PARCELLATION)

# Stimulus rows (transfer targets) and modality columns (order preserved from A–C).
STIM_ROWS = [
    ("movie10", "Movie10"),
    ("harrypotter", "Harry Potter"),
    ("petitprince_fr", "Petit Prince FR"),
    ("petitprince_en", "Petit Prince EN"),
]
MOD_COLS = [
    ("dinov2-large", "Video", 24),
    ("w2v-bert-2.0", "Audio", 24),
    ("llama-3.2-3b", "Text", 28),
]
SIG_ALPHA = 0.05


def load_transfer_cell(stim, model, n_layers):
    """Per (stim, model): per-subject transfer peak relative-depth + FDR-sig.

    BH-FDR is recomputed across each subject's layers here for an explicit,
    self-contained correction (verified numerically identical to the ``p_fdr``
    already stored in the transfer JSON, which 08e computes the same way). Peak
    = argmax of balanced_accuracy − chance_level. Returns dict or None if no data.
    """
    from statsmodels.stats.multitest import multipletests
    paths = sorted(TRANSFER_ROOT.glob(
        f"sub-*/{stim}_{model}/D3a_transfer_{stim}_{model}.json"))
    if not paths:
        return None
    rel_depths, peak_sig, peak_accs = [], [], []
    for p in paths:
        per_layer = json.loads(p.read_text()).get("per_layer", {})
        acc = np.full(n_layers, np.nan)
        pp = np.full(n_layers, np.nan)
        for L in range(n_layers):
            e = per_layer.get(str(L))
            if e is not None:
                acc[L] = e["balanced_accuracy"] - e["chance_level"]
                pp[L] = e["p_perm"]
        fin = np.isfinite(acc)
        if not fin.any():
            continue
        pfdr = np.full(n_layers, np.nan)
        pfdr[fin] = multipletests(pp[fin], method="fdr_bh")[1]
        pk = int(np.nanargmax(acc))
        rel_depths.append(pk / (n_layers - 1))
        peak_sig.append(bool(pfdr[pk] < SIG_ALPHA))
        peak_accs.append(float(acc[pk]))            # peak Δ-above-chance
    if not rel_depths:
        return None
    rel_depths = np.array(rel_depths)
    n_sig = int(np.sum(peak_sig))
    return {
        "n": len(rel_depths),
        "median": float(np.median(rel_depths)),          # peak relative depth
        "iqr": (float(np.percentile(rel_depths, 25)),
                float(np.percentile(rel_depths, 75))),
        "acc_median": float(np.median(peak_accs)),       # peak bal_acc − chance
        "n_sig": n_sig,
        "majority_sig": n_sig >= np.ceil(len(rel_depths) / 2),
    }


def render_panel_d():
    """Transfer peak-depth grid: stimulus × modality, color = median peak depth.

    Three cell states (Res B amended): not-testable (hatched grey), tested but
    not majority-significant (muted, no depth color - a noisy argmax is not shown
    as replicated depth), tested + majority FDR-sig (depth-colored + outline).
    """
    cmap = plt.get_cmap("cividis_r")            # deep (high rel-depth) → dark
    norm = plt.Normalize(0.0, 1.0)
    nr, nc = len(STIM_ROWS), len(MOD_COLS)
    fig, ax = plt.subplots(figsize=(3.9, 3.4))

    for ri, (stim, _) in enumerate(STIM_ROWS):
        y = nr - 1 - ri                          # Movie10 at top
        for ci, (model, _, nl) in enumerate(MOD_COLS):
            cell = load_transfer_cell(stim, model, nl)
            if cell is None:                     # not testable
                ax.add_patch(mpatches.Rectangle(
                    (ci, y), 1, 1, facecolor="0.92", edgecolor="white",
                    hatch="////", linewidth=1.0))
                ax.text(ci + 0.5, y + 0.5, "n/t", ha="center", va="center",
                        fontsize=7, color="0.5")
                continue
            if not cell["majority_sig"]:         # tested but n.s. - muted
                ax.add_patch(mpatches.Rectangle(
                    (ci, y), 1, 1, facecolor="0.96", edgecolor="white",
                    linewidth=1.0))
                ax.text(ci + 0.5, y + 0.58, f"{cell['acc_median']:+.3f}",
                        ha="center", va="center", fontsize=8, color="0.55")
                ax.text(ci + 0.5, y + 0.28,
                        f"{cell['n_sig']}/{cell['n']} ns", ha="center",
                        va="center", fontsize=6, color="0.6")
                continue
            # tested + majority FDR-significant: COLOR = peak depth, NUMBER = peak
            # decoding accuracy (balanced accuracy − chance).
            rgba = cmap(norm(cell["median"]))
            ax.add_patch(mpatches.Rectangle(
                (ci, y), 1, 1, facecolor=rgba, edgecolor="black", linewidth=1.2))
            txt = "white" if sum(rgba[:3]) < 1.5 else "black"
            ax.text(ci + 0.5, y + 0.58, f"{cell['acc_median']:+.3f}", ha="center",
                    va="center", fontsize=9, color=txt, fontweight="bold")
            ax.text(ci + 0.5, y + 0.28, f"{cell['n_sig']}/{cell['n']}",
                    ha="center", va="center", fontsize=6, color=txt)

    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([c + 0.5 for c in range(nc)])
    ax.set_xticklabels([lab for _, lab, _ in MOD_COLS])
    ax.set_yticks([nr - 1 - r + 0.5 for r in range(nr)])
    ax.set_yticklabels([lab for _, lab in STIM_ROWS])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    _save(fig, "fig4_D_transfer_depth_grid")

    # Colorbar emitted as a SEPARATE file for manual assembly - kept off the grid
    # so the grid's width is unencumbered (and matches the per-film panels).
    cfig, cax = plt.subplots(figsize=(0.45, 2.2))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = cfig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1.0])
    cbar.set_label("transfer peak relative depth", fontsize=8)
    cbar.ax.set_yticklabels(["0\nshallow", "0.5\nmid", "1.0\ndeep"], fontsize=6.5)
    # Ticks + label on the LEFT (flipped) so the bar sits to the right of them.
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    _save(cfig, "fig4_D_colorbar")
    print("  D_transfer_depth_grid + D_colorbar (separate) rendered.")


# ── Panel E: Movie10 video per-film depth gradient (secondary companion) ────────
# CB-safe Paul Tol Light film palette, identical to 08e_plots.py's supplementary
# per-film strip so the four films are colour-consistent across main + supp.
# Ordered by social-narrative content (Wolf, Figures = social; Bourne, Life = not).
FILM_SERIES = [
    ("wolf",    "Wolf of Wall St.", "#77AADD"),
    ("figures", "Hidden Figures",   "#EE8866"),
    ("bourne",  "Bourne",           "#44BB99"),
    ("life",    "Life",             "#FFAABB"),
]


# Three small Movie10 per-film line plots (one per modality), to append onto the
# Panel D heatmap in assembly. Letters E/F/G. Video data exists now; audio/text
# render once the 08e --per_subset jobs (w2v-bert, llama) land.
PERFILM_PANELS = [
    ("video", "dinov2-large", 24, "E"),
    ("audio", "w2v-bert-2.0", 24, "F"),
    ("text",  "llama-3.2-3b", 28, "G"),
]


def collect_perfilm(model, n_layers):
    """Per-film Movie10 transfer curves (mean±SEM Δ-above-chance) for one model.

    Returns a list of per-film series dicts, or None if the per-film
    (`D3a_per_subset`) JSONs are not on disk yet for this model.
    """
    paths = sorted(TRANSFER_ROOT.glob(
        f"sub-*/movie10_{model}/D3a_per_subset_movie10_{model}.json"))
    if not paths:
        return None
    series = []
    for film, label, color in FILM_SERIES:
        rows = []
        for p in paths:
            d = json.loads(p.read_text())
            chance = d.get("chance_level_full")
            fl = d.get("per_subset", {}).get(film, {})
            if chance is None or not fl:
                continue
            curve = np.full(n_layers, np.nan)
            for L in range(n_layers):
                e = fl.get(str(L))
                if e is not None:
                    curve[L] = e["balanced_accuracy"] - chance
            rows.append(curve)
        if not rows:
            continue
        arr = np.vstack(rows)
        mean = np.nanmean(arr, axis=0)
        n_per = np.sum(~np.isnan(arr), axis=0).clip(min=1)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_per)
        # Cohort-median of per-subject peak layers (matches §R4b.5 prose), NOT
        # argmax-of-the-mean-curve: on a noisy/plateaued film the mean-curve
        # argmax can land deep (e.g. Bourne L21) while most subjects peak mid
        # (median L15.5). The median is the prose-consistent gradient estimator.
        per_subj_peak = [int(np.nanargmax(r)) for r in arr if np.isfinite(r).any()]
        median_peak_rel = float(np.median(per_subj_peak)) / (n_layers - 1)
        series.append({"label": label, "color": color, "mean": mean,
                       "sem": sem, "n": arr.shape[0], "n_layers": n_layers,
                       "median_peak_rel": median_peak_rel})
    return series or None


def render_perfilm(modality, model, n_layers, letter, series, ylim):
    """One small Movie10 per-film depth-profile line plot for a modality."""
    xs = np.arange(n_layers) / (n_layers - 1)
    fig, ax = plt.subplots(figsize=(0.9, 1.25))     # ~one heatmap-column wide (sparkline)
    for s in series:
        ax.fill_between(xs, s["mean"] - s["sem"], s["mean"] + s["sem"],
                        color=s["color"], alpha=0.18, linewidth=0)
        ax.plot(xs, s["mean"], color=s["color"], linewidth=1.5,
                label=f"{s['label']} (n={s['n']})")
    if ylim:
        ax.set_ylim(*ylim)
    # Cohort-median peak per film as a star on the x-axis (NOT a dot on the
    # mean curve) - marks where the cohort peaks without implying the mean
    # curve's value there. Star (not "^") so it never reads as a subject glyph:
    # SUBJECT_MARKERS uses o/s/^/D/v/P. Matches §R4b.5's cohort-median framing.
    y0 = ax.get_ylim()[0]
    for s in series:
        ax.plot([s["median_peak_rel"]], [y0], marker="*", color=s["color"],
                markersize=9, clip_on=False, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle=":")   # chance reference
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([])           # sparkline: shared y-scale → caption; x-label → caption
    ax.tick_params(labelsize=5.5)
    # Film legend emitted separately (make_perfilm_legend) for manual assembly.
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    _save(fig, f"fig4_{letter}_{modality}_perfilm")
    print(f"  {letter}_{modality}_perfilm rendered ({MODEL_DISPLAY[model]}, "
          f"{len(series)} films)")


def make_perfilm_legend():
    """Standalone 4-film legend for the per-film panels (E/F/G), emitted as a
    separate file so it can be placed once during manual assembly."""
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=c, linewidth=1.6, label=lab)
               for _, lab, c in FILM_SERIES]
    fig, ax = plt.subplots(figsize=(1.7, 0.9))
    ax.legend(handles=handles, frameon=False, loc="center", fontsize=7,
              handlelength=1.6)
    ax.axis("off")
    _save(fig, "fig4_EFG_legend")
    print("  EFG_legend (separate, 4 films) rendered.")


def make_perfilm_yaxis(ylim):
    """Standalone shared y-axis for the per-film sparklines (E/F/G), emitted as
    a separate file for manual assembly (the sparklines themselves drop the
    axis to reach one-column width). Same ylim as the sparklines."""
    fig, ax = plt.subplots(figsize=(0.6, 1.25))
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("bal. acc − chance", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    _save(fig, "fig4_EFG_yaxis")
    print("  EFG_yaxis (separate shared y-axis) rendered.")


def render_perfilm_trio():
    """Render the three small Movie10 per-film line plots, shared y-axis.

    Skips any modality whose 08e --per_subset data is not yet on disk (audio /
    text land after jobs 15640110 / 15640111). Shared y across whatever is
    available so the modalities are directly comparable when appended to D.
    """
    collected = {}
    for modality, model, n_layers, letter in PERFILM_PANELS:
        s = collect_perfilm(model, n_layers)
        if s is None:
            print(f"  [pending] no per-film data yet for {modality} ({model})")
            continue
        collected[modality] = (model, n_layers, letter, s)
    if not collected:
        return
    lo = min(np.nanmin(s["mean"] - s["sem"])
             for _, _, _, ss in collected.values() for s in ss)
    hi = max(np.nanmax(s["mean"] + s["sem"])
             for _, _, _, ss in collected.values() for s in ss)
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    for modality, (model, n_layers, letter, s) in collected.items():
        render_perfilm(modality, model, n_layers, letter, s, ylim)
    make_perfilm_legend()
    make_perfilm_yaxis(ylim)


# ── Companion supplementary: main / neg-control / confound triple (Res C) ──────

SUPP_DIR = SCRATCH_DIR / "output" / "manuscript_figures" / "figS_R4b_negcontrol"


def _neg_cell_nearest(neg_results, best_lag, peak_layer):
    """Neg-control cell at (best_lag, peak_layer), falling back to the nearest
    available lag / layer. Some neg-control grids are sparse (e.g. sub-06 LLaMA
    is missing lag-3); the class-count artifact this panel illustrates holds
    across lags, so a nearest-cell substitute is acceptable for the defusal.
    """
    lag_keys = {int(k.split("_")[1]): k for k in neg_results}
    lag = best_lag if best_lag in lag_keys else min(lag_keys, key=lambda x: abs(x - best_lag))
    layers = neg_results[lag_keys[lag]]
    if str(peak_layer) in layers:
        return layers[str(peak_layer)]
    avail = sorted(int(li) for li in layers)
    nearest = min(avail, key=lambda x: abs(x - peak_layer))
    return layers[str(nearest)]


def load_nes_triple(sub, model, n_layers):
    """NES of (D1_main, neg-control, confound) at the D1_main peak cell.

    Returns dict with the three NES values + the n_classes asymmetry that drives
    the neg-control's apparent advantage (Res C). Confound NES is the timing
    floor; for LLaMA it is reused from the dinov2 donor (already validated by
    load_confound_floor's equivalence guard, called in collect_within).
    """
    profile, meta = load_d1_profile(sub, model)
    best_lag, peak_layer, _, chance, _ = best_lag_layer(profile, n_layers)
    main_cell = profile[best_lag][peak_layer]

    neg_path = D1_ROOT / sub / f"friends_{model}" / "D1_neg_control_run_onset_anchored.json"
    neg = json.loads(neg_path.read_text())
    neg_cell = _neg_cell_nearest(neg["results"], best_lag, peak_layer)

    # Confound floor: own file, else dinov2 donor. The donor reuse is already
    # validated by load_confound_floor's full-key equivalence guard, which
    # main() runs (via collect_within) before this function - do not reorder
    # main() so this fallback executes unguarded.
    conf_path = D1_ROOT / sub / f"friends_{model}" / "D1_confound_baseline.json"
    if not conf_path.exists():
        load_confound_floor(sub, model, best_lag,
                            chance[np.isfinite(chance)][0],
                            meta["n_eligible_states"], meta["eligibility_source"])
        conf_path = D1_ROOT / sub / f"friends_{CONFOUND_DONOR}" / "D1_confound_baseline.json"
    conf = json.loads(conf_path.read_text())

    return {
        "nes_main": main_cell["normalized_effect_size"],
        "nes_neg": neg_cell["normalized_effect_size"],
        "nes_conf": conf["normalized_effect_size"],
        "n_main": main_cell["n_classes"],
        "n_neg": neg_cell["n_classes"],
    }


def render_supp_negcontrol():
    """1×3 modality small-multiples: D1_main vs neg-control vs confound floor.

    Defuses the neg-control: it out-decodes content states (NES_neg > NES_main)
    purely from a 2–8 vs 16–31 class-count asymmetry, while the apples-to-apples
    timing floor (NES_conf) is cleared by NES_main in every cell. Same chart for
    all 3 modalities (supplementary is exempt from the variety rule).
    """
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    for ax, panel in zip(axes, WITHIN_PANELS):
        color = modality_color(panel["modality"])
        xs = np.arange(len(SUBJECTS))
        nmain, nneg, nrange = [], [], []
        for s in SUBJECTS:
            t = load_nes_triple(s, panel["model"], panel["n_layers"])
            ax.plot([xs[SUBJECTS.index(s)]] * 2, [t["nes_conf"], t["nes_main"]],
                    color="0.8", linewidth=1.0, zorder=1)
            ax.scatter(xs[SUBJECTS.index(s)], t["nes_main"], marker="o", s=42,
                       color=color, zorder=3, label="D1 main" if s == "sub-01" else None)
            ax.scatter(xs[SUBJECTS.index(s)], t["nes_neg"], marker="^", s=42,
                       facecolor="white", edgecolor="0.45", linewidth=1.2, zorder=3,
                       label="neg-control" if s == "sub-01" else None)
            ax.scatter(xs[SUBJECTS.index(s)], t["nes_conf"], marker="_", s=90,
                       color="0.5", zorder=2,
                       label="timing floor" if s == "sub-01" else None)
            nmain.append(t["n_main"])
            nneg.append(t["n_neg"])
        ax.set_title(f"{MODEL_DISPLAY[panel['model']]}\n"
                     f"n_classes: main {min(nmain)}–{max(nmain)}, "
                     f"neg {min(nneg)}–{max(nneg)}", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace("sub-", "") for s in SUBJECTS], fontsize=7)
        ax.set_xlabel("subject", fontsize=8)
        ax.axhline(0, color="0.7", linewidth=0.6, linestyle=":")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("NES at D1 peak cell", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper right", frameon=False)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(SUPP_DIR / f"figS_R4b_negcontrol_triple.{ext}",
                    bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  figS_R4b_negcontrol_triple → {SUPP_DIR}")


def main():
    print(f"=== Fig F4 within-Friends → {OUT_DIR} ===")
    all_rows = {p["name"]: collect_within(p) for p in WITHIN_PANELS}
    # Shared y-axis (absolute balanced accuracy) across A/B/C so modalities are
    # directly comparable in magnitude.
    all_acc = np.concatenate([r["acc"][np.isfinite(r["acc"])]
                              for rows in all_rows.values() for r in rows])
    lo, hi = float(all_acc.min()), float(all_acc.max())
    pad = 0.05 * (hi - lo)
    within_ylim = (lo - pad, hi + pad)
    for panel in WITHIN_PANELS:
        render_within_panel(panel, all_rows[panel["name"]], within_ylim)
    make_subject_legend()
    render_panel_d()
    render_perfilm_trio()
    render_supp_negcontrol()
    print("done (Figure 4: A/B/C within-Friends + D transfer grid + E/F/G per-film + supp neg-control).")


if __name__ == "__main__":
    main()
