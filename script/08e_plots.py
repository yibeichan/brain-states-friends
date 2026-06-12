"""08e_plots.py - Figure F3 cross-stimulus depth correspondence (R4b slim).

Companion figure script for ``08e_transformer_cross_stim_aggregate.py``,
analogous to ``08d_plots.py`` for stage 08d.

Three panels, each rendered as small multiples (one horizontal strip per
stim/film; shared x and shared y axis):
  A. Audio (Wav2Vec-BERT 2.0, 24 layers): Movie10, PP-FR, PP-EN  → 3 strips
  B. Text  (LLaMA-3.2-3B,   28 layers): Movie10, HP, PP-FR, PP-EN → 4 strips
  C. Video (DINOv2-large,   24 layers): Movie10 by film
     (wolf, figures, bourne, life) → 4 strips

Why small multiples instead of overlaid lines:
  With 4 stim curves whose Δ values overlap in the same range (Panel B), no
  color choice resolves the overlap - that is a data problem, not a palette
  problem. Stripping each curve onto its own row removes the discrimination
  problem entirely, keeps the SEM uncertainty band, and lets every panel
  share one global y-axis so peak magnitudes remain comparable across A/B/C.

Palettes (CB-safe, used for strip identity not overlap resolution):
  STIM: Paul Tol Vibrant 4-class - distinct from Okabe–Ito network,
        ColorBrewer Set1 taxonomy, Tailwind category, viridis recurrence
        palettes already used elsewhere in this manuscript.
  FILM: Paul Tol Light 4-class - CB-safe pastel companion to Tol Vibrant.

Each panel saves PDF + PNG + SVG. No on-panel legends (each strip self-labels).
Standalone legend files are still emitted for caption / sidebar use:
  - fig3_stim_legend.{pdf,png,svg}: 4 stim colours
  - fig3_film_legend.{pdf,png,svg}: 4 film colours

Source data:
  A, B: 08e .../D3a_transfer_{stim}_{model}.json (pooled, with permutation)
  C:    08e .../D3a_per_subset_movie10_dinov2-large.json (production, with
        permutation). Falls back to .../manuscript_figures/fig3/perfilm/
        sub-*_dinov2-large_movie10_perfilm.json (provisional, point estimates
        only, no permutation) when production JSONs are absent - used for
        the poster build while the production --per_subset run is in flight.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import apply_publication_style

apply_publication_style()

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
PARCELLATION = "atlas-4S156Parcels"
OUT_DIR = SCRATCH_DIR / "output" / "manuscript_figures" / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
D3A_ROOT = SCRATCH_DIR / "output" / "08e_transformer_cross_stim_aggregate" / PARCELLATION
PROVISIONAL_PERFILM_ROOT = OUT_DIR / "perfilm"

# Paul Tol Vibrant 4-class (https://personal.sron.nl/~pault/), CB-safe.
# Distinct from project palettes: Okabe–Ito (networks), ColorBrewer Set1
# (taxonomy), Tailwind (state categories), viridis (recurrence).
STIM_COLORS = {
    "movie10":        "#0077BB",  # vibrant blue
    "harrypotter":    "#EE7733",  # vibrant orange
    "petitprince_fr": "#009988",  # teal
    "petitprince_en": "#CC3311",  # vermilion
}

STIM_LABELS = {
    "movie10":        "Movie10",
    "harrypotter":    "Harry Potter",
    "petitprince_fr": "Petit Prince (FR)",
    "petitprince_en": "Petit Prince (EN)",
}

# Paul Tol Light 4-class, the CB-safe pastel companion to Tol Vibrant.
# One colour per Movie10 film inside Panel C; all solid lines.
FILM_COLORS = [
    ("wolf",    "Wolf",    "#77AADD"),  # light blue
    ("figures", "Figures", "#EE8866"),  # light orange
    ("bourne",  "Bourne",  "#44BB99"),  # light teal-green
    ("life",    "Life",    "#FFAABB"),  # light pink
]

PANELS = [
    {
        "name":     "A_audio",
        "kind":     "pooled",
        "model":    "w2v-bert-2.0",
        "n_layers": 24,
        "stims":    ["movie10", "petitprince_fr", "petitprince_en"],
        "xlabel":   "Wav2Vec-BERT 2.0 layer",
    },
    {
        "name":     "B_text",
        "kind":     "pooled",
        "model":    "llama-3.2-3b",
        "n_layers": 28,
        "stims":    ["movie10", "harrypotter", "petitprince_fr", "petitprince_en"],
        "xlabel":   "LLaMA-3.2-3B layer",
    },
    {
        "name":     "C_video",
        "kind":     "per_subset",
        "model":    "dinov2-large",
        "n_layers": 24,
        "stim":     "movie10",
        "xlabel":   "DINOv2-large layer",
    },
]


def load_pooled_curves(model, stim, n_layers):
    """Return (n_subj, n_layers) of (balanced_acc − chance) for pooled D3a."""
    rows, sub_ids = [], []
    for p in sorted(D3A_ROOT.glob(f"sub-*/{stim}_{model}/D3a_transfer_{stim}_{model}.json")):
        sub = p.parent.parent.name
        d = json.loads(p.read_text())
        per_layer = d.get("per_layer", {})
        curve = np.full(n_layers, np.nan)
        for L in range(n_layers):
            entry = per_layer.get(str(L))
            if entry is None:
                continue
            curve[L] = float(entry["balanced_accuracy"]) - float(entry["chance_level"])
        rows.append(curve)
        sub_ids.append(sub)
    if not rows:
        return None, []
    return np.vstack(rows), sub_ids


def load_per_subset_curves(model, stim, n_layers, film_key):
    """Return (n_subj, n_layers) of (balanced_acc − chance_level_full) per film.

    Prefers the production D3a_per_subset JSON (which has permutation p-values);
    falls back to the provisional perfilm JSON (point estimates only, no
    permutation) if production output isn't on disk yet. The fallback covers
    the poster build window while SLURM 14648842 is still in flight.
    """
    rows, sub_ids, source = [], [], None
    glob = f"sub-*/{stim}_{model}/D3a_per_subset_{stim}_{model}.json"
    prod_paths = sorted(D3A_ROOT.glob(glob))
    if prod_paths:
        source = "production"
        for p in prod_paths:
            sub = p.parent.parent.name
            d = json.loads(p.read_text())
            per_subset = d.get("per_subset", {})
            film_layers = per_subset.get(film_key, {})
            chance = d.get("chance_level_full")
            if chance is None or not film_layers:
                continue
            curve = np.full(n_layers, np.nan)
            for L in range(n_layers):
                entry = film_layers.get(str(L))
                if entry is None:
                    continue
                curve[L] = float(entry["balanced_accuracy"]) - float(chance)
            rows.append(curve)
            sub_ids.append(sub)
    else:
        prov_paths = sorted(PROVISIONAL_PERFILM_ROOT.glob(
            f"sub-*_{model}_{stim}_perfilm.json"
        ))
        if prov_paths:
            source = "provisional"
        for p in prov_paths:
            sub = p.name.split("_")[0]
            d = json.loads(p.read_text())
            chance = d.get("chance_level")
            per_layer = d.get("per_layer", {})
            if chance is None or not per_layer:
                continue
            curve = np.full(n_layers, np.nan)
            for L in range(n_layers):
                layer_entry = per_layer.get(str(L), {})
                film_entry = layer_entry.get(film_key)
                if film_entry is None:
                    continue
                curve[L] = float(film_entry["balanced_accuracy"]) - float(chance)
            rows.append(curve)
            sub_ids.append(sub)
    if not rows:
        return None, [], None
    return np.vstack(rows), sub_ids, source


def _draw_band(ax, xs, mean, sem, color, linewidth=1.6):
    ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.20, linewidth=0)
    ax.plot(xs, mean, color=color, linewidth=linewidth, linestyle="-")


def _mean_sem(arr):
    """Across-subject mean and SEM (NaN-safe)."""
    mean = np.nanmean(arr, axis=0)
    n_per_layer = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_per_layer)
    return mean, sem


def collect_panel_curves(panel):
    """Load every series for *panel* once and return a list of dicts.

    Each dict carries the across-subject mean ± sem so that the same data can
    be (a) inspected to compute the shared y-axis range and (b) handed back to
    the renderer without re-reading from disk.
    """
    series = []
    if panel["kind"] == "pooled":
        for stim_key in panel["stims"]:
            arr, subs = load_pooled_curves(panel["model"], stim_key, panel["n_layers"])
            if arr is None:
                print(f"  [skip] no data for {panel['model']} × {stim_key}")
                continue
            mean, sem = _mean_sem(arr)
            series.append({
                "label":    STIM_LABELS[stim_key],
                "color":    STIM_COLORS[stim_key],
                "mean":     mean,
                "sem":      sem,
                "n":        arr.shape[0],
                "tag":      stim_key,
                "source":   "production",
            })
    elif panel["kind"] == "per_subset":
        for film_key, label, color in FILM_COLORS:
            arr, subs, source = load_per_subset_curves(
                panel["model"], panel["stim"], panel["n_layers"], film_key,
            )
            if arr is None:
                print(f"  [skip] no per-subset data for {panel['stim']} × "
                      f"{panel['model']} × {film_key}")
                continue
            mean, sem = _mean_sem(arr)
            series.append({
                "label":  label,
                "color":  color,
                "mean":   mean,
                "sem":    sem,
                "n":      arr.shape[0],
                "tag":    film_key,
                "source": source,
            })
    else:
        raise ValueError(f"unknown panel kind: {panel['kind']}")
    return series


def panel_ylim_bounds(series):
    """Return (lo, hi) over mean±sem for one panel's series list."""
    if not series:
        return None
    lo = min(np.nanmin(s["mean"] - s["sem"]) for s in series)
    hi = max(np.nanmax(s["mean"] + s["sem"]) for s in series)
    return float(lo), float(hi)


def shared_ylim(panel_series):
    """Compute a single (ymin, ymax) covering all panels, with headroom."""
    bounds = [panel_ylim_bounds(s) for s in panel_series if s]
    bounds = [b for b in bounds if b is not None]
    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    span = hi - lo
    pad = max(0.002, span * 0.08)
    # Always include 0 in view, since the dotted reference line is at 0.
    return min(lo - pad, 0.0), hi + pad


def render_panel(panel, series, ylim):
    n = len(series)
    strip_h = 0.55
    fig_h = strip_h * n + 0.75  # extra for x-axis label and padding
    fig, axes = plt.subplots(n, 1, figsize=(4.0, fig_h),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    xs = np.arange(panel["n_layers"])
    for ax, s in zip(axes, series):
        _draw_band(ax, xs, s["mean"], s["sem"], s["color"])
        ax.axhline(0, color="0.7", linewidth=0.6, linestyle=":")
        ax.text(0.015, 0.92, s["label"], transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5, color=s["color"],
                fontweight="bold")
        peak = int(np.nanargmax(s["mean"]))
        src_tag = f"  [{s['source']}]" if panel["kind"] == "per_subset" else ""
        print(f"  {panel['name']:<8} {panel['model']:<14} "
              f"{panel.get('stim', s['tag']):<10} {s['tag']:<16} "
              f"n={s['n']}  peak L={peak}/{panel['n_layers']-1}  "
              f"peak Δ={s['mean'][peak]:.4f}{src_tag}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(ylim)
    axes[-1].set_xlabel(panel["xlabel"])
    fig.supylabel("Balanced accuracy − chance", fontsize=9, x=0.01)
    fig.subplots_adjust(hspace=0.18, left=0.16, right=0.99,
                        top=0.985, bottom=0.14)
    out_pdf = OUT_DIR / f"fig3_{panel['name']}_depth.pdf"
    out_png = OUT_DIR / f"fig3_{panel['name']}_depth.png"
    out_svg = OUT_DIR / f"fig3_{panel['name']}_depth.svg"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> wrote {out_pdf.name} + {out_png.name}")
    if panel["kind"] == "per_subset" and any(s["source"] == "provisional"
                                             for s in series):
        print(f"  [NOTE] Panel {panel['name']} rendered from PROVISIONAL perfilm "
              f"(point estimates only, no permutation). Re-run after the "
              f"production --per_subset SLURM job finishes for the "
              f"manuscript-grade version.")


def make_stim_legend():
    """4-color stimulus legend (used across Panels A/B/C)."""
    fig, ax = plt.subplots(figsize=(2.2, 1.4))
    handles = []
    for stim_key in ["movie10", "harrypotter", "petitprince_fr", "petitprince_en"]:
        line, = ax.plot([], [], color=STIM_COLORS[stim_key], linewidth=2.0,
                        label=STIM_LABELS[stim_key])
        handles.append(line)
    ax.legend(handles=handles, frameon=False, loc="center", fontsize=9,
              handlelength=2.0)
    ax.axis("off")
    out_pdf = OUT_DIR / "fig3_stim_legend.pdf"
    out_png = OUT_DIR / "fig3_stim_legend.png"
    out_svg = OUT_DIR / "fig3_stim_legend.svg"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> wrote {out_pdf.name} + {out_png.name}")


def make_film_legend():
    """4-colour film legend (Panel C movie10 breakdown, Tol Light palette)."""
    fig, ax = plt.subplots(figsize=(2.2, 1.4))
    handles = []
    for film_key, label, color in FILM_COLORS:
        line, = ax.plot([], [], color=color, linewidth=2.0, label=label)
        handles.append(line)
    ax.legend(handles=handles, frameon=False, loc="center", fontsize=9,
              handlelength=2.0)
    ax.axis("off")
    out_pdf = OUT_DIR / "fig3_film_legend.pdf"
    out_png = OUT_DIR / "fig3_film_legend.png"
    out_svg = OUT_DIR / "fig3_film_legend.svg"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> wrote {out_pdf.name} + {out_png.name}")


def main():
    print(f"=== Fig F3 panels -> {OUT_DIR} ===")
    print(f"D3A_ROOT = {D3A_ROOT}")
    all_series = [collect_panel_curves(p) for p in PANELS]
    ylim = shared_ylim(all_series)
    print(f"shared y-axis: [{ylim[0]:.4f}, {ylim[1]:.4f}]")
    for panel, series in zip(PANELS, all_series):
        if not series:
            print(f"  [skip] no series rendered for {panel['name']}")
            continue
        render_panel(panel, series, ylim)
    print("--- legends ---")
    make_stim_legend()
    make_film_legend()
    print("done.")


if __name__ == "__main__":
    main()
