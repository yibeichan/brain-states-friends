"""fig_S7_individual_differences.py - Supplementary Figure S7.

Subject-level individual differences across F1–F5 findings, rendered as a
1×6 horizontal strip of per-subject radar plots.

One panel per subject (sub-01 to sub-06, left to right). Each radar carries
the same seven axes derived from the five main-text findings:

| # | Axis label | Source finding | Data key                                                                |
|---|------------|----------------|-------------------------------------------------------------------------|
| 1 | K          | F1.R1          | 06b transition_structure_summary.json → n_active_states (canonical)     |
| 2 | %CE        | F2.R2          | 05e_a4 state_flags.csv eligible count / K_active (from 06b)             |
| 3 | FCρ        | F3.R3          | 06b transition_structure_summary.json → A3_fc_transition.rho            |
| 4 | Homo       | F3.R3          | 06b transition_structure_summary.json → A3_network_homophily.ratio      |
| 5 | Aud        | F4.R4b         | 08d friends_w2v-bert-2.0/D1_depth_profile.json → argmax(bal_acc) / 23   |
| 6 | M10        | F5.R5          | m10_05 cross_stimulus_summary.json → A1_recurrence_correlation.rho     |
| 7 | HP+PP      | F5.R5          | mean of hp_05 + pp_05 A1.spearman_rho (sub-04 N/A)                      |

Each axis is scaled to the cohort minimum–maximum with 10% padding on each
end. Sub-04 was not scanned during Harry Potter or Le Petit Prince; sub-04's
axis-7 value is drawn as a hollow marker at the cohort midpoint with dashed
segments to the neighbouring axes, rather than imputed.

Output:
  $SCRATCH_DIR/output/manuscript_figures/figS7/figS7_radar_strip.{pdf,png}

Manuscript-figure conventions: no in-axis titles, no panel letters, minimal
on-figure text (axis labels + subject IDs only). Caption carries axis full
names, cohort absolute ranges, gridline meaning, and sub-04 N/A explanation
(see the figure caption in the Supplementary Material).

Run:
    uv run --no-sync python script/fig_S7_individual_differences.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import (  # noqa: E402
    SUBJECT_MARKERS,
    SUBJECT_NEUTRAL,
    apply_publication_style,
)

apply_publication_style()


# ── Config ────────────────────────────────────────────────────────────────────

SCRATCH_DIR = Path(os.environ["SCRATCH_DIR"])
PARCELLATION = "atlas-4S156Parcels"
VT = "vt0.95"
SUBJECTS: list[str] = [f"sub-0{i}" for i in range(1, 7)]
AXIS_LABELS: list[str] = ["K", "%CE", "FCρ", "Homo", "Aud", "M10", "HP+PP"]
OUT_DIR = SCRATCH_DIR / "output" / "manuscript_figures" / "figS7"
PAD_FRACTION = 0.10
SUB04_HP_PP_FALLBACK_SCALED = 0.5  # cohort midpoint in the scaled [0, 1] range


# ── Per-axis loaders ─────────────────────────────────────────────────────────

def load_06b_summary(sub: str) -> dict:
    """Load 06b transition_structure_summary.json for one subject."""
    path = (
        SCRATCH_DIR / "output" / "06b_transition_structure" / PARCELLATION / sub
        / VT / "transition_structure_summary.json"
    )
    return json.loads(path.read_text())


def load_k_active() -> dict[str, int]:
    """Canonical K_active per subject (recurrence > 0; F1.R1).

    Uses 06b's `n_active_states` (= len([s for s in range(K) if recurrence[s] > 0]))
    rather than 04's `final_refit.n_active_states`, which is a downstream-refit
    count and runs ~5 states lower than the canonical recurrence-based K_active.
    """
    out: dict[str, int] = {}
    for sub in SUBJECTS:
        d = load_06b_summary(sub)
        k = int(d["n_active_states"])
        assert 30 <= k <= 60, f"{sub}: K_active={k} out of expected 30-60 range"
        out[sub] = k
    return out


def load_pct_content_eligible(k_dict: dict[str, int]) -> dict[str, float]:
    """% of K_active that is `eligible_for_content_analysis` (F1.R2)."""
    out: dict[str, float] = {}
    for sub in SUBJECTS:
        path = (
            SCRATCH_DIR / "output" / "05e_temporal_trend_a4" / PARCELLATION / sub
            / VT / "state_flags.csv"
        )
        df = pd.read_csv(path)
        n_elig = int((df["summary_category"] == "eligible_for_content_analysis").sum())
        k = k_dict[sub]
        assert k <= len(df), f"{sub}: K_active={k} > state_flags rows={len(df)}"
        pct = 100.0 * n_elig / k
        assert 0.0 <= pct <= 100.0, f"{sub}: %CE={pct} out of range"
        out[sub] = pct
    return out


def load_fc_rho_and_homo() -> tuple[dict[str, float], dict[str, float]]:
    """FC–transition Mantel ρ and network-homophily ratio (F2.R3)."""
    fc_out: dict[str, float] = {}
    homo_out: dict[str, float] = {}
    for sub in SUBJECTS:
        d = load_06b_summary(sub)
        fc_rho = float(d["A3_fc_transition"]["rho"])
        homo = float(d["A3_network_homophily"]["ratio"])
        assert -1.0 <= fc_rho <= 1.0, f"{sub}: fc_rho={fc_rho} out of [-1, 1]"
        assert 0.0 < homo < 5.0, f"{sub}: homophily ratio={homo} unreasonable"
        fc_out[sub] = fc_rho
        homo_out[sub] = homo
    return fc_out, homo_out


def load_audio_peak_rel_depth() -> dict[str, float]:
    """Wav2Vec-BERT peak-decoding relative depth on Friends (F3.R4b)."""
    out: dict[str, float] = {}
    for sub in SUBJECTS:
        path = (
            SCRATCH_DIR / "output" / "08d_transformer_depth" / PARCELLATION / sub
            / "friends_w2v-bert-2.0" / "D1_depth_profile.json"
        )
        d = json.loads(path.read_text())
        best_layer: int | None = None
        best_bacc = -1.0
        for _lag_key, lag_d in d["results"].items():
            for layer_key, layer_d in lag_d.items():
                b = layer_d.get("balanced_accuracy")
                if b is None:
                    continue
                if b > best_bacc:
                    best_bacc = float(b)
                    best_layer = int(layer_key)
        assert best_layer is not None, f"{sub}: no balanced_accuracy found"
        n_layers = len(d["results"]["lag_0"])  # 24 for w2v-bert (layers 0..23)
        assert n_layers == 24, f"{sub}: w2v n_layers={n_layers}, expected 24"
        rel = best_layer / (n_layers - 1)
        assert 0.3 <= rel <= 0.7, f"{sub}: audio rel-depth={rel} outside expected 0.3-0.7 mid band"
        out[sub] = rel
    return out


def _load_stim_a1_rho(stim_prefix: str) -> dict[str, float | None]:
    """Generic loader for stim_05 cross-stimulus A1 Spearman ρ.

    Returns None for subjects with no data (e.g., sub-04 lacks HP/PP).
    """
    out: dict[str, float | None] = {}
    for sub in SUBJECTS:
        path = (
            SCRATCH_DIR / "output" / f"{stim_prefix}_05_cross_validation"
            / PARCELLATION / sub / VT / "cross_stimulus_summary.json"
        )
        if not path.exists():
            out[sub] = None
            continue
        d = json.loads(path.read_text())
        rho = float(d["A1_recurrence_correlation"]["spearman_rho"])
        assert -1.0 <= rho <= 1.0, f"{sub} {stim_prefix}: rho={rho} out of [-1, 1]"
        out[sub] = rho
    return out


def load_m10_rho() -> dict[str, float | None]:
    """Friends-recurrence → Movie10 occupancy Spearman ρ (F4.R5)."""
    return _load_stim_a1_rho("m10")


def load_hp_pp_avg() -> dict[str, float | None]:
    """Mean of HP and PP Friends-recurrence transfer Spearman ρ (F4.R5).

    Sub-04 has no HP/PP runs → returned as None for sub-04.
    """
    hp = _load_stim_a1_rho("hp")
    pp = _load_stim_a1_rho("pp")
    out: dict[str, float | None] = {}
    for sub in SUBJECTS:
        if hp[sub] is None or pp[sub] is None:
            out[sub] = None
        else:
            out[sub] = 0.5 * (hp[sub] + pp[sub])
    return out


def build_raw_dataframe() -> pd.DataFrame:
    """Load all 7 axes into a (6, 7) DataFrame indexed by SUBJECTS."""
    k_dict = load_k_active()
    pct_dict = load_pct_content_eligible(k_dict)
    fc_dict, homo_dict = load_fc_rho_and_homo()
    aud_dict = load_audio_peak_rel_depth()
    m10_dict = load_m10_rho()
    hp_pp_dict = load_hp_pp_avg()

    raw_df = pd.DataFrame(
        {
            "K": k_dict,
            "%CE": pct_dict,
            "FCρ": fc_dict,
            "Homo": homo_dict,
            "Aud": aud_dict,
            "M10": m10_dict,
            "HP+PP": hp_pp_dict,
        }
    ).reindex(SUBJECTS)

    # Sanity: shape (6, 7); column-7 has exactly one NaN (sub-04)
    assert raw_df.shape == (6, 7), f"raw_df shape {raw_df.shape}, expected (6, 7)"
    assert raw_df["HP+PP"].isna().sum() == 1, "expected exactly one NaN in HP+PP (sub-04)"
    assert pd.isna(raw_df.loc["sub-04", "HP+PP"]), "sub-04 should be the NaN row in HP+PP"
    return raw_df


# ── Cohort min–max scaling ───────────────────────────────────────────────────

def scale_cohort(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], pd.DataFrame]:
    """Per-axis cohort min–max scaling with 10% padding each end.

    Returns
    -------
    scaled_df : (6, 7) DataFrame of values in [0, 1].
    cohort_ranges : {axis_label: (cohort_min, cohort_max)} (unpadded absolute range).
    valid_mask : (6, 7) DataFrame of bool - True where the raw value is real, False where
        the value is a fallback (sub-04 HP+PP).
    """
    scaled_data: dict[str, pd.Series] = {}
    cohort_ranges: dict[str, tuple[float, float]] = {}

    for ax in AXIS_LABELS:
        col = raw_df[ax]
        valid = col.dropna()
        cmin = float(valid.min())
        cmax = float(valid.max())
        rng = cmax - cmin
        low = cmin - PAD_FRACTION * rng
        high = cmax + PAD_FRACTION * rng
        if high == low:
            scaled = pd.Series(SUB04_HP_PP_FALLBACK_SCALED, index=col.index)
        else:
            scaled = (col - low) / (high - low)
            scaled = scaled.fillna(SUB04_HP_PP_FALLBACK_SCALED)
        scaled_data[ax] = scaled
        cohort_ranges[ax] = (cmin, cmax)

    scaled_df = pd.DataFrame(scaled_data).reindex(raw_df.index)

    for ax in AXIS_LABELS:
        v = scaled_df[ax]
        assert ((v >= 0.0) & (v <= 1.0)).all(), f"{ax}: scaled values outside [0, 1]: {v.tolist()}"
    assert (
        abs(scaled_df.loc["sub-04", "HP+PP"] - SUB04_HP_PP_FALLBACK_SCALED) < 1e-9
    ), "sub-04 HP+PP fallback not applied"

    valid_mask = ~raw_df.isna()
    return scaled_df, cohort_ranges, valid_mask


# ── Radar helper ─────────────────────────────────────────────────────────────

def _theta_clockwise_from_top(n_axes: int) -> np.ndarray:
    """Return spoke angles starting at 12 o'clock and going clockwise."""
    base = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False)
    return (np.pi / 2.0 - base) % (2.0 * np.pi)


def plot_one_radar(
    ax,
    sub_id: str,
    scaled_row: list[float],
    valid_row: list[bool],
) -> None:
    """Draw one subject's radar onto the supplied polar axis."""
    n_axes = len(AXIS_LABELS)
    theta = _theta_clockwise_from_top(n_axes)

    # Close the polygon
    values_closed = list(scaled_row) + [scaled_row[0]]
    valid_closed = list(valid_row) + [valid_row[0]]
    theta_closed = np.concatenate([theta, theta[:1]])

    # Polygon outline - segment by segment so we can dash invalid segments
    for i in range(n_axes):
        seg_valid = valid_closed[i] and valid_closed[i + 1]
        ls = "-" if seg_valid else (0, (3, 3))
        ax.plot(
            [theta_closed[i], theta_closed[i + 1]],
            [values_closed[i], values_closed[i + 1]],
            color=SUBJECT_NEUTRAL,
            linewidth=1.2,
            linestyle=ls,
        )

    # Polygon fill (continuous; the dashed outline already signals fallback segments)
    ax.fill(theta_closed, values_closed, color=SUBJECT_NEUTRAL, alpha=0.18, linewidth=0)

    # Spoke-endpoint markers (subject-specific shape; hollow for fallback)
    marker = SUBJECT_MARKERS[sub_id]
    for i in range(n_axes):
        if valid_closed[i]:
            ax.plot(
                theta[i], values_closed[i],
                marker=marker, color=SUBJECT_NEUTRAL,
                markersize=4.5, markerfacecolor=SUBJECT_NEUTRAL, markeredgewidth=0,
            )
        else:
            ax.plot(
                theta[i], values_closed[i],
                marker=marker, color=SUBJECT_NEUTRAL,
                markersize=4.5, markerfacecolor="white",
                markeredgecolor=SUBJECT_NEUTRAL, markeredgewidth=0.8,
            )

    # Axis ticks (drawn as spokes only; labels placed manually below for control)
    ax.set_xticks(theta)
    ax.set_xticklabels([])

    # Manual label placement at a controlled radial offset, with horizontal
    # alignment chosen by spoke angle so labels at top/bottom centre, on the
    # right left-align, on the left right-align. This avoids the default
    # matplotlib behaviour where adjacent short+long labels collide near the top.
    label_r = 1.45  # radial position in data coords (axes go 0..1)
    for i, label in enumerate(AXIS_LABELS):
        t = theta[i]
        cos_t = np.cos(t)
        if cos_t > 0.30:
            ha = "left"
        elif cos_t < -0.30:
            ha = "right"
        else:
            ha = "center"
        # Vertical alignment: top of axes → va='bottom'; bottom → va='top'
        sin_t = np.sin(t)
        if sin_t > 0.50:
            va = "bottom"
        elif sin_t < -0.50:
            va = "top"
        else:
            va = "center"
        ax.text(t, label_r, label, ha=ha, va=va, fontsize=7, color="#333333")

    # Concentric gridlines at 25%, 50%, 75%
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels([])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color="#cccccc", linestyle=":", linewidth=0.6)

    # Spine
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.6)


# ── Strip-render + save ──────────────────────────────────────────────────────

def render_strip(scaled_df: pd.DataFrame, valid_mask: pd.DataFrame, out_dir: Path) -> Path:
    """Render the 1×6 radar strip; save PDF + PNG; return PDF path."""
    fig, axes = plt.subplots(
        1, 6,
        figsize=(8.4, 2.3),
        subplot_kw={"projection": "polar"},
    )

    for ax, sub in zip(axes, SUBJECTS):
        plot_one_radar(
            ax, sub,
            scaled_df.loc[sub].tolist(),
            valid_mask.loc[sub].tolist(),
        )
        ax.text(
            0.5, -0.30, sub,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9, color=SUBJECT_NEUTRAL,
        )

    fig.subplots_adjust(wspace=0.95, left=0.04, right=0.96, top=0.82, bottom=0.16)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figS7_radar_strip.pdf"
    png_path = out_dir / "figS7_radar_strip.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04, transparent=True)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04, transparent=True, dpi=300)
    plt.close(fig)
    return pdf_path


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading axes …")
    raw_df = build_raw_dataframe()
    print(raw_df.round(3).to_string())
    print()

    scaled_df, cohort_ranges, valid_mask = scale_cohort(raw_df)
    print("Cohort ranges (axis: [min, max]):")
    for ax in AXIS_LABELS:
        cmin, cmax = cohort_ranges[ax]
        print(f"  {ax:>6}: [{cmin:.3f}, {cmax:.3f}]")
    print()

    print(f"Rendering 1×6 radar strip → {OUT_DIR} …")
    pdf_path = render_strip(scaled_df, valid_mask, OUT_DIR)
    print(f"saved: {pdf_path}")
    print(f"saved: {pdf_path.with_suffix('.png')}")
    print("done.")


if __name__ == "__main__":
    main()
