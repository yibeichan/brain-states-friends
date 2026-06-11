#!/usr/bin/env python
"""
Annotated brain state plot utility.

Produces composite figures with brain surface renderings alongside sorted
parcel tables (parcel_id, parcel_name, network, value), separated for
cortex and subcortex. Reusable from CLI or as an importable module.

Usage (CLI):
    python script/utils/annotated_brain_plot.py \
        --state_means_path /path/to/state_means_parcel.npy \
        --state_idx 41 \
        --parcellation atlas-4S156Parcels \
        --output_dir /tmp/test_annotated_brain

Usage (import):
    from utils.annotated_brain_plot import (
        build_parcel_table,
        render_annotated_brain_state,
    )
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Subcortical anatomical grouping via canonical v2 partition
# (Alexander-DeLong-Strick 1986; Haber & Knutson 2010). Use
# plot_style.assign_network() per parcel — the parcel-level lookup is the
# source of truth. The legacy atlas_name-based fallback (CIT168Subcortical
# -> single bin) lumped 14 structures including DA midbrain + hypothalamus
# under "Basal Ganglia" which was anatomically wrong; replaced with
# structure-name parsing.

# Yeo-7 network display order
_NETWORK_ORDER = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
]

# Subcortical group display order (v2 canonical, 6 bins matching plot_style)
_SUBCORT_GROUP_ORDER = [
    "BG", "Midbrain-DA", "Midbrain-Diencephalic",
    "Thalamus", "Hipp/Amyg", "Cerebellum",
]

# Schaefer cortical region abbreviation → full name
# Source: Schaefer et al. 2018 (CBIG GitHub README)
_SCHAEFER_REGION_FULL_NAME = {
    "AntTemp": "Anterior Temporal",
    "Aud": "Auditory",
    "Cent": "Central",
    "Cing": "Cingulate",
    "Cinga": "Cingulate Anterior",
    "Cingm": "Mid-Cingulate",
    "Cingp": "Cingulate Posterior",
    "ExStr": "Extrastriate Cortex",
    "ExStrInf": "Extrastriate Inferior",
    "ExStrSup": "Extrastriate Superior",
    "FEF": "Frontal Eye Fields",
    "FPole": "Frontal Pole",
    "FrMed": "Frontal Medial",
    "FrOper": "Frontal Operculum",
    "FrOperIns": "Frontal Operculum/Insula",
    "IFG": "Inferior Frontal Gyrus",
    "Ins": "Insula",
    "IPL": "Inferior Parietal Lobule",
    "IPS": "Intraparietal Sulcus",
    "Med": "Medial",
    "OFC": "Orbital Frontal Cortex",
    "Par": "Parietal",
    "ParMed": "Parietal Medial",
    "ParOcc": "Parietal Occipital",
    "ParOper": "Parietal Operculum",
    "pCun": "Precuneus",
    "pCunPCC": "Precuneus/PCC",
    "PFC": "Prefrontal Cortex",
    "PFCd": "Dorsal PFC",
    "PFCdPFCm": "Dorsal/Medial PFC",
    "PFCl": "Lateral PFC",
    "PFCld": "Lateral Dorsal PFC",
    "PFClv": "Lateral Ventral PFC",
    "PFCm": "Medial PFC",
    "PFCmp": "Medial Posterior PFC",
    "PFCv": "Ventral PFC",
    "PHC": "Parahippocampal Cortex",
    "Post": "Posterior",
    "PostC": "Postcentral",
    "PrC": "Precentral",
    "PrCd": "Precentral Dorsal",
    "PrCv": "Precentral Ventral",
    "RSC": "Retrosplenial Cortex",
    "Rsp": "Retrosplenial",
    "S2": "S2",
    "SPL": "Superior Parietal Lobule",
    "ST": "Superior Temporal",
    "Striate": "Striate Cortex",
    "StriCal": "Striate Calcarine",
    "Temp": "Temporal",
    "TempOcc": "Temporal Occipital",
    "TempOccPar": "Temporal Occipital Parietal",
    "TempPar": "Temporal Parietal",
    "TempPole": "Temporal Pole",
}

# CIT168 subcortical abbreviation → full name
# Source: Pauli et al. 2018, Scientific Data (doi:10.1038/sdata.2018.63)
_CIT168_FULL_NAME = {
    "Pu": "Putamen",
    "Ca": "Caudate Nucleus",
    "NAC": "Nucleus Accumbens",
    "EXA": "Extended Amygdala",
    "GPe": "Globus Pallidus, external",
    "GPi": "Globus Pallidus, internal",
    "SNc_PBP_VTA": "SN compacta/Parabrachial Pigmented N./VTA",
    "RN": "Red Nucleus",
    "SNr": "SN reticulata",
    "VeP": "Ventral Pallidum",
    "HN": "Habenular Nuclei",
    "HTH": "Hypothalamus",
    "MN": "Mammillary Nucleus",
    "STH": "Subthalamic Nucleus",
}

# Thalamus HCP label → full name
_THALAMUS_FULL_NAME = {
    "Pulvinar": "Pulvinar",
    "Anterior": "Anterior",
    "Medio_Dorsal": "Medio Dorsal",
    "Ventral_Latero_Dorsal": "Ventral Latero Dorsal",
    "Central_Lateral-Lateral_Posterior-Medial_Pulvinar": "CL/LP/Medial Pulvinar",
    "Ventral_Anterior": "Ventral Anterior",
    "Ventral_Latero_Ventral": "Ventral Latero Ventral",
}


def _expand_cortical_label(label: str) -> str:
    """Expand Schaefer cortical label to full anatomical name.

    Input:  'LH_Default_pCunPCC_2'
    Output: 'LH Default Precuneus/PCC 2'
    """
    parts = label.split("_")
    if len(parts) < 3:
        return label.replace("_", " ")

    hemi = parts[0]           # LH / RH
    network = parts[1]        # e.g. Default, Vis, SomMot
    idx = parts[-1]           # trailing number

    # Region is everything between network and index (may be empty)
    region_parts = parts[2:-1]
    if region_parts:
        region_abbrev = "_".join(region_parts)
        region_full = _SCHAEFER_REGION_FULL_NAME.get(region_abbrev, region_abbrev)
        return f"{hemi} {network} {region_full} {idx}"
    else:
        # No subregion, e.g. LH_Vis_1
        return f"{hemi} {network} {idx}"


def _expand_subcortical_label(label: str) -> str:
    """Expand abbreviated subcortical label to full anatomical name."""
    parts = label.split("-", 1)
    if len(parts) != 2:
        # SubcorticalHCP / Cerebellum use underscore: LH_Hippocampus
        parts = label.split("_", 1)
        if len(parts) == 2:
            hemi = parts[0]
            name = parts[1].replace("_", " ")
            return f"{hemi} {name}"
        return label

    hemi, abbrev = parts[0], parts[1]
    # CIT168
    if abbrev in _CIT168_FULL_NAME:
        return f"{hemi} {_CIT168_FULL_NAME[abbrev]}"
    # Thalamus
    if abbrev in _THALAMUS_FULL_NAME:
        return f"{hemi} {_THALAMUS_FULL_NAME[abbrev]}"
    # Fallback: replace underscores with spaces
    return f"{hemi} {abbrev.replace('_', ' ')}"


def _subcortical_group(parcel_label: str) -> str:
    """Map a parcel label to its v2 canonical subcortical bin.

    Uses plot_style.assign_network() — the parcel-level source of truth.
    Returns the v2 bin name (BG / Midbrain-DA / Midbrain-Diencephalic /
    Thalamus / Hipp/Amyg / Cerebellum), or "Other" if the label is not a
    recognized subcortical parcel.
    """
    from utils.plot_style import assign_network
    result = assign_network(str(parcel_label))
    return result if result is not None else "Other"


def build_parcel_table(
    pattern: np.ndarray,
    labels_df: pd.DataFrame,
    parcellation: str,
    top_n: int = 10,
    region: str = "cortical",
) -> pd.DataFrame:
    """
    Build a sorted parcel table from a brain state pattern.

    Args:
        pattern: Array of shape (n_parcels,), 0-indexed.
        labels_df: DataFrame from load_parcel_labels() with 1-based index column.
        parcellation: e.g. 'atlas-4S156Parcels'.
        top_n: Number of top activated + deactivated cortical parcels to show.
        region: 'cortical' or 'subcortical'.

    Returns:
        DataFrame with columns [index, label, network, value, abs_value].
        For cortical: top_n positive rows + top_n negative rows.
        For subcortical: all subcortical parcels sorted by abs_value descending,
        with an additional 'group' column.
    """
    from utils.viz_yabplot import _get_cortical_threshold

    threshold = _get_cortical_threshold(parcellation)

    if region == "cortical":
        mask = labels_df["index"] <= threshold
        df = labels_df[mask].copy()
        df["value"] = df["index"].apply(lambda i: float(pattern[int(i) - 1]))
        df["abs_value"] = df["value"].abs()
        # Use Yeo-7 network_label
        df["network"] = df["network_label"].fillna("n/a")
        df["full_name"] = df["label"].apply(_expand_cortical_label)
        df = df[["index", "label", "full_name", "network", "value", "abs_value"]].copy()

        # Top N positive (most activated)
        positive = df[df["value"] > 0].nlargest(top_n, "value")
        # Top N negative (most deactivated)
        negative = df[df["value"] < 0].nsmallest(top_n, "value")

        # Mark section for rendering
        positive = positive.copy()
        negative = negative.copy()
        positive["section"] = "activated"
        negative["section"] = "deactivated"

        result = pd.concat([positive, negative], ignore_index=True)
        return result

    else:  # subcortical
        mask = labels_df["index"] > threshold
        df = labels_df[mask].copy()
        df["value"] = df["index"].apply(lambda i: float(pattern[int(i) - 1]))
        df["abs_value"] = df["value"].abs()
        df["network"] = df["label"].apply(_subcortical_group)
        df["group"] = df["network"]  # alias for grouping
        df["full_name"] = df["label"].apply(_expand_subcortical_label)
        df = df[
            ["index", "label", "full_name", "network", "group", "value", "abs_value"]
        ].copy()
        df = df.sort_values("abs_value", ascending=False).reset_index(drop=True)
        return df


def render_parcel_table_ax(
    ax,
    table_df: pd.DataFrame,
    title: str = "",
    fontsize: int = 7,
    cmap: str = "RdBu_r",
    vmax: float = None,
    show_groups: bool = False,
) -> None:
    """
    Render a parcel table onto a matplotlib Axes using monospace text.

    Args:
        ax: Matplotlib Axes to render onto.
        table_df: DataFrame from build_parcel_table().
        title: Title string shown above the table.
        fontsize: Font size for table text.
        cmap: Colormap name for value color-coding.
        vmax: Max absolute value for colormap normalization.
        show_groups: If True, insert group headers (for subcortical).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if table_df.empty or table_df["value"].abs().max() == 0:
        ax.text(
            0.5, 0.5, "Inactive state",
            ha="center", va="center", fontsize=fontsize + 2,
            color="#999999", fontstyle="italic",
        )
        return

    colormap = plt.get_cmap(cmap)
    if vmax is None:
        vmax = table_df["abs_value"].max()
    if vmax == 0:
        vmax = 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    n_rows = len(table_df)
    # Account for title, header, section separators, group headers
    extra_lines = 2  # title + header
    if show_groups and "group" in table_df.columns:
        n_groups = table_df["group"].nunique()
        extra_lines += n_groups
    elif "section" in table_df.columns:
        extra_lines += 1  # separator between activated/deactivated

    total_lines = n_rows + extra_lines
    row_h = min(1.0 / (total_lines + 1), 0.038)

    y = 1.0 - row_h * 0.5

    # Title
    if title:
        ax.text(
            0.02, y, title,
            fontsize=fontsize + 1, fontweight="bold", fontfamily="sans-serif",
            va="top",
        )
        y -= row_h * 1.3

    # Header
    header = f"{'Idx':>4s}  {'Abbrev':<20s} {'Full Name':<42s} {'Network':<14s} {'Val':>6s}"
    ax.text(
        0.02, y, header,
        fontsize=fontsize, fontweight="bold", fontfamily="monospace",
        va="top", color="#333333",
    )
    y -= row_h
    # Header underline
    ax.axhline(y + row_h * 0.3, xmin=0.02, xmax=0.98, color="#CCCCCC", linewidth=0.5)

    if show_groups and "group" in table_df.columns:
        # Render grouped by anatomy
        for group_name in _SUBCORT_GROUP_ORDER:
            group_df = table_df[table_df["group"] == group_name]
            if group_df.empty:
                continue
            # Group header
            y -= row_h * 0.3
            ax.text(
                0.02, y, f"-- {group_name} --",
                fontsize=fontsize, fontweight="bold", fontfamily="sans-serif",
                va="top", color="#666666",
            )
            y -= row_h
            for _, row in group_df.iterrows():
                _render_table_row(ax, row, y, fontsize, colormap, norm)
                y -= row_h
    elif "section" in table_df.columns:
        # Render activated then deactivated with separator
        for section_name in ["activated", "deactivated"]:
            section_df = table_df[table_df["section"] == section_name]
            if section_df.empty:
                continue
            if section_name == "deactivated":
                y -= row_h * 0.3
                ax.axhline(
                    y + row_h * 0.5, xmin=0.02, xmax=0.98,
                    color="#AAAAAA", linewidth=0.5, linestyle="--",
                )
                ax.text(
                    0.50, y + row_h * 0.7, "--- deactivated ---",
                    fontsize=fontsize - 1, color="#888888",
                    ha="center", va="bottom", fontstyle="italic",
                )
            for _, row in section_df.iterrows():
                _render_table_row(ax, row, y, fontsize, colormap, norm)
                y -= row_h
    else:
        for _, row in table_df.iterrows():
            _render_table_row(ax, row, y, fontsize, colormap, norm)
            y -= row_h


def _render_table_row(ax, row, y, fontsize, colormap, norm):
    """Render a single table row."""
    idx = int(row["index"])
    abbrev = str(row["label"])
    if len(abbrev) > 20:
        abbrev = abbrev[:17] + "..."
    full_name = str(row.get("full_name", abbrev))
    if len(full_name) > 42:
        full_name = full_name[:39] + "..."
    network = str(row["network"])
    if len(network) > 14:
        network = network[:11] + "..."
    value = float(row["value"])

    # Full row in uniform color (no value gradient)
    text_line = f"{idx:>4d}  {abbrev:<20s} {full_name:<42s} {network:<14s} {value:+.3f}"
    ax.text(
        0.02, y, text_line,
        fontsize=fontsize, fontfamily="monospace", va="top", color="#444444",
    )


def render_network_summary_ax(
    ax,
    pattern: np.ndarray,
    labels_df: pd.DataFrame,
    parcellation: str,
    cmap: str = "RdBu_r",
    vmax: float = None,
) -> None:
    """
    Render a horizontal bar chart of mean signed value per Yeo-7 network.

    Args:
        ax: Matplotlib Axes.
        pattern: Array of shape (n_parcels,), 0-indexed.
        labels_df: DataFrame from load_parcel_labels().
        parcellation: e.g. 'atlas-4S156Parcels'.
        cmap: Colormap name.
        vmax: Max absolute value for colormap normalization.
    """
    from utils.viz_yabplot import _get_cortical_threshold

    threshold = _get_cortical_threshold(parcellation)
    cortical = labels_df[labels_df["index"] <= threshold].copy()
    cortical["value"] = cortical["index"].apply(
        lambda i: float(pattern[int(i) - 1])
    )

    network_means = (
        cortical.groupby("network_label")["value"]
        .mean()
        .reindex(_NETWORK_ORDER, fill_value=0.0)
    )

    colormap = plt.get_cmap(cmap)
    if vmax is None:
        vmax = max(abs(network_means.min()), abs(network_means.max()), 0.01)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    colors = [colormap(norm(v)) for v in network_means.values]

    y_pos = np.arange(len(_NETWORK_ORDER))
    ax.barh(y_pos, network_means.values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(_NETWORK_ORDER, fontsize=7)
    ax.axvline(0, color="#333333", linewidth=0.5)
    ax.set_xlabel("Mean z-score", fontsize=7)
    ax.set_title("Network summary", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6)
    ax.invert_yaxis()


def render_annotated_brain_state(
    pattern: np.ndarray,
    labels_df: pd.DataFrame,
    parcellation: str,
    cortical_vmax: float = 0.25,
    subcortical_vmax: float = 0.10,
    cmap: str = "RdBu_r",
    top_n: int = 10,
    title: str = None,
    figsize: tuple = (11.69, 16.54),  # A4 landscape-ish (297mm x 420mm)
) -> dict:
    """
    Create a composite annotated brain state figure.

    Layout:
        Top row: cortical brain surface (left) + subcortical brain surface (right)
        Bottom row: cortical table + network summary (left) + subcortical table (right)

    Args:
        pattern: Array of shape (n_parcels,), 0-indexed.
        labels_df: DataFrame from load_parcel_labels().
        parcellation: e.g. 'atlas-4S156Parcels'.
        cortical_vmax: Max absolute value for cortical colorbar.
        subcortical_vmax: Max absolute value for subcortical colorbar.
        cmap: Matplotlib colormap name.
        top_n: Number of top activated + deactivated cortical parcels.
        title: Optional title for the figure.
        figsize: Figure size in inches.

    Returns:
        {"figure": fig, "cortical_table": DataFrame, "subcortical_table": DataFrame}
    """
    from utils.viz_yabplot import (
        pattern_to_cortical_dict,
        pattern_to_subcortical_dict,
        render_cortical_to_image,
        render_subcortical_to_image,
        get_subcortical_atlas_dir,
    )

    atlas_dir = get_subcortical_atlas_dir()

    # Build data tables
    cortical_table = build_parcel_table(
        pattern, labels_df, parcellation, top_n=top_n, region="cortical",
    )
    subcortical_table = build_parcel_table(
        pattern, labels_df, parcellation, top_n=top_n, region="subcortical",
    )

    # Render brain images
    cort_dict = pattern_to_cortical_dict(pattern, labels_df, parcellation)
    sub_dict = pattern_to_subcortical_dict(pattern, labels_df, parcellation)

    cort_img = render_cortical_to_image(
        cort_dict,
        color_range=(-cortical_vmax, cortical_vmax),
        cmap=cmap,
        figsize=(600, 150),
    )
    sub_img = render_subcortical_to_image(
        sub_dict,
        color_range=(-subcortical_vmax, subcortical_vmax),
        atlas_dir=atlas_dir,
        cmap=cmap,
        figsize=(400, 150),
    )

    # Build composite figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.0, 0.2, 3.0],
        width_ratios=[3, 2],
        hspace=0.08,
        wspace=0.05,
    )

    # Top-left: cortical brain
    ax_cort_brain = fig.add_subplot(gs[0, 0])
    ax_cort_brain.imshow(cort_img)
    ax_cort_brain.axis("off")
    ax_cort_brain.set_title("Cortical", fontsize=10, fontweight="bold")

    # Top-right: subcortical brain
    ax_sub_brain = fig.add_subplot(gs[0, 1])
    ax_sub_brain.imshow(sub_img)
    ax_sub_brain.axis("off")
    ax_sub_brain.set_title("Subcortical", fontsize=10, fontweight="bold")

    # Middle row: colorbars
    ax_cort_cbar = fig.add_subplot(gs[1, 0])
    sm_cort = mcm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=-cortical_vmax, vmax=cortical_vmax),
    )
    sm_cort.set_array([])
    fig.colorbar(
        sm_cort, cax=ax_cort_cbar, orientation="horizontal",
    )
    ax_cort_cbar.set_xlabel("Cortical z-score", fontsize=8)
    ax_cort_cbar.tick_params(labelsize=7)

    ax_sub_cbar = fig.add_subplot(gs[1, 1])
    sm_sub = mcm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=-subcortical_vmax, vmax=subcortical_vmax),
    )
    sm_sub.set_array([])
    fig.colorbar(
        sm_sub, cax=ax_sub_cbar, orientation="horizontal",
    )
    ax_sub_cbar.set_xlabel("Subcortical z-score", fontsize=8)
    ax_sub_cbar.tick_params(labelsize=7)

    # Bottom-left: split into cortical table (top) + network summary (bottom)
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[2, 0], height_ratios=[3, 1], hspace=0.15,
    )
    ax_cort_table = fig.add_subplot(gs_bottom_left[0])
    render_parcel_table_ax(
        ax_cort_table, cortical_table,
        title=f"Top {top_n} Activated / Deactivated Cortical Parcels",
        fontsize=7, cmap=cmap, vmax=cortical_vmax,
    )

    ax_network = fig.add_subplot(gs_bottom_left[1])
    render_network_summary_ax(
        ax_network, pattern, labels_df, parcellation,
        cmap=cmap, vmax=cortical_vmax,
    )

    # Bottom-right: subcortical table
    ax_sub_table = fig.add_subplot(gs[2, 1])
    render_parcel_table_ax(
        ax_sub_table, subcortical_table,
        title="All Subcortical Parcels",
        fontsize=6, cmap=cmap, vmax=subcortical_vmax,
        show_groups=True,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    fig.subplots_adjust(
        left=0.03, right=0.97,
        top=0.95 if title else 0.98,
        bottom=0.03,
    )

    return {
        "figure": fig,
        "cortical_table": cortical_table,
        "subcortical_table": subcortical_table,
    }


def main():
    """CLI entry point."""
    import argparse
    import sys
    import pathlib
    from dotenv import load_dotenv

    matplotlib.use("Agg")

    # Set up paths
    script_dir = pathlib.Path(__file__).resolve().parent.parent
    project_root = script_dir.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    load_dotenv(project_root / ".env")

    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from utils.viz_yabplot import load_parcel_labels, setup_yabplot_headless
    from utils.common import normalize_parcellation_name

    parser = argparse.ArgumentParser(
        description="Generate annotated brain state plot with parcel tables.",
    )
    parser.add_argument(
        "--state_means_path", required=True,
        help="Path to state_means_parcel.npy",
    )
    parser.add_argument(
        "--state_idx", type=int, required=True,
        help="State index to plot (0-based into state_means array)",
    )
    parser.add_argument(
        "--parcellation", default="atlas-4S156Parcels",
        help="Parcellation name (default: atlas-4S156Parcels)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save output PNG and PDF",
    )
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--cortical_vmax", type=float, default=0.25)
    parser.add_argument("--subcortical_vmax", type=float, default=0.10)
    parser.add_argument("--cmap", default="RdBu_r")

    args = parser.parse_args()

    setup_yabplot_headless()

    parc = normalize_parcellation_name(args.parcellation)
    labels_df = load_parcel_labels(parc)
    state_means = np.load(args.state_means_path)

    if args.state_idx < 0 or args.state_idx >= state_means.shape[0]:
        print(
            f"Error: state_idx {args.state_idx} out of range "
            f"[0, {state_means.shape[0] - 1}]"
        )
        sys.exit(1)

    pattern = state_means[args.state_idx]

    result = render_annotated_brain_state(
        pattern=pattern,
        labels_df=labels_df,
        parcellation=parc,
        cortical_vmax=args.cortical_vmax,
        subcortical_vmax=args.subcortical_vmax,
        cmap=args.cmap,
        top_n=args.top_n,
        title=f"State {args.state_idx}",
    )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = result["figure"]
    png_path = out_dir / f"state_{args.state_idx}_annotated.png"
    pdf_path = out_dir / f"state_{args.state_idx}_annotated.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Also save tables as CSV for reference
    csv_cort = out_dir / f"state_{args.state_idx}_cortical_table.csv"
    csv_sub = out_dir / f"state_{args.state_idx}_subcortical_table.csv"
    result["cortical_table"].to_csv(csv_cort, index=False)
    result["subcortical_table"].to_csv(csv_sub, index=False)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_cort}")
    print(f"Saved: {csv_sub}")


if __name__ == "__main__":
    main()
