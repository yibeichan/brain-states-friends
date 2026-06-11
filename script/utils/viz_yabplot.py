#!/usr/bin/env python
"""
Core yabplot visualization utilities for brain state surface plots.

Provides reusable functions for rendering cortical and subcortical brain state
patterns via yabplot (PyVista-based), with headless SLURM support.

Resolution note:
    Parcel index arithmetic uses `pattern[row['index'] - 1]` because
    the TSV uses 1-based indices while the pattern array is 0-indexed.
    Named-key dicts (not raw integer arrays) are used throughout to
    eliminate ordering-mismatch risk between atlases.

Subcortical caveat:
    Subcortical parcels (CIT168 basal ganglia, HCP thalamus nuclei,
    hippocampus/amygdala, cerebellum) are anatomically-defined ROIs, not
    functionally-derived like the Schaefer-100 cortical parcels. Hippocampus
    and amygdala have ~50–70% lower tSNR due to susceptibility artifacts;
    interpret those regions conservatively.
"""

import os
import re
import logging
import numpy as np
import pandas as pd

from utils.env import require_env

logger = logging.getLogger(__name__)


def setup_yabplot_headless() -> None:
    """
    Configure PyVista for SLURM headless rendering.

    Sets PYOPENGL_PLATFORM=osmesa (software rendering) before importing PyVista.
    Falls back to pv.start_xvfb() if osmesa is unavailable.

    Call this BEFORE importing yabplot or pyvista.
    """
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        logger.info("PyVista headless mode: OSMesa")
    except Exception as e:
        logger.warning(f"OSMesa setup issue: {e}. Trying Xvfb fallback.")
        try:
            import pyvista as pv
            pv.start_xvfb()
            logger.info("PyVista headless mode: Xvfb fallback")
        except Exception as e2:
            logger.error(f"Both OSMesa and Xvfb failed: {e2}")
            raise


def load_parcel_labels(parcellation: str) -> pd.DataFrame:
    """
    Load atlas TSV file for the given parcellation.

    Args:
        parcellation: Full parcellation name, e.g. 'atlas-4S156Parcels'.

    Returns:
        DataFrame with at least columns: ['index', 'label', 'atlas_name'].
        TSV index column is 1-based (parcel 1 → pattern[0]).
    """
    tsv_path = os.path.join(require_env("ATLAS_DIR"), parcellation, f"{parcellation}_dseg.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Atlas TSV not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    logger.info(f"Loaded {len(df)} parcel labels from {tsv_path}")
    return df


def _get_cortical_threshold(parcellation: str) -> int:
    """
    Compute the cortical/subcortical boundary index for a 4S atlas.

    The 4S series always places exactly 56 subcortical parcels at the end.
    For atlas-4S156Parcels: cortical = 1–100, subcortical = 101–156.

    Args:
        parcellation: e.g. 'atlas-4S156Parcels'

    Returns:
        Integer threshold: parcels with index <= threshold are cortical.
    """
    match = re.search(r"atlas-4S(\d+)Parcels", parcellation)
    if not match:
        raise ValueError(
            f"Cannot determine cortical threshold from parcellation: {parcellation!r}. "
            f"Expected format 'atlas-4S<N>Parcels'."
        )
    total = int(match.group(1))
    return total - 56  # e.g. 156 - 56 = 100


def pattern_to_cortical_dict(
    pattern: np.ndarray, labels_df: pd.DataFrame, parcellation: str
) -> dict:
    """
    Map cortical parcel activations to a label-keyed dict for yab.plot_cortical.

    TSV labels (e.g. 'LH_Vis_1') match the custom schaefer100_cortical atlas
    LUT built from the project's own CIFTI dlabel (no '7Networks_' prefix).

    Args:
        pattern: Array of shape (n_parcels,), 0-indexed (pattern[0] = parcel 1).
        labels_df: DataFrame from load_parcel_labels().
        parcellation: e.g. 'atlas-4S156Parcels'.

    Returns:
        Dict {label_name: activation_value} for cortical parcels only.
    """
    threshold = _get_cortical_threshold(parcellation)
    cortical_df = labels_df[labels_df["index"] <= threshold]
    cortical_dict = {
        str(row["label_7network"]): float(pattern[int(row["index"]) - 1])
        for _, row in cortical_df.iterrows()
    }
    logger.debug(f"Mapped {len(cortical_dict)} cortical parcels")
    return cortical_dict


def pattern_to_subcortical_dict(
    pattern: np.ndarray, labels_df: pd.DataFrame, parcellation: str
) -> dict:
    """
    Map subcortical parcel activations to a label-keyed dict for yab.plot_subcortical.

    Keys match the VTK filenames in the pre-built 4s156_subcortical atlas dir
    (e.g. 'LH-Pu', 'LH-Ca', 'Thal-Pulvinar-L', 'L_Hipp').

    Args:
        pattern: Array of shape (n_parcels,), 0-indexed.
        labels_df: DataFrame from load_parcel_labels().
        parcellation: e.g. 'atlas-4S156Parcels'.

    Returns:
        Dict {label_name: activation_value} for subcortical parcels only.
    """
    threshold = _get_cortical_threshold(parcellation)
    subcortical_df = labels_df[labels_df["index"] > threshold]
    subcortical_dict = {
        str(row["label"]): float(pattern[int(row["index"]) - 1])
        for _, row in subcortical_df.iterrows()
    }
    logger.debug(f"Mapped {len(subcortical_dict)} subcortical parcels")
    return subcortical_dict


def get_cortical_atlas_dir() -> str:
    """
    Return path to the custom Schaefer-100 cortical atlas built from CIFTI.

    The atlas is built once by running:
        the yabplot repo's tools/build_schaefer100_cortical.py (https://github.com/yibeichan/yabplot)

    Expected structure:
        {atlas_dir}/
            schaefer100_conte69.csv   (parcel ID per vertex, 0 = medial wall)
            schaefer100_LUT.txt       (id name R G B 0)

    Raises:
        RuntimeError: If atlas has not been built yet.
    """
    import yabplot
    atlas_dir = os.path.join(
        os.path.dirname(yabplot.__file__), "data", "atlases", "schaefer100_cortical"
    )
    if not os.path.isdir(atlas_dir):
        raise RuntimeError(
            f"Cortical atlas not found at {atlas_dir}.\n"
            "Run: the yabplot repo's tools/build_schaefer100_cortical.py (https://github.com/yibeichan/yabplot)"
        )
    csv_files = [f for f in os.listdir(atlas_dir) if f.endswith(".csv")]
    lut_files = [f for f in os.listdir(atlas_dir) if f.endswith(".txt")]
    if not csv_files or not lut_files:
        raise RuntimeError(
            f"Cortical atlas directory is incomplete (missing .csv or .txt): {atlas_dir}.\n"
            "Run: the yabplot repo's tools/build_schaefer100_cortical.py (https://github.com/yibeichan/yabplot)"
        )
    logger.info(f"Using cortical atlas: {atlas_dir}")
    return atlas_dir


def get_subcortical_atlas_dir() -> str:
    """
    Return path to the pre-built 4S subcortical VTK directory inside yabplot.

    The directory is shared across all 4S atlas variants (4S156–4S1056) because
    the 56 subcortical parcels (CIT168, HCP thalamus, hippo/amyg, cerebellum)
    have identical MNI coordinates regardless of cortical parcel resolution.

    Raises:
        RuntimeError: If yabplot is not installed or VTK files are missing.
    """
    import yabplot
    atlas_dir = os.path.join(
        os.path.dirname(yabplot.__file__), "data", "atlases", "4s_subcortical"
    )
    if not os.path.isdir(atlas_dir):
        raise RuntimeError(
            f"Subcortical atlas not found at {atlas_dir}.\n"
            "Run tools/build_4s156_subcortical.py in the yabplot fork first.\n"
            "(Any 4S atlas version works: --parcellation atlas-4S156Parcels)"
        )
    vtk_count = len([f for f in os.listdir(atlas_dir) if f.endswith(".vtk")])
    if vtk_count == 0:
        raise RuntimeError(
            f"No .vtk files in {atlas_dir}. "
            "Run tools/build_4s156_subcortical.py to build the atlas."
        )
    logger.info(f"Using subcortical atlas: {atlas_dir} ({vtk_count} VTK files)")
    return atlas_dir


def render_cortical_to_image(
    cortical_dict: dict,
    color_range: tuple,
    cmap: str = "RdBu_r",
    views: list = None,
    figsize: tuple = (900, 350),
) -> np.ndarray:
    """
    Render cortical surface via yab.plot_cortical and return a numpy uint8 RGB array.

    Uses atlas='schaefer_100' (yabplot built-in, 100 Schaefer parcels on Conte69 mesh).
    display_type='object' renders off-screen and returns the plotter for screenshotting.

    Args:
        cortical_dict: {label: value} dict from pattern_to_cortical_dict().
        color_range: (vmin, vmax) tuple for the colormap.
        cmap: Matplotlib colormap name. Default 'RdBu_r'.
        views: List of view names (e.g. ['left_lateral', 'right_lateral']).
               Defaults to all 8 standard views if None.
        figsize: PyVista window size in pixels.

    Returns:
        numpy uint8 array of shape (H, W, 3).
    """
    import yabplot as yab

    if views is None:
        views = ["left_lateral", "right_lateral", "left_medial", "right_medial"]

    plotter = yab.plot_cortical(
        data=cortical_dict,
        custom_atlas_path=get_cortical_atlas_dir(),
        views=views,
        vminmax=list(color_range),
        cmap=cmap,
        figsize=figsize,
        display_type="object",
    )
    plotter.remove_scalar_bar()
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def render_subcortical_to_image(
    subcortical_dict: dict,
    color_range: tuple,
    atlas_dir: str,
    cmap: str = "RdBu_r",
    views: list = None,
    figsize: tuple = (700, 500),
) -> np.ndarray:
    """
    Render subcortical structures via yab.plot_subcortical and return numpy uint8 RGB array.

    Uses pre-built VTK meshes from the custom 4s156_subcortical atlas dir.
    Regions with no data are rendered at reduced opacity (nan_alpha=0.2) so the
    full structure set is still visible as anatomical context.

    Args:
        subcortical_dict: {label: value} dict from pattern_to_subcortical_dict().
        color_range: (vmin, vmax) tuple for the colormap.
        atlas_dir: Path to directory containing .vtk files (from get_subcortical_atlas_dir()).
        cmap: Matplotlib colormap name. Default 'RdBu_r'.
        views: List of view names. Defaults to anterior + superior + posterior if None.
        figsize: PyVista window size in pixels.

    Returns:
        numpy uint8 array of shape (H, W, 3).
    """
    import yabplot as yab

    if views is None:
        views = ["anterior", "superior", "posterior"]

    plotter = yab.plot_subcortical(
        data=subcortical_dict,
        custom_atlas_path=atlas_dir,
        views=views,
        vminmax=list(color_range),
        cmap=cmap,
        nan_alpha=0.2,
        figsize=figsize,
        display_type="object",
    )
    plotter.remove_scalar_bar()
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img
