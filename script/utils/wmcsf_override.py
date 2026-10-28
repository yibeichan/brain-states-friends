"""Swap fMRIPrep's csf/white_matter regressors for the shared clean CSF/WM table.

The CNeuroMod sMRIPrep tissue maps were resampled with trilinear interpolation, so
fMRIPrep's `csf` and `white_matter` confounds are averaged over the wrong voxels.
The shared CNeuroMod CSF/WM table holds, per run and volume, the same mean signals
rebuilt over clean masks (`*_clean`) and over the shipped corrupt masks
(`*_corrupt`) with one recipe, so a clean-vs-corrupt comparison isolates the mask
defect. Table layout: `<root>/friends/<sub>/<sub>_<ses>_<task>_wmcsf.tsv`.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

WMCSF_VARIANTS = ("clean", "corrupt")
WMCSF_COLUMNS = ("csf", "white_matter")


def wmcsf_table_path(table_root, bold_file, dataset="friends"):
    """Table file for a BOLD run, keyed by its sub/ses/task BIDS prefix."""
    stem = os.path.basename(bold_file).split("_space-")[0]
    sub = stem.split("_")[0]
    return Path(table_root) / dataset / sub / f"{stem}_wmcsf.tsv"


def replace_wmcsf(confounds, table_root, bold_file, variant, dataset="friends"):
    """Return a copy of `confounds` with csf/white_matter taken from the table.

    The replacement columns are demeaned, as nilearn's load_confounds demeans
    every regressor it returns. A non-finite or constant column is an error: signal
    cleaning would drop a constant regressor without a word.
    """
    if variant not in WMCSF_VARIANTS:
        raise ValueError(f"wmcsf variant must be one of {WMCSF_VARIANTS}, got {variant!r}")
    missing = [c for c in WMCSF_COLUMNS if c not in confounds.columns]
    if missing:
        raise ValueError(f"confounds lack {missing}; cannot replace csf/white_matter")

    path = wmcsf_table_path(table_root, bold_file, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"No clean CSF/WM table file for this run: {path}")
    table = pd.read_csv(path, sep="\t")
    if len(table) != len(confounds):
        raise ValueError(
            f"{path.name}: {len(table)} TRs in table vs {len(confounds)} in confounds")

    out = confounds.copy()
    for col in WMCSF_COLUMNS:
        values = table[f"{col}_{variant}"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: {col}_{variant} has non-finite values")
        if np.ptp(values) == 0:
            raise ValueError(f"{path.name}: {col}_{variant} is constant")
        out[col] = values - values.mean()
    return out
