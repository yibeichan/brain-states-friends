"""Replacing fMRIPrep's csf/white_matter regressors with the shared clean CSF/WM table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.wmcsf_override import (  # noqa: E402
    replace_wmcsf, wmcsf_table_path,
)


def _confounds(n=5):
    return pd.DataFrame({
        "trans_x": np.arange(n, dtype=float),
        "csf": np.zeros(n),
        "white_matter": np.zeros(n),
        "cosine00": np.ones(n),
    })


def _write_table(root, stem, n=5):
    sub = stem.split("_")[0]
    path = root / "friends" / sub / f"{stem}_wmcsf.tsv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "csf_clean": np.arange(n) + 10.0,
        "white_matter_clean": np.arange(n) * 2.0,
        "csf_corrupt": np.arange(n) * 3.0,
        "white_matter_corrupt": np.arange(n) + 100.0,
    }).to_csv(path, sep="\t", index=False)
    return path


STEM = "sub-04_ses-010_task-s01e19a"
BOLD = f"/x/{STEM}_space-fsLR_den-91k_bold.dtseries.nii"


def test_table_path_from_bold_name(tmp_path):
    assert wmcsf_table_path(tmp_path, BOLD) == (
        tmp_path / "friends" / "sub-04" / f"{STEM}_wmcsf.tsv")


@pytest.mark.parametrize("variant, csf, wm", [
    ("clean", np.arange(5) + 10.0, np.arange(5) * 2.0),
    ("corrupt", np.arange(5) * 3.0, np.arange(5) + 100.0),
])
def test_replaces_columns_demeaned(tmp_path, variant, csf, wm):
    _write_table(tmp_path, STEM)
    out = replace_wmcsf(_confounds(), tmp_path, BOLD, variant)
    np.testing.assert_allclose(out["csf"], csf - csf.mean())
    np.testing.assert_allclose(out["white_matter"], wm - wm.mean())
    # Other regressors are untouched and the column order is kept.
    assert list(out.columns) == ["trans_x", "csf", "white_matter", "cosine00"]
    np.testing.assert_array_equal(out["trans_x"], np.arange(5))


def test_input_frame_is_not_modified(tmp_path):
    _write_table(tmp_path, STEM)
    conf = _confounds()
    replace_wmcsf(conf, tmp_path, BOLD, "clean")
    assert (conf["csf"] == 0).all()


def test_length_mismatch_fails(tmp_path):
    _write_table(tmp_path, STEM, n=4)
    with pytest.raises(ValueError, match="TR"):
        replace_wmcsf(_confounds(5), tmp_path, BOLD, "clean")


def test_missing_table_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        replace_wmcsf(_confounds(), tmp_path, BOLD, "clean")


def test_missing_wmcsf_columns_fails(tmp_path):
    _write_table(tmp_path, STEM)
    with pytest.raises(ValueError, match="csf"):
        replace_wmcsf(_confounds().drop(columns="csf"), tmp_path, BOLD, "clean")


def test_unknown_variant_fails(tmp_path):
    _write_table(tmp_path, STEM)
    with pytest.raises(ValueError, match="variant"):
        replace_wmcsf(_confounds(), tmp_path, BOLD, "shipped")
