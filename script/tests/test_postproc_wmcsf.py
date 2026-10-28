"""Stage 00 run end to end on a tiny synthetic CIFTI: the CSF/WM override and the
exit status.

Claims tested: without the table options the output is unchanged; the corrupt
columns, which here equal fMRIPrep's csf/white_matter up to an offset, give the
same output; the clean columns give a different one; the two table options must
be given together; one failed run makes the script exit 1 while the other runs
are still written, and a clean run exits 0.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "00_postproc.py"
SUB, N_TRS, N_GRAY = "sub-01", 60, 12
STEM = f"{SUB}_ses-001_task-s01e02a"


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    """fMRIPrep-style input (one run), its confounds file and a CSF/WM table."""
    root = tmp_path_factory.mktemp("stage00")
    rng = np.random.default_rng(0)
    func = root / "fmriprep" / SUB / "ses-001" / "func"
    func.mkdir(parents=True)
    t = np.arange(N_TRS)
    conf = pd.DataFrame({c: rng.normal(size=N_TRS) for c in
                         ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")})
    conf["csf"] = rng.normal(size=N_TRS)
    conf["white_matter"] = rng.normal(size=N_TRS)
    conf["cosine00"] = np.cos(np.pi * (t + 0.5) / N_TRS)
    conf.to_csv(func / f"{STEM}_desc-confounds_timeseries.tsv", sep="\t", index=False)

    data = rng.normal(size=(N_TRS, N_GRAY)) + np.outer(conf["csf"], rng.normal(size=N_GRAY))
    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(N_GRAY, dtype=bool), name="CortexLeft")
    series = nib.cifti2.SeriesAxis(start=0, step=1.49, size=N_TRS)
    nib.Cifti2Image(data.astype(np.float32), header=(series, bm)).to_filename(
        func / f"{STEM}_space-fsLR_den-91k_bold.dtseries.nii")

    table = root / "table" / "friends" / SUB
    table.mkdir(parents=True)
    pd.DataFrame({
        "csf_corrupt": conf["csf"] + 5.0,              # the shipped signal, offset
        "white_matter_corrupt": conf["white_matter"],
        "csf_clean": rng.normal(size=N_TRS),
        "white_matter_clean": rng.normal(size=N_TRS),
    }).to_csv(table / f"{STEM}_wmcsf.tsv", sep="\t", index=False)
    return root


def _run(layout, scratch, *extra):
    env = os.environ | {"BASE_DIR": str(layout), "SCRATCH_DIR": str(layout / scratch),
                        "DATA_DIR": str(layout)}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), SUB, "friends", "--n_jobs", "1",
         "--fmriprep_dir", str(layout / "fmriprep" / SUB), *extra],
        env=env, cwd=layout, capture_output=True, text=True)
    out = layout / scratch / "output" / "00_postproc" / SUB / \
        f"{STEM}_space-fsLR_den-91k_bold_cleaned.dtseries.nii"
    return proc, out


@pytest.fixture(scope="module")
def outputs(layout):
    res = {}
    for name, extra in (("default", ()),
                        ("corrupt", ("--wmcsf_table", str(layout / "table"),
                                     "--wmcsf_variant", "corrupt")),
                        ("clean", ("--wmcsf_table", str(layout / "table"),
                                   "--wmcsf_variant", "clean"))):
        proc, out = _run(layout, f"scratch_{name}", *extra)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        res[name] = np.asarray(nib.load(out).get_fdata())
    return res


def test_default_output_is_nilearn_cleaning_of_the_input(layout, outputs):
    spec = importlib.util.spec_from_file_location("postproc00", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from nilearn import signal
    bold = layout / "fmriprep" / SUB / "ses-001" / "func" / f"{STEM}_space-fsLR_den-91k_bold.dtseries.nii"
    confounds = mod.load_confounds_minimal(str(bold))
    expected = signal.clean(nib.load(bold).get_fdata(), confounds=confounds, detrend=True,
                            standardize="zscore_sample", t_r=1.49)
    np.testing.assert_allclose(outputs["default"], expected, rtol=0, atol=1e-5)


def test_corrupt_columns_reproduce_the_default_output(outputs):
    np.testing.assert_allclose(outputs["corrupt"], outputs["default"], rtol=0, atol=1e-5)


def test_clean_columns_change_the_output(outputs):
    assert np.abs(outputs["clean"] - outputs["default"]).max() > 1e-2


def test_table_options_must_come_together(layout):
    proc, out = _run(layout, "scratch_half", "--wmcsf_table", str(layout / "table"))
    assert proc.returncode == 2
    assert "must be given together" in proc.stderr
    assert not (layout / "scratch_half").exists()


def test_one_failed_run_exits_1_and_keeps_the_others(layout, tmp_path):
    # A second session whose BOLD file is not a CIFTI.
    bad = layout / "fmriprep" / SUB / "ses-002" / "func"
    bad.mkdir(parents=True)
    (bad / f"{SUB}_ses-002_task-s01e03a_space-fsLR_den-91k_bold.dtseries.nii").write_text("x")
    try:
        proc, out = _run(layout, "scratch_fail")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert out.exists()
        # The failure is caught per run and the summary is printed, whichever
        # run is processed first.
        assert "Successfully processed: 1/2 files" in proc.stdout, proc.stdout
    finally:
        for f in bad.iterdir():
            f.unlink()
        bad.rmdir()
        bad.parent.rmdir()
    proc, out = _run(layout, "scratch_ok")
    assert proc.returncode == 0, proc.stdout + proc.stderr
