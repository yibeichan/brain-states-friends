import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rv04", Path(__file__).resolve().parents[1] / "04rv_reliability_vis.py")


def _module():
    # The module requires SCRATCH_DIR at import; these tests touch no files.
    import os
    import tempfile
    os.environ.setdefault("SCRATCH_DIR", tempfile.gettempdir())
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)
    return m


def _corr(rng, p=12, n=200):
    x = rng.normal(size=(n, p))
    return np.corrcoef(x, rowvar=False)


def test_offdiag_r_ignores_diagonal_and_detects_identity():
    m = _module()
    rng = np.random.default_rng(0)
    a, b = _corr(rng), _corr(rng)
    assert abs(m.offdiag_fc_r(a, a) - 1.0) < 1e-12
    # two independent correlation matrices: off-diagonal r near 0
    assert abs(m.offdiag_fc_r(a, b)) < 0.4


def test_pair_record_reliability_flag():
    m = _module()
    rng = np.random.default_rng(1)
    fc = np.stack([_corr(rng), _corr(rng)])
    n = np.array([29, 30])
    rec = m._pair_record(fc, fc, 0, 1, n, n, 30)
    assert rec["reliable"] is False and rec["n_trs_A"] == 29
    rec = m._pair_record(fc, fc, 1, 1, n, n, 30)
    assert rec["reliable"] is True and abs(rec["offdiag_r"] - 1.0) < 1e-12
