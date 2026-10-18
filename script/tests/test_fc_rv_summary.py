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


def test_json_safe_replaces_non_finite_values():
    m = _module()
    obj = {"a": float("nan"), "b": [1.0, np.inf, np.float64("-inf"), np.float32(2.5)],
           "c": {"d": np.nan, "e": "x", "f": None, "g": 3, "h": True}}
    out = m._json_safe(obj)
    assert out == {"a": None, "b": [1.0, None, None, 2.5],
                   "c": {"d": None, "e": "x", "f": None, "g": 3, "h": True}}


def _write_fc(dirpath, fc, n_trs, name, min_trs=30):
    import json
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / "state_empirical_corr.npy", fc)
    (dirpath / name).write_text(json.dumps(
        {"n_trs_per_state": [int(x) for x in n_trs], "min_trs": min_trs}))


def test_write_fc_rv_summary_writes_strict_json(tmp_path):
    import json
    m = _module()
    m.SCRATCH_DIR = str(tmp_path)
    parc, sub = "atlas-test", "sub-01"
    rng = np.random.default_rng(2)
    fc = np.stack([_corr(rng) for _ in range(4)])
    fc_nan = fc.copy()
    fc_nan[3] = np.nan                      # a state whose FC is undefined
    n_trs = np.array([100, 80, 20, 60])     # state 2 is below the 30-TR rule

    out = tmp_path / "output"
    _write_fc(out / "05f_state_fc" / parc / sub / "vt0.95", fc, n_trs, "metadata.json")
    sub_dir = out / "04_combined_hdphmm" / parc / sub
    _write_fc(sub_dir / "loso" / "season_1", fc, n_trs, "fc_metadata.json")
    _write_fc(sub_dir / "split_half" / "A", fc, n_trs, "fc_metadata.json")
    _write_fc(sub_dir / "split_half" / "B", fc_nan, n_trs, "fc_metadata.json")

    loso = {"sub_id": sub, "hungarian_matching": {"per_fold": {"1": {"matches": [
        {"primary_state": 0, "fold_state": 0, "well_matched": True},
        {"primary_state": 1, "fold_state": 1, "well_matched": True},
        {"primary_state": 2, "fold_state": 2, "well_matched": False}]}}}}
    sh = {"hungarian_matching": {"matching": {"pairs": [
        {"state_A": s, "state_B": s, "correlation": c}
        for s, c in zip(range(4), (0.9, 0.8, 0.7, 0.6))]}}}

    m.write_fc_rv_summary([loso], [sh], parc, str(tmp_path / "multi"))

    def reject(const):
        raise AssertionError(f"non-standard JSON constant {const}")
    rec = json.loads((tmp_path / "multi" / "fc_rv_summary.json").read_text(),
                     parse_constant=reject)["subjects"][sub]

    # LOSO keeps only well-matched pairs; identical FC gives RV 1.
    assert [p["season"] for p in rec["loso"]["pairs"]] == [1, 1]
    assert all(abs(p["rv"] - 1.0) < 1e-9 for p in rec["loso"]["pairs"])

    pairs = rec["split_half"]["pairs"]
    assert [p["reliable"] for p in pairs] == [True, True, False, True]
    assert pairs[3]["rv"] is None and pairs[3]["offdiag_r"] is None
    # Summaries skip the undefined pair: reliable pairs are states 0 and 1.
    assert rec["split_half"]["summary"]["reliable"]["rv"]["n"] == 2
    assert rec["split_half"]["summary"]["all"]["rv"]["n"] == 3
