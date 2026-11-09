"""Unit tests for the P7 sensitivity helpers.

Claims tested: (a) the seed rule picks the best validation LL among successful
seeds, and refuses ties, a disagreeing config_summary, and all-failed configs;
(b) resolves_inside follows symlinks, including for paths that do not exist
yet; (c) the side-root layout links only read stages, makes regenerated
stages real directories, aliases vt0.9 <-> vt0.90 for 05e_a4 in the right
direction, and its check fails when a regenerated stage is a link into the
published tree. All on synthetic data in tmp_path.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

m04s = importlib.import_module("04s_sweep_config_final")
side = importlib.import_module("p7_side_root")


def _summary(vals, best_by_valid, status=None):
    status = status or ["success"] * len(vals)
    return {"config_name": "cfg", "best_seed_by_valid": best_by_valid,
            "seed_results": [{"seed": i, "status": s, "valid_ll_per_sample": v}
                             for i, (v, s) in enumerate(zip(vals, status))]}


def test_choose_seed_best_valid():
    seed, res = m04s.choose_seed(_summary([-3.0, -1.0, -2.0], 1))
    assert seed == 1 and res["valid_ll_per_sample"] == -1.0


def test_choose_seed_skips_failed():
    s = _summary([-3.0, None, -2.0], 2, ["success", "failed", "success"])
    assert m04s.choose_seed(s)[0] == 2


def test_choose_seed_failed_seed_with_value_is_ignored():
    s = _summary([-3.0, -1.0, -2.0], 2, ["success", "failed", "success"])
    assert m04s.choose_seed(s)[0] == 2


def test_choose_seed_rejects_mismatch():
    with pytest.raises(ValueError, match="best_seed_by_valid"):
        m04s.choose_seed(_summary([-3.0, -1.0, -2.0], 0))


def test_choose_seed_rejects_tie():
    with pytest.raises(ValueError, match="tied"):
        m04s.choose_seed(_summary([-1.0, -1.0], 0))


def test_choose_seed_rejects_all_failed():
    with pytest.raises(ValueError, match="no successful"):
        m04s.choose_seed(_summary([None], 0, ["failed"]))


def test_resolves_inside_follows_links(tmp_path):
    prot = tmp_path / "published"
    (prot / "stage").mkdir(parents=True)
    side_out = tmp_path / "side"
    side_out.mkdir()
    os.symlink(prot / "stage", side_out / "stage")
    assert m04s.resolves_inside(side_out / "stage" / "new" / "dir", prot)
    assert not m04s.resolves_inside(side_out / "real" / "new", prot)
    assert m04s.resolves_inside(prot, prot)
    assert not m04s.resolves_inside(tmp_path / "published2" / "x", prot)


def _published(tmp_path):
    pub = tmp_path / "pub" / "output"
    for st in side.LINKED + ("m10_03_projected",):
        (pub / st).mkdir(parents=True)
    return pub


def test_setup_layout_vt095(tmp_path):
    pub = _published(tmp_path)
    side.setup(str(tmp_path / "root"), "cfg", "0.95", ["sub-01"], str(pub),
               link_m10_03=True, published_final=False)
    out = tmp_path / "root" / "cfg" / "output"
    for st in side.LINKED + ("m10_03_projected",):
        assert (out / st).is_symlink()
    for st in side.REGEN:
        assert (out / st).is_dir() and not (out / st).is_symlink()
    assert (out / "04_combined_hdphmm" / side.PARC / "sub-01" / "final" / "vt0.95").is_dir()
    assert not (out / "05a_recurrence_analysis" / side.PARC / "sub-01" / "vt0.9").exists()


def test_setup_alias_vt090(tmp_path):
    pub = _published(tmp_path)
    side.setup(str(tmp_path / "root"), "cfg", "0.90", ["sub-01"], str(pub),
               link_m10_03=False, published_final=False)
    out = tmp_path / "root" / "cfg" / "output"
    assert (out / "m10_03_projected").is_dir() and not (out / "m10_03_projected").is_symlink()
    rec = out / "05a_recurrence_analysis" / side.PARC / "sub-01"
    assert (rec / "vt0.90").is_dir() and not (rec / "vt0.90").is_symlink()
    assert (rec / "vt0.9").is_symlink() and os.readlink(rec / "vt0.9") == "vt0.90"
    fin = out / "04_combined_hdphmm" / side.PARC / "sub-01" / "final"
    assert os.readlink(fin / "vt0.9") == "vt0.90"
    a4 = out / "05e_temporal_trend_a4" / side.PARC / "sub-01"
    assert (a4 / "vt0.9").is_dir() and not (a4 / "vt0.9").is_symlink()
    assert os.readlink(a4 / "vt0.90") == "vt0.9"
    # a file written by a4 under vt0.9 is visible to readers of vt0.90
    (a4 / "vt0.9" / "state_flags.csv").write_text("x")
    assert (a4 / "vt0.90" / "state_flags.csv").read_text() == "x"
    # rerunning setup is idempotent
    side.setup(str(tmp_path / "root"), "cfg", "0.90", ["sub-01"], str(pub),
               link_m10_03=False, published_final=False)


def test_check_fails_on_link_into_dataset(tmp_path):
    pub = _published(tmp_path)
    (pub / "05a_recurrence_analysis").mkdir()
    out = tmp_path / "root" / "cfg" / "output"
    out.mkdir(parents=True)
    os.symlink(pub / "05a_recurrence_analysis", out / "05a_recurrence_analysis")
    with pytest.raises(SystemExit, match="resolve into the dataset"):
        side.check(str(tmp_path / "root"), "cfg", "0.95", ["sub-01"], str(pub), True)


def test_published_final_links(tmp_path):
    pub = _published(tmp_path)
    fin = pub / "04_combined_hdphmm" / side.PARC / "sub-01" / "final" / "vt0.95"
    fin.mkdir(parents=True)
    for fn in side.FINAL_FILES:
        (fin / fn).write_text(fn)
    side.setup(str(tmp_path / "root"), "gate", "0.95", ["sub-01"], str(pub),
               link_m10_03=True, published_final=True)
    dst = tmp_path / "root" / "gate" / "output" / "04_combined_hdphmm" / side.PARC / \
        "sub-01" / "final" / "vt0.95"
    assert not dst.is_symlink()
    for fn in side.FINAL_FILES:
        assert (dst / fn).is_symlink() and (dst / fn).read_text() == fn


def test_state_covars_for_save():
    import numpy as np
    d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    full = np.stack([np.diag(r) for r in d])
    assert np.array_equal(m04s.state_covars_for_save(full, "diag"), d)
    assert m04s.state_covars_for_save(d, "diag") is not None
    assert np.array_equal(m04s.state_covars_for_save(d, "diag"), d)
    assert m04s.state_covars_for_save(full, "full").shape == (2, 3, 3)
    bad = full.copy()
    bad[0, 0, 1] = 1e-12
    with pytest.raises(ValueError, match="off-diagonal"):
        m04s.state_covars_for_save(bad, "diag")
