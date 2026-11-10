"""Tests for script/preproc_by_season.py (Table S7)."""
import importlib.util
import itertools
from math import comb
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "preproc_by_season.py"
_spec = importlib.util.spec_from_file_location("preproc_by_season", _PATH)
pbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbs)


def test_report_parsing_reads_release_and_counts():
    text = ("<p>fMRIPrep version: 20.2.3</p>"
            "<li>Slice timing correction: Applied</li><li>Slice timing correction: Applied</li>")
    assert pbs.parse_report(text) == ("20.2.3", 2, 0)
    assert pbs.parse_report("no version here") == (None, 0, 0)


@pytest.mark.parametrize("release,app,not_app,sidecar,expected", [
    ("20.2.3", 3, 0, None, "STC"),
    ("20.2.3", 0, 4, None, "noSTC"),
    ("20.2.3", 1, 1, None, "mixed?"),       # a report with both is not decided
    ("20.2.3", 0, 0, None, "mixed?"),       # nor is a report with neither
    ("20.2.5", 0, 0, False, "noSTC"),       # later releases: the run sidecar decides
    ("20.2.6", 0, 0, True, "STC"),
    ("20.2.5", 5, 0, "missing", "missing"),  # a 20.2.5 report line does not override the sidecar
])
def test_stc_status(release, app, not_app, sidecar, expected):
    assert pbs.stc_status(release, app, not_app, sidecar) == expected


def test_batch_l1_matches_single_mask():
    rng = np.random.default_rng(0)
    F = rng.dirichlet(np.ones(5), size=12)
    masks = np.stack([rng.permutation(np.r_[np.ones(4, bool), np.zeros(8, bool)]) for _ in range(7)])
    np.testing.assert_allclose(pbs.l1_batch(F, masks), [pbs.l1_from_mask(F, m) for m in masks], atol=1e-14)


def test_session_null_enumerates_whole_sessions():
    # 5 sessions of 2 runs; group A = sessions 0 and 1, which differ strongly from the rest.
    F = np.array([[1, 0]] * 4 + [[0, 1]] * 6, float)
    units = [u for u in range(5) for _ in range(2)]
    null, n = pbs.session_null(F, units, {u: u < 2 for u in range(5)})
    assert n == comb(5, 2) == len(null)
    obs = pbs.l1_from_mask(F, np.array([u < 2 for u in units]))
    # the observed split is the most extreme assignment, so p equals the floor
    assert np.mean(null >= obs - 1e-12) == pytest.approx(1 / n)
    # every assignment keeps sessions whole: each null value is one of the C(5,2) session splits
    expected = sorted(pbs.l1_from_mask(F, np.isin(units, c)) for c in itertools.combinations(range(5), 2))
    np.testing.assert_allclose(sorted(null), expected, atol=1e-14)


def test_block_reference_uses_only_groups_of_at_least_2k_runs():
    F = np.eye(3)[[0, 0, 1, 1, 2, 2, 0, 1, 2, 0]]
    groups = {("S1", "x"): list(range(6)), ("S2", "y"): [6, 7, 8]}  # second group is too short for k=2
    ref, first = pbs.block_reference(F, groups, k=2)
    assert len(ref) == 5                      # 6 - 2 + 1 blocks, all from the first group
    assert [g for g, _ in first] == [("S1", "x")]
    assert first[0][1] == pytest.approx(pbs.l1_from_mask(F[:6], np.r_[True, True, False, False, False, False]))
