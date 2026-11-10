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


# ---------------------------------------------------------------- fail-closed guards

import json as _json
import pickle as _pickle


def _tree(tmp_path, reports, runs, sidecars=None):
    """Fake fMRIPrep and output trees for sub-02.

    reports: {ses: (release, n_applied, n_not)}; runs: {run: [ses, ...]} (several = duplicate);
    sidecars: {(ses, run): SliceTimingCorrected value}.
    """
    fp = tmp_path / "fmriprep"
    out = tmp_path / "output"
    fp.mkdir()
    for ses, (rel, a, n) in reports.items():
        body = (f"fMRIPrep version: {rel}" if rel else "") + " Slice timing correction: Applied" * a \
            + " Slice timing correction: Not applied" * n
        (fp / f"sub-02_{ses}.html").write_text(body)
    for (ses, run), val in (sidecars or {}).items():
        d = fp / "sub-02" / ses / "func"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"sub-02_{ses}_task-{run}_space-fsLR_den-91k_bold.json").write_text(_json.dumps({"SliceTimingCorrected": val}))
    dec_dir = out / "04_combined_hdphmm" / pbs.PARC / "sub-02" / "final" / "vt0.95"
    dec_dir.mkdir(parents=True)
    with open(dec_dir / "decoded_states.pkl", "wb") as fh:
        _pickle.dump({run: np.zeros(10, int) for run in runs}, fh)
    ts = out / "02_parcel_ts_avg" / pbs.PARC / "sub-02"
    ts.mkdir(parents=True)
    for run, sess in runs.items():
        for ses in sess:
            (ts / f"sub-02_{ses}_task-{run}_parcel_avg.npy").write_bytes(b"")
    return str(fp), str(out)


def test_duplicate_run_uses_its_pinned_session(tmp_path, monkeypatch):
    monkeypatch.setattr(pbs, "SUBJECTS", ["sub-02"])
    fp, out = _tree(tmp_path, {"ses-036": ("20.2.3", 0, 3), "ses-037": ("20.2.3", 2, 0)},
                    {"s05e09a": ["ses-036", "ses-037"]})  # pinned to ses-037 in DUPLICATE_RUN_SESSIONS
    (row,) = pbs.build_run_table(fp, out)
    assert (row["ses"], row["stc"]) == (37, "STC")


def test_unlisted_duplicate_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pbs, "SUBJECTS", ["sub-02"])
    fp, out = _tree(tmp_path, {"ses-001": ("20.2.3", 1, 0), "ses-002": ("20.2.3", 1, 0)},
                    {"s01e01a": ["ses-001", "ses-002"]})
    with pytest.raises(ValueError):
        pbs.build_run_table(fp, out)


def test_missing_sidecar_or_report_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pbs, "SUBJECTS", ["sub-02"])
    fp, out = _tree(tmp_path, {"ses-040": ("20.2.5", 0, 0)},           # no sidecar for this run
                    {"s05e10a": ["ses-040"], "s05e11a": ["ses-041"]})  # and no report for ses-041
    rows = {r["run"]: r for r in pbs.build_run_table(fp, out)}
    assert rows["s05e10a"]["stc"] == "nofile"
    assert rows["s05e11a"]["version"] is None
    with pytest.raises(RuntimeError, match="2 run"):
        pbs.validate_rows(list(rows.values()))


def _rows(spec):
    return [dict(sub=s, season=se, version=v, stc=c, run=f"r{i}") for i, (s, se, v, c) in enumerate(spec)]


def test_contrast_list_must_match_the_mixed_seasons():
    rows = _rows([("sub-01", 1, "20.2.3", "STC"), ("sub-01", 1, "20.2.5", "noSTC"),
                  ("sub-01", 2, "20.2.5", "noSTC"), ("sub-01", 2, "20.2.6", "noSTC")])
    ok = [("sub-01", 1, "20.2.3/STC", "20.2.5/noSTC", "k"), ("sub-01", 2, "20.2.6/noSTC", "20.2.5/noSTC", "k")]
    pbs.validate_contrasts(rows, ok)
    with pytest.raises(RuntimeError, match="S2: mixed but no contrast"):
        pbs.validate_contrasts(rows, ok[:1])
    with pytest.raises(RuntimeError, match="not in that season"):
        pbs.validate_contrasts(rows, ok + [("sub-01", 2, "20.2.3/STC", "20.2.5/noSTC", "k")])


def test_mixed_preprocessing_within_a_session_is_refused():
    rr = [dict(lab="20.2.3/STC"), dict(lab="20.2.5/noSTC"), dict(lab="20.2.5/noSTC")]
    pbs.check_sessions_uniform(rr, [(1, 1), (1, 2), (1, 2)])
    with pytest.raises(RuntimeError, match="within session"):
        pbs.check_sessions_uniform(rr, [(1, 1), (1, 1), (1, 2)])


def test_specificity_check_fails_on_a_mismatch():
    F = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    seasons = [1, 1, 2]
    ref = [0.5, 1.0, 0.5]  # per-state max - min of season recurrence (FO > 0.02)
    assert pbs.check_specificity(F, seasons, ref) == 0.0
    with pytest.raises(RuntimeError, match="specificity"):
        pbs.check_specificity(F, seasons, [0.5, 0.9, 0.5])


def test_nothing_is_written_when_a_check_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(pbs, "build_run_table", lambda *a: [dict(sub="sub-01", run="s01e01a", season=1, episode=1,
                                                                part="a", ses=1, version=None, stc="noSTC", n_tr=5)])
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        pbs.main(["--out_dir", str(out)])
    assert not any(out.iterdir())
