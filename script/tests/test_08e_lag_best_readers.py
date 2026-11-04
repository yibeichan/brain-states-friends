"""Tests that the Figure 5 transfer panels and Figure S7 read the best-lag 08e outputs.

08e writes the best-lag transfer to ``<stim>_<model>_lag-best`` and keeps the
older lag-0 outputs in ``<stim>_<model>``. These tests pin:
  (a) the shared reader returns only the best-lag folder's JSONs, one per
      expected participant, by exact path (no ``sub-*`` glob);
  (b) it raises naming any missing participant, on a JSON that does not
      record the Friends-best lag, on an excluded lag, and on a lag that
      differs from the best lag recomputed from the participant's current
      08d D1 output;
  (c) 08e writes to the folder the reader looks in;
  (d) fig_F5_within_friends (Fig 5D-G) and 08e_plots (Fig S7) both go through
      the shared reader, so neither can drift back to the lag-0 folder;
  (e) Fig 5D decides "not testable" from the stimulus × model check, and a
      testable cell with missing outputs raises instead of being drawn "n/t".
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from utils.transformer_analysis import (  # noqa: E402
    D3A_LAG_BEST_SUFFIX,
    D3A_VT,
    d3a_expected_subjects,
    friends_d1_dir,
    friends_d1_file,
    load_d3a_lag_best,
)

PARC = "atlas-4S156Parcels"
BEST_LAG = 3
ELIGIBLE = (0, 1, 2)          # current 05e content-eligible states
INTERSECTION = [0, 1]         # recorded in each test D3a file


def _d1_dir(scratch):
    return Path(friends_d1_dir(str(scratch), PARC))


def _scratch_of(d1_dir):
    """The scratch root of a D1 dir made by _d1_dir (output/08d/<parc>)."""
    return Path(d1_dir).parents[2]


def _write_flags(scratch, sub, eligible=ELIGIBLE, other=(7,)):
    """05e state_flags.csv where ``eligible`` are content-eligible."""
    d = (Path(scratch) / "output" / "05e_temporal_trend_a4" / PARC / sub
         / f"vt{D3A_VT}")
    d.mkdir(parents=True, exist_ok=True)
    rows = ["state,summary_category"]
    rows += [f"{s},eligible_for_content_analysis" for s in eligible]
    rows += [f"{s},run_onset_anchored" for s in other]
    (d / "state_flags.csv").write_text("\n".join(rows) + "\n")


def _read(root, d1, *args, **kwargs):
    return load_d3a_lag_best(root, d1, *args, scratch_dir=str(_scratch_of(d1)),
                             parcellation=PARC, **kwargs)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(root, sub, folder, kind, stim, model, payload):
    d = root / sub / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"D3a_{kind}_{stim}_{model}.json"
    p.write_text(json.dumps(payload))
    return p


def _write_d1(d1_dir, sub, model, best_lag=BEST_LAG):
    """A D1 profile whose best lag (lag 0 excluded) is ``best_lag``, and the
    participant's current 05e flags under the same scratch root."""
    _write_flags(_scratch_of(d1_dir), sub)
    p = Path(friends_d1_file(d1_dir, sub, model))
    p.parent.mkdir(parents=True, exist_ok=True)
    results = {f"lag_{lag}": {"0": {"balanced_accuracy": 0.05}} for lag in range(5)}
    results["lag_0"]["0"]["balanced_accuracy"] = 0.9   # ignored: lag 0 excluded
    results[f"lag_{best_lag}"]["1"] = {"balanced_accuracy": 0.2}
    p.write_text(json.dumps({"results": results}))
    return p


def _elig_fields(states=INTERSECTION):
    """Eligibility fields as written by older 08e code (all real files)."""
    return {"eligibility_source": "05e_a4", "intersection_states": list(states),
            "n_classes": len(states)}


def _best(extra=None, lag=BEST_LAG):
    return {"lag": lag, "lag_source": "friends_best", **_elig_fields(),
            **(extra or {})}


def _two_lags(root, d1, stim="movie10", model="m", subjects=("sub-01", "sub-02")):
    """Each participant at lag 0 (acc 0.1) and at the best lag (acc 0.9)."""
    for sub in subjects:
        _write_d1(d1, sub, model)
        _write(root, sub, f"{stim}_{model}", "transfer", stim, model,
               {"lag": 0, "lag_source": "explicit", "acc": 0.1})
        _write(root, sub, f"{stim}_{model}{D3A_LAG_BEST_SUFFIX}", "transfer",
               stim, model, _best({"acc": 0.9}))


SUBS = ["sub-01", "sub-02"]


def test_reader_returns_only_best_lag_jsons(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1)
    out = _read(root, d1, "movie10", "m", subjects=SUBS)
    assert [sub for sub, _ in out] == SUBS
    assert all(d["acc"] == 0.9 for _, d in out)


def test_reader_ignores_folders_outside_the_expected_participants(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1, subjects=("sub-01", "sub-02", "sub-06_old"))
    out = _read(root, d1, "movie10", "m", subjects=SUBS)
    assert [sub for sub, _ in out] == SUBS


def test_reader_raises_naming_every_missing_participant(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1, subjects=("sub-01",))
    with pytest.raises(FileNotFoundError) as exc:
        _read(root, d1, "movie10", "m",
                          subjects=["sub-01", "sub-02", "sub-03"])
    assert "sub-02, sub-03" in str(exc.value)


def test_reader_raises_when_nothing_on_disk(tmp_path):
    with pytest.raises(FileNotFoundError, match="sub-01"):
        _read(tmp_path, _d1_dir(tmp_path), "movie10", "m")


def test_expected_participants_follow_the_data_provenance_rule():
    assert d3a_expected_subjects("movie10") == [f"sub-0{i}" for i in range(1, 7)]
    for stim in ("harrypotter", "petitprince_fr", "petitprince_en"):
        assert d3a_expected_subjects(stim) == ["sub-01", "sub-02", "sub-03",
                                               "sub-05", "sub-06"]


def test_reader_rejects_json_without_friends_best_lag(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", {"lag": BEST_LAG, "lag_source": "explicit"})
    with pytest.raises(ValueError, match="friends_best"):
        _read(root, d1, "movie10", "m", subjects=["sub-01"])


def test_reader_rejects_lag0_file_labelled_friends_best(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", _best(lag=0))
    with pytest.raises(ValueError, match="not a candidate"):
        _read(root, d1, "movie10", "m", subjects=["sub-01"])


def test_reader_rejects_file_whose_lag_is_no_longer_the_d1_best(tmp_path):
    """A D1 rerun that moves the peak (lag 3 -> 4) invalidates the old file."""
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1)
    _write_d1(d1, "sub-02", "m", best_lag=4)
    with pytest.raises(ValueError, match="lag 3 on disk vs best lag 4"):
        _read(root, d1, "movie10", "m", subjects=SUBS)


def test_reader_message_names_both_sides_and_the_resubmit_remedy(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1)
    _write_d1(d1, "sub-02", "m", best_lag=4)
    with pytest.raises(ValueError) as exc:
        _read(root, d1, "movie10", "m", subjects=SUBS)
    msg = str(exc.value)
    assert "sub-02" in msg and "Either side may be stale" in msg
    assert "--array=1" in msg and "without FORCE" in msg
    assert msg.isascii()


def test_reader_rejects_intersection_state_no_longer_eligible(tmp_path):
    """Older files (no content_eligible list): an intersection state that
    05e no longer calls content-eligible is caught."""
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _two_lags(root, d1)
    _write_flags(tmp_path, "sub-02", eligible=(0, 2))
    with pytest.raises(ValueError, match=r"sub-02.*intersection states \[1\]"):
        _read(root, d1, "movie10", "m", subjects=SUBS)


def test_reader_rejects_state_count_that_disagrees_with_the_states(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "per_subset",
           "movie10", "m", _best({"n_classes_intersection": 3}))
    with pytest.raises(ValueError, match="n_classes_intersection=3"):
        _read(root, d1, "movie10", "m", kind="per_subset", subjects=["sub-01"])


def test_reader_rejects_other_eligibility_source(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", _best({"eligibility_source": "sub_hrf_fallback"}))
    with pytest.raises(ValueError, match="eligibility_source"):
        _read(root, d1, "movie10", "m", subjects=["sub-01"])


def test_reader_rejects_file_without_intersection_states(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    payload = _best()
    del payload["intersection_states"]
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", payload)
    with pytest.raises(ValueError, match="no intersection_states"):
        _read(root, d1, "movie10", "m", subjects=["sub-01"])


@pytest.mark.parametrize("recorded, ok", [
    (list(ELIGIBLE), True),
    ([0, 1], False),            # 05e added state 2 since: caught only here
    ([0, 1, 2, 5], False),
])
def test_reader_requires_recorded_content_eligible_to_equal_05e(tmp_path, recorded, ok):
    """New files record the full content_eligible list; it must be equal."""
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", _best({"input_identity": {"content_eligible": recorded}}))
    if ok:
        assert len(_read(root, d1, "movie10", "m", subjects=["sub-01"])) == 1
    else:
        with pytest.raises(ValueError, match="content_eligible differs"):
            _read(root, d1, "movie10", "m", subjects=["sub-01"])


def test_reader_raises_when_d1_is_missing(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", _best())
    with pytest.raises(FileNotFoundError, match="D1"):
        _read(root, d1, "movie10", "m", subjects=["sub-01"])


def test_reader_reads_per_subset_kind(tmp_path):
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "per_subset",
           "movie10", "m", _best({"x": 1}))
    _write(root, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", _best({"x": 2}))
    out = _read(root, d1, "movie10", "m", kind="per_subset",
                            subjects=["sub-01"])
    assert [d["x"] for _, d in out] == [1]


def test_08e_writes_where_the_reader_looks(tmp_path):
    """08e's output folder is the one the reader opens."""
    m08e = _load("_m08e_readers", "08e_transformer_cross_stim_aggregate.py")
    root, d1 = tmp_path / "d3a", _d1_dir(tmp_path)
    _write_d1(d1, "sub-01", "m")
    folder = m08e.output_subdir("movie10", "m", m08e.LAG_FRIENDS_BEST)
    _write(root, "sub-01", folder, "transfer", "movie10", "m", _best({"x": 7}))
    out = _read(root, d1, "movie10", "m", subjects=["sub-01"])
    assert out[0][1]["x"] == 7


@pytest.fixture
def fig_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    return tmp_path


def _fig_d1(fig_env, model, subjects):
    d1 = fig_env / "output" / "08d_transformer_depth" / PARC
    for sub in subjects:
        _write_d1(d1, sub, model)
    return d1


ALL_SUBS = d3a_expected_subjects("movie10")


def _write_transfer_both_lags(root, subjects, stim="movie10", model="m"):
    for sub in subjects:
        for folder, acc, lag, source in (
            (f"{stim}_{model}", 0.10, 0, "explicit"),
            (f"{stim}_{model}{D3A_LAG_BEST_SUFFIX}", 0.90, BEST_LAG, "friends_best"),
        ):
            _write(root, sub, folder, "transfer", stim, model, {
                "lag": lag, "lag_source": source, **_elig_fields(),
                "per_layer": {
                    "0": {"balanced_accuracy": acc, "chance_level": 0.05, "p_perm": 0.001},
                    "1": {"balanced_accuracy": 0.06, "chance_level": 0.05, "p_perm": 0.5},
                },
            })


def test_fig5_transfer_cell_reads_best_lag(fig_env):
    f5 = _load("_fig_f5_readers", "fig_F5_within_friends.py")
    root = fig_env / "transfer_root"
    f5.TRANSFER_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    _write_transfer_both_lags(root, ALL_SUBS)
    cell = f5.load_transfer_cell("movie10", "m", 2)
    assert cell is not None
    assert cell["n"] == len(ALL_SUBS)
    assert cell["acc_median"] == pytest.approx(0.85)


def test_fig5_transfer_cell_raises_on_partial_cohort(fig_env):
    f5 = _load("_fig_f5_partial", "fig_F5_within_friends.py")
    root = fig_env / "transfer_root"
    f5.TRANSFER_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    _write_transfer_both_lags(root, ["sub-04"])
    with pytest.raises(FileNotFoundError, match="sub-01"):
        f5.load_transfer_cell("movie10", "m", 2)


def test_fig5_panel_d_not_testable_comes_from_the_modality_check(fig_env, monkeypatch):
    """HP x audio is drawn "n/t" without reading files; a testable cell with
    nothing on disk (llama x Petit Prince FR here) raises instead."""
    f5 = _load("_fig_f5_paneld", "fig_F5_within_friends.py")
    f5.TRANSFER_ROOT = fig_env / "empty_root"
    asked = []

    def fake_cell(stim, model, n_layers):
        asked.append((stim, model))
        if (stim, model) == ("petitprince_fr", "llama-3.2-3b"):
            return f5.load_d3a_lag_best(f5.TRANSFER_ROOT, f5.D1_ROOT, stim, model,
                                        **f5.ELIGIBILITY_SOURCE)
        return {"n": 5, "median": 0.5, "iqr": (0.4, 0.6), "acc_median": 0.01,
                "n_sig": 5, "majority_sig": True}

    monkeypatch.setattr(f5, "load_transfer_cell", fake_cell)
    monkeypatch.setattr(f5, "_save", lambda fig, stem: f5.plt.close(fig))
    with pytest.raises(FileNotFoundError, match="sub-01"):
        f5.render_panel_d()
    assert ("harrypotter", "w2v-bert-2.0") not in asked
    assert ("harrypotter", "dinov2-large") not in asked


def test_fig5_panel_d_testable_cell_without_data_raises(fig_env, monkeypatch):
    """A testable cell that yields no data is an error, never an "n/t" cell."""
    f5 = _load("_fig_f5_paneld_none", "fig_F5_within_friends.py")
    monkeypatch.setattr(f5, "load_transfer_cell", lambda stim, model, n: None)
    monkeypatch.setattr(f5, "_save", lambda fig, stem: f5.plt.close(fig))
    with pytest.raises(ValueError, match="no finite layer"):
        f5.render_panel_d()


def test_s7_pooled_curves_read_best_lag(fig_env):
    s7 = _load("_s7_readers", "08e_plots.py")
    root = fig_env / "d3a_root"
    s7.D3A_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    _write_transfer_both_lags(root, ALL_SUBS)
    curves, subs = s7.load_pooled_curves("m", "movie10", 1)
    assert subs == ALL_SUBS
    assert curves[0, 0] == pytest.approx(0.85)


def _write_per_subset_both_lags(root, subjects, model="m"):
    for sub in subjects:
        for folder, acc, lag, source in (
            (f"movie10_{model}", 0.10, 0, "explicit"),
            (f"movie10_{model}{D3A_LAG_BEST_SUFFIX}", 0.90, BEST_LAG, "friends_best"),
        ):
            _write(root, sub, folder, "per_subset", "movie10", model, {
                "lag": lag, "lag_source": source, **_elig_fields(),
                "chance_level_full": 0.05,
                "per_subset": {"wolf": {"0": {"balanced_accuracy": acc},
                                        "1": {"balanced_accuracy": 0.06}}},
            })


def test_s7_per_film_reads_best_lag(fig_env):
    s7 = _load("_s7_perfilm", "08e_plots.py")
    root = fig_env / "d3a_root"
    s7.D3A_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    _write_per_subset_both_lags(root, ALL_SUBS)
    curves, subs, source = s7.load_per_subset_curves("m", "movie10", 2, "wolf")
    assert subs == ALL_SUBS and source == "production"
    assert curves[:, 0] == pytest.approx([0.85] * len(ALL_SUBS))


def test_s7_per_film_raises_when_best_lag_outputs_are_missing(fig_env):
    """Only lag-0 per-film files on disk: no fallback to them, an error."""
    s7 = _load("_s7_nofallback", "08e_plots.py")
    root = fig_env / "d3a_root"
    s7.D3A_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    for sub in ALL_SUBS:
        _write(root, sub, "movie10_m", "per_subset", "movie10", "m", {
            "lag": 0, "lag_source": "explicit", "chance_level_full": 0.05,
            "per_subset": {"wolf": {"0": {"balanced_accuracy": 0.5}}},
        })
    with pytest.raises(FileNotFoundError):
        s7.load_per_subset_curves("m", "movie10", 1, "wolf")


def test_fig5_per_film_reads_best_lag(fig_env):
    f5 = _load("_fig_f5_perfilm", "fig_F5_within_friends.py")
    root = fig_env / "transfer_root"
    f5.TRANSFER_ROOT = root
    _fig_d1(fig_env, "m", ALL_SUBS)
    _write_per_subset_both_lags(root, ALL_SUBS)
    series = f5.collect_perfilm("m", 2)
    assert [s["label"] for s in series] == ["Wolf of Wall St."]
    assert series[0]["n"] == len(ALL_SUBS)
    assert series[0]["mean"][0] == pytest.approx(0.85)


def test_wrapper_rules_match_the_reader_constants():
    """The 08e wrapper's participant list, sub-04 skip rule and VT default
    are the ones the reader assumes (D3A_SUBJECTS, D3A_STIMULI_WITHOUT_SUB04,
    D3A_VT)."""
    import re
    from utils.transformer_analysis import D3A_STIMULI_WITHOUT_SUB04, D3A_SUBJECTS
    sh = (_SCRIPT_DIR / "08e_transformer_cross_stim_aggregate.sh").read_text()
    subjects = re.search(r"^SUBJECTS=\(([^)]*)\)", sh, re.M).group(1).split()
    assert tuple(subjects) == D3A_SUBJECTS
    skip = [line for line in sh.splitlines()
            if line.startswith("if ") and '"sub-04"' in line]
    assert len(skip) == 1
    assert set(re.findall(r'== "([a-z_]+)"', skip[0])) == set(
        D3A_STIMULI_WITHOUT_SUB04)
    assert re.search(r'^VT="\$\{VT:-([0-9.]+)\}"', sh, re.M).group(1) == D3A_VT
