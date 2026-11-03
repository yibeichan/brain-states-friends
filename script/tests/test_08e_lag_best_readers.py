"""Tests that the Figure 5 transfer panels and Figure S7 read the best-lag 08e outputs.

08e writes the best-lag transfer to ``<stim>_<model>_lag-best`` and keeps the
older lag-0 outputs in ``<stim>_<model>``. These tests pin:
  (a) the shared reader returns only the best-lag folder's JSONs, one per
      participant, and fails loudly on a JSON that does not record the
      Friends-best lag;
  (b) 08e writes to the folder the reader looks in;
  (c) fig_F5_within_friends (Fig 5D-G) and 08e_plots (Fig S7) both go through
      the shared reader, so neither can drift back to the lag-0 folder;
  (d) Fig S7 no longer fills a missing per-film panel with lag-0 estimates.
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
    load_d3a_lag_best,
)


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


def _two_lags(root, stim="movie10", model="m"):
    """sub-01 and sub-02 at lag 0 (acc 0.1) and at the best lag (acc 0.9)."""
    for sub in ("sub-01", "sub-02"):
        _write(root, sub, f"{stim}_{model}", "transfer", stim, model,
               {"lag": 0, "lag_source": "explicit", "acc": 0.1})
        _write(root, sub, f"{stim}_{model}{D3A_LAG_BEST_SUFFIX}", "transfer",
               stim, model, {"lag": 3, "lag_source": "friends_best", "acc": 0.9})


def test_reader_returns_only_best_lag_jsons(tmp_path):
    _two_lags(tmp_path)
    out = load_d3a_lag_best(tmp_path, "movie10", "m")
    assert [sub for sub, _ in out] == ["sub-01", "sub-02"]
    assert all(d["acc"] == 0.9 for _, d in out)


def test_reader_rejects_json_without_friends_best_lag(tmp_path):
    _write(tmp_path, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", {"lag": 0, "lag_source": "explicit"})
    with pytest.raises(ValueError, match="friends_best"):
        load_d3a_lag_best(tmp_path, "movie10", "m")


def test_reader_reads_per_subset_kind(tmp_path):
    _write(tmp_path, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "per_subset",
           "movie10", "m", {"lag": 3, "lag_source": "friends_best", "x": 1})
    _write(tmp_path, "sub-01", f"movie10_m{D3A_LAG_BEST_SUFFIX}", "transfer",
           "movie10", "m", {"lag": 3, "lag_source": "friends_best", "x": 2})
    out = load_d3a_lag_best(tmp_path, "movie10", "m", kind="per_subset")
    assert [d["x"] for _, d in out] == [1]


def test_reader_returns_empty_list_when_nothing_on_disk(tmp_path):
    assert load_d3a_lag_best(tmp_path, "movie10", "m") == []


def test_08e_writes_where_the_reader_looks():
    m08e = _load("_m08e_readers", "08e_transformer_cross_stim_aggregate.py")
    assert m08e.output_subdir("movie10", "m", m08e.LAG_FRIENDS_BEST) == \
        f"movie10_m{D3A_LAG_BEST_SUFFIX}"


@pytest.fixture
def fig_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    return tmp_path


def test_fig5_transfer_cell_reads_best_lag(fig_env):
    f5 = _load("_fig_f5_readers", "fig_F5_within_friends.py")
    root = fig_env / "transfer_root"
    f5.TRANSFER_ROOT = root
    for sub in ("sub-01", "sub-02"):
        for folder, acc in (("movie10_m", 0.10), (f"movie10_m{D3A_LAG_BEST_SUFFIX}", 0.90)):
            lag_source = "friends_best" if folder.endswith(D3A_LAG_BEST_SUFFIX) else "explicit"
            _write(root, sub, folder, "transfer", "movie10", "m", {
                "lag_source": lag_source,
                "per_layer": {
                    "0": {"balanced_accuracy": acc, "chance_level": 0.05, "p_perm": 0.001},
                    "1": {"balanced_accuracy": 0.06, "chance_level": 0.05, "p_perm": 0.5},
                },
            })
    cell = f5.load_transfer_cell("movie10", "m", 2)
    assert cell is not None
    assert cell["n"] == 2
    assert cell["acc_median"] == pytest.approx(0.85)


def test_s7_pooled_curves_read_best_lag(fig_env):
    s7 = _load("_s7_readers", "08e_plots.py")
    root = fig_env / "d3a_root"
    s7.D3A_ROOT = root
    _two_lags_per_layer = {"movie10_m": 0.10, f"movie10_m{D3A_LAG_BEST_SUFFIX}": 0.90}
    for folder, acc in _two_lags_per_layer.items():
        lag_source = "friends_best" if folder.endswith(D3A_LAG_BEST_SUFFIX) else "explicit"
        _write(root, "sub-01", folder, "transfer", "movie10", "m", {
            "lag_source": lag_source,
            "per_layer": {"0": {"balanced_accuracy": acc, "chance_level": 0.05}},
        })
    curves, subs = s7.load_pooled_curves("m", "movie10", 1)
    assert subs == ["sub-01"]
    assert curves[0, 0] == pytest.approx(0.85)


def test_s7_per_film_has_no_lag0_fallback(fig_env):
    s7 = _load("_s7_nofallback", "08e_plots.py")
    s7.D3A_ROOT = fig_env / "d3a_root"
    prov = fig_env / "provisional"
    prov.mkdir()
    s7.PROVISIONAL_PERFILM_ROOT = prov
    (prov / "sub-01_m_movie10_perfilm.json").write_text(json.dumps({
        "chance_level": 0.05,
        "per_layer": {"0": {"wolf": {"balanced_accuracy": 0.5}}},
    }))
    curves, subs, source = s7.load_per_subset_curves("m", "movie10", 1, "wolf")
    assert curves is None and subs == [] and source is None


def test_fig5_per_film_reads_best_lag(fig_env):
    f5 = _load("_fig_f5_perfilm", "fig_F5_within_friends.py")
    root = fig_env / "transfer_root"
    f5.TRANSFER_ROOT = root
    for sub in ("sub-01", "sub-02"):
        for folder, acc in (("movie10_m", 0.10), (f"movie10_m{D3A_LAG_BEST_SUFFIX}", 0.90)):
            lag_source = "friends_best" if folder.endswith(D3A_LAG_BEST_SUFFIX) else "explicit"
            _write(root, sub, folder, "per_subset", "movie10", "m", {
                "lag_source": lag_source,
                "chance_level_full": 0.05,
                "per_subset": {"wolf": {"0": {"balanced_accuracy": acc},
                                        "1": {"balanced_accuracy": 0.06}}},
            })
    series = f5.collect_perfilm("m", 2)
    assert [s["label"] for s in series] == ["Wolf of Wall St."]
    assert series[0]["n"] == 2
    assert series[0]["mean"][0] == pytest.approx(0.85)
