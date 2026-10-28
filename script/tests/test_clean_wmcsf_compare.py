"""clean_wmcsf_compare.py: how the seed fits of the two variants are wired into the
verdict and the permutation test, stage 1's vt pick, and the output checks.

Claims tested: the verdict is judged against within-variant seed pairs only; the
selected comparison uses each variant's best seed; the permutation test groups the
pooled fits by variant; stage 1's vt is the per-dimension validation LL pick over
the stage-1 grid, skipping configs without fits; a decode mismatch is reported as
a failed check; the JSON is written whole. Synthetic data only.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clean_wmcsf_compare as cmp  # noqa: E402
from config.combined_hmm_config import build_stage1_grid, config_name  # noqa: E402
from utils.state_compare import compare_fits  # noqa: E402

N_STATES, N_PARCELS, N_TRS = 4, 30, 200


def _fit(means, decoded):
    return {"means": means, "decoded": decoded, "k_used": N_STATES}


def _variant_seeds(rng, base_means, base_labels, n_seeds, flip):
    """Seed fits around one solution: small noise on the maps, `flip` TRs relabelled."""
    seeds = {}
    for s in range(n_seeds):
        means = base_means + 0.05 * rng.normal(size=base_means.shape)
        decoded = {}
        for run, labels in base_labels.items():
            lab = labels.copy()
            idx = rng.choice(lab.size, size=flip, replace=False)
            lab[idx] = rng.integers(0, N_STATES, size=flip)
            decoded[run] = lab
        seeds[s] = _fit(means, decoded)
    return seeds


@pytest.fixture
def fits():
    rng = np.random.default_rng(7)
    means_a = rng.normal(size=(N_STATES, N_PARCELS))
    labels_a = {r: np.repeat(np.arange(N_STATES), N_TRS // N_STATES)[rng.permutation(N_TRS)]
                for r in ("r1", "r2")}
    # The clean solution: different maps and an unrelated labelling.
    means_b = rng.normal(size=(N_STATES, N_PARCELS))
    labels_b = {r: rng.integers(0, N_STATES, size=N_TRS) for r in ("r1", "r2")}
    # Seeds differ only by noise, so any seed other than the best one gives a
    # different selected comparison.
    corrupt = _variant_seeds(rng, means_a, labels_a, 3, flip=4)
    clean = _variant_seeds(rng, means_b, labels_b, 3, flip=4)
    return {"corrupt": {"seeds": corrupt, "best_seed": 2, "selected_config": "cfg"},
            "clean": {"seeds": clean, "best_seed": 1, "selected_config": "cfg"}}


def test_selected_comparison_uses_each_variant_best_seed(fits):
    out = cmp.compare_variants(fits)
    expected = compare_fits(fits["corrupt"]["seeds"][2], fits["clean"]["seeds"][1])
    assert out["selected_fits"] == expected
    assert out["same_selected_config"] is True


def test_verdict_is_judged_against_within_variant_pairs(fits):
    out = cmp.compare_variants(fits)
    within = [compare_fits(seeds[i], seeds[j])
              for v in ("corrupt", "clean")
              for seeds in [fits[v]["seeds"]]
              for i, j in itertools.combinations(sorted(seeds), 2)]
    for name in ("median_r", "ari", "fo_distance"):
        vals = [w[name] for w in within]
        e = out["verdict"]["by_metric"][name]
        assert e["within_min"] == pytest.approx(min(vals))
        assert e["within_max"] == pytest.approx(max(vals))
    # The clean solution differs from the corrupt one far beyond seed noise.
    assert out["verdict"]["moved"] is True
    assert out["verdict"]["by_metric"]["ari"]["outside"] is True


def test_permutation_test_groups_fits_by_variant(fits):
    out = cmp.compare_variants(fits)
    pooled = [f for v in ("corrupt", "clean") for _, f in sorted(fits[v]["seeds"].items())]
    variant = [0] * 3 + [1] * 3
    within, cross = [], []
    for i, j in itertools.combinations(range(6), 2):
        (within if variant[i] == variant[j] else cross).append(
            compare_fits(pooled[i], pooled[j])["ari"])
    d = out["distribution_test"]
    assert d["n_splits"] == 20 and d["exact"] is True
    assert d["by_metric"]["ari"]["within_median"] == pytest.approx(np.median(within))
    assert d["by_metric"]["ari"]["cross_median"] == pytest.approx(np.median(cross))


def _write_stage1_configs(configs_dir, scores):
    """config_summary.json per stage-1 config: {vt: (mean valid LL/sample, n_pcs)}."""
    for cfg in build_stage1_grid():
        vt = cfg["variance_threshold"]
        if vt not in scores:
            continue
        ll, n_pcs = scores[vt]
        d = Path(configs_dir) / config_name(cfg)
        d.mkdir(parents=True)
        (d / "config_summary.json").write_text(json.dumps({
            "n_pcs": n_pcs, "n_seeds_successful": 5,
            "mean_valid_ll_per_sample": ll, "best_valid_ll_per_sample": ll,
            "std_valid_ll_per_sample": 0.1}))


def test_stage1_vt_is_the_per_dimension_ll_pick(tmp_path):
    # Raw LL would pick 0.99; LL per PC picks 0.90 (1.0 vs 0.5 per dimension).
    _write_stage1_configs(tmp_path / "configs", {
        0.80: (5.0, 10), 0.85: (8.0, 14), 0.90: (20.0, 20), 0.95: (24.0, 40),
        0.99: (50.0, 100)})
    assert cmp.stage1_vt(tmp_path) == (0.90, 5, 5)


def test_stage1_vt_skips_configs_without_fits(tmp_path):
    _write_stage1_configs(tmp_path / "configs", {0.85: (8.0, 10), 0.95: (5.0, 10)})
    assert cmp.stage1_vt(tmp_path) == (0.85, 2, 5)
    assert cmp.stage1_vt(tmp_path / "nothing_here") == (None, 0, 5)


def _result(n_seeds=10, matches=True, dist=None):
    return {"variants": {"corrupt": {"n_seeds": 10, "saved_decode_matches_best_seed": True},
                         "clean": {"n_seeds": n_seeds, "saved_decode_matches_best_seed": matches}},
            "distribution_test": dist or {"n_splits": 20}}


def test_failed_checks_are_listed():
    assert cmp.check_failures(_result()) == []
    assert cmp.check_failures(_result(matches=False)) == [
        "clean: the selected fit's saved decode differs from its seed model's"]
    assert cmp.check_failures(_result(n_seeds=9)) == [
        "clean: 9 final seed models, expected 10"]
    assert cmp.check_failures(_result(dist={"error": "ari: 1 of 15 seed pairs are not finite"})) == [
        "permutation test not run: ari: 1 of 15 seed pairs are not finite"]


def test_a_permutation_test_error_leaves_the_verdict(fits, monkeypatch):
    def boom(pairs, labels):
        raise ValueError("ari: 1 of 15 seed pairs are not finite")
    monkeypatch.setattr(cmp, "distribution_test", boom)
    out = cmp.compare_variants(fits)
    d = out["distribution_test"]
    assert d["error"] == "ari: 1 of 15 seed pairs are not finite"
    assert sorted(d["by_metric"]) == ["ari", "fo_distance", "median_r"]
    assert all(np.isnan(m["p"]) for m in d["by_metric"].values())
    assert out["verdict"]["moved"] is True


def test_main_writes_the_json_then_exits_1_on_a_failed_check(tmp_path, monkeypatch, fits):
    result = {"sub_id": "sub-00", "parcellation": "p"} | cmp.compare_variants(fits)
    for v in ("corrupt", "clean"):
        result["variants"][v]["stage1_selected_vt"] = 0.99
        result["variants"][v]["stage1_configs_scored"] = "5/5"
        result["variants"][v]["saved_decode_matches_best_seed"] = True
    monkeypatch.setattr(cmp, "run", lambda *a: result)
    out = tmp_path / "sub-00.json"
    argv = ["x", "--run_root", str(tmp_path), "--sub_id", "sub-00", "--out", str(out),
            "--n_final_seeds", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    cmp.main()                                   # 3 seeds each, as expected: exit 0
    result["variants"]["clean"]["saved_decode_matches_best_seed"] = False
    out.unlink()
    with pytest.raises(SystemExit) as e:
        cmp.main()
    assert e.value.code == 1
    assert json.loads(out.read_text())["variants"]["clean"]["saved_decode_matches_best_seed"] is False


def test_load_variant_compares_the_saved_decode_with_the_best_seed(tmp_path, monkeypatch):
    final = tmp_path / "output" / "04_combined_hdphmm" / "p" / "sub-00" / "final" / "vt0.95"
    (final / "seeds").mkdir(parents=True)
    (final / "final_results.json").write_text(json.dumps({
        "data_info": {"n_pcs": 3}, "final_refit": {"best_seed": 1},
        "selected_config_name": "cfg"}))
    for name in ("seeds/seed_0_model.pkl", "seeds/seed_1_model.pkl", "pca_model.pkl",
                 "decoded_states.pkl"):
        (final / name).write_bytes(b"")
    models = {f"seed_{s}_model.pkl": {"decoded": {"r": np.array([s, s, 1])}} for s in (0, 1)}
    saved = {"r": np.array([1, 1, 1])}           # seed 1's decode
    monkeypatch.setattr(cmp, "_load_pickle", lambda path: saved if Path(path).name ==
                        "decoded_states.pkl" else models.get(Path(path).name))
    monkeypatch.setattr(cmp, "back_project_states", lambda m, pca, n: (np.zeros((2, 4)), None))
    monkeypatch.setattr(cmp, "decode_all_runs", lambda m, split, d, n: m["decoded"])
    monkeypatch.setattr(cmp, "load_split", lambda base: None)
    monkeypatch.setattr(cmp, "get_projected_dir", lambda base: None)
    monkeypatch.setattr(cmp, "used_state_indices", lambda m: np.arange(2))
    monkeypatch.setattr(cmp, "stage1_vt", lambda sub_dir: (0.99, 5, 5))
    fit = cmp.load_variant(tmp_path, "sub-00", "p")
    assert sorted(fit["seeds"]) == [0, 1] and fit["best_seed"] == 1
    assert fit["saved_decode_matches_best_seed"] is True
    saved["r"] = np.array([0, 0, 1])             # seed 0's decode, not the best seed's
    assert cmp.load_variant(tmp_path, "sub-00", "p")["saved_decode_matches_best_seed"] is False


def test_write_json_replaces_the_file_whole(tmp_path):
    path = tmp_path / "compare" / "sub-00.json"
    cmp.write_json({"a": 1}, path)
    cmp.write_json({"b": 2}, path)
    assert json.loads(path.read_text()) == {"b": 2}
    assert [p.name for p in path.parent.iterdir()] == ["sub-00.json"]


def test_write_json_keeps_the_old_file_when_writing_fails(tmp_path, monkeypatch):
    path = tmp_path / "sub-00.json"
    cmp.write_json({"a": 1}, path)

    def partial(obj, f, **kw):
        f.write('{"trunc')
        raise OSError("disk full")
    monkeypatch.setattr(cmp.json, "dump", partial)
    with pytest.raises(OSError):
        cmp.write_json({"b": 2}, path)
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["sub-00.json"]


def test_write_json_keeps_or_sets_the_file_mode(tmp_path):
    old = os.umask(0o022)
    try:
        path = tmp_path / "new.json"
        cmp.write_json({"a": 1}, path)
        assert path.stat().st_mode & 0o777 == 0o644
        os.chmod(path, 0o664)
        cmp.write_json({"b": 2}, path)
        assert path.stat().st_mode & 0o777 == 0o664
    finally:
        os.umask(old)
