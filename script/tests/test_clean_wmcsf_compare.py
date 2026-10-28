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


def test_decode_mismatch_is_a_failed_check():
    ok = {"variants": {"corrupt": {"saved_decode_matches_best_seed": True},
                       "clean": {"saved_decode_matches_best_seed": True}}}
    assert cmp.check_failures(ok) == []
    bad = {"variants": {"corrupt": {"saved_decode_matches_best_seed": True},
                        "clean": {"saved_decode_matches_best_seed": False}}}
    assert cmp.check_failures(bad) == [
        "clean: the selected fit's saved decode differs from its seed model's"]


def test_write_json_replaces_the_file_whole(tmp_path):
    path = tmp_path / "compare" / "sub-00.json"
    cmp.write_json({"a": 1}, path)
    cmp.write_json({"b": 2}, path)
    assert json.loads(path.read_text()) == {"b": 2}
    assert [p.name for p in path.parent.iterdir()] == ["sub-00.json"]
