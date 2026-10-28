"""clean_wmcsf_check_fits.py: the gate between the fit attempts and the final refit.

Claims tested: every config of the stage-1 and stage-2 grids must have a summary
with all its seeds; a missing summary or a short seed count is listed, and the
script exits 1; a complete set exits 0. Synthetic data only.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_wmcsf_check_fits import incomplete_configs  # noqa: E402
from config.combined_hmm_config import (  # noqa: E402
    build_stage1_grid, build_stage2_grid, config_name,
)

SCRIPT = Path(__file__).resolve().parents[1] / "clean_wmcsf_check_fits.py"


def _names():
    return list(dict.fromkeys(config_name(c) for c in
                              build_stage1_grid() + build_stage2_grid(0.95)))


def _write(configs, seeds):
    for name, n in seeds.items():
        d = configs / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config_summary.json").write_text(json.dumps({"n_seeds_successful": n}))


def test_complete_grids_pass(tmp_path):
    _write(tmp_path, {n: 5 for n in _names()})
    assert incomplete_configs(tmp_path, 0.95, 5) == []


def test_missing_and_short_configs_are_listed(tmp_path):
    names = _names()
    _write(tmp_path, {n: 5 for n in names[1:]} | {names[2]: 4})
    assert incomplete_configs(tmp_path, 0.95, 5) == [
        f"{names[0]}: no config_summary.json", f"{names[2]}: 4 of 5 seeds"]


def test_stage2_only_configs_are_checked(tmp_path):
    stage1 = {config_name(c) for c in build_stage1_grid()}
    stage2_only = [config_name(c) for c in build_stage2_grid(0.95)
                   if config_name(c) not in stage1]
    _write(tmp_path, {n: 5 for n in _names() if n != stage2_only[-1]})
    assert incomplete_configs(tmp_path, 0.95, 5) == [
        f"{stage2_only[-1]}: no config_summary.json"]


def test_script_exit_status(tmp_path):
    configs = tmp_path / "output" / "04_combined_hdphmm" / "atlas-4S156Parcels" / "sub-02" / "configs"
    _write(configs, {n: 5 for n in _names()[1:]})
    cmd = [sys.executable, str(SCRIPT), "--scratch_dir", str(tmp_path), "--sub_id", "sub-02",
           "--fixed_vt", "0.95"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1 and "INCOMPLETE sub-02" in proc.stdout
    _write(configs, {_names()[0]: 5})
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
