"""Shared active-state (decoded) and used-state (fitted model) definitions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.recurrence_utils import (  # noqa: E402
    ACTIVE_FO_THRESHOLD, active_state_indices, used_state_indices,
)


def test_active_threshold_is_two_percent():
    assert ACTIVE_FO_THRESHOLD == 0.02


def test_active_states_need_fo_above_threshold_in_one_run():
    # 100-TR runs. State 1: 3 TRs in run a (FO 0.03) -> active.
    # State 2: 2 TRs in each run (FO 0.02, not > 0.02) -> not active, even
    # though it is present in every run. State 3 never appears.
    run_a = np.array([0] * 97 + [1] * 3)
    run_b = np.array([0] * 98 + [2] * 2)
    run_a[:2] = 2
    decoded = {"a": run_a, "b": run_b}
    assert active_state_indices(decoded, 4).tolist() == [0, 1]


def test_used_states_read_last_usage_iteration():
    model = SimpleNamespace(history={"state_usage": [
        [0.5, 0.5, 0.0], [0.6, 0.395, 0.005]]})
    assert used_state_indices(model).tolist() == [0, 1]
    assert used_state_indices(model, min_state_usage=0.4).tolist() == [0]


def test_used_states_empty_without_history():
    assert used_state_indices(SimpleNamespace()).size == 0
    assert used_state_indices(SimpleNamespace(history={})).size == 0


def test_states_present_in_runs_uses_any_run_rule():
    from utils.recurrence_utils import states_present_in_runs
    fo = [np.array([0.5, 0.019, 0.0]), np.array([0.5, 0.021, 0.02])]
    out = states_present_in_runs(fo, [1, 2])
    assert list(out) == [1]                 # state 2 only reaches 0.02, not > 0.02
    assert out[1]["n_runs_active"] == 1
    assert out[1]["max_run_fo"] == 0.021
    assert states_present_in_runs([], [1]) == {}


def test_figure_presence_threshold_matches_shared_rule():
    # fig_S08 and fig_F5 are marimo notebooks that set PRESENCE_FO as a literal
    # inside a cell; keep it equal to ACTIVE_FO_THRESHOLD.
    import re
    script_dir = Path(__file__).resolve().parents[1]
    for name in ("fig_S08_cross_stimulus_validity.py", "fig_F5_cross_stimulus_transfer.py"):
        text = (script_dir / name).read_text()
        values = re.findall(r"^\s*PRESENCE_FO\s*=\s*([0-9.]+)\s*$", text, re.M)
        assert values and all(float(v) == ACTIVE_FO_THRESHOLD for v in values), name


def test_used_state_threshold_matches_fit_defaults():
    import inspect
    import re
    from utils.recurrence_utils import USED_STATE_THRESHOLD
    src = (Path(__file__).resolve().parents[1] / "utils" / "hdphmm.py").read_text()
    m = re.search(r"min_state_usage=([0-9.]+)", src)
    assert m and float(m.group(1)) == USED_STATE_THRESHOLD
    from utils.recurrence_utils import used_state_indices
    default = inspect.signature(used_state_indices).parameters["min_state_usage"].default
    assert default == USED_STATE_THRESHOLD


def test_fo_threshold_check_against_05a(tmp_path):
    import json
    import pytest
    from utils.recurrence_utils import check_fo_threshold_matches_05a
    (tmp_path / "recurrence_summary.json").write_text(json.dumps({"fo_active_threshold": 0.02}))
    check_fo_threshold_matches_05a(str(tmp_path), 0.02)
    with pytest.raises(ValueError, match="differs"):
        check_fo_threshold_matches_05a(str(tmp_path), 0.01)
