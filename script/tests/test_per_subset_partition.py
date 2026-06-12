"""Unit tests for 08e per-subset partition logic.

Verifies the per-subset partition helper (used to break Movie10 into 4 films
for Fig 3 Panel C):
  (a) every run_id maps to exactly one subset under the canonical patterns
  (b) per-film run counts match expected viewing structure
  (c) an unanticipated run_id (e.g. ``figures_remake01``) raises rather than
      silently merging into ``figures``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The 08e module name starts with a digit ("08e_transformer_..."), which is
# not a valid Python identifier. Load it via importlib.util so the test can
# pull `subset_partition_for` from it directly without renaming the module.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
_spec = importlib.util.spec_from_file_location(
    "_module_08e",
    _SCRIPT_DIR / "08e_transformer_cross_stim_aggregate.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
subset_partition_for = _module.subset_partition_for
PER_SUBSET_DEFS = _module.PER_SUBSET_DEFS


@pytest.fixture
def canonical_movie10_run_ids():
    """Synthetic run_ids that mirror the actual m10_04 decoded_states keys.

    Wolf: 17 runs (one viewing each).
    Hidden Figures: 12 clips × 2 viewings = 24 run_ids.
    Bourne: 10 runs.
    Life: 5 clips × 2 viewings = 10 run_ids.
    Total: 61 run_ids - matches sub-05 m10_04_decoded.
    """
    rids = []
    for i in range(1, 18):
        rids.append(f"wolf{i:02d}")
    for i in range(1, 13):
        rids.append(f"figures{i:02d}_run-1")
        rids.append(f"figures{i:02d}_run-2")
    for i in range(1, 11):
        rids.append(f"bourne{i:02d}")
    for i in range(1, 6):
        rids.append(f"life{i:02d}_run-1")
        rids.append(f"life{i:02d}_run-2")
    return rids


def test_partition_exact_assignment(canonical_movie10_run_ids):
    """Every run_id maps to exactly one subset; per-film counts match."""
    labels, by_subset = subset_partition_for("movie10", canonical_movie10_run_ids)
    assert labels == ["wolf", "figures", "bourne", "life"]
    assert len(by_subset["wolf"]) == 17
    assert len(by_subset["figures"]) == 24, (
        "Hidden Figures has 12 clips × 2 viewings = 24 runs"
    )
    assert len(by_subset["bourne"]) == 10
    assert len(by_subset["life"]) == 10, (
        "Life has 5 clips × 2 viewings = 10 runs"
    )
    # Total = 61 (no orphans, no double-counting)
    assert sum(len(v) for v in by_subset.values()) == 61


def test_partition_unknown_run_id_raises_orphan():
    """A run_id matching no pattern raises ValueError (orphan)."""
    rids = ["wolf01", "figures01_run-1", "bourne01", "life01_run-1",
            "unknown_movie01"]
    with pytest.raises(ValueError, match=r"orphan|matches no subset"):
        subset_partition_for("movie10", rids)


def test_partition_remake_pattern_does_not_silently_merge():
    """``figures_remake01`` must NOT silently merge into ``figures``.

    The canonical pattern is ``^figures\\d+(_run-\\d+)?$``. The character
    after ``figures`` must be a digit, so ``figures_remake01`` fails to match
    and raises an orphan. Catches an inadvertent regression where a stricter
    pattern is loosened to ``startswith``.
    """
    rids = ["wolf01", "figures01_run-1", "bourne01", "life01_run-1",
            "figures_remake01"]
    with pytest.raises(ValueError, match=r"orphan|matches no subset"):
        subset_partition_for("movie10", rids)


def test_partition_unknown_stimulus_raises():
    """Calling with a stimulus that has no defined partition raises."""
    with pytest.raises(ValueError, match=r"No per-subset partition"):
        subset_partition_for("harrypotter", ["wolf01"])
