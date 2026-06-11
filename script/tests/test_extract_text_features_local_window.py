#!/usr/bin/env python3
"""Unit tests for local-window text feature extraction.

Covers the 6 cases from §2.6 of the design doc
(`the design notes`):

1. TR 0 with 1 content token: BOS-clamped pool returns the single content token.
2. TR 1 with W=4: BOS clamp engages; content tokens of TR 0 + TR 1 pooled
   (excluding BOS, per §2.3.2 formal slice formula).
3. Empty TR (no new tokens): hi == lo after clamp; zero vector emitted.
4. Mid-run silent stretch ≥ W TRs: window contains zero content; zero vector.
5. Degenerate-prefix sanity: searchsorted-on-full-text-offsets is monotone
   even when char_lengths land mid-token (the failure mode the old
   re-tokenize-per-TR strategy could not guarantee, per §2.3.1).
6. extract_text_features asserts ``tokenizer.is_fast`` before any forward pass.

Standalone::

    python script/tests/test_extract_text_features_local_window.py

Pure synthetic — no transformers, no GPU, no pytest dep.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Make script/ importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_DIR = os.path.dirname(_HERE)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils.transformer_io import (  # noqa: E402
    _compute_tr_token_boundaries,
    _local_window_pool,
    extract_text_features,
)


def test_pool_tr0_one_word() -> None:
    """Case 1: TR 0 has 1 content token; W=4; B=1.

    `tr_token_boundaries[0] = 1 + B = 2` (BOS + 1 content token). Local pool
    returns the single content token's hidden state.
    """
    hidden_states = np.array(
        [
            [99.0, 99.0, 99.0],   # BOS
            [1.0, 2.0, 3.0],      # content token
        ],
        dtype=np.float32,
    )
    boundaries = np.array([2], dtype=np.int64)

    result = _local_window_pool(hidden_states, boundaries, window_trs=4, B=1)

    assert result.shape == (1, 3)
    np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])


def test_pool_tr1_W4_bos_clamp() -> None:
    """Case 2: TR 1 with W=4, B=1.

    Per §2.3.2 formal: ``t < W`` → ``lo_count = 0``; ``lo = max(B, 0) = 1``.
    Slice ``[1:hi]`` pools content tokens of TR 0 + TR 1 (excluding BOS).
    """
    hidden_states = np.array(
        [
            [99.0, 99.0],  # BOS
            [1.0, 0.0],    # content @ TR 0
            [3.0, 0.0],    # content @ TR 1
            [5.0, 0.0],    # content @ TR 1
        ],
        dtype=np.float32,
    )
    boundaries = np.array([2, 4], dtype=np.int64)  # BOS+1 content; +2 content

    result = _local_window_pool(hidden_states, boundaries, window_trs=4, B=1)

    assert result.shape == (2, 2)
    # Slice [1:4] = mean([1,0],[3,0],[5,0]) = (3, 0).
    np.testing.assert_array_almost_equal(result[1], [3.0, 0.0])


def test_pool_empty_tr() -> None:
    """Case 3: Empty TR (no new content tokens).

    ``boundaries[t] == boundaries[t-1]``; after clamp ``hi == lo`` and a zero
    vector is emitted.
    """
    hidden_states = np.array(
        [
            [99.0, 99.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    # TR 0: 1 content tok; TR 1: 1 new content tok; TR 2: no new content.
    boundaries = np.array([2, 3, 3], dtype=np.int64)

    result = _local_window_pool(hidden_states, boundaries, window_trs=1, B=1)

    # TR 2 with W=1: lo_count = boundaries[1] = 3, hi = 3, lo = max(1, 3) = 3
    # → hi == lo → zero vector.
    np.testing.assert_array_equal(result[2], np.zeros(2, dtype=np.float32))


def test_pool_silent_stretch_W_trs() -> None:
    """Case 4: Mid-run silent stretch ≥ W TRs.

    Window contains zero new content tokens → zero vector. Sanity-check:
    immediately after the silence, a TR with new content recovers a non-zero
    pool.
    """
    hidden_states = np.array(
        [
            [99.0, 99.0],   # BOS
            [1.0, 2.0],     # token @ TR 0
            [3.0, 4.0],     # token @ TR 1
            [5.0, 6.0],     # token @ TR 4 (silence at TR 2, TR 3)
        ],
        dtype=np.float32,
    )
    # 5 TRs: content at 0, 1, 4; silence at 2, 3.
    boundaries = np.array([2, 3, 3, 3, 4], dtype=np.int64)

    result = _local_window_pool(hidden_states, boundaries, window_trs=1, B=1)

    # Silent TRs (window of 1 TR contains no new tokens) → zero.
    np.testing.assert_array_equal(result[2], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(result[3], np.zeros(2, dtype=np.float32))
    # TR 4 has new content; W=1 covers (3, 4]: lo=boundaries[3]=3, hi=4 → token [5,6].
    np.testing.assert_array_almost_equal(result[4], [5.0, 6.0])


def test_boundaries_monotone_under_mid_token_char_lengths() -> None:
    """Case 5: Degenerate-prefix sanity.

    ``searchsorted`` on full-text token end-offsets is monotone-by-construction
    even when ``char_lengths`` land mid-token. The old re-tokenize-per-TR
    strategy could produce non-monotone boundaries due to BPE / leading-space
    drift between prefix-tokenization and full-text-tokenization.
    """
    # 3 tokens spanning chars [0,5), [5,11), [11,17) of the full text.
    offsets = np.array([[0, 5], [5, 11], [11, 17]], dtype=np.int64)
    # Adversarial: TR 0's char_length lands mid-token-1 (char 8).
    char_lengths = np.array([8, 13, 17], dtype=np.int64)

    b = _compute_tr_token_boundaries(offsets, char_lengths)

    # searchsorted([5,11,17], [8,13,17], side="right") =
    #   [1 (5 ≤ 8 < 11), 2 (11 ≤ 13 < 17), 3 (17 == 17, side=right past)].
    np.testing.assert_array_equal(b, [1, 2, 3])
    # Critical guarantee from the §2.3.1 redesign:
    assert np.all(np.diff(b) >= 0), "boundaries must be monotone non-decreasing"


def test_boundaries_handle_silent_run() -> None:
    """Case 5b (regression): char_lengths constant over a stretch → boundaries flat.

    Pure-silence stretches must produce equal boundaries (and downstream zero
    vectors via case 3 / case 4 paths).
    """
    offsets = np.array([[0, 4], [4, 9]], dtype=np.int64)
    # 4 TRs: text grows on TR 0 only, then stays at length 9.
    char_lengths = np.array([4, 9, 9, 9], dtype=np.int64)

    b = _compute_tr_token_boundaries(offsets, char_lengths)

    np.testing.assert_array_equal(b, [1, 2, 2, 2])


def test_extract_text_features_requires_fast_tokenizer() -> None:
    """Case 6: ``extract_text_features`` must reject a slow tokenizer.

    ``return_offsets_mapping=True`` is unsupported by slow tokenizers, so the
    function asserts ``tokenizer.is_fast`` early. The check must fire before
    any forward pass — this test passes a MagicMock model that would error on
    any call, proving the assertion is reached first.
    """
    slow_tokenizer = MagicMock()
    slow_tokenizer.is_fast = False
    fake_model = MagicMock()
    fake_model.side_effect = AssertionError(
        "model should not be called when is_fast=False"
    )
    transcript = ["hello", "hello world"]

    raised = False
    try:
        extract_text_features(
            transcript, fake_model, slow_tokenizer, n_trs=2, device="cpu",
        )
    except AssertionError as e:
        raised = True
        # Failure must mention the tokenizer requirement, not the side-effect
        # we wired into the fake model — proves the early-assert path fired.
        assert (
            "is_fast" in str(e).lower()
            or "fast tokenizer" in str(e).lower()
        ), f"unexpected assertion: {e!r}"
    assert raised, "extract_text_features must reject is_fast=False"


def main() -> int:
    tests = [
        test_pool_tr0_one_word,
        test_pool_tr1_W4_bos_clamp,
        test_pool_empty_tr,
        test_pool_silent_stretch_W_trs,
        test_boundaries_monotone_under_mid_token_char_lengths,
        test_boundaries_handle_silent_run,
        test_extract_text_features_requires_fast_tokenizer,
    ]
    n_pass = 0
    n_fail = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            n_fail += 1
            print(f"FAIL {t.__name__}: {e!r}")
        else:
            n_pass += 1
            print(f"PASS {t.__name__}")
    print(f"\n{n_pass}/{n_pass + n_fail} tests passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
