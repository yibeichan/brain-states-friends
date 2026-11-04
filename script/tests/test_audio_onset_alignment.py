"""Audio TR windows are placed on scan time when the audio starts after the scan."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from utils.transformer_io import TR_DURATION, _aggregate_frames_to_trs  # noqa: E402

RATE = 50  # w2v-BERT 2.0 frames per second


def _frames(n):
    # frame i holds the value i, so a pooled mean shows which frames went in
    return np.arange(n, dtype=np.float32)[:, None]


def test_zero_onset_matches_plain_binning():
    frames = _frames(1000)
    out = _aggregate_frames_to_trs(frames, 10, RATE)
    fpt = TR_DURATION * RATE
    for t in range(10):
        start, end = int(t * fpt), int((t + 1) * fpt)
        assert out[t, 0] == frames[start:end].mean()


def test_onset_places_audio_on_scan_time():
    onset_s = 4.0  # LPP: 4 s of silence before each section
    frames = _frames(1000)
    out = _aggregate_frames_to_trs(frames, 8, RATE, onset_s=onset_s)
    fpt = TR_DURATION * RATE
    off = onset_s * RATE
    # TRs 0 and 1 (0-2.98 s) end before the audio starts.
    assert np.all(out[:2] == 0)
    # TR 2 (2.98-4.47 s) holds only the audio after 4.0 s.
    assert out[2, 0] == frames[0:int(3 * fpt - off)].mean()
    # Later TRs hold the audio heard during them.
    for t in range(3, 8):
        start, end = int(t * fpt - off), int((t + 1) * fpt - off)
        assert out[t, 0] == frames[start:end].mean()
    # The audio at scan time 5.0 s (audio time 1.0 s, frame 50) is in TR 3,
    # which covers 4.47-5.96 s.
    t_of_frame_50 = int((onset_s + 50 / RATE) / TR_DURATION)
    assert t_of_frame_50 == 3
    start, end = int(3 * fpt - off), int(4 * fpt - off)
    assert start <= 50 < end


def test_frames_run_out_before_last_tr():
    out = _aggregate_frames_to_trs(_frames(100), 10, RATE, onset_s=4.0)
    assert np.all(out[5:] == 0)  # 100 frames = 2 s of audio, ending at 6 s


def test_08c_passes_onset_and_full_tr_count(tmp_path, monkeypatch):
    # The script requires these at import.
    for var in ("SCRATCH_DIR", "DATA_DIR", "ALGONAUTS_DIR"):
        monkeypatch.setenv(var, str(tmp_path))
    spec = importlib.util.spec_from_file_location("m08c", SCRIPT_DIR / "08c_transformer_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = {}

    def fake_extract(audio_path, model, processor, n_trs, device="cuda", onset_s=0.0, **kw):
        seen.update(n_trs=n_trs, onset_s=onset_s)
        return {0: np.ones((n_trs, 2), dtype=np.float32)}

    saved = {}
    monkeypatch.setattr(mod, "extract_audio_features", fake_extract)
    monkeypatch.setattr(mod, "save_layer_features",
                        lambda out_dir, layer, run_id, feats: saved.update({layer: feats}))
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    run_info = {"run_id": "lppFR_run-01", "audio_path": str(wav), "n_trs": 12, "onset_s": 4.0}
    mod.process_run_audio(run_info, None, None, str(tmp_path), "cpu", str(tmp_path))
    assert seen == {"n_trs": 12, "onset_s": 4.0}
    # No extra zero TRs are prepended after extraction.
    assert saved[0].shape == (12, 2)
