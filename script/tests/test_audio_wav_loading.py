"""WAV files are read with scipy and scaled to float32 in [-1, 1], as torchaudio did."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

torch = pytest.importorskip("torch")

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from utils import transformer_io  # noqa: E402


def _write(tmp_path, data, sr=16000, name="a.wav"):
    path = tmp_path / name
    wavfile.write(str(path), sr, data)
    return path


def test_int16_mono_scaled_by_32768(tmp_path):
    data = np.array([0, 1, -1, 16384, -16384, 32767, -32768], dtype=np.int16)
    waveform, sr = transformer_io.load_wav(_write(tmp_path, data))
    assert sr == 16000
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert tuple(waveform.shape) == (1, data.size)
    expected = data.astype(np.float32) / 32768.0
    np.testing.assert_array_equal(waveform.numpy()[0], expected)
    assert waveform.min().item() == -1.0


def test_int16_stereo_is_channels_by_samples(tmp_path):
    left = np.arange(-50, 50, dtype=np.int16) * 300
    right = -left
    data = np.stack([left, right], axis=1)  # scipy layout: (samples, channels)
    waveform, sr = transformer_io.load_wav(_write(tmp_path, data, sr=44100))
    assert sr == 44100
    assert tuple(waveform.shape) == (2, left.size)
    np.testing.assert_array_equal(waveform.numpy()[0], left.astype(np.float32) / 32768.0)
    np.testing.assert_array_equal(waveform.numpy()[1], right.astype(np.float32) / 32768.0)


def test_int32_uint8_and_float_wavs(tmp_path):
    i32 = np.array([0, 2**30, -(2**31), 2**31 - 1], dtype=np.int32)
    w, _ = transformer_io.load_wav(_write(tmp_path, i32, name="i32.wav"))
    np.testing.assert_array_equal(w.numpy()[0], (i32.astype(np.float64) / 2**31).astype(np.float32))

    u8 = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    w, _ = transformer_io.load_wav(_write(tmp_path, u8, name="u8.wav"))
    np.testing.assert_array_equal(w.numpy()[0], (u8.astype(np.float32) - 128) / 128)

    f32 = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)
    w, _ = transformer_io.load_wav(_write(tmp_path, f32, name="f32.wav"))
    assert w.dtype == torch.float32
    np.testing.assert_array_equal(w.numpy()[0], f32)


class _ReachedProcessor(Exception):
    pass


def test_extract_audio_features_does_not_use_torchaudio_load(tmp_path, monkeypatch):
    # torchaudio.load goes through torchcodec, which cannot load on the cluster.
    torchaudio = pytest.importorskip("torchaudio")

    def broken_load(*a, **kw):
        raise RuntimeError("torchcodec unavailable")

    monkeypatch.setattr(torchaudio, "load", broken_load)
    data = (np.sin(np.arange(16000) / 10.0) * 10000).astype(np.int16)
    path = _write(tmp_path, data)
    seen = {}

    def processor(chunk, sampling_rate, **kw):
        seen.update(chunk=chunk, sr=sampling_rate)
        raise _ReachedProcessor

    with pytest.raises(_ReachedProcessor):
        transformer_io.extract_audio_features(path, None, processor, n_trs=5, device="cpu")
    assert seen["sr"] == 16000
    np.testing.assert_array_equal(seen["chunk"], data.astype(np.float32) / 32768.0)
