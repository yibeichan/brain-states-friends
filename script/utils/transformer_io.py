#!/usr/bin/env python
"""
transformer_io.py - I/O utilities for transformer feature extraction (08c).

Handles:
  - Loading pretrained models with layer-wise output
  - Audio chunking for Wav2VecBert (30s chunks, 5s overlap) to avoid OOM
  - Text context building for LLaMA (cumulative context per TR)
  - Temporal aggregation of native-rate activations to TR resolution
  - Transcript loading for Friends/Movie10 (TR-aligned TSV), HP (word-level
    events.tsv), and PP (TBD)

Models (TRIBEv2-validated backbones, frozen, all layers):
  - Audio: Wav2VecBert 2.0 (facebook/w2v-bert-2.0), 24 layers, 1024 dim
  - Video: DINOv2-large (facebook/dinov2-large), 24 layers, 1024 dim
  - Text:  LLaMA 3.2 3B (meta-llama/Llama-3.2-3B), 28 layers, 3072 dim
"""

import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TR_DURATION = 1.49  # seconds

# Model registry: HuggingFace IDs and metadata
MODEL_REGISTRY = {
    "w2v-bert-2.0": {
        "hf_id": "facebook/w2v-bert-2.0",
        "modality": "audio",
        "n_layers": 24,
        "hidden_dim": 1024,
        "native_rate_hz": 50,  # Conformer output rate
        "loader": "Wav2Vec2BertModel",
    },
    "dinov2-large": {
        "hf_id": "facebook/dinov2-large",
        "modality": "video",
        "n_layers": 24,
        "hidden_dim": 1024,
        "native_rate_hz": None,  # frame-level (1 frame per TR)
        "loader": "AutoModel",
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "modality": "text",
        "n_layers": 28,
        "hidden_dim": 3072,
        "native_rate_hz": None,  # token-level
        "loader": "AutoModelForCausalLM",
    },
}

# Stimulus → available modalities (ground truth for which models can run on
# which stimulus). Used by ``validate_stimulus_model`` to fail fast on
# incompatible combinations (e.g. Harry Potter has no audio).
STIMULUS_MODALITIES = {
    "friends":        {"audio", "video", "text"},
    "movie10":        {"audio", "video", "text"},
    "harrypotter":    {"text"},              # 2Hz RSVP reading — no audio, no scene video
    "petitprince_fr": {"audio", "text"},     # audiobook (no video)
    "petitprince_en": {"audio", "text"},     # audiobook (no video)
}


def validate_stimulus_model(stimulus, model_key):
    """Raise ValueError if a (stimulus, model) pair is incompatible.

    The check is derived from ``MODEL_REGISTRY[model_key]['modality']`` and
    ``STIMULUS_MODALITIES[stimulus]``. Used by all 08-series correspondence
    scripts that accept ``--stimulus`` and ``--model`` flags.

    Parameters
    ----------
    stimulus : str
        Stimulus key. Must be present in ``STIMULUS_MODALITIES``.
    model_key : str
        Model key. Must be present in ``MODEL_REGISTRY``.

    Raises
    ------
    ValueError
        If either key is unknown, or if the model's modality is not available
        for the stimulus.
    """
    if stimulus not in STIMULUS_MODALITIES:
        raise ValueError(
            f"Unknown stimulus '{stimulus}'. Known stimuli: "
            f"{sorted(STIMULUS_MODALITIES.keys())}"
        )
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Known models: "
            f"{sorted(MODEL_REGISTRY.keys())}"
        )
    model_modality = MODEL_REGISTRY[model_key]["modality"]
    available = STIMULUS_MODALITIES[stimulus]
    if model_modality not in available:
        raise ValueError(
            f"Model '{model_key}' (modality={model_modality}) is not "
            f"compatible with stimulus '{stimulus}' "
            f"(available modalities={sorted(available)})."
        )


# Audio chunking parameters (review resolution C1)
AUDIO_CHUNK_SECONDS = 30.0
AUDIO_OVERLAP_SECONDS = 5.0
AUDIO_SAMPLE_RATE = 16_000  # Wav2VecBert expects 16 kHz

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_key, device="cuda", dtype=None):
    """Load a pretrained model with output_hidden_states=True.

    Args:
        model_key: Key into MODEL_REGISTRY (e.g., 'w2v-bert-2.0').
        device: Torch device string.
        dtype: Optional torch dtype (e.g., torch.float16). If None, uses
            float16 for models >= 1B params, float32 otherwise.

    Returns:
        (model, processor_or_tokenizer): Model in eval mode on device,
            plus the appropriate processor/tokenizer.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

    info = MODEL_REGISTRY[model_key]
    hf_id = info["hf_id"]

    if dtype is None:
        # Use fp16 for large models to save VRAM
        dtype = torch.float16 if info["hidden_dim"] >= 3072 else torch.float32

    logger.info(f"Loading {hf_id} (dtype={dtype}, device={device})")

    if info["loader"] == "Wav2Vec2BertModel":
        from transformers import Wav2Vec2BertModel

        model = Wav2Vec2BertModel.from_pretrained(
            hf_id, output_hidden_states=True, torch_dtype=dtype
        )
        processor = AutoFeatureExtractor.from_pretrained(hf_id)

    elif info["loader"] == "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, output_hidden_states=True, torch_dtype=dtype
        )
        processor = AutoTokenizer.from_pretrained(hf_id)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token

    else:  # AutoModel (DINOv2)
        from transformers import AutoImageProcessor

        model = AutoModel.from_pretrained(
            hf_id, output_hidden_states=True, torch_dtype=dtype
        )
        processor = AutoImageProcessor.from_pretrained(hf_id)

    model = model.to(device).eval()
    return model, processor


# ---------------------------------------------------------------------------
# Audio feature extraction (Wav2VecBert)
# ---------------------------------------------------------------------------


def extract_audio_features(
    audio_path,
    model,
    processor,
    n_trs,
    device="cuda",
    chunk_seconds=AUDIO_CHUNK_SECONDS,
    overlap_seconds=AUDIO_OVERLAP_SECONDS,
):
    """Extract layer-wise audio features from Wav2VecBert with chunking.

    Processes audio in overlapping chunks to avoid OOM (review C1).
    Each chunk: [overlap | center | overlap]. Only center frames are kept.
    After chunking, mean-pools native-rate frames within each TR window.

    Args:
        audio_path: Path to 16 kHz mono WAV file.
        model: Loaded Wav2VecBert model.
        processor: Wav2Vec2 feature extractor.
        n_trs: Number of TRs (authoritative count).
        device: Torch device.
        chunk_seconds: Center duration of each chunk in seconds.
        overlap_seconds: Overlap on each side in seconds.

    Returns:
        dict: {layer_idx: np.ndarray of shape (n_trs, hidden_dim)}
            layer_idx 0 = first transformer layer (not embedding).
    """
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if sr != AUDIO_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = AUDIO_SAMPLE_RATE

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (n_samples,)

    n_samples = waveform.shape[0]
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    stride_samples = chunk_samples  # stride = center size (no overlap in output)

    # Compute chunks
    all_hidden_states = None  # will be list of lists
    total_center_frames = 0

    pos = 0
    while pos < n_samples:
        # Chunk boundaries with overlap
        chunk_start = max(0, pos - overlap_samples)
        chunk_end = min(n_samples, pos + chunk_samples + overlap_samples)
        chunk_waveform = waveform[chunk_start:chunk_end]

        # How many samples of overlap were actually prepended
        actual_left_overlap = pos - chunk_start

        # Run through model
        inputs = processor(
            chunk_waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, dim)
        hidden_states = outputs.hidden_states

        # Compute frame boundaries for the center region
        # Conformer downsamples by ~320x (16kHz -> 50Hz)
        downsample_factor = sr / MODEL_REGISTRY["w2v-bert-2.0"]["native_rate_hz"]
        left_frames = int(actual_left_overlap / downsample_factor)
        total_chunk_frames = hidden_states[0].shape[1]

        # Center end: account for right overlap
        actual_right_overlap = chunk_end - min(n_samples, pos + chunk_samples)
        right_frames = int(actual_right_overlap / downsample_factor)
        center_end = total_chunk_frames - right_frames if right_frames > 0 else total_chunk_frames

        if all_hidden_states is None:
            n_layers = len(hidden_states) - 1  # exclude embedding layer
            all_hidden_states = [[] for _ in range(n_layers)]

        # Extract center frames from each transformer layer (skip embedding at idx 0)
        for layer_idx in range(n_layers):
            center = hidden_states[layer_idx + 1][0, left_frames:center_end, :]
            all_hidden_states[layer_idx].append(center.cpu().float().numpy())

        total_center_frames += center_end - left_frames
        pos += stride_samples

    # Concatenate all center frames per layer
    layer_features = {}
    for layer_idx in range(n_layers):
        frames = np.concatenate(all_hidden_states[layer_idx], axis=0)  # (total_frames, dim)
        layer_features[layer_idx] = _aggregate_frames_to_trs(
            frames, n_trs, MODEL_REGISTRY["w2v-bert-2.0"]["native_rate_hz"]
        )

    return layer_features


def _aggregate_frames_to_trs(frames, n_trs, native_rate_hz):
    """Mean-pool native-rate frames into TR bins.

    Args:
        frames: np.ndarray of shape (n_frames, dim).
        n_trs: Number of TRs.
        native_rate_hz: Frame rate of the model output.

    Returns:
        np.ndarray of shape (n_trs, dim).
    """
    frames_per_tr = TR_DURATION * native_rate_hz
    dim = frames.shape[1]
    result = np.zeros((n_trs, dim), dtype=np.float32)

    for t in range(n_trs):
        start = int(t * frames_per_tr)
        end = int((t + 1) * frames_per_tr)
        if start >= len(frames):
            break
        chunk = frames[start : min(end, len(frames))]
        if len(chunk) > 0:
            result[t] = chunk.mean(axis=0)

    return result


# ---------------------------------------------------------------------------
# Video feature extraction (DINOv2)
# ---------------------------------------------------------------------------


def extract_video_features(
    video_path,
    model,
    processor,
    n_trs,
    device="cuda",
):
    """Extract layer-wise video features from DINOv2 (1 frame per TR center).

    Extracts one frame at the center of each TR window, processes through
    DINOv2, and returns the CLS token embedding per layer.

    Args:
        video_path: Path to video file (mkv).
        model: Loaded DINOv2 model.
        processor: DINOv2 image processor.
        n_trs: Number of TRs.
        device: Torch device.

    Returns:
        dict: {layer_idx: np.ndarray of shape (n_trs, hidden_dim)}
    """
    import torch

    frames = _extract_video_frames(video_path, n_trs)
    n_layers = MODEL_REGISTRY["dinov2-large"]["n_layers"]
    dim = MODEL_REGISTRY["dinov2-large"]["hidden_dim"]

    layer_features = {i: np.zeros((n_trs, dim), dtype=np.float32) for i in range(n_layers)}

    # Process frames in batches to manage memory
    batch_size = 16
    for batch_start in range(0, n_trs, batch_size):
        batch_end = min(batch_start + batch_size, n_trs)
        batch_frames = frames[batch_start:batch_end]

        if len(batch_frames) == 0:
            continue

        inputs = processor(images=batch_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states: tuple of (n_layers+1) tensors, each (batch, n_patches+1, dim)
        hidden_states = outputs.hidden_states

        for layer_idx in range(n_layers):
            # Mean-pool patch tokens (exclude CLS at index 0).
            # Patch tokens encode spatial structure (V1-IT hierarchy);
            # CLS is classification-biased and discards spatial info.
            patch_tokens = hidden_states[layer_idx + 1][:, 1:, :]
            pooled = patch_tokens.mean(dim=1)  # (batch, dim)
            layer_features[layer_idx][batch_start:batch_end] = (
                pooled.cpu().float().numpy()
            )

    return layer_features


def _extract_video_frames(video_path, n_trs):
    """Extract 1 frame per TR center from video using a single ffmpeg call.

    Uses ffmpeg's select filter to extract frames at specific timestamps
    in one pass, avoiding 500+ subprocess invocations.

    Args:
        video_path: Path to video file.
        n_trs: Number of TRs.

    Returns:
        list of PIL.Image: One frame per TR.
    """
    import subprocess
    import tempfile
    from PIL import Image

    # Build select filter for TR center timestamps
    # select='eq(n,F0)+eq(n,F1)+...' is too long; use fps + trim approach instead.
    # Extract at ~1/TR_DURATION fps, which gives ~1 frame per TR.
    target_fps = 1.0 / TR_DURATION  # ~0.671 fps

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = str(Path(tmpdir) / "frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={target_fps:.6f}",
            "-q:v", "2",
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Load extracted frames (ffmpeg numbers from 1)
        frames = []
        for i in range(1, n_trs + 1):
            fpath = Path(tmpdir) / f"frame_{i:05d}.jpg"
            if fpath.exists():
                frames.append(Image.open(fpath).convert("RGB"))
            else:
                # Fewer frames than TRs — duplicate last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # Create blank frame as fallback
                    frames.append(Image.new("RGB", (224, 224)))
                logger.warning("Missing video frame %d — using fallback", i)

    return frames


# ---------------------------------------------------------------------------
# Text feature extraction (LLaMA)
# ---------------------------------------------------------------------------


def _compute_tr_token_boundaries(
    offsets: np.ndarray,
    char_lengths: np.ndarray,
) -> np.ndarray:
    """Map cumulative per-TR character lengths to token-count boundaries.

    Per §2.3.1 of `2026-05-01_08c_llama_local_window_design.md`: a single
    full-text tokenization yields token end-character offsets; per-TR
    boundaries fall out of `np.searchsorted` on those offsets. The result is
    monotone non-decreasing by construction (cumulative `char_lengths` are
    non-decreasing and `searchsorted` is monotone).

    Args:
        offsets: (seq_len, 2) int array of (char_start, char_end) per token,
            from `tokenizer(..., return_offsets_mapping=True)`.
        char_lengths: (n_trs,) int array — cumulative character length of the
            transcript through TR `t`.

    Returns:
        (n_trs,) int array — `tr_token_boundaries[t]` is the number of tokens
        whose end-character is ≤ `char_lengths[t]` (BOS-inclusive).
    """
    return np.searchsorted(offsets[:, 1], char_lengths, side="right").astype(
        np.int64
    )


def _local_window_pool(
    hidden_states: np.ndarray,
    tr_token_boundaries: np.ndarray,
    window_trs: int,
    B: int,
) -> np.ndarray:
    """Causal local-window mean-pool of hidden states per TR.

    Per §2.3.2 + §2.6 of the design doc. At TR `t` the window covers TRs
    `(t - W, t]` (W TRs of cumulative transcript). Token range is
    `[lo : hi]` half-open with::

        lo_count = tr_token_boundaries[t - W] if t >= W else 0
        hi       = tr_token_boundaries[t]
        lo       = max(B, lo_count)        # BOS clamp

    `B = 1` excludes the LLaMA BOS (`<|begin_of_text|>`) at index 0; `B = 0`
    means no BOS in the input. Empty windows (`hi <= lo`) emit a zero vector
    per §2.5 v1 policy (cortical-encoding precedent: Vaidya et al. 2022).

    Args:
        hidden_states: (seq_len, dim) array — hidden states for one layer.
        tr_token_boundaries: (n_trs,) array of token-count boundaries.
        window_trs: W in TR units; must be ≥ 1.
        B: BOS clamp (1 if BOS at index 0, else 0).

    Returns:
        (n_trs, dim) float32 features.
    """
    if window_trs < 1:
        raise ValueError(f"window_trs must be ≥ 1, got {window_trs}")
    n_trs = len(tr_token_boundaries)
    seq_len, dim = hidden_states.shape
    features = np.zeros((n_trs, dim), dtype=np.float32)
    for t in range(n_trs):
        if t >= window_trs:
            lo_count = int(tr_token_boundaries[t - window_trs])
        else:
            lo_count = 0
        hi = int(tr_token_boundaries[t])
        # `tr_token_boundaries` is computed against the full text but the
        # forward pass may truncate (`max_length`); clamp to actual seq_len.
        hi = min(hi, seq_len)
        lo = max(B, lo_count)
        if hi > lo:
            features[t] = hidden_states[lo:hi].mean(axis=0)
        # else: leave the pre-zeroed row.
    return features


def extract_text_features(
    transcript_data,
    model,
    tokenizer,
    n_trs,
    window_trs: int = 4,
    device: str = "cuda",
):
    """Extract layer-wise text features via single forward pass + local-window pool.

    Implements the local-window readout from
    `the design notes`. The
    earlier cumulative-mean readout was a martingale that converged to a
    per-run constant by mid-run (adjacent-TR cosine = 1.0000); this fix
    integrates only tokens whose end-character falls in the most recent
    `window_trs` TRs of cumulative transcript.

    Strategy:
    1. Single forward pass on the FULL run (LLaMA's causal attention is
       unchanged — token at position `i` still attends only to `0..i`).
    2. Single full-text tokenization with `return_offsets_mapping=True` to
       guarantee monotone per-TR boundaries (§2.3.1).
    3. Per-TR readout = mean of hidden states whose end-character is in
       `(char_lengths[t - W], char_lengths[t]]`, with BOS excluded (§2.3.2,
       §2.6). Empty windows emit zero (§2.5 v1).

    Args:
        transcript_data: list of str, length `n_trs`. `transcript_data[t]` is
            the cumulative text through TR `t`.
        model: Loaded LLaMA model.
        tokenizer: A *fast* tokenizer (e.g. `LlamaTokenizerFast`); the slow
            tokenizer does not support `return_offsets_mapping`.
        n_trs: Number of TRs.
        window_trs: Local-window span in TRs (default 4; sweep grid {1,3,6,9}
            per §2.4). Must be ≥ 1.
        device: Torch device.

    Returns:
        dict: {layer_idx: np.ndarray of shape (n_trs, hidden_dim)}.

    References:
        - Caucheteux & King 2022 (mean-pooling for distributed semantics).
        - Caucheteux et al. 2023 (NHB) — bounded-context cortical encoding.
        - Antonello et al. 2024 (NeurIPS) — local-window LLM-fMRI alignment.
    """
    assert getattr(tokenizer, "is_fast", False), (
        "extract_text_features requires a fast tokenizer "
        "(`return_offsets_mapping=True` is unsupported by slow tokenizers); "
        "got is_fast=False"
    )
    if window_trs < 1:
        raise ValueError(f"window_trs must be ≥ 1, got {window_trs}")

    import torch

    n_layers = MODEL_REGISTRY["llama-3.2-3b"]["n_layers"]
    dim = MODEL_REGISTRY["llama-3.2-3b"]["hidden_dim"]
    layer_features = {
        i: np.zeros((n_trs, dim), dtype=np.float32) for i in range(n_layers)
    }

    # Cumulative transcript: the last non-empty TR holds the full run text.
    full_text = ""
    for t in range(n_trs - 1, -1, -1):
        if transcript_data[t] and transcript_data[t].strip():
            full_text = transcript_data[t]
            break
    if not full_text:
        logger.warning("No text in any TR — returning zeros")
        return layer_features

    # Per-TR cumulative character lengths (§2.3.1).
    char_lengths = np.array(
        [len(transcript_data[t] or "") for t in range(n_trs)],
        dtype=np.int64,
    )

    # Single full-text tokenize with offset mapping (§2.3.1).
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    offsets = enc["offset_mapping"][0].cpu().numpy()  # (seq_len, 2)
    input_ids = enc["input_ids"][0]                   # (seq_len,)

    # BOS clamp — §2.3.2 / §2.6.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    B = 1 if (bos_id is not None and int(input_ids[0]) == int(bos_id)) else 0

    tr_token_boundaries = _compute_tr_token_boundaries(offsets, char_lengths)

    # Forward pass — `return_offsets_mapping` is not a model input.
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }
    with torch.no_grad():
        outputs = model(**model_inputs)
    hidden_states = outputs.hidden_states  # tuple of (n_layers + 1) tensors

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1][0].cpu().float().numpy()  # (seq_len, dim)
        layer_features[layer_idx] = _local_window_pool(
            h, tr_token_boundaries, window_trs=window_trs, B=B,
        )

    return layer_features


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------


def load_transcript_friends_m10(tsv_path, n_trs=None):
    """Load TR-aligned transcript from algonauts Friends/Movie10 TSV.

    Builds cumulative text context per TR.

    Args:
        tsv_path: Path to TR-aligned TSV with columns: text_per_tr,
            words_per_tr, onsets_per_tr, durations_per_tr.
        n_trs: If provided, truncate/pad to this many TRs.

    Returns:
        list of str: cumulative_text[t] = all text from TR 0 through TR t.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    texts_per_tr = df["text_per_tr"].fillna("").tolist()
    actual_trs = len(texts_per_tr)

    if n_trs is not None and actual_trs > n_trs:
        texts_per_tr = texts_per_tr[:n_trs]
    elif n_trs is not None and actual_trs < n_trs:
        texts_per_tr.extend([""] * (n_trs - actual_trs))

    # Build cumulative context
    cumulative = []
    running = ""
    for text in texts_per_tr:
        if text:
            running += text + " "
        cumulative.append(running.strip())

    return cumulative


def load_transcript_hp(events_path, n_trs):
    """Load word-level HP events.tsv and build cumulative text per TR.

    HP is RSVP at 2 Hz: each word presented for 0.5s on screen.
    Maps words to TRs using onset time, then builds cumulative context.

    Args:
        events_path: Path to BIDS events.tsv with columns: word, onset,
            duration. Silence markers are '+'.
        n_trs: Number of TRs.

    Returns:
        list of str: cumulative_text[t] = all words from TR 0 through TR t.
    """
    df = pd.read_csv(events_path, sep="\t")

    # Filter out silence markers
    words_df = df[df["word"] != "+"].copy()
    words_df = words_df[words_df["word"].notna()].copy()

    # Map each word to its TR
    words_per_tr = [""] * n_trs
    for _, row in words_df.iterrows():
        tr_idx = int(row["onset"] / TR_DURATION)
        if 0 <= tr_idx < n_trs:
            words_per_tr[tr_idx] += row["word"] + " "

    # Build cumulative context
    cumulative = []
    running = ""
    for text in words_per_tr:
        if text:
            running += text
        cumulative.append(running.strip())

    return cumulative


def load_transcript_pp(transcript_path, section, n_trs, onset_s):
    """Load Petit Prince transcript and build cumulative text per TR.

    PP is audiobook listening. The transcript is a shared word-level CSV
    (``lpp{EN,FR}_word_information.csv``) with columns ``word, lemma, onset,
    offset, section, ...``. Word onsets are in seconds **relative to the
    start of the section audio** (not scan-time), so the scan-time TR index
    for each word is computed from the events.tsv pre-stimulus silence:

        tr_idx = int((onset_s + word_onset) / TR_DURATION)

    Args:
        transcript_path: Path to ``lpp{EN,FR}_word_information.csv``.
        section: Section number (1–9) to filter words for.
        n_trs: Target number of TRs (pre-silence padding + audio duration).
        onset_s: Scan-time onset of the section audio, read live from the
            BIDS events.tsv (typically 4.0 s).

    Returns:
        list of str: cumulative_text[t] = all words spoken from TR 0 through
        TR t. TRs before the first word return an empty string.
    """
    df = pd.read_csv(transcript_path)
    # Filter to this section
    df = df[df["section"] == section].copy()
    if df.empty:
        return [""] * n_trs

    # Drop any rows with missing word text (shouldn't happen but be safe)
    df = df[df["word"].notna()].copy()
    df = df.sort_values("onset").reset_index(drop=True)

    # Bucket each word into its scan-time TR
    words_per_tr = [""] * n_trs
    for _, row in df.iterrows():
        tr_idx = int((onset_s + float(row["onset"])) / TR_DURATION)
        if 0 <= tr_idx < n_trs:
            words_per_tr[tr_idx] += str(row["word"]) + " "

    # Build cumulative context (same convention as Friends/HP)
    cumulative = []
    running = ""
    for text in words_per_tr:
        if text:
            running += text
        cumulative.append(running.strip())

    return cumulative


# ---------------------------------------------------------------------------
# Audio extraction from video files
# ---------------------------------------------------------------------------


def extract_audio_from_video(video_path, output_wav_path, sample_rate=AUDIO_SAMPLE_RATE):
    """Extract mono audio from video file using ffmpeg.

    Args:
        video_path: Path to input video (mkv).
        output_wav_path: Path for output WAV file.
        sample_rate: Target sample rate (default 16 kHz for Wav2VecBert).

    Returns:
        Path: output_wav_path
    """
    import subprocess

    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),  # resample
        "-ac", "1",  # mono
        str(output_wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Extracted audio: {video_path} -> {output_wav_path}")
    return output_wav_path


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------


def zscore_within_run(features):
    """Z-score features within a single run (per component).

    Consistent with 07a/08a normalization pattern.

    Args:
        features: np.ndarray of shape (n_trs, dim).

    Returns:
        np.ndarray of shape (n_trs, dim), z-scored per column.
    """
    mean = np.nanmean(features, axis=0, keepdims=True)
    std = np.nanstd(features, axis=0, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for constant features
    return (features - mean) / std


# ---------------------------------------------------------------------------
# Metadata and summary helpers
# ---------------------------------------------------------------------------


def save_extraction_metadata(
    out_dir,
    model_key,
    stimulus,
    n_runs,
    n_trs_per_run,
    run_ids,
    **extra_fields,
):
    """Save extraction metadata JSON.

    Args:
        out_dir: Output directory for this model/stimulus.
        model_key: Key into MODEL_REGISTRY.
        stimulus: Stimulus name.
        n_runs: Number of runs processed.
        n_trs_per_run: dict of {run_id: n_trs}.
        run_ids: list of run IDs processed.
        **extra_fields: Per-model audit fields merged into the output JSON
            (e.g. ``pooling="local_window"``, ``window_trs=N``, ``design_doc=...``
            for LLaMA; ``pooling="chunked_mean"`` for audio if desired).
    """
    info = MODEL_REGISTRY[model_key]
    metadata = {
        "model_key": model_key,
        "hf_id": info["hf_id"],
        "n_layers": info["n_layers"],
        "hidden_dim": info["hidden_dim"],
        "stimulus": stimulus,
        "n_runs": n_runs,
        "run_ids": run_ids,
        "n_trs_per_run": n_trs_per_run,
        "tr_duration_s": TR_DURATION,
    }
    metadata.update(extra_fields)

    out_path = Path(out_dir) / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {out_path}")


def save_layer_features(out_dir, layer_idx, run_id, features):
    """Save features for one layer and one run.

    Args:
        out_dir: Base output directory for model/stimulus.
        layer_idx: Layer index (0-based, transformer layers only).
        run_id: Run identifier string.
        features: np.ndarray of shape (n_trs, dim).
    """
    layer_dir = Path(out_dir) / f"layer_{layer_idx:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    out_path = layer_dir / f"{run_id}_raw.npy"
    np.save(out_path, features.astype(np.float32))
