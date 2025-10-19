import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    get_speech_timestamps_np,
)


def test_compare_np_and_torch_on_wav():
    model = load_silero_vad(onnx=True)

    audio_torch = read_audio("tests/data/test.wav", sampling_rate=16000)
    audio_np = audio_torch.numpy().astype(np.float32, copy=False)

    ts_torch = get_speech_timestamps(
        audio_torch,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    ts_np = get_speech_timestamps_np(
        audio_np,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    assert ts_np == ts_torch


def test_compare_np_and_torch_on_opus():
    model = load_silero_vad(onnx=True)

    audio_torch = read_audio("tests/data/test.opus", sampling_rate=16000)
    audio_np = audio_torch.numpy().astype(np.float32, copy=False)

    ts_torch = get_speech_timestamps(
        audio_torch,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    ts_np = get_speech_timestamps_np(
        audio_np,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    assert ts_np == ts_torch


def test_compare_np_and_torch_on_mp3():
    model = load_silero_vad(onnx=True)

    audio_torch = read_audio("tests/data/test.mp3", sampling_rate=16000)
    audio_np = audio_torch.numpy().astype(np.float32, copy=False)

    ts_torch = get_speech_timestamps(
        audio_torch,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    ts_np = get_speech_timestamps_np(
        audio_np,
        model,
        threshold=0.5,
        sampling_rate=16000,
        visualize_probs=False,
        return_seconds=False,
    )

    assert ts_np == ts_torch