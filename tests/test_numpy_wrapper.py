import os
import sys
import numpy as np

# 优先使用本地源码目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from silero_vad import load_silero_vad, get_speech_timestamps_np, NumpyOnnxWrapper, read_audio


def _locate_onnx_path() -> str:
    """Locate an ONNX model file within the source tree for tests.
    Preference order: silero_vad_v6.onnx -> silero_vad_v6_16k_op15.onnx -> silero_vad_half.onnx
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "silero_vad", "data"))
    candidates = [
        "silero_vad_v6.onnx",
        "silero_vad_v6_16k_op15.onnx",
    ]
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No ONNX model file found under {base}")


def test_numpy_wrapper_matches_existing_wrapper_on_wav():
    # 1) 读取测试音频为 numpy
    wav_torch = read_audio("tests/data/test.wav", sampling_rate=16000)
    wav_np = wav_torch.numpy().astype(np.float32, copy=False)

    # 2) 现有 OnnxWrapper（通过 loader）
    model_existing = load_silero_vad(onnx=True)
    ts_existing = get_speech_timestamps_np(
        wav_np,
        model_existing,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )

    # 3) 新的纯 numpy 包装器
    onnx_path = _locate_onnx_path()
    model_numpy = NumpyOnnxWrapper(onnx_path, force_onnx_cpu=True)
    ts_numpy = get_speech_timestamps_np(
        wav_np,
        model_numpy,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )

    assert ts_numpy == ts_existing

def test_numpy_wrapper_matches_existing_wrapper_on_opus():
    audio_torch = read_audio("tests/data/test.opus", sampling_rate=16000)
    audio_np = audio_torch.numpy().astype(np.float32, copy=False)

    model_existing = load_silero_vad(onnx=True)
    ts_existing = get_speech_timestamps_np(
        audio_np,
        model_existing,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )

    onnx_path = _locate_onnx_path()
    model_numpy = NumpyOnnxWrapper(onnx_path, force_onnx_cpu=True)
    ts_numpy = get_speech_timestamps_np(
        audio_np,
        model_numpy,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )
    assert ts_numpy == ts_existing
def test_numpy_wrapper_matches_existing_wrapper_on_mp3():
    audio_torch = read_audio("tests/data/test.mp3", sampling_rate=16000)
    audio_np = audio_torch.numpy().astype(np.float32, copy=False)
    onnx_path = _locate_onnx_path()
    model_numpy = NumpyOnnxWrapper(onnx_path, force_onnx_cpu=True)
    ts_numpy = get_speech_timestamps_np(
        audio_np,
        model_numpy,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )

    model_existing = load_silero_vad(onnx=True)
    ts_existing = get_speech_timestamps_np(
        audio_np,
        model_existing,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=False,
        visualize_probs=False,
    )
    assert ts_numpy == ts_existing