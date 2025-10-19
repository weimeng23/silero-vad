import os
import sys
import numpy as np
import torch

# 优先导入本地源码版本
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from silero_vad import NumpyOnnxWrapper, load_silero_vad


def _onnx_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "silero_vad", "data"))
    for name in ["silero_vad_v6.onnx", "silero_vad_v6_16k_op15.onnx"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(base)


def test_forward_single_window_equivalence():
    sr = 16000
    win = 512  # for 16k
    x_np = np.random.randn(win).astype(np.float32)

    model_np = NumpyOnnxWrapper(_onnx_path(), force_onnx_cpu=True)
    model_np.reset_states(1)
    out_np = model_np(x_np, sr)

    model_torch = load_silero_vad(onnx=True)
    model_torch.reset_states(1)
    x_t = torch.from_numpy(x_np)
    out_t = model_torch(x_t, sr).cpu().numpy()

    assert np.size(out_np) == 1 and np.size(out_t) == 1
    np.testing.assert_allclose(np.asarray(out_np).reshape(-1)[0], np.asarray(out_t).reshape(-1)[0], rtol=1e-6, atol=1e-6)


def test_forward_batch_window_equivalence():
    sr = 16000
    win = 512
    batch = 3
    x_np = np.random.randn(batch, win).astype(np.float32)

    model_np = NumpyOnnxWrapper(_onnx_path(), force_onnx_cpu=True)
    model_np.reset_states(batch)
    out_np = model_np(x_np, sr)

    model_torch = load_silero_vad(onnx=True)
    model_torch.reset_states(batch)
    x_t = torch.from_numpy(x_np)
    out_t = model_torch(x_t, sr).cpu().numpy()

    assert out_np.shape[0] == batch and out_t.shape[0] == batch
    np.testing.assert_allclose(out_np.reshape(batch, -1), out_t.reshape(batch, -1), rtol=1e-6, atol=1e-6)


def test_audio_forward_batch_equivalence():
    sr = 16000
    win = 512
    batch = 2
    num_chunks = 5
    x_np = np.random.randn(batch, win * num_chunks).astype(np.float32)

    model_np = NumpyOnnxWrapper(_onnx_path(), force_onnx_cpu=True)
    model_np.reset_states(batch)
    seq_np = model_np.audio_forward(x_np, sr)

    model_torch = load_silero_vad(onnx=True)
    model_torch.reset_states(batch)
    x_t = torch.from_numpy(x_np)
    seq_t = model_torch.audio_forward(x_t, sr).cpu().numpy()

    assert seq_np.shape == (batch, num_chunks)
    assert seq_t.shape == (batch, num_chunks)
    np.testing.assert_allclose(seq_np, seq_t, rtol=1e-6, atol=1e-6)


