import os
import sys
import argparse
from typing import List

import numpy as np
import soundfile as sf


def _add_src_to_path():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_src_to_path()

from silero_vad import NumpyOnnxWrapper, get_speech_timestamps_np  # noqa: E402


def _locate_onnx_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "silero_vad", "data"))
    for name in ["silero_vad_v6.onnx", "silero_vad_v6_16k_op15.onnx"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No ONNX model found in {base}")


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    return audio


def _validate_sr(sr: int):
    if sr not in (8000, 16000) and (sr % 16000 != 0):
        raise ValueError("Sampling rate must be 8000, 16000, or a multiple of 16000")


def split_by_timestamps(audio: np.ndarray, sr: int, tss: List[dict]) -> List[np.ndarray]:
    return [audio[seg['start']:seg['end']] for seg in tss]


def main():
    parser = argparse.ArgumentParser(description="Split long audio into segments using Silero VAD (numpy + onnxruntime)")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--out_dir", default="./out_segments", help="Output directory to save segments")
    parser.add_argument("--threshold", type=float, default=0.5, help="Speech threshold")
    parser.add_argument("--min_speech_ms", type=int, default=250, help="Minimum speech duration in ms")
    parser.add_argument("--min_silence_ms", type=int, default=2000, help="Minimum silence duration in ms")
    parser.add_argument("--max_speech_s", type=float, default=60, help="Maximum speech duration in seconds")
    parser.add_argument("--min_silence_at_max_speech", type=float, default=100, help="Minimum silence (ms) used when max_speech_s is reached")
    parser.add_argument("--pad_ms", type=int, default=30, help="Padding around each speech segment in ms")
    parser.add_argument("--visualize", action="store_true", help="Visualize probability curve")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    audio, sr = sf.read(args.input, dtype="float32")
    audio = _ensure_mono_float32(audio)
    _validate_sr(sr)

    model_path = _locate_onnx_path()
    model = NumpyOnnxWrapper(model_path, force_onnx_cpu=True)

    tss = get_speech_timestamps_np(
        audio,
        model,
        threshold=args.threshold,
        sampling_rate=sr,
        min_speech_duration_ms=args.min_speech_ms,
        max_speech_duration_s=args.max_speech_s,
        min_silence_duration_ms=args.min_silence_ms,
        min_silence_at_max_speech=args.min_silence_at_max_speech,
        speech_pad_ms=args.pad_ms,
        visualize_probs=args.visualize,
        return_seconds=False,
    )

    if not tss:
        print("No speech detected.")
        return

    segments = split_by_timestamps(audio, sr, tss)
    total_audio_sec = len(audio) / sr
    kept_audio_sec = sum(len(seg) for seg in segments) / sr
    kept_ratio = (kept_audio_sec / total_audio_sec * 100.0) if total_audio_sec > 0 else 0.0
    print(f"Kept {kept_audio_sec:.2f}s / {total_audio_sec:.2f}s ({kept_ratio:.2f}%)")
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    for idx, seg in enumerate(segments):
        out_path = os.path.join(args.out_dir, f"{base_name}_seg{idx:03d}.wav")
        # 保存为16位 PCM；soundfile 会根据 dtype 推断，显式写 subtype 更稳
        sf.write(out_path, seg, sr, subtype="PCM_16")
        print(f"Saved: {out_path} ({len(seg)/sr:.2f}s)")


if __name__ == "__main__":
    main()


