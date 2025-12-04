import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf


def _add_src_to_path():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_src_to_path()

from silero_vad import NumpyOnnxWrapper, get_speech_timestamps_np  # noqa: E402

# 进程内缓存模型
_process_model = None


def _locate_onnx_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "silero_vad", "data"))
    for name in ["silero_vad_v6_16k_op15.onnx", "silero_vad_v6.onnx"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No ONNX model found in {base}")


def _get_model():
    """每个进程懒加载一次模型"""
    global _process_model
    if _process_model is None:
        model_path = _locate_onnx_path()
        _process_model = NumpyOnnxWrapper(model_path, force_onnx_cpu=True)
    return _process_model


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    return audio


def _validate_sr(sr: int):
    if sr not in (8000, 16000) and (sr % 16000 != 0):
        raise ValueError("Sampling rate must be 8000, 16000, or a multiple of 16000")


def concat_speech_segments(audio: np.ndarray, tss: List[dict]) -> np.ndarray:
    if not tss:
        return np.array([], dtype=np.float32)
    segments = [audio[seg['start']:seg['end']] for seg in tss]
    return np.concatenate(segments)


def _worker_process_file(task: dict) -> tuple:
    """子进程处理函数，返回 (filename, success)"""
    filepath = task['filepath']
    filename = os.path.basename(filepath)
    
    try:
        model = _get_model()
        
        if task['overwrite']:
            out_path = filepath
        else:
            base, _ = os.path.splitext(filepath)
            out_path = f"{base}_no_silence.wav"

        audio, sr = sf.read(filepath, dtype="float32")
        audio = _ensure_mono_float32(audio)
        _validate_sr(sr)

        tss = get_speech_timestamps_np(
            audio,
            model,
            threshold=task['threshold'],
            sampling_rate=sr,
            min_speech_duration_ms=task['min_speech_ms'],
            max_speech_duration_s=task['max_speech_s'],
            min_silence_duration_ms=task['min_silence_ms'],
            min_silence_at_max_speech=task['min_silence_at_max_speech'],
            speech_pad_ms=task['pad_ms'],
            return_seconds=False,
        )

        if tss:
            concatenated = concat_speech_segments(audio, tss)
            sf.write(out_path, concatenated, sr, subtype="PCM_16")
        
        return (filename, True)
    except Exception:
        return (filename, False)


def get_audio_files(directory: str, recursive: bool = False) -> List[str]:
    extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if Path(f).suffix.lower() in extensions:
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(directory, f))
    return files


def load_processed_log(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def process_directory(args):
    """多进程处理目录"""
    input_dir = args.input
    audio_files = get_audio_files(input_dir, recursive=args.recursive)
    
    if not audio_files:
        print("没有找到音频文件")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.basename(os.path.realpath(input_dir))
    log_path = os.path.join(script_dir, f"{dir_name}.processed.log")
    
    processed = load_processed_log(log_path)
    pending_files = [f for f in audio_files if os.path.basename(f) not in processed]
    
    total = len(audio_files)
    done_count = len(processed)
    pending_count = len(pending_files)
    
    print(f"总文件数: {total} | 已处理: {done_count} | 待处理: {pending_count} | 进程数: {args.workers}")
    
    if pending_count == 0:
        print("所有文件已处理完成")
        return

    # 构建任务列表
    tasks = [{
        'filepath': f,
        'overwrite': args.overwrite,
        'threshold': args.threshold,
        'min_speech_ms': args.min_speech_ms,
        'min_silence_ms': args.min_silence_ms,
        'max_speech_s': args.max_speech_s,
        'min_silence_at_max_speech': args.min_silence_at_max_speech,
        'pad_ms': args.pad_ms,
    } for f in pending_files]

    completed = 0
    success = 0
    
    # 主进程统一写日志，避免并发问题
    with open(log_path, 'a') as log_file:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker_process_file, t): t for t in tasks}
            
            for future in as_completed(futures):
                filename, ok = future.result()
                completed += 1
                if ok:
                    success += 1
                    log_file.write(filename + '\n')
                    log_file.flush()
                
                print(f"\r进度: {completed}/{pending_count} (成功: {success})", end="", flush=True)

    print(f"\n完成! 成功: {success}/{pending_count}")


def process_single(args):
    """处理单个文件"""
    if args.overwrite:
        output_path = args.input
    elif args.output is None:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_no_silence.wav"
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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
        print("No speech detected, output file will not be created.")
        return

    concatenated = concat_speech_segments(audio, tss)

    total_audio_sec = len(audio) / sr
    kept_audio_sec = len(concatenated) / sr
    removed_sec = total_audio_sec - kept_audio_sec
    kept_ratio = (kept_audio_sec / total_audio_sec * 100.0) if total_audio_sec > 0 else 0.0

    print(f"Original: {total_audio_sec:.2f}s")
    print(f"After removing silence: {kept_audio_sec:.2f}s ({kept_ratio:.1f}%)")
    print(f"Removed: {removed_sec:.2f}s of silence")
    print(f"Detected {len(tss)} speech segments")

    sf.write(output_path, concatenated, sr, subtype="PCM_16")
    if args.overwrite:
        print(f"Overwritten: {output_path}")
    else:
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove silence and concatenate speech segments using Silero VAD"
    )
    parser.add_argument("input", help="Path to input audio file or directory")
    parser.add_argument("-o", "--output", default=None, help="Output file path (single file mode only)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original input file(s)")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel workers (directory mode)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Speech threshold")
    parser.add_argument("--min_speech_ms", type=int, default=250, help="Minimum speech duration in ms")
    parser.add_argument("--min_silence_ms", type=int, default=1000, help="Minimum silence duration in ms")
    parser.add_argument("--max_speech_s", type=float, default=10000, help="Maximum speech duration in seconds")
    parser.add_argument("--min_silence_at_max_speech", type=float, default=98, help="Minimum silence (ms) at max speech")
    parser.add_argument("--pad_ms", type=int, default=30, help="Padding around each speech segment in ms")
    parser.add_argument("--visualize", action="store_true", help="Visualize probability curve (single file mode)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search subdirectories")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args)
    else:
        process_single(args)


if __name__ == "__main__":
    main()
