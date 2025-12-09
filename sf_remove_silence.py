import os
import sys
import argparse
import multiprocessing as mp
import time
from pathlib import Path
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

# 子进程共享变量（通过 initializer 传入）
_shared_counter = None
_shared_lock = None


def _init_worker(counter, lock):
    """子进程初始化：接收共享对象"""
    global _shared_counter, _shared_lock
    _shared_counter = counter
    _shared_lock = lock


def _locate_onnx_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "silero_vad", "data"))
    for name in ["silero_vad_v6_16k_op15.onnx", "silero_vad_v6.onnx"]:
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


def concat_speech_segments(audio: np.ndarray, tss: List[dict]) -> np.ndarray:
    if not tss:
        return np.array([], dtype=np.float32)
    segments = [audio[seg['start'] : seg['end']] for seg in tss]
    return np.concatenate(segments)


def _split_long_segment(seg: np.ndarray, max_samples: int) -> List[np.ndarray]:
    """单个段过长时再拆分"""
    if len(seg) <= max_samples:
        return [seg]
    return [seg[i : i + max_samples] for i in range(0, len(seg), max_samples)]


def _chunk_by_timestamps(audio: np.ndarray, tss: List[dict], sr: int, max_hours: float) -> List[np.ndarray]:
    """按时间戳聚合，累计长度超出限制则换新段"""
    if not tss:
        return []
    max_samples = int(max_hours * 3600 * sr) if max_hours > 0 else len(audio)
    if max_samples <= 0:
        max_samples = len(audio)

    chunks: List[np.ndarray] = []
    cur_segments: List[np.ndarray] = []
    cur_len = 0

    for ts in tss:
        seg = audio[ts["start"] : ts["end"]]
        seg_len = len(seg)

        # 如果单个段就超过 max，先把它切小块再处理
        if seg_len > max_samples:
            # 按照 max_samples or len(seg) 拆分
            split_parts = _split_long_segment(seg, max_samples)
            for part in split_parts:
                if cur_segments and cur_len + len(part) > max_samples:
                    chunks.append(np.concatenate(cur_segments))
                    cur_segments = []
                    cur_len = 0
                cur_segments.append(part)
                cur_len += len(part)
            continue

        if cur_segments and cur_len + seg_len > max_samples:
            chunks.append(np.concatenate(cur_segments))
            cur_segments = []
            cur_len = 0

        cur_segments.append(seg)
        cur_len += seg_len

    if cur_segments:
        chunks.append(np.concatenate(cur_segments))

    return chunks


def _write_output_segments(out_path: str, segments: List[np.ndarray], sr: int) -> List[str]:
    """写出多个段，1 段用原名，多段带 _partNNN"""
    os.makedirs(Path(out_path).parent, exist_ok=True)

    base = str(Path(out_path).with_suffix(""))
    ext = Path(out_path).suffix or ".wav"

    written_paths = []
    if not segments:
        return written_paths

    if len(segments) == 1:
        sf.write(out_path, segments[0], sr, subtype="PCM_16")
        return [out_path]

    for idx, seg in enumerate(segments):
        part_path = f"{base}_part{idx:03d}{ext}"
        sf.write(part_path, seg, sr, subtype="PCM_16")
        written_paths.append(part_path)

    return written_paths


def _worker_batch(args_tuple) -> tuple:
    """子进程：处理一批文件，返回 (成功文件名列表, 失败文件及原因列表, 删除文件列表)"""
    file_list, params = args_tuple
    global _shared_counter, _shared_lock

    # 模型初始化失败时，整批标记失败并推进进度，避免主进程卡死
    try:
        model = NumpyOnnxWrapper(_locate_onnx_path(), force_onnx_cpu=True)
    except Exception as e:
        failed_files = []
        with _shared_lock:
            for fp in file_list:
                failed_files.append((os.path.basename(fp), f"model_init: {str(e)[:100]}"))
                _shared_counter.value += 1
        return [], failed_files, []

    success_files = []
    failed_files = []  # [(filename, error_msg), ...]
    removed_files = []

    for filepath in file_list:
        filename = os.path.basename(filepath)
        try:
            if params['overwrite']:
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
                threshold=params['threshold'],
                sampling_rate=sr,
                min_speech_duration_ms=params['min_speech_ms'],
                max_speech_duration_s=params['max_speech_s'],
                min_silence_duration_ms=params['min_silence_ms'],
                min_silence_at_max_speech=params['min_silence_at_max_speech'],
                speech_pad_ms=params['pad_ms'],
                return_seconds=False,
            )

            has_speech = bool(tss)
            if has_speech:
                segments = _chunk_by_timestamps(audio, tss, sr, params['max_output_hours'])
                written = _write_output_segments(out_path, segments, sr)
                # 无写出也算失败
                if not written:
                    has_speech = False
                else:
                    success_files.append(filename)
            if not has_speech:
                # 无语音则删除已存在的输出文件（覆盖模式会删除原文件）
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                        removed_files.append(filename)
                except Exception:
                    pass
        except Exception as e:
            failed_files.append((filename, str(e)[:100]))  # 截断错误信息

        # 更新进度计数
        with _shared_lock:
            _shared_counter.value += 1

    return success_files, failed_files, removed_files


def get_audio_files(directory: str, recursive: bool = False) -> List[str]:
    # extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    extensions = {'.wav'}
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


def split_list(lst: list, n: int) -> List[list]:
    """将列表均匀分成 n 份"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_directory(args):
    """多进程处理目录（预分片模式）"""
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
    num_workers = min(args.workers, pending_count) if pending_count > 0 else 1

    print(f"总文件数: {total} | 已处理: {done_count} | 待处理: {pending_count} | 进程数: {num_workers}", flush=True)

    if pending_count == 0:
        print("所有文件已处理完成")
        return

    params = {
        'overwrite': args.overwrite,
        'threshold': args.threshold,
        'min_speech_ms': args.min_speech_ms,
        'min_silence_ms': args.min_silence_ms,
        'max_speech_s': args.max_speech_s,
        'min_silence_at_max_speech': args.min_silence_at_max_speech,
        'pad_ms': args.pad_ms,
        'max_output_hours': args.max_output_hours,
    }

    # 分片
    chunks = split_list(pending_files, num_workers)

    # 共享计数器（通过 initializer 传给子进程）
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # 启动进程池，用 initializer 传递共享对象
    pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(counter, lock))

    # 提交任务
    async_results = [pool.apply_async(_worker_batch, ((chunk, params),)) for chunk in chunks]
    pool.close()

    # 主进程轮询进度
    while True:
        with lock:
            current = counter.value
        print(f"\r进度: {current}/{pending_count}", end="", flush=True)
        if current >= pending_count:
            break
        time.sleep(1.0)

    pool.join()
    print()

    # 收集结果
    all_success = []
    all_failed = []
    all_removed = []
    for r in async_results:
        success, failed, removed = r.get()
        all_success.extend(success)
        all_failed.extend(failed)
        all_removed.extend(removed)

    # 写成功日志
    with open(log_path, 'a') as f:
        for name in all_success:
            f.write(name + '\n')

    # 写失败日志
    if all_failed:
        error_log_path = log_path.replace('.processed.log', '.errors.log')
        with open(error_log_path, 'a') as f:
            for name, err in all_failed:
                f.write(f"{name}\t{err}\n")
        print(f"错误日志: {error_log_path}")

    if all_removed:
        removed_log_path = log_path.replace('.processed.log', '.removed.log')
        with open(removed_log_path, 'a') as f:
            for name in all_removed:
                f.write(f"{name}\n")
        print(f"移除日志: {removed_log_path}")

    print(f"完成! 成功: {len(all_success)}/{pending_count}, 失败: {len(all_failed)}, 移除: {len(all_removed)}")


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

    has_speech = bool(tss)
    if has_speech:
        segments = _chunk_by_timestamps(audio, tss, sr, args.max_output_hours)
        if segments:
            total_audio_sec = len(audio) / sr
            kept_audio_sec = sum(len(seg) for seg in segments) / sr
            removed_sec = total_audio_sec - kept_audio_sec
            kept_ratio = (kept_audio_sec / total_audio_sec * 100.0) if total_audio_sec > 0 else 0.0

            print(f"Original: {total_audio_sec:.2f}s")
            print(f"After removing silence: {kept_audio_sec:.2f}s ({kept_ratio:.1f}%)")
            print(f"Removed: {removed_sec:.2f}s of silence")
            print(f"Detected {len(tss)} speech segments")

            written = _write_output_segments(output_path, segments, sr)
            if args.overwrite and len(written) == 1:
                print(f"Overwritten: {written[0]}")
            else:
                print("Saved:")
                for p in written:
                    print(f"  {p}")
        else:
            has_speech = False

    if not has_speech:
        # 无语音则删除输出（覆盖模式删除原文件）
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            print("No speech detected, output removed.")
        except Exception as e:
            print(f"No speech detected, failed to remove output: {e}")


def main():
    # 避免 fork 下第三方库卡死，强制使用 spawn
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove silence and concatenate speech segments using Silero VAD",
    )
    parser.add_argument("input", help="Path to input audio file or directory")
    parser.add_argument("-o", "--output", default=None, help="Output file path (single file mode only)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original input file(s)")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel workers (directory mode)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Speech threshold")
    parser.add_argument("--min_speech_ms", type=int, default=250, help="Minimum speech duration in ms")
    parser.add_argument("--min_silence_ms", type=int, default=1000, help="Minimum silence duration in ms")
    parser.add_argument("--max_speech_s", type=float, default=10000, help="Maximum speech duration in seconds")
    parser.add_argument(
        "--min_silence_at_max_speech", type=float, default=98, help="Minimum silence (ms) at max speech"
    )
    parser.add_argument("--pad_ms", type=int, default=30, help="Padding around each speech segment in ms")
    parser.add_argument("--visualize", action="store_true", help="Visualize probability curve (single file mode)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search subdirectories")
    parser.add_argument("--max_output_hours", type=float, default=3.0, help="Maximum duration per output file in hours")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args)
    else:
        process_single(args)


if __name__ == "__main__":
    main()
