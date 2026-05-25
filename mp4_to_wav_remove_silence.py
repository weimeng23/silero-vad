"""
MP4 -> 剔除静音后的 WAV 一体化脚本

功能：
  1. 使用 ffmpeg 将 mp4 转换为 16kHz/mono PCM（通过管道，不落盘中间文件）
  2. 使用 Silero VAD (numpy + onnxruntime) 检测语音段
  3. 拼接语音段并写出最终 WAV

用法：
  # 处理单个文件
  python mp4_to_wav_remove_silence.py input.mp4 -o output.wav

  # 批量处理目录
  python mp4_to_wav_remove_silence.py /path/to/mp4s --out_dir /path/to/wavs -j 16

  # 递归搜索子目录
  python mp4_to_wav_remove_silence.py /path/to/mp4s --out_dir /path/to/wavs -j 16 -r
"""

import os
import sys
import argparse
import subprocess
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


def _add_src_to_path():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_src_to_path()

from silero_vad import NumpyOnnxWrapper, get_speech_timestamps_np  # noqa: E402

# 子进程共享变量
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


def ffmpeg_decode_to_numpy(input_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """
    使用 ffmpeg 将音频/视频文件解码为 numpy float32 数组。
    通过管道传输，不生成中间文件。

    Returns:
        numpy float32 数组 (mono, 归一化到 [-1, 1])，失败返回 None
    """
    cmd = [
        "ffmpeg",
        "-threads", "1",
        "-i", input_path,
        "-vn",                  # 不要视频
        "-ar", str(sr),         # 采样率
        "-ac", "1",             # 单声道
        "-f", "s16le",          # 原始 PCM 16-bit little-endian
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        "pipe:1",               # 输出到 stdout
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,  # 10分钟超时
        )
        if proc.returncode != 0:
            return None

        # PCM s16le -> float32
        audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) == 0:
            return None
        return audio

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def _split_long_segment(seg: np.ndarray, max_samples: int) -> List[np.ndarray]:
    """单个段过长时再拆分"""
    if len(seg) <= max_samples:
        return [seg]
    return [seg[i: i + max_samples] for i in range(0, len(seg), max_samples)]


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
        seg = audio[ts["start"]: ts["end"]]
        seg_len = len(seg)

        if seg_len > max_samples:
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
    """子进程：处理一批文件，返回 (成功列表, 失败列表, 跳过列表)"""
    file_list, params = args_tuple
    global _shared_counter, _shared_lock

    # 模型初始化
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
    failed_files = []   # [(filename, error_msg), ...]
    skipped_files = []  # 无语音的文件

    sr = params['sample_rate']

    for filepath in file_list:
        filename = os.path.basename(filepath)
        stem = Path(filename).stem
        out_path = os.path.join(params['out_dir'], f"{stem}.wav")

        try:
            # 1. ffmpeg 解码
            audio = ffmpeg_decode_to_numpy(filepath, sr=sr)
            if audio is None:
                failed_files.append((filename, "ffmpeg decode failed"))
                with _shared_lock:
                    _shared_counter.value += 1
                continue

            # 2. VAD 检测
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

            # 3. 写出结果
            if tss:
                segments = _chunk_by_timestamps(audio, tss, sr, params['max_output_hours'])
                if segments:
                    _write_output_segments(out_path, segments, sr)
                    success_files.append(filename)
                else:
                    skipped_files.append(filename)
            else:
                skipped_files.append(filename)

        except Exception as e:
            failed_files.append((filename, str(e)[:100]))

        with _shared_lock:
            _shared_counter.value += 1

    return success_files, failed_files, skipped_files


def get_media_files(directory: str, recursive: bool = False) -> List[str]:
    """获取目录下的媒体文件"""
    extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.m4a', '.mp3', '.flac', '.ogg', '.wav'}
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
    return sorted(files)


def load_processed_log(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def split_list(lst: list, n: int) -> List[list]:
    """将列表均匀分成 n 份"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _fmt_duration(seconds: float) -> str:
    """格式化耗时：>1h 显示 HH:MM:SS，>1min 显示 MM:SS，否则显示秒"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def process_directory(args):
    """多进程批量处理目录"""
    start_time = time.time()
    input_dir = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    media_files = get_media_files(input_dir, recursive=args.recursive)
    if not media_files:
        print("没有找到媒体文件")
        return

    # 日志路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.basename(os.path.realpath(input_dir))
    log_path = os.path.join(script_dir, f"{dir_name}_mp4wav.processed.log")

    processed = load_processed_log(log_path)
    pending_files = [f for f in media_files if os.path.basename(f) not in processed]

    total = len(media_files)
    done_count = len(processed)
    pending_count = len(pending_files)
    num_workers = min(args.workers, pending_count) if pending_count > 0 else 1

    print(f"总文件数: {total} | 已处理: {done_count} | 待处理: {pending_count} | 进程数: {num_workers}", flush=True)

    if pending_count == 0:
        print("所有文件已处理完成")
        return

    params = {
        'out_dir': out_dir,
        'sample_rate': args.sample_rate,
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

    # 共享计数器
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(counter, lock))
    async_results = [pool.apply_async(_worker_batch, ((chunk, params),)) for chunk in chunks]
    pool.close()

    # 主进程轮询进度
    while True:
        with lock:
            current = counter.value
        elapsed = time.time() - start_time
        if current > 0:
            avg = elapsed / current
            eta = avg * (pending_count - current)
            print(f"\r进度: {current}/{pending_count} | 已用 {_fmt_duration(elapsed)} | 预计剩余 {_fmt_duration(eta)}",
                  end="", flush=True)
        else:
            print(f"\r进度: {current}/{pending_count} | 已用 {_fmt_duration(elapsed)}", end="", flush=True)
        if current >= pending_count:
            break
        time.sleep(1.0)

    pool.join()
    print()

    # 收集结果
    all_success = []
    all_failed = []
    all_skipped = []
    for r in async_results:
        success, failed, skipped = r.get()
        all_success.extend(success)
        all_failed.extend(failed)
        all_skipped.extend(skipped)

    # 写成功日志（成功 + 跳过都算已处理，不再重复处理）
    with open(log_path, 'a') as f:
        for name in all_success:
            f.write(name + '\n')
        for name in all_skipped:
            f.write(name + '\n')

    # 写失败日志
    if all_failed:
        error_log_path = log_path.replace('.processed.log', '.errors.log')
        with open(error_log_path, 'a') as f:
            for name, err in all_failed:
                f.write(f"{name}\t{err}\n")
        print(f"错误日志: {error_log_path}")

    # 写跳过日志（无语音）
    if all_skipped:
        skipped_log_path = log_path.replace('.processed.log', '.skipped.log')
        with open(skipped_log_path, 'a') as f:
            for name in all_skipped:
                f.write(name + '\n')
        print(f"跳过日志(无语音): {skipped_log_path}")

    total_elapsed = time.time() - start_time
    avg_per_file = total_elapsed / pending_count if pending_count else 0.0
    print(f"完成! 成功: {len(all_success)} | 跳过(无语音): {len(all_skipped)} | 失败: {len(all_failed)} / 总待处理: {pending_count}")
    print(f"总耗时: {_fmt_duration(total_elapsed)} | 平均: {avg_per_file:.2f}s/文件 | 进程数: {num_workers}")


def process_single(args):
    """处理单个文件"""
    start_time = time.time()
    if args.output:
        out_path = args.output
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = Path(args.input).stem
        out_path = os.path.join(args.out_dir, f"{stem}.wav")
    else:
        stem = Path(args.input).stem
        out_path = os.path.join(os.path.dirname(args.input), f"{stem}.wav")

    sr = args.sample_rate

    # 1. ffmpeg 解码
    print(f"解码: {args.input}")
    audio = ffmpeg_decode_to_numpy(args.input, sr=sr)
    if audio is None:
        print("错误: ffmpeg 解码失败，请检查输入文件和 ffmpeg 是否可用")
        sys.exit(1)

    print(f"音频长度: {len(audio) / sr:.2f}s")

    # 2. VAD
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
        return_seconds=False,
    )

    if not tss:
        print("未检测到语音，不生成输出文件")
        return

    # 3. 写出
    segments = _chunk_by_timestamps(audio, tss, sr, args.max_output_hours)
    if not segments:
        print("未检测到语音，不生成输出文件")
        return

    total_sec = len(audio) / sr
    kept_sec = sum(len(seg) for seg in segments) / sr
    removed_sec = total_sec - kept_sec
    kept_ratio = (kept_sec / total_sec * 100.0) if total_sec > 0 else 0.0

    print(f"原始: {total_sec:.2f}s | 保留: {kept_sec:.2f}s ({kept_ratio:.1f}%) | 剔除静音: {removed_sec:.2f}s")
    print(f"检测到 {len(tss)} 个语音段")

    written = _write_output_segments(out_path, segments, sr)
    for p in written:
        print(f"已保存: {p}")

    total_elapsed = time.time() - start_time
    print(f"总耗时: {_fmt_duration(total_elapsed)}")


def main():
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="MP4/媒体文件 -> 剔除静音后的 WAV（一体化，无中间文件）",
    )
    parser.add_argument("input", help="输入文件路径或目录")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径（单文件模式）")
    parser.add_argument("--out_dir", default=None, help="输出目录（目录模式必须指定，单文件模式可选）")
    parser.add_argument("-j", "--workers", type=int, default=8, help="并行进程数（目录模式）")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument("--sample_rate", type=int, default=16000, help="目标采样率")
    parser.add_argument("--threshold", type=float, default=0.5, help="语音概率阈值")
    parser.add_argument("--min_speech_ms", type=int, default=250, help="最小语音时长 (ms)")
    parser.add_argument("--min_silence_ms", type=int, default=2000, help="最小静音时长 (ms)")
    parser.add_argument("--max_speech_s", type=float, default=3600, help="最大语音段时长 (s)")
    parser.add_argument("--min_silence_at_max_speech", type=float, default=98, help="最大语音段处最小静音 (ms)")
    parser.add_argument("--pad_ms", type=int, default=30, help="语音段两侧填充 (ms)")
    parser.add_argument("--max_output_hours", type=float, default=1.0, help="单个输出文件最大时长 (小时)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        if not args.out_dir:
            print("错误: 目录模式必须指定 --out_dir")
            sys.exit(1)
        process_directory(args)
    else:
        process_single(args)


if __name__ == "__main__":
    main()
