#!/bin/bash
# File              : batch_mp4_to_wav_remove_silence.sh
# Description       : MP4 -> 剔除连续静音后的 WAV，仅落盘最终结果（中间 wav 用临时文件，处理完即删）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOVE_SILENCE_SCRIPT="$SCRIPT_DIR/sf_remove_silence.py"

if [ $# -lt 2 ]; then
    echo "用法: $0 <输入目录(mp4)> <输出目录(wav)> [并发数] [VAD参数...]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/mp4s /path/to/output"
    echo "  $0 /path/to/mp4s /path/to/output 8"
    echo "  $0 /path/to/mp4s /path/to/output 8 --threshold 0.5 --min_silence_ms 2000"
    echo ""
    echo "说明: 每个 mp4 经 ffmpeg 写入临时 wav，VAD 处理后直接写出到输出目录，"
    echo "      不保留含静音的完整中间 wav。目录模式参数 -j/-r 会被忽略。"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

JOBS=8
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    JOBS="$1"
    shift
fi

# 过滤目录模式专用参数（单文件流水线不需要）
VAD_PARAMS=""
while [ $# -gt 0 ]; do
    case "$1" in
        -j|--workers)
            shift 2
            ;;
        -r|--recursive)
            shift
            ;;
        *)
            VAD_PARAMS="$VAD_PARAMS $1"
            shift
            ;;
    esac
done

if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$REMOVE_SILENCE_SCRIPT" ]; then
    echo "错误: 找不到静音剔除脚本: $REMOVE_SILENCE_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}.log"
: > "$LOG_FILE"

process_file() {
    local input_file="$1"
    local output_dir="$2"
    local filename
    filename=$(basename "$input_file" .mp4)
    local output_file="$output_dir/${filename}.wav"
    local tmp_wav

    tmp_wav=$(mktemp --tmpdir="${TMPDIR:-/tmp}" --suffix=".wav" 2>/dev/null || mktemp --suffix=".wav")
    trap 'rm -f "$tmp_wav"' RETURN

    if ! ffmpeg -threads 1 \
        -i "$input_file" \
        -vn \
        -ar 16000 \
        -ac 1 \
        -sample_fmt s16 \
        -acodec pcm_s16le \
        -y \
        "$tmp_wav" \
        -loglevel error 2>> "$LOG_FILE"; then
        echo "✗ 转换失败: $(basename "$input_file")"
        return 1
    fi

    # 从临时 wav 读取，仅将去静音结果写入 output_file
    if ! python3 "$REMOVE_SILENCE_SCRIPT" "$tmp_wav" -o "$output_file" $VAD_PARAMS >> "$LOG_FILE" 2>&1; then
        echo "✗ 剔除静音失败: $(basename "$input_file")"
        rm -f "$output_file"
        return 1
    fi

    if [ ! -f "$output_file" ]; then
        echo "○ 无语音，已跳过: $(basename "$input_file")"
        return 0
    fi

    echo "✓ 完成: $(basename "$input_file") -> $(basename "$output_file")"
}

export -f process_file
export REMOVE_SILENCE_SCRIPT LOG_FILE VAD_PARAMS

total_files=$(find "$INPUT_DIR" -name "*.mp4" -type f | wc -l)
total_files=$(echo "$total_files" | tr -d ' \n')

echo "========================================"
echo "MP4 -> 去静音 WAV (单文件流水线)"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "并发数:   $JOBS"
echo "VAD参数:  ${VAD_PARAMS:-<默认>}"
echo "发现 mp4: $total_files 个"
echo "========================================"

if [ "${total_files:-0}" -eq 0 ]; then
    echo "错误: 输入目录中没有找到 mp4 文件"
    exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
    echo "错误: 需要 GNU parallel，请先安装"
    exit 1
fi

find "$INPUT_DIR" -name "*.mp4" -type f \
    | parallel -j "$JOBS" process_file {} "$OUTPUT_DIR"

echo ""
echo "全部完成! 输出目录: $OUTPUT_DIR"
echo "详细日志: $LOG_FILE"
