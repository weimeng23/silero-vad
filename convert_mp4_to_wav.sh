#!/bin/bash

# 使用方法: ./convert_mp4_to_wav.sh <输入目录> <输出目录> [并发数]

INPUT_DIR="$1"
OUTPUT_DIR="$2"
JOBS="${3:-8}" # 默认8个并发

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <输入目录> <输出目录> [并发数]"
    echo "示例: $0 /path/to/mp4s /path/to/wavs 8"
    exit 1
fi

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 统一日志文件（与输出 wav 同目录），每次运行前清空
LOG_FILE="${OUTPUT_DIR}.log"
: > "$LOG_FILE"

# 转换函数
convert_file() {
    local input_file="$1"
    local output_dir="$2"
    local filename=$(basename "$input_file" .mp4)
    local output_file="$output_dir/${filename}.wav"

    if ffmpeg -threads 1 \
        -i "$input_file" \
        -vn \
        -ar 16000 \
        -ac 1 \
        -sample_fmt s16 \
        -acodec pcm_s16le \
        -y \
        "$output_file" \
        -loglevel error 2>> "$LOG_FILE"; then
        echo "✓ 成功转换: $(basename "$input_file")"
    else
        echo "✗ 转换失败: $(basename "$input_file")"
    fi
}

# 导出函数供parallel使用
export -f convert_file
export LOG_FILE

# 查找MP4文件并并发处理
find "$INPUT_DIR" -name "*.mp4" -type f \
                                        | parallel -j "$JOBS" convert_file {} "$OUTPUT_DIR"

echo "转换完成!"
