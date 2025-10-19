#!/bin/bash

# 并发批量音频拆分脚本
# 用法: ./batch_split_parallel.sh <输入目录> <输出目录> [并发数] [其他参数...]

set -e

# 参数检查
if [ $# -lt 2 ]; then
    echo "用法: $0 <输入目录> <输出目录> [并发数] [其他VAD参数...]"
    echo "示例: $0 ./audio_input ./audio_output 8 --threshold 0.5 --min_speech_ms 250"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
PARALLEL_JOBS="${3:-8}"  # 默认8个并发

# 提取其他 VAD 参数 (从第4个参数开始)
shift 2
if [ $# -gt 0 ] && [ "$1" != "" ]; then
    # 如果第3个参数看起来像数字,则跳过它(并发数)
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        shift
    fi
fi
VAD_PARAMS="$@"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取脚本所在目录，用于定位 Python 脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$(dirname "$SCRIPT_DIR")/sf_split_numpy.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 临时文件用于记录统计信息
TEMP_LOG=$(mktemp)
PROGRESS_FILE=$(mktemp)
trap "rm -f $TEMP_LOG $PROGRESS_FILE" EXIT

# 处理单个音频文件的函数
process_audio() {
    local input_file="$1"
    shift
    
    # 提取文件名(不含扩展名)作为音频ID
    local filename=$(basename "$input_file")
    local audio_id="${filename%.*}"
    
    # 创建该音频ID的输出目录
    local out_dir="${OUTPUT_DIR}/${audio_id}"
    mkdir -p "$out_dir"
    
    # 执行 VAD 拆分 - 剩余参数 $@ 作为 VAD 参数
    local start_time=$(date +%s)
    if python3 "$PYTHON_SCRIPT" "$input_file" --out_dir "$out_dir" "$@" > /dev/null 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local segment_count=$(ls -1 "$out_dir" 2>/dev/null | wc -l)
        segment_count=$(echo "$segment_count" | tr -d ' \n')
        echo "SUCCESS|${audio_id}|${segment_count}|${duration}" >> "$TEMP_LOG"
        echo "1" >> "$PROGRESS_FILE"
        echo "[✓] $audio_id (${segment_count} segments, ${duration}s)"
    else
        echo "FAILED|${audio_id}|0|0" >> "$TEMP_LOG"
        echo "1" >> "$PROGRESS_FILE"
        echo "[✗] $audio_id (处理失败)"
    fi
}

# 导出函数和变量，使其在子shell中可用
export -f process_audio
export PYTHON_SCRIPT
export OUTPUT_DIR
export TEMP_LOG
export PROGRESS_FILE

echo "========================================"
echo "批量音频拆分任务启动"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "并发数: $PARALLEL_JOBS"
echo "VAD参数: $VAD_PARAMS"
echo "----------------------------------------"

# 统计音频文件总数
total_files=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.opus" \) | wc -l)
total_files=$(echo "$total_files" | tr -d ' \n')
echo "发现音频文件: $total_files 个"
echo "========================================"

if [ "${total_files:-0}" -eq 0 ]; then
    echo "错误: 输入目录中没有找到音频文件"
    exit 1
fi

# 后台进度监控函数
monitor_progress() {
    local total=$1
    local progress_file=$2
    local last_count=0
    
    while true; do
        if [ -f "$progress_file" ]; then
            local current=$(wc -l < "$progress_file" 2>/dev/null | tr -d ' ')
            current=${current:-0}
            
            if [ "$current" != "$last_count" ]; then
                local percent=$((current * 100 / total))
                local success=$(grep -c "^SUCCESS" "$TEMP_LOG" 2>/dev/null || echo 0)
                success=$(echo "$success" | tr -d ' \n')
                success=${success:-0}
                local failed=$(grep -c "^FAILED" "$TEMP_LOG" 2>/dev/null || echo 0)
                failed=$(echo "$failed" | tr -d ' \n')
                failed=${failed:-0}
                
                # 使用 \r 实现同行刷新，清除整行避免残留字符
                printf "\r\033[K进度: %d/%d (%d%%) | 成功: %d | 失败: %d " \
                    "$current" "$total" "$percent" "$success" "$failed"
                
                last_count=$current
                
                # 处理完成
                if [ "$current" -ge "$total" ]; then
                    printf "\n"
                    break
                fi
            fi
        fi
        sleep 1
    done
}

# 启动后台进度监控
monitor_progress "$total_files" "$PROGRESS_FILE" &
MONITOR_PID=$!

# 使用 GNU parallel 或 xargs 实现并发
if command -v parallel >/dev/null 2>&1; then
    # 如果安装了 GNU parallel (禁用自带进度条,使用我们的监控)
    echo "使用 GNU parallel 进行并发处理..."
    echo ""
    find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.opus" \) \
        | parallel -j "$PARALLEL_JOBS" process_audio {} $VAD_PARAMS
else
    # 使用 xargs 作为后备方案
    echo "使用 xargs 进行并发处理..."
    echo ""
    find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.opus" \) \
        | xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c "process_audio \"{}\" $VAD_PARAMS"
fi

# 等待监控进程完成
wait $MONITOR_PID 2>/dev/null

echo ""
echo "========================================"
echo "处理完成 - 统计信息"
echo "========================================"

# 统计结果 (确保所有变量都是纯数字,移除空格和换行)
if [ -f "$TEMP_LOG" ] && [ -s "$TEMP_LOG" ]; then
    success_count=$(grep -c "^SUCCESS" "$TEMP_LOG" 2>/dev/null || echo "0")
    failed_count=$(grep -c "^FAILED" "$TEMP_LOG" 2>/dev/null || echo "0")
    total_segments=$(grep "^SUCCESS" "$TEMP_LOG" 2>/dev/null | cut -d'|' -f3 | awk '{s+=$1} END {print s+0}')
    total_time=$(grep "^SUCCESS" "$TEMP_LOG" 2>/dev/null | cut -d'|' -f4 | awk '{s+=$1} END {print s+0}')
    avg_time=$(grep "^SUCCESS" "$TEMP_LOG" 2>/dev/null | cut -d'|' -f4 | awk '{s+=$1; n++} END {if(n>0) print s/n; else print 0}')
else
    success_count=0
    failed_count=0
    total_segments=0
    total_time=0
    avg_time=0
fi

# 清理变量
success_count=$(echo "$success_count" | tr -d ' \n' | grep -o '[0-9]*' | head -1)
success_count=${success_count:-0}
failed_count=$(echo "$failed_count" | tr -d ' \n' | grep -o '[0-9]*' | head -1)
failed_count=${failed_count:-0}
total_segments=$(echo "$total_segments" | tr -d ' \n' | grep -o '[0-9]*' | head -1)
total_segments=${total_segments:-0}
total_time=$(echo "$total_time" | tr -d ' \n' | grep -o '[0-9]*' | head -1)
total_time=${total_time:-0}
avg_time=$(echo "$avg_time" | tr -d ' \n.')
avg_time=${avg_time:-0}

echo "总文件数: $total_files"
echo "成功处理: $success_count"
echo "处理失败: $failed_count"
echo "总片段数: $total_segments"
echo "总耗时: ${total_time}s"
echo "平均耗时: $(printf "%.2f" "$avg_time")s/文件"
echo "========================================"

# 验证输出目录结构
output_dirs=$(find "$OUTPUT_DIR" -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)
output_dirs=$(echo "$output_dirs" | tr -d ' \n')
echo "输出目录中创建的音频ID文件夹: $output_dirs 个"

# 使用默认值保护比较操作
if [ "${success_count:-0}" -eq "${total_files:-0}" ] && [ "${total_files:-0}" -gt 0 ]; then
    echo "✓ 所有文件处理完成!"
    exit 0
else
    echo "⚠ 部分文件处理失败，请查看上述日志"
    exit 1
fi

