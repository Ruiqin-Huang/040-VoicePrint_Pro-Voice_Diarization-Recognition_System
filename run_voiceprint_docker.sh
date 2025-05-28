#!/bin/bash
# 使用说明: 
#   ./run_voiceprintpro.sh [--input <宿主机输入路径>] [--output <宿主机输出路径>] [--gpu] [--port <端口>]

# 默认值（从docker/settings.py读取容器路径）
CONFIG_DIR="$(dirname "$0")/docker"
CONTAINER_INPUT=$(python3 -c "from pathlib import Path; from settings import INPUT_DIR; print(str(Path(INPUT_DIR).resolve()))" "$CONFIG_DIR")
CONTAINER_OUTPUT_ROOT=$(python3 -c "from pathlib import Path; from settings import SEGMENTATION_OUTPUT_DIR; print(str(Path(SEGMENTATION_OUTPUT_DIR).parent.resolve()))" "$CONFIG_DIR")

# 参数解析
POSITIONAL_ARGS=()
HOST_INPUT=""
HOST_OUTPUT=""
USE_GPU=$(python3 -c "from pathlib import Path; from settings import USE_GPU; print(str(USE_GPU))" "$CONFIG_DIR")
PORT=$(python3 -c "from pathlib import Path; from settings import PORT; print(str(PORT))" "$CONFIG_DIR")

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            HOST_INPUT="$2"
            shift; shift ;;
        --output)
            HOST_OUTPUT="$2"
            shift; shift ;;
        --gpu)
            USE_GPU=true
            shift ;;
        --port)
            PORT="$2"
            shift; shift ;;
        -*|--*)
            echo "未知选项: $1"
            exit 1 ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift ;;
    esac
done

# GPU参数
GPU_ARGS=""
if [ "$USE_GPU" = true ]; then
    GPU_ARGS="--gpus all"
fi

echo "======================================"
echo "VoicePrintPro 服务启动配置"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "容器端口: $PORT"
echo "GPU状态: $USE_GPU"
echo "容器内映射:"
echo "  - 输入: $INPUT_DIR -> /app/data/input"
echo "  - 输出: $OUTPUT_DIR -> /app/data/output"
echo "======================================"

docker run -it --rm \
    $GPU_ARGS \
    -v "$HOST_INPUT:$CONTAINER_INPUT" \
    -v "$HOST_OUTPUT:$CONTAINER_OUTPUT_ROOT" \
    -p "$PORT:8000" \
    voiceprint-pro:latest