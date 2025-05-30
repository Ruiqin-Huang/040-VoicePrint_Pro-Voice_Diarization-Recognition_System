#!/bin/bash
# 使用说明: 
#   ./run_voiceprintpro.sh --name <容器名称> [--models <宿主机预训练模型路径>] [--input <宿主机输入路径>] [--output <宿主机输出路径>] [--gpu] [--port <端口>]

# 默认值（从docker/settings.py读取容器路径）
CONFIG_DIR="$(dirname "$0")"
CONTAINER_MODELS=$(python3 -c "from pathlib import Path; from settings import MODEL_DIR; print(MODEL_DIR)" "$CONFIG_DIR")
CONTAINER_INPUT=$(python3 -c "from pathlib import Path; from settings import INPUT_DIR; print(INPUT_DIR)" "$CONFIG_DIR")
CONTAINER_OUTPUT=$(python3 -c "from pathlib import Path; from settings import OUTPUT_DIR; print(OUTPUT_DIR)" "$CONFIG_DIR")

# 参数解析
POSITIONAL_ARGS=()
NAME=""
HOST_MODELS=""
HOST_INPUT=""
HOST_OUTPUT=""
USE_GPU=$(python3 -c "from pathlib import Path; from settings import USE_GPU; print(str(USE_GPU))" "$CONFIG_DIR")
PORT=$(python3 -c "from pathlib import Path; from settings import PORT; print(str(PORT))" "$CONFIG_DIR")

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift; shift ;;
        --models)
            HOST_MODELS="$2"
            shift; shift ;;
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
echo "输入目录: $HOST_INPUT"
echo "输出目录: $HOST_OUTPUT"
echo "容器端口: $PORT"
echo "GPU状态: $USE_GPU"
echo "容器内映射:"
echo "  - 输入: $HOST_INPUT -> $CONTAINER_INPUT"
echo "  - 输出: $HOST_OUTPUT -> $CONTAINER_OUTPUT_ROOT"
echo "======================================"

docker run -it --rm \
    $GPU_ARGS \
    -d --name "voiceprint-api" \
    --gpus all \
    -v "$HOST_MODELS:$CONTAINER_MODELS"
    -v "$HOST_INPUT:$CONTAINER_INPUT" \
    -v "$HOST_OUTPUT:$CONTAINER_OUTPUT" \
    -e "HOST_INPUT_PATH=$HOST_INPUT" \
    -e "HOST_OUTPUT_PATH=$HOST_OUTPUT" \
    -p "$PORT:8000" \
    voiceprint-pro-api:latest