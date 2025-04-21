#!/bin/bash
set -e
. ./path.sh || exit 1 # source path.sh

# Usage: 
# ./run_audio_diarization-cluster.sh --help to see the help message

# 定义默认值
DEFAULT_AUDIO_DIR="./example/audio_7_speakers"
DEFAULT_WORKSPACE="./workspaces/demo_aishell-1_7_speakers"
DEFAULT_NUM_SPEAKERS=2
DEFAULT_LANGUAGE="zh"
DEFAULT_GPUS=""
DEFAULT_PROC_PER_NODE=8
DEFAULT_RUN_STAGE="1 2 3 4"
DEFAULT_USE_GPU=false

# 帮助函数
show_usage() {
    echo "Usage: ./run_audio_diarization-cluster.sh [OPTIONS]"
    echo "Options:"
    echo "  --audio_dir DIR           包含需要聚类的wav音频文件的目录 (默认: $DEFAULT_AUDIO_DIR)"
    echo "  --workspace DIR           工作目录，用于存放输出结果 (默认: $DEFAULT_WORKSPACE)"
    echo "  --num_speakers NUM        输入原始音频中的说话人数量 (默认: $DEFAULT_NUM_SPEAKERS)"
    echo "  --language \"zh, en, ru\"        音频及文本语言（zh=中文，en=英语，ru=俄语） (默认: $DEFAULT_LANGUAGE)"
    echo "  --gpus \"ID1 ID2...\"       要使用的GPU设备ID (默认: \"$DEFAULT_GPUS\")"
    echo "  --use_gpu                 使用GPU进行计算 (需同时指定 --gpus)"
    echo "  --proc_per_node NUM       每个节点的进程数量 (默认: $DEFAULT_PROC_PER_NODE)"
    echo "  --run_stage \"STAGES...\"     指定要执行的阶段 (1-5)，用空格分隔 (默认: \"$DEFAULT_RUN_STAGE\")"
    echo "  --help                    显示帮助信息"
    exit 1
}

# 初始化参数为默认值
audio_dir=$DEFAULT_AUDIO_DIR
workspace=$DEFAULT_WORKSPACE
num_speakers=$DEFAULT_NUM_SPEAKERS
language=$DEFAULT_LANGUAGE
gpus=$DEFAULT_GPUS
proc_per_node=$DEFAULT_PROC_PER_NODE
run_stage=$DEFAULT_RUN_STAGE
use_gpu=$DEFAULT_USE_GPU


# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --audio_dir)
            if [[ $# -lt 2 ]]; then
                echo "错误: --audio_dir 参数缺少值"
                show_usage
            fi
            audio_dir="$2"
            shift 2
            ;;
        --workspace)
            if [[ $# -lt 2 ]]; then
                echo "错误: --workspace 参数缺少值"
                show_usage
            fi
            workspace="$2"
            shift 2
            ;;
        --num_speakers)
            if [[ $# -lt 2 ]]; then
                echo "错误: --num_speakers 参数缺少值"
                show_usage
            fi
            num_speakers="$2"
            shift 2
            ;;
        --language)
            if [[ $# -lt 2 ]]; then
                echo "错误: --language 参数缺少值"
                show_usage
            fi
            language="$2"
            shift 2
            ;;
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "错误: --gpus 参数缺少值"
                show_usage
            fi
            gpus="$2"
            shift 2
            ;;
        --proc_per_node)
            if [[ $# -lt 2 ]]; then
                echo "错误: --proc_per_node 参数缺少值"
                show_usage
            fi
            proc_per_node="$2"
            shift 2
            ;;
        --run_stage)
            if [[ $# -lt 2 ]]; then
                echo "错误: --run_stage 参数缺少值"
                show_usage
            fi
            run_stage="$2"
            shift 2
            ;;
        --use_gpu)
            use_gpu=true
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            ;;
    esac
done

# 验证 gpu 和 use_gpu 参数的组合是否合理
if [[ -n "$gpus" && "$use_gpu" == "false" ]]; then # 检查$gpus是否非空
    echo "错误: 指定了 --gpus 参数但未指定 --use_gpu 参数"
    show_usage
fi

if [[ "$use_gpu" == "true" && -z "$gpus" ]]; then # 检查$gpus是否为空
    echo "错误: 指定了 --use_gpu 参数但未指定 --gpus 参数"
    show_usage
fi

# 显示已配置的参数
echo "配置参数:"
echo "  audio_dir: $audio_dir"
echo "  workspace: $workspace"
echo "  language: $language"
echo "  use_gpu: $use_gpu"
echo "  gpus: $gpus"
echo "  proc_per_node: $proc_per_node"
echo "  run_stage: $run_stage"

# 在切换目录【前】保存原始脚本目录的绝对路径
SCRIPT_DIR=$(dirname "$(realpath "$0")") # 读取脚本及指令文件需要需要使用来源目录的绝对路径（本.sh脚本所在路径），但是在workspace工作目录下执行
conf_file=${SCRIPT_DIR}/conf/diar.yaml # 模型配置文件路径，指定extract_diar_embeddings及cluster_and_postprocess过程中模型设置

# Create workspace directory
mkdir -p ${workspace}

if [[ "$use_gpu" == "true" ]]; then
    python ${SCRIPT_DIR}/local/speech_recognition.py --input_dir "$audio_dir" --workspace "$workspace" --language "$language" --gpu "$gpus" --use_gpu
else
    python ${SCRIPT_DIR}/local/speech_recognition.py --input_dir "$audio_dir" --workspace "$workspace" --language "$language"
fi

echo "++++++++ Completed ++++++++"
echo "Speech recognition completed."
echo "results can be found in ${workspace}/results/text"