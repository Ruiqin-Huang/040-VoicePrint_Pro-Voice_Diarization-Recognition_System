#!/bin/bash
set -e
. ./path.sh || exit 1 # source path.sh

# Usage: 
# ./run_speaker_identification.sh --audio_dir <audio_dir> --workspace <workspace> --gpus <gpus> --proc_per_node <proc_per_node> --speaker_model_id <speaker_model_id>
# ./run_speaker_identification.sh --help to see the help message

# 定义默认值
DEFAULT_AUDIO_DIR="./example/audio_to_identify"
DEFAULT_WORKSPACE="./workspaces/demo_speaker_identification"
DEFAULT_GPUS="0 1 2 3"
DEFAULT_PROC_PER_NODE=8
DEFAULT_SPEAKER_MODEL_ID="iic/speech_campplus_sv_zh_en_16k-common_advanced"
DEFAULT_RUN_STAGE="1 2 3 4"

# 帮助函数
show_usage() {
    echo "Usage: ./run_speaker_identification.sh [OPTIONS]"
    echo "Options:"
    echo "  --audio_dir DIR           包含需要识别身份的wav音频文件的目录 (默认: $DEFAULT_AUDIO_DIR)"
    echo "  --workspace DIR           工作目录，用于存放输出结果 (默认: $DEFAULT_WORKSPACE)"
    echo "  --gpus \"ID1 ID2...\"       要使用的GPU设备ID (默认: \"$DEFAULT_GPUS\")"
    echo "  --proc_per_node NUM       每个节点的进程数量 (默认: $DEFAULT_PROC_PER_NODE)"
    echo "  --speaker_model_id ID     声纹模型ID (默认: $DEFAULT_SPEAKER_MODEL_ID)"
    echo "  --run_stage \"STAGES...\"   指定要执行的阶段 (1-4)，用空格分隔 (默认: \"$DEFAULT_RUN_STAGE\")"
    echo "  --help                    显示帮助信息"
    exit 1
}

# 初始化参数为默认值
audio_dir=$DEFAULT_AUDIO_DIR
workspace=$DEFAULT_WORKSPACE
gpus=$DEFAULT_GPUS
proc_per_node=$DEFAULT_PROC_PER_NODE
speaker_model_id=$DEFAULT_SPEAKER_MODEL_ID
run_stage=$DEFAULT_RUN_STAGE

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
        --speaker_model_id)
            if [[ $# -lt 2 ]]; then
                echo "错误: --speaker_model_id 参数缺少值"
                show_usage
            fi
            speaker_model_id="$2"
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
        --help)
            show_usage
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            ;;
    esac
done

# 显示已配置的参数
echo "配置参数:"
echo "  audio_dir: $audio_dir"
echo "  workspace: $workspace"
echo "  gpus: $gpus"
echo "  proc_per_node: $proc_per_node"
echo "  speaker_model_id: $speaker_model_id"
echo "  run_stage: $run_stage"

# 在切换目录【前】保存原始脚本目录的绝对路径
SCRIPT_DIR=$(dirname "$(realpath "$0")") # 读取脚本及指令文件需要需要使用来源目录的绝对路径（本.sh脚本所在路径）
conf_file=${SCRIPT_DIR}/conf/diar.yaml # 模型配置文件路径，指定extract_diar_embeddings过程中模型设置

# Create workspace directory
mkdir -p ${workspace}

# Stage 1: Generate dataset for identification
# 构建得到的workspace/dataset文件夹格式应如
# - workspace/dataset
#     - audio
#         - 1.wav
#         - 2.wav
#         - ...
#     - wav_list.txt
if [[ $run_stage =~ (^|[[:space:]])1($|[[:space:]]) ]]; then
    echo "++++++++ Stage 1: Generate dataset for identification ++++++++"
    python ${SCRIPT_DIR}/local/generate_identification_dataset.py --audio_dir $audio_dir --workspace ${workspace}
    # 生成wav_list.txt文件，内容为音频文件的绝对路径
    wav_file_list="${workspace}/dataset/wav_to_identify_list.txt"
else
    echo "++++++++ Skipping Stage 1: Generate dataset for identification ++++++++"
    wav_file_list="${workspace}/dataset/wav_to_identify_list.txt"
fi

# Stage 2: Perform VAD on each audio file
if [[ $run_stage =~ (^|[[:space:]])2($|[[:space:]]) ]]; then
    echo "++++++++ Stage 2: Perform VAD on each audio file ++++++++"
    torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/voice_activity_detection.py --wav_list $wav_file_list --workspace ${workspace} --gpu $gpus --use_gpu
else
    echo "++++++++ Skipping Stage 2: Perform VAD on each audio file ++++++++"
fi

# Stage 3: Extract speaker embeddings
if [[ $run_stage =~ (^|[[:space:]])3($|[[:space:]]) ]]; then
    echo "++++++++ Stage 3: Extract speaker embeddings ++++++++"
    torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/prepare_subseg_json.py --wav_list $wav_file_list --workspace ${workspace} --dur 1.5 --shift 0.75 --min_seg_len 0.75 --max_seg_num 100
    torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/extract_diar_embeddings.py --wav_list $wav_file_list --workspace ${workspace} --model_id $speaker_model_id --conf $conf_file --batchsize 64 --gpu $gpus --use_gpu
else
    echo "++++++++ Skipping Stage 3: Extract speaker embeddings ++++++++"
fi

# Stage 4: Identify speakers and generate report
if [[ $run_stage =~ (^|[[:space:]])4($|[[:space:]]) ]]; then
    echo "++++++++ Stage 4: Identify speakers and generate report ++++++++"
    python ${SCRIPT_DIR}/local/identify_speakers.py --wav_to_identify_list $wav_file_list --workspace ${workspace} --threshold 1000
else
    echo "++++++++ Skipping Stage 4: Identify speakers and generate report ++++++++"
fi

echo "++++++++ All stages completed ++++++++"
echo "Speaker identification completed."
echo "Results can be found in ${workspace}/results"