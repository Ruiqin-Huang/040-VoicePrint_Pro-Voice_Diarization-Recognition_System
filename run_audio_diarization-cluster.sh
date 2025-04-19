#!/bin/bash
set -e
. ./path.sh || exit 1 # source path.sh

# Usage: 
# ./run_audio_diarization-cluster.sh --audio_dir <audio_dir> --workspace <workspace> --gpus <gpus> --proc_per_node <proc_per_node> --speaker_model_id <speaker_model_id>
# ./run_audio_diarization-cluster.sh --help to see the help message

# 定义默认值
DEFAULT_AUDIO_DIR="./example/audio_7_speakers"
DEFAULT_WORKSPACE="./workspaces/demo_aishell-1_7_speakers"
DEFAULT_NUM_SPEAKERS=2
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
echo "  num_speakers: $num_speakers"
echo "  use_gpu: $use_gpu"
echo "  gpus: $gpus"
echo "  proc_per_node: $proc_per_node"
echo "  run_stage: $run_stage"

# 在切换目录【前】保存原始脚本目录的绝对路径
SCRIPT_DIR=$(dirname "$(realpath "$0")") # 读取脚本及指令文件需要需要使用来源目录的绝对路径（本.sh脚本所在路径），但是在workspace工作目录下执行
conf_file=${SCRIPT_DIR}/conf/diar.yaml # 模型配置文件路径，指定extract_diar_embeddings及cluster_and_postprocess过程中模型设置

# Create workspace directory
mkdir -p ${workspace}

# Stage 1: Generate metadata for the dataset
if [[ $run_stage =~ (^|[[:space:]])1($|[[:space:]]) ]]; then
    echo "++++++++ Stage 1: Generate metadata for the dataset ++++++++"
    if [[ "$use_gpu" == "true" ]]; then
        python ${SCRIPT_DIR}/local/audio_diarization.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers" --gpu "$gpus" --use_gpu
    else
        python ${SCRIPT_DIR}/local/audio_diarization.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers"
    fi
    
    # 执行语音识别
    # 修改metadata.csv文件，填充transcription列（注意transcription列可以放到audio_diarization.py中创建）
    # python ${SCRIPT_DIR}/local/语音识别.py --input_dir "$audio_dir" --workspace "$workspace"

    # 将在workspace目录下构建dataset文件夹
    # 构建得到的workspace/dataset文件夹格式应如
    # - workspace/dataset
    #    - dataset  
    #      - audio
    #         - 1.wav
    #         - 2.wav
    #         - ...
    #      - audio_source
    #      - metadata.csv
    # metadata.csv包含了音频文件的路径和对应的说话人标签
else
    echo "++++++++ Skipping Stage 1: Generate metadata for the dataset ++++++++"
fi

# Stage 2: Perform VAD on each audio file
if [[ $run_stage =~ (^|[[:space:]])2($|[[:space:]]) ]]; then
    echo "++++++++ Stage 2: Perform VAD on each audio file ++++++++"
    if [[ "$use_gpu" == "true" ]]; then
        torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/voice_activity_detection.py --workspace ${workspace} --gpu $gpus --use_gpu
    else
        torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/voice_activity_detection.py --workspace ${workspace}
    fi
else
    echo "++++++++ Skipping Stage 2: Perform VAD on each audio file ++++++++"
fi

# Stage 3: Extract speaker embeddings
if [[ $run_stage =~ (^|[[:space:]])3($|[[:space:]]) ]]; then
    echo "++++++++ Stage 3: Extract speaker embeddings ++++++++"
    torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/prepare_subseg_json.py --workspace ${workspace} --dur 1.0 --shift 0.5 --min_seg_len 0.5 --max_seg_num 150
    speaker_model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced" # 预训练声纹提取模型
    if [[ "$use_gpu" == "true" ]]; then
        torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/extract_diar_embeddings.py --workspace ${workspace} --model_id $speaker_model_id --conf $conf_file --batchsize 64 --gpu $gpus --use_gpu
    else
        torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/extract_diar_embeddings.py --workspace ${workspace} --model_id $speaker_model_id --conf $conf_file --batchsize 64
    fi
else
    echo "++++++++ Skipping Stage 3: Extract speaker embeddings ++++++++"
fi

# Stage 4: Cluster embeddings
if [[ $run_stage =~ (^|[[:space:]])4($|[[:space:]]) ]]; then
    echo "++++++++ Stage 4: Cluster embeddings ++++++++"
    python ${SCRIPT_DIR}/local/cluster_and_postprocess.py --workspace ${workspace} --conf $conf_file 
else
    echo "++++++++ Skipping Stage 4: Cluster embeddings ++++++++"
fi

# TODO: Need to add the evaluation stage
# Stage 5: Evaluate and generate results
# if [[ $run_stage =~ (^|[[:space:]])5($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 5: Evaluate and generate results ++++++++"
#     python ${SCRIPT_DIR}/local/evaluate_cluster_result.py --workspace ${workspace}
# else
#     echo "++++++++ Skipping Stage 5: Evaluate and generate results ++++++++"
# fi

echo "++++++++ All stages completed ++++++++"
echo "Speaker clustering completed."
echo "results can be found in ${workspace}/results"