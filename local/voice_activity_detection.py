import os
import json
import csv
import argparse
import sys

import torch
import torchaudio
import torch.distributed as dist

from utils.io_suppressor import suppress_stdout_stderr
# 1. 先检查当前目录下是否有对应的包/模块
# 2. 再检查PYTHONPATH目录下是否有对应的包/模块
# 3. Python安装时设置的默认目录

try:
    import modelscope
except ImportError:
    raise ImportError("Package \"modelscope\" not found. Please install them first.")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

try:
    from speakerlab.utils.utils import merge_vad
except:
    pass

parser = argparse.ArgumentParser(description='Voice activity detection')
parser.add_argument('--workspace', default='.', type=str, help='workspace path')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

# # 若gpus!=''(不为空), 从gpus中选择第一个gpu生成string 'cuda:n'，否则使用cpu
# if [ -n "$gpus" ]; then
#     vad_gpus="cuda:$(echo $gpus | cut -d' ' -f1)"
# else
#     vad_gpus="cpu"
# fi


# Use pretrained model from modelscope. So "model_id" and "model_revision" are necessary.
VAD_PRETRAINED = {
    'model_id': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'model_revision': 'v2.0.4',
}

def main():
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK']) # 获取当前进程在本节点（服务器）的排名
    threads_num = int(os.environ['WORLD_SIZE']) # 获取当前节点的进程总数
    dist.init_process_group(backend='gloo') # 初始化分布式进程组，使用gloo作为后端
    # TODO: 虽然初始化了分布式环境，但仅作为基础设施，没有实际使用同步功能，此行可被注释
    
    # 在workspace目录下创建一个名为vad的子目录，用于存放所有音频的VAD处理的结果
    vad_dir = os.path.join(args.workspace, 'vad')
    os.makedirs(vad_dir, exist_ok=True)
    
    # out_dir = os.path.dirname(os.path.abspath(args.out_file))
    
    # 如果{载入的wav音频文件所在文件夹}/segmentation.pkl文件存在，程序会加载这些分段信息，并在后续的 VAD 处理逻辑中使用它来过滤或调整 VAD 结果。
    # 程序会调用 merge_vad 函数，将 VAD 结果与分段信息进行合并处理。否则程序直接使用原始的 VAD 结果，不进行额外的分段处理。
    # segmentation_file = os.path.join(out_dir, 'segmentation.pkl')
    # if os.path.exists(segmentation_file):
    #     consider_segmentation = True
    #     with open(segmentation_file, 'rb') as f:
    #         segmentations = pickle.load(f)
    # else:
    #     consider_segmentation = False

    wavs = []
    try:
        # 从 metadata.csv 读取 wav_name 列
        metadata_path = os.path.join(args.workspace, 'dataset', 'metadata.csv')
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path = row['wav_name']
                full_wav_path = os.path.join(args.workspace, 'dataset', 'audio', wav_path)
                wavs.append(full_wav_path)
        
        if not wavs:
            raise Exception('[ERROR]: No wav files found in metadata.csv')
    except Exception as e:
        raise Exception(f'[ERROR]: Error reading metadata.csv: {str(e)}')

    if len(wavs) <= rank:
        print("[WARNING]: The number of threads exceeds the number of wavs.")
        sys.exit()

    # print(f'[INFO]: Start computing VAD...')
    # 子进程工作分配
    local_wavs = wavs[rank::threads_num]

    # 子进程设备分配
    if args.use_gpu:
        if not args.gpu:
            print("[WARNING]: GPU flag set but no GPU IDs provided. Using CPU instead.")
            vad_device = 'cpu'
        else:
            gpu_id = int(args.gpu[rank % len(args.gpu)])
            if gpu_id < torch.cuda.device_count():
                vad_device = 'cuda:%d' % gpu_id
            else:
                print("[WARNING]: GPU %s is not available. Using CPU instead." % gpu_id)
                vad_device = 'cpu'
    else:
        vad_device = 'cpu'
    # TODO: (可能存在联网操作)加载模型并初始化VAD管道
    with suppress_stdout_stderr():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection, 
            model='./pretrained_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', 
            model_revision='v2.0.4',
            # device='cuda' if torch.cuda.is_available() else 'cpu',
            device=vad_device,
        )

    
    for wpath in local_wavs:
        # 为每个文件创建新的空字典
        json_dict = {}
        # 使用预训练的VAD模型处理音频文件，返回检测到的语音片段
        with suppress_stdout_stderr():
            vad_time = vad_pipeline(wpath)[0]
        # 将时间从毫秒转换为秒，并保留三位小数
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in vad_time['value']]
        # if consider_segmentation:
        #     basename = os.path.basename(wpath).rsplit('.', 1)[0]
        #     vad_time = merge_vad(vad_time, segmentations[basename]['valid_field'])
        vad_time = [[round(vad_t[0], 3), round(vad_t[1], 3)] for vad_t in vad_time]
        # vad_time 是一个二维列表，格式为：
        # [[start1, end1], [start2, end2], ..., [startN, endN]]
            # start：语音片段的开始时间（秒）
            # end：语音片段的结束时间（秒）
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        for strt, end in vad_time:
            # 子段名称示例：2speakers_example_0.08_23.84
            subsegmentid = wid + '_' + str(strt) + '_' + str(end)
            json_dict[subsegmentid] = {
                        'file': wpath,
                        'start': strt,
                        'stop': end,
                }
            
        output_file = os.path.join(vad_dir, wid + '_vad.json')
        with open(output_file, 'w') as f:
            # 将该音频文件的VAD结果写入到对应的json文件中
            json.dump(json_dict, f, indent=2)
            
    # print(f'[INFO]: VAD finished for {len(local_wavs)} wavs.')
    # print(f'[INFO]: VAD results are saved in {vad_dir}.')

if __name__ == '__main__':
    main()