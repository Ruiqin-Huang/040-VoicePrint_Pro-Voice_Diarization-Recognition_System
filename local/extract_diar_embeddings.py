# filepath: /speaker-diarization-system/speaker-diarization-system/local/extract_diar_embeddings.py
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script extracts speaker embeddings from audio files using a pretrained model.
It computes embeddings for each sub-segment and saves the results in a specified format.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np

import torch
import torchaudio
import torch.distributed as dist

from speakerlab.utils.config import yaml_config_loader, Config
from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad

from utils.io_suppressor import suppress_stdout_stderr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress FutureWarning(In torch.load(...))

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract speaker embeddings for diarization.')
parser.add_argument('--workspace', default='.', type=str, help='workspace path')
parser.add_argument('--model_id', default=None, help='Model id in modelscope')
parser.add_argument('--pretrained_model', default=None, type=str, help='Path of local pretrained model')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--batchsize', default=64, type=int, help='Batch of segements for extracting embeddings')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

# 依据support中的模型选择对应的参数
# FEATURE_COMMON为【特征提取器的配置】
# 特征提取器负责将音频信号转换为特征表示，是音频预处理阶段使用的组件，所有模型【共用】相同的特征提取配置
# 特征提取和嵌入生成是两个独立步骤，所有语音识别模型都使用相同的特征提取参数FEATURE_COMMON
FEATURE_COMMON = {
    'obj': 'speakerlab.process.processor.FBank',
    'args': {
        'n_mels': 80,
        'sample_rate': 16000,
        'mean_nor': True,
    },
}

# CAMPPLUS_VOX和CAMPPLUS_COMMON为【声纹嵌入提取器的配置】
# 声纹嵌入提取器负责将提取的特征转换为声纹嵌入向量
CAMPPLUS_VOX = { # 针对VoxCeleb语料库优化，生成512维声纹嵌入向量
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

# Model configurations
# 依据args.model_id选择对应的模型
supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
}   

def main():
    args = parser.parse_args()
    # 确保workspace目录下创建一个名为emb的子目录，用于存放所有音频的声纹嵌入
    # 该目录用于存储所有音频文件的声纹嵌入向量
    emb_dir = os.path.join(args.workspace, 'emb')
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir, exist_ok=True)
    
    conf = yaml_config_loader(args.conf)
    rank = int(os.environ['LOCAL_RANK']) # 获取当前进程在本节点（服务器）的排名
    threads_num = int(os.environ['WORLD_SIZE']) # 获取当前节点的进程总数
    dist.init_process_group(backend='gloo') # 初始化分布式进程组，使用gloo作为后端
        # 进程组初始化：创建一个分布式进程组，使多个独立进程能够相互通信
        # 通信通道建立：为数据交换、状态同步和协调工作提供必要的底层通信机制
    if args.model_id is not None:
        # with suppress_stdout_stderr():
        #     assert isinstance(args.model_id, str) and is_official_hub_path(args.model_id), "Invalid modelscope model id."
        #     # 检查args.model_id是否为字符串类型
        #     # TODO: (可能存在联网操作)通过is_official_hub_path()函数验证是否符合ModelScope官方模型路径格式
        # assert args.model_id in supports, "Model id not currently supported."
        model_config = supports[args.model_id]
        
        # 只在rank=0进程下载模型，然后广播给其他进程
        # if rank == 0:
        #     with suppress_stdout_stderr():
        #         cache_dir = snapshot_download(args.model_id, revision=model_config['revision'])
        #     # TODO: （可能存在联网操作）首先检查缓存目录是否有对应 ID 和版本的模型。如检测到现有缓存，会直接返回缓存路径，不重新下载。即使有缓存，代码仍可能尝试进行轻量级网络连接以验证缓存状态。
        #     obj_list = [cache_dir] # 将Python对象(cache_dir字符串)序列化为字节流
        # else:
        #     obj_list = [None]
        # dist.broadcast_object_list(obj_list, 0) # 从rank 0广播到所有进程（使用基础通信原语广播这些字节，包含进程同步，确保所有进程暂停等待接收数据。非主进程的obj_list会被自动填充接收到的内容）
        # cache_dir = obj_list[0] # 广播完成后，所有进程的obj_list[0]都包含相同的cache_dir值
        pretrained_model = os.path.join('./pretrained_models/iic/speech_campplus_sv_zh_en_16k-common_advanced', model_config['model_pt'])
        conf['embedding_model'] = model_config['model']
        conf['pretrained_model'] = pretrained_model
        conf['feature_extractor'] = FEATURE_COMMON
    else:
        # 若未指定modelscope中的args.model_id，默认使用args.pretrained_model（需要给出本地路径）
        assert args.pretrained_model is not None, "[ERROR] One of the params `model_id` and `pretrained_model` must be set."
        conf['pretrained_model'] = args.pretrained_model
        conf['feature_extractor'] = FEATURE_COMMON
        conf['embedding_model'] = CAMPPLUS_COMMON
    
    wavs_list = []
    try:
        # 从 metadata.csv 读取 wav_name 列
        import csv
        metadata_path = os.path.join(args.workspace, 'dataset', 'metadata.csv')
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path = row['wav_name']
                full_wav_path = os.path.join(args.workspace, 'dataset', 'audio', wav_path)
                wavs_list.append(full_wav_path)
        
        if not wavs_list:
            raise Exception('[ERROR]: No wav files found in metadata.csv')
    except Exception as e:
        raise Exception(f'[ERROR]: Error reading metadata.csv: {str(e)}')

    if len(wavs_list) <= rank:
        print("[WARNING]: The number of threads exceeds the number of wavs.")
        sys.exit()
    
    # print(f'[INFO]: Start segmentation...')
    # 子进程工作分配
    local_wavs = wavs_list[rank::threads_num]
    # 子进程设备分配
    if args.use_gpu:
        gpu_id = int(args.gpu[rank % len(args.gpu)])
        # 基于进程编号 (rank) 和指定的 GPU 列表 (args.gpu)，通过取模操作 (rank % len(args.gpu)) 来循环分配 GPU。
        # 假设有 6 个进程 (rank = 0, 1, 2, 3, 4, 5)，GPU 列表为 [3, 5, 7]。
            # 进程 0、3 使用 GPU 3。
            # 进程 1、4 使用 GPU 5。
            # 进程 2、5 使用 GPU 7。
        if gpu_id < torch.cuda.device_count():
            device = torch.device('cuda:%d' % gpu_id)
        else:
            print("[WARNING]: GPU %s is not available. Use CPU instead." % gpu_id)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    pretrained_state = torch.load(config.pretrained_model, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    embedding_model.to(device)
    
    # 为每个进程分配的音频文件列表进行处理
    for wpath in local_wavs:
        # 获取音频文件的名称
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        # 为每个音频文件单独加载其对应的subseg_json文件
        subseg_json_path = os.path.join(args.workspace, 'vad', wid + '_subseg.json')
        if not os.path.exists(subseg_json_path):
            print(f"[WARNING]: Sub-segment json file {subseg_json_path} not found, skipping.")
            continue
        
        # 读取该音频文件的子段信息
        with open(subseg_json_path, "r") as f:
            subseg_json = json.load(f)
            
        # 输出文件路径
        emb_file_name = wid + ".pkl"
        stat_emb_file = os.path.join(emb_dir, emb_file_name)
        
        if not os.path.isfile(stat_emb_file):
            # 检查当前音频文件的嵌入文件是否已经存在
            # 如果不存在，则进行嵌入计算
            embeddings = []
            # 提取子段信息
            meta = subseg_json
            wav_path = list(meta.values())[0]['file']  # 所有子段都来自同一个音频文件
            obj_fs = feature_extractor.sample_rate
            # print(f"[INFO]: Processing {wav_path}...")
            wav = load_audio(wav_path, obj_fs=obj_fs)
            
            # 提取每个子段对应的音频片段
            # wavs = [wav[0, int(meta[i]['start'] * obj_fs):int(meta[i]['stop'] * obj_fs)] for i in meta]
            # max_len = max([x.shape[0] for x in wavs])
            # wavs = [circle_pad(x, max_len) for x in wavs]
            
            # 先提取音频片段
            wavs = [wav[0, int(meta[i]['start'] * obj_fs):int(meta[i]['stop'] * obj_fs)] for i in meta]
            
            # 过滤掉长度为零的片段
            valid_indices = []
            valid_wavs = []
            for idx, x in enumerate(wavs):
                if x.shape[0] > 0:  # 检查长度是否大于零
                    valid_wavs.append(x)
                    valid_indices.append(idx)
                else:
                    print(f"[WARNING]: Segment {idx} in file {wid} has zero length. Skipping.")
            # 如果所有片段都无效，则跳过此文件
            if len(valid_wavs) == 0:
                print(f"[WARNING]: All segments in file {wid} have zero length. Skipping file.")
                continue
            # 使用有效片段继续处理
            wavs = valid_wavs
            max_len = max([x.shape[0] for x in wavs])
            wavs = [circle_pad(x, max_len) for x in wavs]
            
            
            wavs = torch.stack(wavs).unsqueeze(1)
            # 批量计算嵌入向量
            embeddings = []
            batch_st = 0
            with torch.no_grad():
                while batch_st < wavs.shape[0]:
                    # args.batchsize表示每个批次处理的音频片段数量
                    wavs_batch = wavs[batch_st: batch_st + args.batchsize].to(device)
                    feats_batch = torch.vmap(feature_extractor)(wavs_batch)
                    embeddings_batch = embedding_model(feats_batch).cpu()
                    embeddings.append(embeddings_batch)
                    batch_st += args.batchsize
            # 所有子片段的嵌入向量都存储在embeddings数组中
            # 系统按照录音文件而非片段组织数据，每个录音生成一个PKL文件,减少文件数量，避免大量小文件造成的I/O开销
            embeddings = torch.cat(embeddings, dim=0).numpy()
            # 计算所有子片段嵌入向量的平均值
            avg_embedding = np.mean(embeddings, axis=0)  # 沿着第0轴求平均，得到一个192维的向量
            # 创建包含所有嵌入向量的字典对象
            valid_meta_keys = [list(meta.keys())[i] for i in valid_indices]
            stat_obj = {
                'embeddings': embeddings,
                'times': [[meta[key]['start'], meta[key]['stop']] for key in valid_meta_keys],
                'avg_embedding': avg_embedding
            }
            # stat_obj = {
            #     'embeddings': embeddings,
            #     'times': [[meta[i]['start'], meta[i]['stop']] for i in meta],
            #     'avg_embedding': avg_embedding  # 添加整个音频的平均嵌入向量
            # }
            
            # 保存结果
            pickle.dump(stat_obj, open(stat_emb_file, 'wb'))
        else:
            print(f"[WARNING]: Embeddings for {wid} have been saved previously. Skipping.")
            
    # print(f'[INFO]: Finished extracting embeddings for {len(local_wavs)} files.')

if __name__ == "__main__":
    main()