import os
import sys
import argparse
import pickle
import pathlib
import numpy as np
import shutil

from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build

from utils.io_suppressor import suppress_stdout_stderr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress FutureWarning(In torch.load(...))

parser = argparse.ArgumentParser(description='Cluster embeddings and output speaker libraries')
parser.add_argument('--workspace', default='.', type=str, help='workspace path')
parser.add_argument('--conf', default=None, help='Config file')

# --audio_embs_dir "${workspace}/emb" --result_dir "${workspace}/result"

def collect_embeddings(audio_embs_dir, wav_list):
    """收集所有音频文件的平均嵌入向量"""
    # print("[INFO] Start collecting embeddings...")
    all_embeddings = []  # 所有wav文件的avg_embedding集合
    all_paths = []  # 所有wav文件的路径集合
    
    for wav_file in wav_list:
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]  # 去除扩展名
        embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        
        if not os.path.exists(embs_file):
            print(f"[WARNING]: {embs_file} does not exist, skipping {wav_file}")
            continue
        
        with open(embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            if 'avg_embedding' not in stat_obj:
                print(f"[WARNING]: No average embedding found in {embs_file}, skipping {wav_file}")
                continue
            
            all_embeddings.append(stat_obj['avg_embedding'])
            all_paths.append(wav_file)
    
    # print("[INFO] All embeddings collected.")
    return all_embeddings, all_paths


def perform_clustering(all_embeddings, all_paths, config):
    """执行聚类操作"""
    if not all_embeddings:
        print("[WARNING]: No valid average embeddings found.")
        return {}
    
    # print(f"[INFO] Starting clustering of {len(all_embeddings)} audio files...")
    
    # 转换为numpy数组进行聚类
    all_embeddings_array = np.array(all_embeddings)
    
    # 执行聚类
    cluster = build('cluster', config)
    labels = cluster(all_embeddings_array)
    
    # 将聚类标签重新编号(从0开始)
    new_labels = np.zeros(len(labels), dtype=int)
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i
    
    # 构建聚类结果
    clusters = {}
    for i, label in enumerate(new_labels):
        speaker_id = f"speaker_{label}"
        if speaker_id not in clusters:
            clusters[speaker_id] = []
        clusters[speaker_id].append(all_paths[i])
    
    return clusters


def create_result_files(clusters, output_dir):
    """生成聚类结果文本文件"""
    # 生成txt文件
    result_file = os.path.join(output_dir, 'cluster_result.txt')
    with open(result_file, 'w') as f:
        for speaker_id, audio_files in sorted(clusters.items()):
            f.write(f"{speaker_id}:\n")
            for audio_file in audio_files:
                f.write(f"    {audio_file}\n")
            f.write("\n")
    
    # 生成csv文件
    csv_file = os.path.join(output_dir, 'cluster_result.csv')
    with open(csv_file, 'w') as f:
        # 写入表头
        f.write("wav_name,speaker_id_pred,language\n")
        
        # 写入数据行
        for speaker_id, audio_files in sorted(clusters.items()):
            # 从speaker_id (形如 "speaker_6") 中提取数字部分
            speaker_num = speaker_id.split('_')[1]
            
            for audio_file in audio_files:
                # 获取文件名（不含路径）
                audio_filename = os.path.basename(audio_file)
                # 写入一行数据
                f.write(f"{audio_filename},{speaker_num},zh-cn\n") # TODO: 默认语言为中文，后续可根据需要修改


def create_voiceprint_library(clusters, output_dir, audio_embs_dir):
    """创建声纹库目录结构"""
    voiceprintlib_dir = os.path.join(output_dir, 'voiceprintlib')
    os.makedirs(voiceprintlib_dir, exist_ok=True)
    
    for speaker_id, audio_files in sorted(clusters.items()):
        # 创建说话人目录
        speaker_dir = os.path.join(voiceprintlib_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # 创建audio子目录
        audio_dir = os.path.join(speaker_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 处理音频文件
        audio_filenames = []
        speaker_avg_embeddings = []
        
        for audio_file in audio_files:
            # 获取音频文件名
            audio_filename = os.path.basename(audio_file)
            audio_filenames.append(audio_filename)
            dst_path = os.path.join(audio_dir, audio_filename)
            
            # 复制音频文件
            try:
                shutil.copy2(audio_file, dst_path)
            except Exception as e:
                print(f"[WARNING]: Failed to copy {audio_file}: {str(e)}")
            
            # 收集平均嵌入
            rec_id = audio_filename.rsplit('.', 1)[0]
            embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
            with open(embs_file, 'rb') as f:
                stat_obj = pickle.load(f)
                speaker_avg_embeddings.append(stat_obj['avg_embedding'])
        
        # 计算说话人的平均声纹
        speaker_voiceprint = np.mean(speaker_avg_embeddings, axis=0)
        
        # 保存声纹文件
        voiceprint_data = {
            'audio': audio_filenames,
            'avg_embeddings': speaker_avg_embeddings,
            'avg_voiceprint': speaker_voiceprint
        }
        
        with open(os.path.join(speaker_dir, f"{speaker_id}_voiceprint.pkl"), 'wb') as f:
            pickle.dump(voiceprint_data, f)
    
    return voiceprintlib_dir


def main():
    args = parser.parse_args()
    
    # 检查result_dir是否存在，不存在则创建
    result_dir = os.path.join(args.workspace, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    audio_embs_dir = os.path.join(args.workspace, 'emb')
    if not os.path.exists(audio_embs_dir):
        print(f"[WARNING]: {audio_embs_dir} does not exist. Please run the embedding extraction first.")
        return
    
    # 读取音频文件列表    
    wav_list = []
    try:
        # 从 metadata.csv 读取 wav_name 列
        import csv
        metadata_path = os.path.join(args.workspace, 'dataset', 'metadata.csv')
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path = row['wav_name']
                full_wav_path = os.path.join(args.workspace, 'dataset', 'audio', wav_path)
                wav_list.append(full_wav_path)
        
        if not wav_list:
            raise Exception('[ERROR]: No wav files found in metadata.csv')
    except Exception as e:
        raise Exception(f'[ERROR]: Error reading metadata.csv: {str(e)}')
    wav_list.sort()

    # 1. 收集所有音频文件的嵌入向量
    # print(f"[INFO] Collecting embeddings from {len(wav_list)} audio files...")
    all_embeddings, all_paths = collect_embeddings(audio_embs_dir, wav_list)
    if not all_embeddings:
        print("[WARNING]: No valid embeddings found.")
        return
    # print(f"[INFO] Collected {len(all_embeddings)} embeddings.")
    
    # 2. 执行聚类
    # print(f"[INFO] Clustering {len(all_embeddings)} audio files...")
    if not args.conf:
        print("[WARNING]: No config file provided for clustering.")
        return
    config = build_config(args.conf)
    clusters = perform_clustering(all_embeddings, all_paths, config)
    print(f"[INFO] Clustering completed. Found {len(clusters)} speakers.")
    
    # 检查聚类结果，若无有效聚类，则返回
    if not clusters:
        print("[WARNING]: No valid clusters found.")
        return
        
    # 3. 生成结果文件
    # print(f"[INFO] Generating result files...")
    create_result_files(clusters, result_dir)
    # print(f"[INFO] Result files generated at {result_dir}")
    
    # 4. 创建声纹库
    # print(f"[INFO] Creating voiceprint library...")
    voiceprintlib_dir = create_voiceprint_library(clusters, result_dir, audio_embs_dir)
    print(f"[INFO] Voiceprint library created at {voiceprintlib_dir}")
    
    print(f"[INFO] Results saved to {result_dir}/cluster_result.txt and {result_dir}/cluster_result.csv")


if __name__ == "__main__":
    main()