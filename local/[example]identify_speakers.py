import os
import sys
import argparse
import pickle
import numpy as np
import csv
from collections import defaultdict

parser = argparse.ArgumentParser(description='Identify speakers using voiceprint library')
parser.add_argument('--wav_to_identify_list', default='', type=str, help='wav path list to identify')
parser.add_argument('--workspace', default='.', type=str, help='workspace path')
parser.add_argument('--threshold', default=0.5, type=float, help='Distance threshold for speaker identification')

def load_voiceprint_library(voiceprintlib_dir):
    """加载声纹库中所有说话人的声纹"""
    speaker_voiceprints = {}
    speaker_audio_files = {}
    
    # 遍历声纹库目录
    for speaker_dir in os.listdir(voiceprintlib_dir):
        speaker_path = os.path.join(voiceprintlib_dir, speaker_dir)
        
        # 确保是目录
        if not os.path.isdir(speaker_path):
            continue
        
        # 查找声纹文件
        voiceprint_file = os.path.join(speaker_path, f"{speaker_dir}_voiceprint.pkl")
        if not os.path.exists(voiceprint_file):
            print(f"[WARNING]: No voiceprint file found for {speaker_dir}")
            continue
        
        # 加载声纹数据
        with open(voiceprint_file, 'rb') as f:
            voiceprint_data = pickle.load(f)
            
        # 存储说话人的平均声纹和相关音频文件列表
        speaker_voiceprints[speaker_dir] = voiceprint_data['avg_voiceprint']
        speaker_audio_files[speaker_dir] = voiceprint_data['audio']
    
    return speaker_voiceprints, speaker_audio_files

def collect_embeddings_to_identify(audio_embs_dir, wav_list):
    """收集待识别音频的嵌入向量"""
    embeddings_dict = {}
    
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
            
            embeddings_dict[wav_name] = stat_obj['avg_embedding']
    
    return embeddings_dict

def compute_cosine_distance(emb1, emb2):
    """计算两个嵌入向量之间的余弦距离"""
    # 归一化
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    # 计算余弦相似度
    cos_sim = np.dot(emb1_norm, emb2_norm)
    # 转换为余弦距离 (1 - 余弦相似度)
    cos_dist = 1.0 - cos_sim
    return cos_dist

def identify_speakers(embeddings_to_identify, speaker_voiceprints, threshold):
    """识别说话人身份"""
    identification_results = {}
    
    for wav_name, embedding in embeddings_to_identify.items():
        # 计算与每个说话人的距离
        distances = {}
        for speaker_id, voiceprint in speaker_voiceprints.items():
            distance = compute_cosine_distance(embedding, voiceprint)
            distances[speaker_id] = distance
        
        # 找到距离最小的说话人
        if distances:
            min_speaker = min(distances, key=distances.get)
            min_distance = distances[min_speaker]
            
            # 如果距离小于等于阈值，则识别为该说话人
            if min_distance <= threshold:
                identification_results[wav_name] = {
                    'identified_speaker': min_speaker,
                    'distances': distances
                }
            else:
                identification_results[wav_name] = {
                    'identified_speaker': 'unknown',
                    'distances': distances
                }
        else:
            identification_results[wav_name] = {
                'identified_speaker': 'unknown',
                'distances': {}
            }
    
    return identification_results

def save_identification_results(results, speaker_audio_files, output_file):
    """保存识别结果到CSV文件"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["wav_name", "identified_speaker_and_distances", "speaker_audio_files"])
        
        # 写入数据行
        for wav_name, result in results.items():
            speaker_id = result['identified_speaker']
            
            # 构建距离字典的字符串表示
            distances_str = "|".join([f"{spk}-{dist:.4f}" for spk, dist in result['distances'].items()])
            
            # 获取说话人的音频文件列表
            audio_files = []
            if speaker_id != 'unknown' and speaker_id in speaker_audio_files:
                audio_files = speaker_audio_files[speaker_id]
            
            audio_files_str = ", ".join(audio_files)
            
            # 写入一行数据
            writer.writerow([wav_name, f"{speaker_id}|{distances_str}", audio_files_str])

def save_identification_results_txt(results, speaker_audio_files, output_file):
    """保存识别结果到TXT文件，使用指定格式"""
    with open(output_file, 'w') as f:
        for wav_name, result in results.items():
            speaker_id = result['identified_speaker']
            
            # 获取说话人的音频文件列表
            audio_files = []
            if speaker_id != 'unknown' and speaker_id in speaker_audio_files:
                audio_files = speaker_audio_files[speaker_id]
                
            # 构建输出字符串
            output_str = f"{wav_name} 属于类别 {speaker_id}, 该类别说话人的音频库包含{len(audio_files)}条注册音频，分别为{', '.join(audio_files)}"
            
            # 写入文件
            f.write(output_str + '\n')

def main():
    args = parser.parse_args()
    
    # 检查声纹库目录是否存在
    voiceprintlib_dir = os.path.join(args.workspace, 'result', 'voiceprintlib')
    if not os.path.exists(voiceprintlib_dir):
        print(f"[ERROR]: Voiceprint library directory {voiceprintlib_dir} does not exist.")
        return
    
    # 检查嵌入向量目录是否存在
    audio_embs_dir = os.path.join(args.workspace, 'emb')
    if not os.path.exists(audio_embs_dir):
        print(f"[ERROR]: Embeddings directory {audio_embs_dir} does not exist.")
        return
    
    # 读取待识别的音频文件列表
    wav_list = []
    with open(args.wav_to_identify_list, 'r') as f:
        wav_list_read = f.readlines()
        for wav_path in wav_list_read:
            # 去除每行末尾的换行符和空格
            wav_path = wav_path.strip()
            wav_path = os.path.join(args.workspace, 'dataset', 'audio', wav_path)
            wav_list.append(wav_path)
    
    # 1. 加载声纹库
    # print(f"[INFO]: Loading voiceprint library from {voiceprintlib_dir}...")
    speaker_voiceprints, speaker_audio_files = load_voiceprint_library(voiceprintlib_dir)
    if not speaker_voiceprints:
        print("[ERROR]: No valid voiceprints found in the library.")
        return
    # print(f"[INFO]: Loaded {len(speaker_voiceprints)} speaker voiceprints.")
    
    # 2. 收集待识别音频的嵌入向量
    # print(f"[INFO]: Collecting embeddings for {len(wav_list)} audio files to identify...")
    embeddings_to_identify = collect_embeddings_to_identify(audio_embs_dir, wav_list)
    if not embeddings_to_identify:
        print("[ERROR]: No valid embeddings found for identification.")
        return
    # print(f"[INFO]: Collected {len(embeddings_to_identify)} embeddings for identification.")
    
    # 3. 识别说话人
    # print(f"[INFO]: Identifying speakers with threshold {args.threshold}...")
    identification_results = identify_speakers(embeddings_to_identify, speaker_voiceprints, args.threshold)
    
    # 4. 保存识别结果
    result_dir = os.path.join(args.workspace, 'result')
    os.makedirs(result_dir, exist_ok=True)
    output_file_csv = os.path.join(result_dir, 'identify_result.csv')
    output_file_txt = os.path.join(result_dir, 'identify_result.txt')  
    # print(f"[INFO]: Saving identification results to {output_file}...")
    save_identification_results(identification_results, speaker_audio_files, output_file_csv)
    save_identification_results_txt(identification_results, speaker_audio_files, output_file_txt) 
    
    # 统计识别结果
    identified_count = sum(1 for result in identification_results.values() if result['identified_speaker'] != 'unknown')
    print(f"[INFO]: Identification completed. {identified_count}/{len(identification_results)} audio files were successfully identified.")
    print(f"[INFO]: Results saved to {output_file_csv} and {output_file_txt}.")

if __name__ == "__main__":
    main()