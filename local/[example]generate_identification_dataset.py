#!/usr/bin/env python3
# 生成待识别音频的数据集目录结构，用于声纹识别系统

import os
import sys
import shutil
import argparse
import glob

import pandas as pd

def generate_identification_dataset(audio_dir, workspace):
    # 创建必要的目录
    dataset_dir = os.path.join(workspace, 'dataset')
    audio_to_identify_dir = os.path.join(dataset_dir, 'audio')
    os.makedirs(audio_to_identify_dir, exist_ok=True)
    
    # 查找源目录中的所有WAV文件
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if not wav_files:
        print(f"错误: 在 {audio_dir} 中没有找到WAV文件")
        sys.exit(1)
    
    # 准备元数据收集
    metadata = []
    default_language = "zh-cn"  # 默认语种
    
    # 创建wav_to_identify_list.txt文件
    wav_to_identify_list_path = os.path.join(dataset_dir, 'wav_to_identify_list.txt')
    
    with open(wav_to_identify_list_path, 'w') as wav_to_identify_list:
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            output_path = os.path.join(audio_to_identify_dir, filename)
            
            # 复制文件
            shutil.copy2(wav_file, output_path)
            
            # 写入文件名到列表文件
            wav_to_identify_list.write(f"{filename}\n")
            
            # 添加元数据
            metadata.append({
                "wav_name": filename,
                "file_path": output_path,
                "language": default_language
            })
    
    # 创建并保存元数据CSV
    if metadata:
        df = pd.DataFrame(metadata)
        metadata_path = os.path.join(dataset_dir, 'wav_to_identify_metadata.csv')
        df.to_csv(metadata_path, index=False)
        print(f"已生成元数据文件: {metadata_path}，包含 {len(metadata)} 条记录")
        print(f"数据集中共有 {len(metadata)} 个待识别音频文件")

def main():
    parser = argparse.ArgumentParser(description='为声纹识别系统生成待识别音频的数据集目录结构')
    parser.add_argument('--audio_dir', required=True, help='包含待识别WAV文件的目录')
    parser.add_argument('--workspace', required=True, help='工作空间目录')
    args = parser.parse_args()
    
    generate_identification_dataset(args.audio_dir, args.workspace)

if __name__ == "__main__":
    main()