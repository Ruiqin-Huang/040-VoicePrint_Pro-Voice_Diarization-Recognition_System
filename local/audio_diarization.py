from pydub import AudioSegment
import os
import sys
import argparse
from tqdm import tqdm

from modelscope.pipelines import pipeline

from utils.io_suppressor import suppress_stdout_stderr

import librosa
import soundfile as sf
import numpy as np
import csv

from unittest import mock
import socket

# 禁用 socket 连接
def raise_error(*args, **kwargs):
    raise OSError("Network access disabled for offline test")

def extract_speaker_audio(wav_path, results, target_speaker, save_path):
    """
    从原始音频中提取目标说话人的语音，其他人语音置为静音
    :param wav_path: 原始音频文件路径
    :param results: Diarization 结果，格式为 [[start_time, end_time, speaker_id], ...]
    :param target_speaker: 目标说话人的 ID (例如 0 或 1)
    :param save_path: 保存提取结果的路径
    """
    # 读取音频
    audio, sr = librosa.load(wav_path, sr=None)
    audio_out = np.zeros_like(audio)  # 初始化输出音频为零（静音）

    # 遍历所有说话人的语音段落
    for seg in results:
        start_time, end_time, speaker_id = seg

        if speaker_id == target_speaker:
            # 获取音频的起始和结束位置
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # 将目标说话人的语音段复制到输出音频中
            audio_out[start_sample:end_sample] = audio[start_sample:end_sample]

    # 保存输出音频
    sf.write(save_path, audio_out, sr)

def get_language_from_filename(filename):
    if filename.startswith('en'):
        return 'English'
    elif filename.startswith('zh'):
        return 'Chinese'
    elif filename.startswith('ru'):
        return 'Russian'
    else:
        return 'Unknown'

def seperate_speakers(input_dir, workspace, num_speakers=2, gpu="", use_gpu=True):
    """
    根据说话人分离音频
    :param input_dir: 输入目录
    :param workspace: 工作目录
    :param num_speakers: 说话人数量
    :param gpu: 可用的GPU列表，如 "0 1 2 3"
    :param use_gpu: 是否使用GPU
    """
    # 创建输出目录
    os.makedirs(workspace, exist_ok=True)
    
    # 创建audio子文件夹用于存放分割后的音频
    audio_output_path = os.path.join(workspace, "dataset", "audio")
    os.makedirs(audio_output_path, exist_ok=True)
    
    # 创建audio_source子文件夹用于存放原始音频
    audio_source_path = os.path.join(workspace, "dataset", "audio_source")
    os.makedirs(audio_source_path, exist_ok=True)
    
    output_csv_path = os.path.join(workspace, "dataset", "metadata.csv")
    csv_data = []

    # 设置计算设备
    if not use_gpu:
        # 强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif gpu:
        # 设置使用特定GPU，该环境变量只影响当前进程。
        # 无论指定多少块可用gpu，实际仅仅会使用可视gpu中的第一块gpu，即cuda:0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu.replace(" ", ",")

    with suppress_stdout_stderr():
    # with mock.patch("socket.socket", side_effect=raise_error):
        sd_pipeline = pipeline(
            task='speaker-diarization',
            model='./pretrained_models/iic/speech_campplus_speaker-diarization_common',
            model_revision='v1.0.0'
        )

    input_files = os.listdir(input_dir)
    for file in tqdm(input_files, desc=f"Seperate the speakers "):
        try:
            file_path = os.path.join(input_dir, file)
            
            # 复制原始音频到audio_source文件夹
            import shutil
            shutil.copy2(file_path, os.path.join(audio_source_path, file))

            # 分割说话人
            with suppress_stdout_stderr():
                result = sd_pipeline(file_path, oracle_num=num_speakers)
                # 若不指定说话人数量，则会自动检测说话人数量并依据此进行音频分离
                # result = sd_pipeline(file_path)

            # 保存音频
            file_name, ext = os.path.splitext(file)
            language = get_language_from_filename(file_name)

            for i in range(num_speakers):
                filename = f"{file_name}_speaker{i}.wav"
                # 将分割后的音频保存到audio子文件夹
                output_audio_path = os.path.join(audio_output_path, filename)
                extract_speaker_audio(file_path, result['text'], i, output_audio_path)
                csv_data.append([filename, file, i, language, ''])

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wav_name', 'source_name', 'speaker_id', 'language', 'character_id'])
        writer.writerows(csv_data)

def main(input_dir, workspace, num_speakers=2, gpu="", use_gpu=True):
    print(f"[INFO] 原始音频文件读取自：{input_dir}")
    print(f"[INFO] 原始音频文件保存至：{workspace}/dataset/audio_source")
    print(f"[INFO] 按说话人分离后的音频文件保存至：{workspace}/dataset/audio")
    
    # if not use_gpu:
    #     print("[INFO]使用 CPU 进行处理")
    # elif gpu:
    #     print(f"[INFO]使用 GPU: {gpu}")
    # else:
    #     print("[INFO]使用默认 GPU 配置，使用cuda:0训练")

    seperate_speakers(input_dir, workspace, num_speakers, gpu, use_gpu)

    return workspace

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="分离说话人脚本")
    
    # 添加输入路径参数
    parser.add_argument('--input_dir', type=str, help="输入的音频文件夹路径")
    
    # 添加输出路径参数
    parser.add_argument('--workspace', type=str, help="指定的工作目录路径")

    # 添加说话人数量参数
    parser.add_argument('--num_speakers', type=int, default=2, help="说话人数量")
    
    # 修改设备选择参数
    parser.add_argument('--gpu', type=str, default="", help="指定可用的GPU列表，例如 '0 1 2 3'")
    parser.add_argument('--use_gpu', action='store_true', help="是否使用GPU")
    
    # 解析命令行参数
    args = parser.parse_args()

    if not args.input_dir or not args.workspace:
        parser.print_help()
        exit(1)

    main(args.input_dir, args.workspace, args.num_speakers, args.gpu, args.use_gpu)