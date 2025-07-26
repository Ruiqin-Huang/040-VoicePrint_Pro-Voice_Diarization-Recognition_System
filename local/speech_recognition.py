import whisper
import os
import argparse
from tqdm import tqdm
import json

import torch
import time
import soundfile as sf

def speech2text(input_dir, workspace, language, gpu="", use_gpu=True):
    # 加载 Whisper 模型（可选: "tiny", "base", "small", "medium", "large"）
    model = whisper.load_model("medium")  # "base" 平衡速度和精
    if language == "zh":
        initial_prompt = "这是一段双人对话。生于忧患，死于安乐。岂不快哉？"

    # 创建text子文件夹用于存放语音识别结果
    output_path = os.path.join(workspace, "result/text")
    os.makedirs(output_path, exist_ok=True)

    # 设置计算设备
    if not use_gpu:
        # 强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif gpu:
        # 设置使用特定GPU，该环境变量只影响当前进程。
        # 无论指定多少块可用gpu，实际仅仅会使用可视gpu中的第一块gpu，即cuda:0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu.replace(" ", ",")

    # 处理音频文件
    input_files = os.listdir(input_dir)
    for file in tqdm(input_files, desc=f"Recognize speech context "):
        try:
            file_path = os.path.join(input_dir, file)

            if language == "zh":
                result = model.transcribe(file_path, language=language, initial_prompt=initial_prompt)
            else:
                result = model.transcribe(file_path, language=language)

            # # 输出识别结果
            # print(result["text"])

            # 获取文本
            # transcript = result["text"]
            
            # 提取带时间戳的转录段落
            segments = result["segments"]  # 每段包含 start, end, text

            # 只保留需要的字段
            output_segments = [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "flag": segment["no_speech_prob"]
                }
                for segment in segments
            ]

            # 保存为 JSON 文件
            file_name, ext = os.path.splitext(file)
            file_name += ".json"
            output_file = os.path.join(output_path, file_name)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_segments, f, ensure_ascii=False, indent=2)

            # # 将结果写入 TXT 文件
            # file_name, ext = os.path.splitext(file)
            # file_name += ".txt"
            # output_file = os.path.join(output_path, file_name)
            # with open(output_file, "w", encoding="utf-8") as f:
            #     f.write(transcript)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

def transcribe_audio_file(whisper_model, file_path: str, language: str = "zh"):
    """转录单个音频文件"""
    # 设置prompt（中文优化）
    initial_prompt = "这是一段双人对话。生于忧患，死于安乐。岂不快哉？" if language == "zh" else None
    
    # 执行语音识别
    result = whisper_model.transcribe(
        file_path, 
        language=language,
        initial_prompt=initial_prompt
    )
    
    # 提取完整文本
    full_text = result["text"]
    
    # 提取带时间戳的段落
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "no_speech_prob": segment["no_speech_prob"]
        }
        for segment in result["segments"]
    ]
    
    return {
        "full_text": full_text,
        "segments": segments,
        "language": language
    }

def main(input_dir, workspace, language, gpu="", use_gpu=True):
    print(f"[INFO] 音频文件读取自：{input_dir}")
    print(f"[INFO] 转录的json文件保存至：{workspace}/result/text")

    speech2text(input_dir, workspace, language)

    return workspace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语音转文字脚本")
    
    # 添加输入路径参数
    parser.add_argument('--input_dir', type=str, help="输入的音频文件夹路径")
    
    # 添加输出路径参数
    parser.add_argument('--workspace', type=str, help="指定的工作目录路径")
    
    # 添加语言参数
    parser.add_argument('--language', type=str, help="音频及文本语言")

    # 修改设备选择参数
    parser.add_argument('--gpu', type=str, default="", help="指定可用的GPU列表，例如 '0 1 2 3'")
    parser.add_argument('--use_gpu', action='store_true', help="是否使用GPU")

    # 解析命令行参数
    args = parser.parse_args()

    if not args.input_dir or not args.workspace or not args.language:
        parser.print_help()
        exit(1)

    main(args.input_dir, args.workspace, args.language, args.gpu, args.use_gpu)
