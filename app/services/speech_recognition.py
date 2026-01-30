"""
语音识别服务模块

提供基于Whisper的语音转文本功能，支持音频文件转录和说话人分离结果的合并。
通过预加载的Whisper模型实现高效的语音识别。

主要功能：
- 音频文件转录
- 说话人分离结果合并
- URL文件下载支持
- 批量音频处理
- 结果格式化和保存

依赖：
- whisper: OpenAI的语音识别模型
- torch: PyTorch深度学习框架
- requests: HTTP请求处理
- tqdm: 进度条显示
- app.config.settings: 配置管理
"""

import os
import uuid
import json
import requests
import tempfile
from tqdm import tqdm  # 添加tqdm导入
from urllib.parse import urlparse
from typing import Any, List, Dict

from app.models.speech_recognition import RecognitionFileRequest
from app.config.path_mapper import PathMapper
from app.config.settings import settings
from utils.helpers import format_datetime, generate_phone_number
import whisper
import requests

# 加载Whisper模型（全局加载，避免重复加载）
load_model_on_device = f"cuda:{settings.GPU_ID}" if settings.USE_GPU else "cpu"
# 使用前先检查gpu是否可用，如果不可用则使用cpu
if settings.USE_GPU:
    import torch
    if not torch.cuda.is_available():
        load_model_on_device = "cpu"
        print("[WARN] GPU不可用，已切换到CPU模式。")
whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=load_model_on_device, download_root=settings.WHISPER_CACHE_DIR)

async def is_url(path: str) -> bool:
    """
    检查给定的路径是否为有效的URL

    Args:
        path: 要检查的路径字符串

    Returns:
        bool: 如果是URL返回True，否则False
    """
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """
    从URL异步下载文件到临时目录

    Args:
        url: 文件的URL地址

    Returns:
        str: 下载后的本地文件路径

    Raises:
        Exception: 下载失败时抛出
    """
    try:
        # 发送HTTP请求下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功

        # 创建临时文件路径
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

        # 以二进制模式写入文件
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_filename
    except Exception as e:
        raise Exception(f"下载文件失败: {str(e)}")

def transcribe_audio_file(whisper_model, file_path: str):
    """
    使用Whisper模型转录单个音频文件

    Args:
        whisper_model: 已加载的Whisper模型实例
        file_path: 音频文件路径

    Returns:
        Dict: 包含完整文本和分段信息的字典
            - full_text: 完整的识别文本
            - segments: 带时间戳的文本段列表
    """
    # 设置初始提示词（优化中文识别）
    initial_prompt = "这是一段双人对话。生于忧患，死于安乐。岂不快哉？"

    # 执行语音识别转录
    result = whisper_model.transcribe(
        file_path
    )

    # 提取完整文本
    full_text = result["text"]

    # 提取带时间戳的段落信息
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
        "segments": segments
    }

def merge_by_speaker_segments(whisper_results: List[Dict], speaker_segments: Dict[str, Any]) -> List[Dict]:
    """
    按说话人片段合并Whisper识别结果

    将Whisper的时间戳分段结果与说话人分离结果进行合并，
    生成按说话人分组的识别结果。

    Args:
        whisper_results: Whisper识别的时间戳分段结果
        speaker_segments: 说话人分离的元数据字典

    Returns:
        List[Dict]: 按说话人合并后的结果列表
    """
    merged_results = []
    whisper_idx = 0  # 全局指针，跟踪当前处理的Whisper分段
    whisper_len = len(whisper_results)

    # 遍历每个说话人片段
    for speaker_seg in speaker_segments["segments"]:
        # 初始化当前说话人的合并结果
        current_speaker = {
            "seg_id": speaker_seg["id"],
            "speaker": speaker_seg["speaker"],
            "identity": speaker_seg["identity"],
            "start_time": speaker_seg["start_time"],
            "end_time": speaker_seg["end_time"],
            "duration": speaker_seg["duration"],
            "file_path": speaker_seg["file_path"],
            "text": "",
            "no_speech_prob": 0
        }

        # 只检查当前指针之后的Whisper片段（利用时间有序性优化）
        while whisper_idx < whisper_len:
            whisper_seg = whisper_results[whisper_idx]

            # 如果Whisper片段完全在当前说话人片段之前，跳过
            if whisper_seg["end"] <= speaker_seg["start_time"]:
                whisper_idx += 1
                continue

            # 如果Whisper片段已经超过当前说话人片段，停止检查
            if whisper_seg["start"] >= speaker_seg["end_time"]:
                break

            # 避免VAD误差导致的错误向前合并：
            # 如果whisper片段在speaker片段内的部分占比过小，归到下一个speaker片段
            overlap_duration = speaker_seg["end_time"] - whisper_seg["start"]
            whisper_duration = whisper_seg["end"] - whisper_seg["start"]
            if whisper_duration > 0 and (overlap_duration / whisper_duration) < 0.3:
                break  # 不增加whisper_idx，留给下一个speaker处理

            # 记录匹配的片段，合并文本
            if current_speaker["text"]:
                current_speaker["text"] += " " + whisper_seg["text"].strip()
                current_speaker["end_time"] = whisper_seg["end"]  # 更新结束时间为最后一个匹配片段的结束时间
            else:
                current_speaker["text"] = whisper_seg["text"].strip()
                current_speaker["start_time"] = whisper_seg["start"]  # 更新开始时间为第一个匹配片段的开始时间
            # 更新无语音概率为最新值
            current_speaker["no_speech_prob"] = whisper_seg["no_speech_prob"]
            whisper_idx += 1

        merged_results.append(current_speaker)

    return merged_results

def save_segments_to_file(results: List[Dict], seg_metadata: str, file_path: str):
    """
    将转录分段结果保存为JSON文件

    Args:
        results: Whisper识别的分段结果
        seg_metadata: 说话人分离元数据文件路径
        file_name: 输出文件名（不含扩展名）

    Returns:
        Tuple[str, List]: (保存的JSON文件路径, 合并后的识别结果)
    """
    # 构建输出目录路径
    output_dir = os.path.join(settings.OUTPUT_DIR, settings.RECOGNITION_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 构建识别结果字典
    recognitions = {}

    if seg_metadata:
        # 读取说话人分离元数据
        with open(seg_metadata) as f:
            speaker_segments = json.load(f)
        
        # recognitions["audio_source"] = speaker_segments["audio_source"]
        recognitions["recognition"] = merge_by_speaker_segments(results["segments"], speaker_segments)
    else:
        start_time = results["segments"][0]["start"] if results["segments"] else 0
        end_time = results["segments"][-1]["end"] if results["segments"] else 0
        recognitions["recognition"] = [
            {
                "seg_id": str(uuid.uuid4()),
                "speaker": "unknown",
                "identity": "未知",
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "file_path": file_path,
                "text": results["full_text"],
                "no_speech_prob": results["segments"][0]["no_speech_prob"] if results["segments"] else 0
            }
        ]

    # # 生成JSON文件名和路径
    # json_filename = f"{file_name}.json"
    # json_path = os.path.join(output_dir, json_filename)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # # 保存为JSON文件
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(recognitions, f, ensure_ascii=False, indent=2)

    return recognitions["recognition"]

async def process_speech_files(file_requests: List[RecognitionFileRequest]) -> tuple:
    """
    批量处理语音识别文件

    对多个音频文件执行语音识别，支持本地文件和URL。
    需要对应的说话人分离结果文件存在。

    Args:
        file_requests: 文件请求列表，每个包含文件ID和路径

    Returns:
        Tuple[List[Dict], List[str]]:
            - 成功处理的文件结果列表
            - 处理失败的文件列表，包含错误信息
    """
    processed_files = []
    invalid_files = []
    temp_files = []  # 存储需要清理的临时文件

    # 使用进度条显示处理进度
    for file_request in tqdm(file_requests, desc="Processing audio files"):
        file_id = file_request.id
        file_path = file_request.file_path
        seg_file_path = file_request.seg_file_path
        try:
            # 转换为容器内路径
            # local_path = path_mapper.host_to_container(file_path)
            local_path = os.path.join(settings.INPUT_DIR, file_path)

            # 检查是否为URL，如果是则下载到本地临时文件
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)

            # 验证本地文件是否存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue

            # 检查对应的说话人分离结果是否存在
            # base_name, ext = os.path.splitext(os.path.basename(local_path))
            # seg_metadata = os.path.join(settings.OUTPUT_DIR, settings.SEGMENTATION_OUTPUT_DIR, base_name, (base_name + '.json'))
            # if not os.path.exists(seg_metadata):
            #     invalid_files.append(f"语音切分结果不存在：{file_path}")
            #     continue  # 确保有continue语句
            seg_metadata = None
            if seg_file_path:
                # 转换为容器内路径
                seg_metadata = os.path.join(settings.INPUT_DIR, seg_file_path)

                # 检查是否为URL，如果是则下载到本地临时文件
                if await is_url(seg_file_path):
                    seg_metadata = await download_file(seg_file_path)
                    temp_files.append(seg_metadata)

                # 验证本地文件是否存在
                if not os.path.exists(seg_metadata):
                    invalid_files.append(f"文件不存在: {seg_file_path}")
                    continue
                
            # 执行语音识别转录
            result = transcribe_audio_file(whisper_model, local_path)

            # 保存结果并合并说话人信息
            recognitions = save_segments_to_file(result, seg_metadata, file_path)
            # file_url = path_mapper.container_to_host(local_data_path)

            # 添加到成功处理列表
            processed_files.append({
                "file_id": file_id,
                "call_original": result["full_text"],
                "recognitions": recognitions
            })
        except Exception as e:
            # 记录处理失败的文件及错误信息
            invalid_files.append(f"{file_path}: {str(e)}")

    # 清理下载的临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

    return processed_files, invalid_files