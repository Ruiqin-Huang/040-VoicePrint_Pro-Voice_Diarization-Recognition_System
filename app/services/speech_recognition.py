import os
import uuid
import json
import requests
import tempfile
from tqdm import tqdm  # 添加tqdm导入
from urllib.parse import urlparse
from typing import Any, List, Dict

from app.models.speech_segmentation import FileRequest
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
    """检查路径是否为URL"""
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """从URL下载文件到本地临时目录"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # 写入文件
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_filename
    except Exception as e:
        raise Exception(f"下载文件失败: {str(e)}")

def transcribe_audio_file(whisper_model, file_path: str):
    """转录单个音频文件"""
    # 设置prompt（中文优化）
    initial_prompt = "这是一段双人对话。生于忧患，死于安乐。岂不快哉？"
    
    # 执行语音识别
    result = whisper_model.transcribe(
        file_path
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
        "segments": segments
    }

def merge_by_speaker_segments(whisper_results: List[Dict], speaker_segments: Dict[str, Any]) -> List[Dict]:
    """
    按说话人片段合并Whisper识别结果
    :param whisper_results: Whisper识别结果列表
    :param speaker_segments: 说话人分割元数据
    :return: 合并后的结果列表（按说话人片段分组）
    """
    merged_results = []
    whisper_idx = 0  # 全局指针
    whisper_len = len(whisper_results)
    
    for speaker_seg in speaker_segments["segments"]:
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
     
        # 只检查当前指针之后的Whisper片段（利用时间有序性）
        while whisper_idx < whisper_len:
            whisper_seg = whisper_results[whisper_idx]
            
            # 如果Whisper片段完全在当前说话人片段之前，跳过
            if whisper_seg["end"] <= speaker_seg["start_time"]:
                whisper_idx += 1
                continue
            
            # 如果Whisper片段已经超过当前说话人片段，停止检查
            if whisper_seg["start"] >= speaker_seg["end_time"]:
                break
                
            # 记录匹配的片段
            if current_speaker["text"]:
                current_speaker["text"] += " " + whisper_seg["text"].strip()
            else:
                current_speaker["text"] = whisper_seg["text"].strip()
            current_speaker["no_speech_prob"] = whisper_seg["no_speech_prob"]  # 更新为最新值
            whisper_idx += 1
        
        merged_results.append(current_speaker)
    
    return merged_results

def save_segments_to_file(results: List[Dict], seg_metadata: str, file_name: str):
    """将转录segments保存为JSON文件到固定目录"""
    output_dir = os.path.join(settings.OUTPUT_DIR, settings.RECOGNITION_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    with open(seg_metadata) as f:
        speaker_segments = json.load(f)

    recognitions = {}
    recognitions["audio_source"] = speaker_segments["audio_source"]
    recognitions["recognition"] = merge_by_speaker_segments(results, speaker_segments)

    json_filename = f"{file_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recognitions, f, ensure_ascii=False, indent=2)
    
    return str(json_path), recognitions["recognition"]

async def process_speech_files(file_requests: List[FileRequest]) -> tuple:
    """处理语音识别文件"""
    processed_files = []
    invalid_files = []
    temp_files = []  # 存储需要清理的临时文件
    
    # 添加tqdm进度条
    for file_request in tqdm(file_requests, desc="Processing audio files"):
        file_id = file_request.id
        file_path = file_request.file_path
        try:
            # 转换为容器内路径
            # local_path = path_mapper.host_to_container(file_path)
            local_path = os.path.join(settings.INPUT_DIR, file_path)
            
            # 检查是否为URL，如果是则下载到本地
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)
            
            # 验证文件存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            base_name, ext = os.path.splitext(os.path.basename(local_path))
            seg_metadata = os.path.join(settings.OUTPUT_DIR, settings.SEGMENTATION_OUTPUT_DIR, base_name, (base_name + '.json'))
            if not os.path.exists(seg_metadata):
                invalid_files.append(f"语音切分结果不存在：{file_path}")
            
            result = transcribe_audio_file(whisper_model, local_path)
            
            local_data_path, recognitions = save_segments_to_file(result["segments"], seg_metadata, base_name)
            # file_url = path_mapper.container_to_host(local_data_path)
            
            processed_files.append({
                "file_id": file_id,
                "call_original": result["full_text"],
                "recognitions": recognitions
            })
        except Exception as e:
            invalid_files.append(f"{file_path}: {str(e)}")
    
    # 清理临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
            
    return processed_files, invalid_files