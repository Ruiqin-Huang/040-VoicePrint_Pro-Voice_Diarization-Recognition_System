import os
import uuid
import json
import requests
import tempfile
from tqdm import tqdm  # 添加tqdm导入
from urllib.parse import urlparse
from typing import List, Dict
from app.config.settings import settings
from utils.helpers import format_datetime, generate_phone_number
import whisper
import requests

# 加载Whisper模型（全局加载，避免重复加载）
whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE, download_root=settings.WHISPER_CACHE_DIR)

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

def save_segments_to_file(segments: List[Dict], source_file_path: str, file_id: str) -> str:
    """将转录segments保存为JSON文件到固定目录"""
    output_dir = "./data/speech_recognition"  # 改为固定输出目录
    json_filename = f"{file_id}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return str(json_path)

async def process_speech_files(files: List[str], language: str = "zh") -> tuple:
    """处理语音识别文件"""
    processed_files = []
    invalid_files = []
    temp_files = []  # 存储需要清理的临时文件
    
    # 添加tqdm进度条
    for file_path in tqdm(files, desc="Processing audio files"):
        local_path = file_path
        try:
            # 检查是否为URL，如果是则下载到本地
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)
            
            # 验证文件存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            result = transcribe_audio_file(whisper_model, local_path, language)
            
            file_id = str(uuid.uuid4())
            phone_number = generate_phone_number()
            file_url = save_segments_to_file(result["segments"], local_path, file_id)
            
            processed_files.append({
                "file_id": file_id,
                "phone_number": phone_number,
                "identity": "主叫" if len(processed_files) == 0 else "被叫",
                "call_record": result["full_text"],
                "create_time": format_datetime(),
                "file_url": file_url
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