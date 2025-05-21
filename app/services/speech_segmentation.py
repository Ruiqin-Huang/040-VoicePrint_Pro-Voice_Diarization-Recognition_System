import os
import uuid
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from typing import List, Tuple, Dict
from modelscope.pipelines import pipeline

from app.config.settings import settings
from utils.helpers import get_file_type
from utils.io_suppressor import suppress_stdout_stderr
import requests
import tempfile 
from urllib.parse import urlparse 

async def extract_speaker_audio(wav_path: str, results: List, target_speaker: int, save_path: str) -> str:
    """从原始音频中提取目标说话人的语音，其他人语音置为静音"""
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
            if end_sample <= len(audio):  # 确保索引不超出范围
                audio_out[start_sample:end_sample] = audio[start_sample:end_sample]

    # 保存输出音频
    sf.write(save_path, audio_out, sr)
    return save_path

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

async def process_audio_files(file_paths: List[str]) -> Tuple[List[Dict], List[str], List[str]]:
    """处理语音分离"""
    # 创建输出目录
    output_dir = settings.SEGMENTATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 检测是否存在GPU
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print(f"使用设备: {device}")
    
    # 初始化说话人分离模型
    try:
        with suppress_stdout_stderr():
            sd_pipeline = pipeline(
                task='speaker-diarization',
                model=settings.DIARIZATION_MODEL_PATH,
                model_revision=settings.DIARIZATION_MODEL_REVISION
            )
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")
    
    results = []
    invalid_files = []
    file_types = []  # 存储每个输入文件的类型
    temp_files = []  # 存储需要清理的临时文件
    
    for file_path in tqdm(file_paths, desc="Processing audio files"):
        local_path = file_path
        try:
            # 检查是否为URL，如果是则下载到本地
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)
            
            # 验证文件存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                file_types.append("未知")
                continue
                
            # 分割说话人
            with suppress_stdout_stderr():
                result = sd_pipeline(local_path)

            # 获取实际的说话人数量
            speaker_ids = set()
            for segment in result['text']:
                speaker_ids.add(segment[2])
            actual_speakers = len(speaker_ids)
            
            file_name = os.path.basename(file_path if not await is_url(file_path) else local_path)
            base_name, ext = os.path.splitext(file_name)
            
            file_type = get_file_type(actual_speakers)
            file_types.append(file_type)
            
            # 处理每个说话人
            for i in range(actual_speakers):
                file_id = str(uuid.uuid4())
                output_filename = f"{base_name}_speaker{i}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 提取说话人音频
                await extract_speaker_audio(local_path, result['text'], i, output_path)
                
                # 添加到结果列表
                results.append({
                    "file_id": file_id,
                    "source_url": file_path,
                    "file_url": output_path
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            invalid_files.append(f"{file_path}: {str(e)}")
            file_types.append("错误")
    
    # 清理临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
    
    if invalid_files and not results:
        raise Exception(f"所有文件处理失败: {'; '.join(invalid_files)}")
    
    return results, invalid_files, file_types