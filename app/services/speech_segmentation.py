import json
import os
import shutil
import uuid
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from typing import Any, List, Tuple, Dict
from modelscope.pipelines import pipeline
from app.models.speech_segmentation import FileRequest

from app.config.settings import settings
from app.config.path_mapper import PathMapper
from utils.helpers import get_file_type
from utils.io_suppressor import suppress_stdout_stderr
import requests
import tempfile 
from urllib.parse import urlparse

def extend_audio_if_needed(audio_path: str, min_duration: float = 20.0, temp_dir: str = None) -> tuple:
    """
    检查音频长度，如果不足指定时长则通过重复堆砌的方式延长至至少指定时长
    返回处理后的临时文件路径和是否需要清理的标记
    
    Args:
        audio_path: 原始音频文件路径
        min_duration: 最小时长（秒），默认20秒
        temp_dir: 临时文件目录，如果为None则使用系统临时目录
    
    Returns:
        tuple: (处理后的文件路径, 是否需要清理临时文件, 原始时长)
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(audio) / sr
    
    # 如果音频长度已经满足要求，直接返回原路径
    if audio_duration >= min_duration:
        return audio_path, False, audio_duration
    
    # 需要扩展音频
    target_samples = int(min_duration * sr)
    repeats_needed = int(target_samples / len(audio)) + 1
    
    # 重复堆砌音频
    extended_audio = np.tile(audio, repeats_needed)[:target_samples]
    
    # 创建临时文件
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_filename = os.path.join(temp_dir, f"extended_{os.path.basename(audio_path)}")
    sf.write(temp_filename, extended_audio, sr)
    
    return temp_filename, True, audio_duration 

async def extract_speaker_audio(wav_path: str, results: List, file_dir: str, num_speakers: int) -> str:
    """从原始音频中提取目标说话人的语音，其他人语音置为静音"""
    save_dir = os.path.join(settings.OUTPUT_DIR, file_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 读取音频
    audio, sr = librosa.load(wav_path, sr=None)
    # 创建speaker人数长度的音频列表，每个元素都是静音数组
    audio_out_list = [np.zeros_like(audio) for _ in range(num_speakers)]

    # 按照真实出现顺序编号说话人（原始输出不一定编号连续）
    speakers = {}

    # 遍历所有说话人的语音段落
    for seg in results:
        start_time, end_time, speaker_id = seg

        if speaker_id not in speakers:
            speakers[speaker_id] = len(speakers)
        real_speaker_id = speakers[speaker_id]

        # 获取音频的起始和结束位置
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 将目标说话人的语音段复制到输出音频中
        if end_sample <= len(audio):  # 确保索引不超出范围
            audio_out_list[real_speaker_id][start_sample:end_sample] = audio[start_sample:end_sample]
    
    segment_files = []

    # 保存输出音频
    for audio_out in audio_out_list:
        segment_id = str(uuid.uuid4())
        filename = f"{segment_id}.wav"
        save_path = os.path.join(save_dir, filename)

        sf.write(save_path, audio_out, sr)
        
        segment_files.append(
            {
                "id": segment_id,
                "file_url": save_path
            }
        )
    return segment_files

async def extract_and_save_speaker_segments(wav_path: str, results: List, file_dir: str) -> Dict[str, Any]:
    """提取语音段（自动合并连续的相同说话人段）"""
    audio, sr = librosa.load(wav_path, sr=None)
    save_dir = os.path.join(settings.OUTPUT_DIR, file_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 复制原始音频到输出目录
    original_filename = os.path.basename(wav_path)
    original_copy_path = os.path.join(save_dir, original_filename)
    shutil.copy2(wav_path, original_copy_path)  # 保留元数据

    # 按照真实出现顺序编号说话人（原始输出不一定编号连续）
    speakers = {}
    
    # 合并连续的同说话人段
    merged_segments = []
    for seg in results:
        start_time, end_time, speaker_id = seg

        if speaker_id not in speakers:
            speakers[speaker_id] = len(speakers)
        real_speaker_id = speakers[speaker_id]

        if not merged_segments:
            merged_segments.append([start_time, end_time, real_speaker_id])
        else:
            last_seg = merged_segments[-1]
            # 如果说话人相同且当前段起始 <= 上一段结束，则合并
            if real_speaker_id == last_seg[2]:
                last_seg[1] = max(last_seg[1], end_time)  # 扩展结束时间
            else:
                merged_segments.append([start_time, end_time, real_speaker_id])

    metadata = {"audio_source": original_filename, "segments": []}

    for seg_idx, (start_time, end_time, speaker_id) in enumerate(merged_segments):
        segment_id = str(uuid.uuid4())
        filename = f"{segment_id}.wav"
        filepath = os.path.join(save_dir, filename)
        output_filepath = os.path.join(file_dir, filename)
        
        # # 提取并保存音频
        # start_sample = int(start_time * sr)
        # end_sample = int(end_time * sr)
        # sf.write(filepath, audio[start_sample:end_sample], sr)

        identity = ["主叫", "被叫"]

        # 记录元数据
        metadata["segments"].append({
            "id": segment_id,
            "speaker": f"speaker{speaker_id}",
            "identity": identity[speaker_id] if speaker_id < 2 else "其他",
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "file_path": output_filepath
        })

    base_name, ext = os.path.splitext(original_filename)
    # 保存元数据
    with open(os.path.join(save_dir, f"{base_name}.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata

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

async def process_audio_files(file_requests: List[FileRequest]) -> List[Dict]:
    """处理语音分离"""
    # 创建输出目录
    output_dir = settings.SEGMENTATION_OUTPUT_DIR
    os.makedirs(os.path.join(settings.OUTPUT_DIR, output_dir), exist_ok=True)
    
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
    temp_files = []  # 存储需要清理的临时文件
    
    for file_request in file_requests:
        file_id = file_request.id
        file_path = file_request.file_path
        local_path = file_path
        
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
                results.append({
                    "file_id": file_id,
                    "file_type": "未知，该文件不存在",
                    "segment_files": []
                })
                continue
            
            # 检查音频长度，如果不足20秒则扩展（仅在内存中处理，使用临时文件）
            processing_path, needs_cleanup, original_duration = extend_audio_if_needed(
                local_path, min_duration=20.0, temp_dir=None
            )
            if needs_cleanup:
                temp_files.append(processing_path)
                print(f"[INFO] 音频 {file_path} 原始时长 {original_duration:.2f}秒，已扩展至至少20秒")
                
            # 分割说话人
            try:
                with suppress_stdout_stderr():
                    result = sd_pipeline(processing_path)
            except Exception as e:
                if "The effective audio duration is too short" in str(e):
                    print(f"Skipping {file_path} due to short audio duration.")
                    invalid_files.append(f"{file_path}: The effective audio duration is too short")
                    results.append({
                        "file_id": file_id,
                        "file_type": "未知，该条音频过短或者未找到两个以上的不同说话人的声音",
                        "segment_files": []
                    })
                    continue
                else:
                    raise e

            # 如果音频被扩展过，需要过滤掉超出原始音频长度的切分结果
            if needs_cleanup:
                filtered_segments = []
                for seg_start, seg_end, spk_id in result['text']:
                    if seg_start < original_duration:
                        # 限制结束时间不超过原始音频长度
                        seg_end = min(seg_end, original_duration)
                        if seg_end > seg_start:  # 确保还有有效时长
                            filtered_segments.append((seg_start, seg_end, spk_id))
                result['text'] = filtered_segments

            # 获取实际的说话人数量
            speaker_ids = set()
            for segment in result['text']:
                speaker_ids.add(segment[2])
            actual_speakers = len(speaker_ids)
            
            file_name = os.path.basename(local_path)
            base_name, ext = os.path.splitext(file_name)
            
            file_type = get_file_type(actual_speakers)
            segment_files = []

            # 提取说话人分割元数据
            metadata = await extract_and_save_speaker_segments(local_path, result['text'], os.path.join(output_dir, base_name))

            # segment_files = [
            #     {
            #         "id": seg["id"],
            #         # "file_url": path_mapper.container_to_host(seg["file_path"])
            #         "file_url": seg["file_path"]
            #     }
            #     for seg in metadata["segments"]
            # ]

            segment_files = await extract_speaker_audio(local_path, result['text'], os.path.join(output_dir, base_name), actual_speakers)
            
            # 添加到结果列表
            results.append({
                "file_id": file_id,
                "file_type": file_type,
                "segment_files": segment_files
            })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            invalid_files.append(f"{file_path}: {str(e)}")
            results.append({
                "file_id": file_id,
                "file_type": "错误",
                "segment_files": []
            })
    
    # 清理临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
    
    if invalid_files and not any(len(r["segment_files"]) > 0 for r in results):
        raise Exception(f"所有文件处理失败: {'; '.join(invalid_files)}")
    
    return results, invalid_files