"""
语音分割服务模块

该模块提供音频说话人分割功能，支持将多人语音分离为单个说话人的音频片段。
主要功能包括：
- 说话人分割（Speaker Diarization）
- 音频文件扩展（如果音频过短）
- 说话人音频提取和保存
- 支持本地文件和URL文件处理

依赖：
- modelscope: 用于说话人分割
- librosa: 用于音频处理
- soundfile: 用于音频文件读写
- requests: 用于URL文件下载
"""

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
from utils.io_suppressor import suppress_stdout_stderr
import requests
import tempfile 
from urllib.parse import urlparse

def get_file_type(speaker_count):
    """根据说话人数量确定语音类别"""
    if speaker_count == 1:
        return "单人"
    elif speaker_count >= 2:
        return "双人"
    else:
        return "未知，该条音频过短或未检测到人声"

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

async def extract_speaker_audio_from_channels(wav_path: str, file_dir: str) -> List[Dict]:
    """
    从左右声道切分音频（用于JY开头的文件）
    
    左声道作为主叫，右声道作为被叫。
    输出的音频文件都是双声道，主叫音频的左声道和右声道都是左声道内容，
    被叫音频的左声道和右声道都是右声道内容。
    
    Args:
        wav_path: 原始音频文件的绝对路径
        file_dir: 保存分割后音频文件的目录（相对于OUTPUT_DIR）
        
    Returns:
        List[Dict]: 分割后的文件信息列表，每个字典包含：
            - id: 分割文件的唯一标识符
            - file_url: 分割后的音频文件绝对路径
    """
    # 构建保存目录的绝对路径
    save_dir = os.path.join(settings.OUTPUT_DIR, file_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取原始音频（以双声道读取）
    audio, sr = librosa.load(wav_path, sr=None, mono=False)
    
    # 如果音频是单声道，转换为双声道（左右声道相同）
    if audio.ndim == 1:
        audio = np.array([audio, audio])
    
    # 确保是双声道
    if audio.shape[0] != 2:
        # 如果不是双声道，将单声道复制到两个声道
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        else:
            # 如果是多声道，只取前两个
            audio = audio[:2]
    
    # 左声道（主叫）
    left_channel = audio[0]
    # 右声道（被叫）
    right_channel = audio[1]
    
    # 存储分割后的文件信息
    segment_files = []
    
    # 生成主叫音频（左声道内容复制到左右两个声道）
    segment_id_caller = str(uuid.uuid4())
    filename_caller = f"{segment_id_caller}.wav"
    save_path_caller = os.path.join(save_dir, filename_caller)
    # 创建双声道音频：左右声道都是左声道内容
    stereo_caller = np.array([left_channel, left_channel])
    sf.write(save_path_caller, stereo_caller.T, sr)
    segment_files.append({
        "id": segment_id_caller,
        "file_url": save_path_caller
    })
    
    # 生成被叫音频（右声道内容复制到左右两个声道）
    segment_id_callee = str(uuid.uuid4())
    filename_callee = f"{segment_id_callee}.wav"
    save_path_callee = os.path.join(save_dir, filename_callee)
    # 创建双声道音频：左右声道都是右声道内容
    stereo_callee = np.array([right_channel, right_channel])
    sf.write(save_path_callee, stereo_callee.T, sr)
    segment_files.append({
        "id": segment_id_callee,
        "file_url": save_path_callee
    })
    
    return segment_files

async def extract_speaker_audio(wav_path: str, results: List, file_dir: str, num_speakers: int) -> str:
    """
    从原始音频中提取目标说话人的语音，其他人语音置为静音
    
    根据说话人分割结果，为每个说话人创建一个独立的音频文件，
    其中只包含该说话人的语音，其他说话人的语音部分被置为静音。
    
    Args:
        wav_path: 原始音频文件的绝对路径
        results: 说话人分割结果列表，每个元素为(start_time, end_time, speaker_id)元组
        file_dir: 保存分割后音频文件的目录（相对于OUTPUT_DIR）
        num_speakers: 说话人数量
        
    Returns:
        List[Dict]: 分割后的文件信息列表，每个字典包含：
            - id: 分割文件的唯一标识符
            - file_url: 分割后的音频文件绝对路径
    """
    # 构建保存目录的绝对路径
    save_dir = os.path.join(settings.OUTPUT_DIR, file_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 读取原始音频
    audio, sr = librosa.load(wav_path, sr=None)
    # 创建speaker人数长度的音频列表，每个元素都是静音数组
    audio_out_list = [np.zeros_like(audio) for _ in range(num_speakers)]

    # 按照真实出现顺序编号说话人（原始输出不一定编号连续）
    speakers = {}

    # 遍历所有说话人的语音段落
    for seg in results:
        start_time, end_time, speaker_id = seg

        # 为每个说话人分配连续编号
        if speaker_id not in speakers:
            speakers[speaker_id] = len(speakers)
        real_speaker_id = speakers[speaker_id]

        # 获取音频的起始和结束位置（采样点）
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 将目标说话人的语音段复制到输出音频中
        if end_sample <= len(audio):  # 确保索引不超出范围
            audio_out_list[real_speaker_id][start_sample:end_sample] = audio[start_sample:end_sample]
    
    # 存储分割后的文件信息
    segment_files = []

    # 保存输出音频
    for audio_out in audio_out_list:
        # 生成唯一的分割文件ID
        segment_id = str(uuid.uuid4())
        filename = f"{segment_id}.wav"
        save_path = os.path.join(save_dir, filename)

        # 保存音频文件
        sf.write(save_path, audio_out, sr)
        
        # 记录文件信息
        segment_files.append(
            {
                "id": segment_id,
                "file_url": save_path
            }
        )
    return segment_files

async def extract_and_save_speaker_segments(wav_path: str, results: List, file_dir: str) -> Dict[str, Any]:
    """
    提取语音段（自动合并连续的相同说话人段）
    
    根据说话人分割结果，合并连续的相同说话人段，并生成元数据文件。
    注意：此函数目前只生成元数据，不实际提取和保存音频片段。
    
    Args:
        wav_path: 原始音频文件的绝对路径
        results: 说话人分割结果列表，每个元素为(start_time, end_time, speaker_id)元组
        file_dir: 保存元数据文件的目录（相对于OUTPUT_DIR）
        
    Returns:
        Dict[str, Any]: 包含音频源和分割段信息的元数据字典
    """
    # 读取原始音频
    audio, sr = librosa.load(wav_path, sr=None)
    # 构建保存目录的绝对路径
    save_dir = os.path.join(settings.OUTPUT_DIR, file_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 复制原始音频到输出目录
    original_filename = os.path.basename(wav_path)
    # original_copy_path = os.path.join(save_dir, original_filename)
    # shutil.copy2(wav_path, original_copy_path)  # 保留元数据

    # 按照真实出现顺序编号说话人（原始输出不一定编号连续）
    speakers = {}
    
    # 合并连续的同说话人段
    merged_segments = []
    for seg in results:
        start_time, end_time, speaker_id = seg

        # 为每个说话人分配连续编号
        if speaker_id not in speakers:
            speakers[speaker_id] = len(speakers)
        real_speaker_id = speakers[speaker_id]

        # 如果当前段与上一段是同一说话人且时间连续，则合并
        if not merged_segments:
            merged_segments.append([start_time, end_time, real_speaker_id])
        else:
            last_seg = merged_segments[-1]
            # 如果说话人相同且当前段起始 <= 上一段结束，则合并
            if real_speaker_id == last_seg[2]:
                last_seg[1] = max(last_seg[1], end_time)  # 扩展结束时间
            else:
                merged_segments.append([start_time, end_time, real_speaker_id])

    # 构建元数据字典
    metadata = {"audio_source": original_filename, "segments": []}

    # 为每个合并后的段生成元数据
    for seg_idx, (start_time, end_time, speaker_id) in enumerate(merged_segments):
        segment_id = str(uuid.uuid4())
        filename = f"{segment_id}.wav"
        filepath = os.path.join(save_dir, filename)
        output_filepath = os.path.join(file_dir, filename)
        
        # # 提取并保存音频（已注释）
        # start_sample = int(start_time * sr)
        # end_sample = int(end_time * sr)
        # sf.write(filepath, audio[start_sample:end_sample], sr)

        # 说话人身份标识
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

    # 保存元数据到JSON文件
    base_name, ext = os.path.splitext(original_filename)
    metadata_path = os.path.join(save_dir, f"{base_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path

async def is_url(path: str) -> bool:
    """
    检查路径是否为URL
    
    Args:
        path: 待检查的路径字符串
        
    Returns:
        bool: 如果是URL返回True，否则返回False
    """
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """
    从URL下载文件到本地临时目录
    
    Args:
        url: 文件的URL地址
        
    Returns:
        str: 下载后的本地文件路径
        
    Raises:
        Exception: 下载失败时抛出异常
    """
    try:
        # 发送HTTP请求下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # 以二进制模式写入文件
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_filename
    except Exception as e:
        raise Exception(f"下载文件失败: {str(e)}")

async def process_audio_files(file_requests: List[FileRequest]) -> List[Dict]:
    """
    处理语音分离
    
    对多个音频文件执行说话人分割，将多人语音分离为单个说话人的音频片段。
    支持本地文件和URL文件，支持批量处理。
    
    Args:
        file_requests: 文件请求列表，每个请求包含文件ID和文件路径
        
    Returns:
        Tuple[List[Dict], List[str]]: 
            - 成功处理的文件结果列表，每个包含file_id, file_type, segment_files
            - 处理失败的文件列表，包含错误信息
            
    Raises:
        Exception: 当所有文件处理失败或模型加载失败时抛出
    """
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
    
    # 存储处理结果
    results = []
    # 存储处理失败的文件信息
    invalid_files = []
    # 存储需要清理的临时文件
    temp_files = []
    
    # 遍历每个文件请求进行处理
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
                    "segment_files": [],
                    "metadata": None
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
                # 处理音频过短的异常
                if "The effective audio duration is too short" in str(e):
                    print(f"Skipping {file_path} due to short audio duration.")
                    invalid_files.append(f"{file_path}: The effective audio duration is too short")
                    results.append({
                        "file_id": file_id,
                        "file_type": "未知，该条音频过短或未检测到人声",
                        "segment_files": [],
                        "metadata": None
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
            
            # 提取文件名和基础名称
            file_name = os.path.basename(local_path)
            base_name, ext = os.path.splitext(file_name)
            
            # 检查文件名是否以"JY"开头
            is_jy_file = file_name.upper().startswith("JY")
            
            # 根据说话人数量确定文件类型
            file_type = get_file_type(actual_speakers)
            segment_files = []

            # 处理不同情况
            if actual_speakers == 0:
                # 未检测到人声，保持现有逻辑
                results.append({
                    "file_id": file_id,
                    "file_type": "未知，该条音频过短或未检测到人声",
                    "segment_files": [],
                    "metadata": None
                })
                continue
            elif actual_speakers == 1:
                # 只检测到一个人，保持现有逻辑
                # 提取说话人分割元数据
                metadata = await extract_and_save_speaker_segments(local_path, result['text'], os.path.join(output_dir, base_name))
                # 提取说话人音频片段
                segment_files = await extract_speaker_audio(local_path, result['text'], os.path.join(output_dir, base_name), actual_speakers)
            elif actual_speakers >= 2:
                # 检测到两个及以上说话人

                try:
                    with suppress_stdout_stderr():
                        result = sd_pipeline(processing_path, oracle_num=2)
                except Exception as e:
                    print(f"Error processing {file_path} with oracle_num=2: {str(e)}")
                    invalid_files.append(f"{file_path}: oracle_num=2处理失败: {str(e)}")
                    results.append({
                        "file_id": file_id,
                        "file_type": "未知，该条音频过短或未检测到人声",
                        "segment_files": [],
                        "metadata": None
                    })
                    continue
                
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
                
                # 提取说话人分割元数据
                metadata = await extract_and_save_speaker_segments(local_path, result['text'], os.path.join(output_dir, base_name))

                if is_jy_file:
                    # JY文件：使用左右声道切分
                    file_type = "双人"
                    # # 复制原始音频到输出目录
                    # save_dir = os.path.join(settings.OUTPUT_DIR, os.path.join(output_dir, base_name))
                    # os.makedirs(save_dir, exist_ok=True)
                    # original_copy_path = os.path.join(save_dir, file_name)
                    # shutil.copy2(local_path, original_copy_path)
                    # 提取左右声道音频
                    segment_files = await extract_speaker_audio_from_channels(local_path, os.path.join(output_dir, base_name))
                else:
                    # 非JY文件：使用oracle_num=2重新执行sd_pipeline
                    file_type = "双人"
                    # 提取说话人音频片段（固定为2个说话人）
                    segment_files = await extract_speaker_audio(local_path, result['text'], os.path.join(output_dir, base_name), 2)
            
            # 添加到结果列表
            results.append({
                "file_id": file_id,
                "file_type": file_type,
                "segment_files": segment_files,
                "metadata": metadata
            })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            invalid_files.append(f"{file_path}: {str(e)}")
            results.append({
                "file_id": file_id,
                "file_type": "未知，该条音频过短或未检测到人声",
                "segment_files": [],
                "metadata": None
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