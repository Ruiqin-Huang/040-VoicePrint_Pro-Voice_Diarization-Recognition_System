"""
音频声纹注册服务模块

该模块提供音频声纹特征提取和注册功能，支持将音频文件转换为声纹特征向量并存储到Milvus向量数据库。
主要功能包括：
- VAD（语音活动检测）处理
- 音频分段和特征提取
- 声纹嵌入向量生成
- Milvus向量数据库存储

依赖：
- modelscope: 用于VAD和声纹特征提取
- speakerlab: 用于音频处理和特征提取
- pymilvus: 用于向量数据库操作
- librosa: 用于音频处理
- torch: 用于深度学习模型推理
"""

import os
import sys
import json
import pickle
import shutil
from copy import deepcopy
from tqdm import tqdm
import warnings
from typing import List, Dict, Optional, Tuple
import uuid

import numpy as np
import torch
import librosa  # 新增导入
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad

from pymilvus import utility, Collection
from app.config.settings import settings
from utils.io_suppressor import suppress_stdout_stderr
from utils.Milvus import MilvusClient

warnings.filterwarnings("ignore", category=FutureWarning)

class AudioRegistrationService:
    """
    音频声纹注册服务类
    
    负责处理音频文件的声纹特征提取和注册到Milvus向量数据库。
    """
    def __init__(self):
        """
        初始化音频声纹注册服务
        
        创建临时工作目录，设置计算设备（GPU/CPU），准备处理环境。
        """
        # 工作空间根目录
        self.workspace = os.path.join(settings.OUTPUT_DIR, "audio_registration")
        # VAD（语音活动检测）结果存储目录
        self.vad_dir = os.path.join(self.workspace, "vad")  # 新增VAD目录
        # 声纹嵌入向量存储目录
        self.emb_dir = os.path.join(self.workspace, "emb")  # 新增嵌入目录
        # 计算设备（GPU或CPU）
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        # 创建必要的目录
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)  # 创建目录
        os.makedirs(self.emb_dir, exist_ok=True)  # 创建目录
        print(f"[INFO] Created temporary workspace for registration: {self.workspace}, Device: {self.device}")

    def _extract_embedding_for_file(self, audio_path: str, feature_extractor, embedding_model) -> Optional[np.ndarray]:
        """
        为单个音频文件执行VAD、分段和特征提取，返回192维声纹嵌入向量
        
        处理流程：
        1. 执行VAD（语音活动检测）识别有效语音段
        2. 将语音段切分为1秒的子片段
        3. 提取每个子片段的声纹特征
        4. 计算所有子片段的平均嵌入向量
        
        Args:
            audio_path: 音频文件的绝对路径
            feature_extractor: 特征提取器对象，用于提取音频特征
            embedding_model: 嵌入模型对象，用于生成声纹嵌入向量
            
        Returns:
            Optional[np.ndarray]: 192维的声纹嵌入向量，如果处理失败则返回None
        """
        # 提取文件名（不含扩展名）作为工作ID
        wid = os.path.basename(audio_path).rsplit('.', 1)[0]
        
        # 1. VAD（语音活动检测）- 添加异常处理和静音检测
        try:
            # 静音检测（新增）：检查音频是否包含有效语音信号
            audio_data, sr = librosa.load(audio_path, sr=None)
            # 如果音频最大振幅小于0.01，认为是静音文件
            if np.max(np.abs(audio_data)) < 0.01:
                print(f"[WARN] Silent audio detected: {audio_path}")
                return None
                
            # 初始化VAD（语音活动检测）管道
            vad_pipeline = pipeline(Tasks.voice_activity_detection, 
                                    model=settings.VAD_MODEL_PATH, 
                                    model_revision=settings.VAD_MODEL_REVISION, 
                                    device=self.device)
            
            # 执行VAD检测，抑制标准输出和错误输出
            with suppress_stdout_stderr():
                vad_result = vad_pipeline(audio_path)
            
            # 处理VAD结果格式（同DiarizationService）
            # 如果结果是列表，取第一个元素；否则直接使用
            if vad_result and isinstance(vad_result, list):
                result_dict = vad_result[0]
            else:
                result_dict = vad_result
                
            # 提取语音段信息，支持'text'或'value'键
            segments = result_dict.get('text', result_dict.get('value'))
            if not segments:
                print(f"[WARNING] No speech segments found in {audio_path}")
                return None
                
            # 将时间戳从毫秒转换为秒
            vad_time = [[seg[0]/1000, seg[1]/1000] for seg in segments]
            
        except Exception as e:
            print(f"[ERROR] VAD processing failed for {audio_path}: {e}")
            return None

        # 2. 子分段处理（Sub-segmentation）- 添加长度校验
        # 将VAD检测到的语音段切分为1秒的子片段，步长为0.5秒（有重叠）
        subseg_data = []
        for st, ed in vad_time:
            subseg_st = st  # 子片段起始时间
            while subseg_st < ed:
                # 计算子片段结束时间，最大不超过1秒或原段结束时间
                subseg_ed = min(subseg_st + 1.0, ed)
                # 如果子片段长度小于0.5秒，跳过（太短的片段质量不佳）
                if subseg_ed - subseg_st < 0.5:  # 跳过短片段
                    break
                # 记录子片段信息
                subseg_data.append({'start': round(subseg_st, 2), 'stop': round(subseg_ed, 2)})
                subseg_st += 0.5  # 步长0.5秒（50%重叠）
        
        if not subseg_data: 
            print(f"[WARN] No valid subsegments for {audio_path}")
            return None

        # 3. 声纹嵌入提取（Embedding Extraction）- 添加固定长度处理
        try:
            # 加载音频文件，采样率与特征提取器一致
            wav = load_audio(audio_path, obj_fs=feature_extractor.sample_rate)
            sr = feature_extractor.sample_rate  # 采样率
            target_length = int(1.0 * sr)  # 固定1秒长度的采样点数
            
            # 存储处理后的音频片段
            wav_segments = []
            for seg_info in subseg_data:
                # 计算片段的起始和结束采样点
                start_sample = int(seg_info['start'] * sr)
                end_sample = int(seg_info['stop'] * sr)
                segment = wav[0, start_sample:end_sample]
                
                # 跳过静音或太短的片段（同DiarizationService）
                # 如果片段长度小于0.5秒，跳过
                if segment.shape[0] < int(0.5 * sr):
                    continue
                # 如果片段长度超过目标长度，截断
                if segment.shape[0] > target_length:
                    segment = segment[:target_length]  # 超长截断
                
                # 使用环形填充将片段填充到固定长度（1秒）
                padded_segment = circle_pad(segment, target_length)
                wav_segments.append(padded_segment)
            
            # 如果没有有效的片段，返回None
            if not wav_segments:
                print(f"[WARNING] No valid segments after filtering for {audio_path}")
                return None
                
            # 将所有片段堆叠为张量，添加通道维度
            wav_tensor = torch.stack(wav_segments).unsqueeze(1)
            
            # 提取特征和嵌入向量（不计算梯度，节省内存）
            with torch.no_grad():
                # 使用vmap批量提取特征
                feats = torch.vmap(feature_extractor)(wav_tensor.to(self.device))
                # 通过嵌入模型生成嵌入向量，并转移到CPU
                embeddings = embedding_model(feats).cpu().numpy()
            
            # 计算所有子片段嵌入向量的平均值，得到该音频文件的声纹特征
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
            
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed for {audio_path}: {e}")
            return None

    async def run_pipeline(self, person_ids: List[str], audio_files: List[str], collection_name: Optional[str]) -> Dict:
        """
        执行完整的音频声纹注册流程
        
        处理流程：
        1. 初始化特征提取器和嵌入模型
        2. 对每个音频文件提取声纹特征
        3. 将声纹特征插入到Milvus向量数据库
        
        Args:
            person_ids: 人员ID列表，与音频文件列表一一对应
            audio_files: 待注册的音频文件绝对路径列表
            collection_name: 目标Milvus集合名称，如果为None则使用默认配置
            
        Returns:
            Dict: 包含以下键的字典：
                - collection_name: 数据插入的目标集合名称
                - inserted_count: 成功插入的记录数量
                - inserted_result: 成功插入的记录详情列表
                
        Raises:
            RuntimeError: 当无法从任何音频文件中提取声纹特征或插入失败时抛出
        """
        # 配置文件路径
        conf_path = os.path.join(self.workspace, 'diar.yaml')
        try:
            # 准备配置和模型
            # 将配置内容写入临时配置文件
            with open(conf_path, 'w', encoding='utf-8') as f:
                f.write(settings.DIAR_CLUSTER_CONFIG_CONTENT)
            # 构建配置对象
            conf = build_config(conf_path)
            
            # 构建特征提取器和嵌入模型
            feature_extractor = build('feature_extractor', conf)
            embedding_model = build('embedding_model', conf)
            # 加载预训练的嵌入模型权重
            model_path = os.path.join(settings.SPEAKER_EMBEDDING_MODEL_PATH, settings.SPEAKER_EMBEDDING_MODEL_FILE)
            embedding_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            # 设置为评估模式并移动到指定设备
            embedding_model.eval().to(self.device)

            # 处理每个音频文件
            all_embeddings = []  # 存储所有有效的声纹嵌入向量
            valid_person_ids = []  # 存储对应的人员ID
            valid_file_ids = []  # 存储对应的文件ID

            # 遍历每个音频文件，提取声纹特征
            for person_id, audio_file in tqdm(zip(person_ids, audio_files), total=len(audio_files), desc="Processing audio for registration"):
                # 检查文件是否存在
                if not os.path.exists(audio_file):
                    print(f"[WARN] File not found, skipping: {audio_file}")
                    continue
                
                # 提取声纹嵌入向量
                embedding = self._extract_embedding_for_file(audio_file, feature_extractor, embedding_model)
                
                # 如果成功提取嵌入向量，添加到列表中
                if embedding is not None:
                    all_embeddings.append(embedding.tolist())  # 转换为列表格式
                    valid_person_ids.append(person_id)
                    # 提取文件名（不含扩展名）作为文件ID
                    valid_file_ids.append(os.path.splitext(os.path.basename(audio_file))[0])
                else:
                    print(f"[WARN] Failed to extract embedding for {audio_file}, skipping.")

            # 检查是否有有效的嵌入向量
            if not all_embeddings:
                raise RuntimeError("未能从任何音频文件中成功提取声纹特征。")

            # 连接Milvus并插入数据
            # 确定目标集合名称（使用传入的集合名或默认配置）
            target_collection = collection_name or settings.MILVUS_COLLECTION
            print(f"[INFO] Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            # 创建Milvus客户端
            mc = MilvusClient(config={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT})
            
            # 确保集合存在，维度为192
            mc.ensure_collection(target_collection, dim=192)
            
            # 插入声纹嵌入向量到Milvus
            print(f"[INFO] Inserting {len(all_embeddings)} embeddings into collection '{target_collection}'...")
            inserted_ids = mc.insert(
                collection_name=target_collection,
                person_ids=valid_person_ids,
                file_ids=valid_file_ids,
                embeddings=all_embeddings
            )

            # 检查插入是否成功
            if inserted_ids is None:
                raise RuntimeError("Failed to insert data into Milvus.")

            # 构建插入结果详情列表
            inserted_result = [
                {
                    "audio_file": audio_file,
                    "person_id": person_id,
                    "id": str(inserted_id)
                }
                for audio_file, person_id, inserted_id in zip(valid_file_ids, valid_person_ids, inserted_ids)
            ]

            return {
                "collection_name": target_collection,
                "inserted_count": len(inserted_ids),
                "inserted_result": inserted_result
            }

        finally:
            # 清理临时文件和目录
            print(f"[INFO] Cleaning up temporary workspace: {self.workspace}")
            if os.path.exists(self.workspace):
                shutil.rmtree(self.workspace)
            print("[INFO] Cleanup complete for registration service.")