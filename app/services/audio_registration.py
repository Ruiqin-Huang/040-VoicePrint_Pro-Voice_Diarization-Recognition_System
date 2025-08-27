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
    def __init__(self):
        self.workspace = os.path.join(settings.OUTPUT_DIR, "audio_registration")
        self.vad_dir = os.path.join(self.workspace, "vad")  # 新增VAD目录
        self.emb_dir = os.path.join(self.workspace, "emb")  # 新增嵌入目录
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)  # 创建目录
        os.makedirs(self.emb_dir, exist_ok=True)  # 创建目录
        print(f"[INFO] Created temporary workspace for registration: {self.workspace}, Device: {self.device}")

    def _extract_embedding_for_file(self, audio_path: str, feature_extractor, embedding_model) -> Optional[np.ndarray]:
        """为单个音频文件执行VAD、分段和特征提取，返回192维向量"""
        wid = os.path.basename(audio_path).rsplit('.', 1)[0]
        
        # 1. VAD - 添加异常处理和静音检测
        try:
            # 静音检测（新增）
            audio_data, sr = librosa.load(audio_path, sr=None)
            if np.max(np.abs(audio_data)) < 0.01:
                print(f"[WARN] Silent audio detected: {audio_path}")
                return None
                
            vad_pipeline = pipeline(Tasks.voice_activity_detection, 
                                    model=settings.VAD_MODEL_PATH, 
                                    model_revision=settings.VAD_MODEL_REVISION, 
                                    device=self.device)
            
            with suppress_stdout_stderr():
                vad_result = vad_pipeline(audio_path)
            
            # 处理VAD结果格式（同DiarizationService）
            if vad_result and isinstance(vad_result, list):
                result_dict = vad_result[0]
            else:
                result_dict = vad_result
                
            segments = result_dict.get('text', result_dict.get('value'))
            if not segments:
                print(f"[WARNING] No speech segments found in {audio_path}")
                return None
                
            vad_time = [[seg[0]/1000, seg[1]/1000] for seg in segments]
            
        except Exception as e:
            print(f"[ERROR] VAD processing failed for {audio_path}: {e}")
            return None

        # 2. Sub-segmentation - 添加长度校验
        subseg_data = []
        for st, ed in vad_time:
            subseg_st = st
            while subseg_st < ed:
                subseg_ed = min(subseg_st + 1.0, ed)
                if subseg_ed - subseg_st < 0.5:  # 跳过短片段
                    break
                subseg_data.append({'start': round(subseg_st, 2), 'stop': round(subseg_ed, 2)})
                subseg_st += 0.5  # 步长0.5秒
        
        if not subseg_data: 
            print(f"[WARN] No valid subsegments for {audio_path}")
            return None

        # 3. Embedding Extraction - 添加固定长度处理
        try:
            wav = load_audio(audio_path, obj_fs=feature_extractor.sample_rate)
            sr = feature_extractor.sample_rate
            target_length = int(1.0 * sr)  # 固定1秒长度
            
            wav_segments = []
            for seg_info in subseg_data:
                start_sample = int(seg_info['start'] * sr)
                end_sample = int(seg_info['stop'] * sr)
                segment = wav[0, start_sample:end_sample]
                
                # 跳过静音或太短的片段（同DiarizationService）
                if segment.shape[0] < int(0.5 * sr):
                    continue
                if segment.shape[0] > target_length:
                    segment = segment[:target_length]  # 超长截断
                
                # 环形填充到固定长度
                padded_segment = circle_pad(segment, target_length)
                wav_segments.append(padded_segment)
            
            if not wav_segments:
                print(f"[WARNING] No valid segments after filtering for {audio_path}")
                return None
                
            # 转换为张量
            wav_tensor = torch.stack(wav_segments).unsqueeze(1)
            
            with torch.no_grad():
                feats = torch.vmap(feature_extractor)(wav_tensor.to(self.device))
                embeddings = embedding_model(feats).cpu().numpy()
            
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
            
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed for {audio_path}: {e}")
            return None

    async def run_pipeline(self, person_ids: List[str], audio_files: List[str], collection_name: Optional[str]) -> Dict:
        conf_path = os.path.join(self.workspace, 'diar.yaml')
        try:
            # 准备配置和模型
            with open(conf_path, 'w', encoding='utf-8') as f:
                f.write(settings.DIAR_CLUSTER_CONFIG_CONTENT)
            conf = build_config(conf_path)
            
            feature_extractor = build('feature_extractor', conf)
            embedding_model = build('embedding_model', conf)
            model_path = os.path.join(settings.SPEAKER_EMBEDDING_MODEL_PATH, settings.SPEAKER_EMBEDDING_MODEL_FILE)
            embedding_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            embedding_model.eval().to(self.device)

            # 处理每个音频文件
            all_embeddings = []
            valid_person_ids = []
            valid_file_ids = []

            for person_id, audio_file in tqdm(zip(person_ids, audio_files), total=len(audio_files), desc="Processing audio for registration"):
                if not os.path.exists(audio_file):
                    print(f"[WARN] File not found, skipping: {audio_file}")
                    continue
                
                embedding = self._extract_embedding_for_file(audio_file, feature_extractor, embedding_model)
                
                if embedding is not None:
                    all_embeddings.append(embedding.tolist())
                    valid_person_ids.append(person_id)
                    valid_file_ids.append(os.path.splitext(os.path.basename(audio_file))[0])
                else:
                    print(f"[WARN] Failed to extract embedding for {audio_file}, skipping.")

            if not all_embeddings:
                raise RuntimeError("未能从任何音频文件中成功提取声纹特征。")

            # 连接Milvus并插入数据
            target_collection = collection_name or settings.MILVUS_COLLECTION
            print(f"[INFO] Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            mc = MilvusClient(config={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT})
            
            mc.ensure_collection(target_collection, dim=192)
            
            print(f"[INFO] Inserting {len(all_embeddings)} embeddings into collection '{target_collection}'...")
            inserted_ids = mc.insert(
                collection_name=target_collection,
                person_ids=valid_person_ids,
                file_ids=valid_file_ids,
                embeddings=all_embeddings
            )

            if inserted_ids is None:
                raise RuntimeError("Failed to insert data into Milvus.")

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