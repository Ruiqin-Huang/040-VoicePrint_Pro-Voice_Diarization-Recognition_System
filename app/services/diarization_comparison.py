"""
说话人切分及声纹比对服务模块

该模块提供音频说话人切分、声纹特征提取、聚类分析和声纹比对功能。
主要功能包括：
- 主被叫切分（Speaker Diarization）
- VAD（语音活动检测）处理
- 声纹特征提取和嵌入向量生成
- t-SNE降维和HDBSCAN聚类
- 与Milvus声纹库进行相似度比对

依赖：
- modelscope: 用于说话人切分和VAD
- speakerlab: 用于音频处理和特征提取
- sklearn: 用于t-SNE降维
- hdbscan: 用于聚类分析
- librosa: 用于音频处理
- torch: 用于深度学习模型推理
- pymilvus: 用于向量数据库操作
"""

import os
import shutil
import uuid
import json
import pickle
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import librosa
import soundfile as sf
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import hdbscan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad

from app.config.settings import settings
from utils.io_suppressor import suppress_stdout_stderr
from utils.Milvus import MilvusClient

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
    import tempfile
    
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

class DiarizationComparisonService:
    """
    说话人切分及声纹比对服务类
    
    负责处理音频文件的主被叫切分、声纹特征提取、聚类分析和声纹比对。
    """
    def __init__(self):
        """
        初始化说话人切分及声纹比对服务
        
        创建临时工作目录，设置计算设备（GPU/CPU），准备处理环境。
        """
        # 工作空间根目录
        self.workspace = os.path.join(settings.OUTPUT_DIR, "diarization_comparison")
        # 音频切分结果存储目录
        self.audio_segmentation_dir = os.path.join(self.workspace, "audio_segmentation")
        # VAD（语音活动检测）结果存储目录
        self.vad_dir = os.path.join(self.workspace, "vad")
        # 声纹嵌入向量存储目录
        self.emb_dir = os.path.join(self.workspace, "emb")
        # 计算设备（GPU或CPU）
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        
        # 创建必要的目录
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.audio_segmentation_dir, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        print(f"[INFO] Created temporary workspace for comparison: {self.workspace}, Device: {self.device}")

    def _run_diarization(self, audio_files: List[str]) -> List[Dict]:
        """
        对每个音频文件进行主被叫切分，返回切分后的文件信息列表
        
        处理流程：
        1. 检查音频长度，如果不足20秒则扩展
        2. 使用说话人切分模型识别主叫和被叫
        3. 提取并保存主叫和被叫的音频片段
        
        Args:
            audio_files: 待处理的音频文件绝对路径列表
            
        Returns:
            List[Dict]: 切分后的文件信息列表，每个字典包含：
                - origin_audio_file: 原始音频文件名
                - segment_audio_file: 切分后的音频文件名
                - segment_audio_path: 切分后的音频文件绝对路径
                - calling_called: 主叫或被叫标识（'calling' 或 'called'）
        """
        print("++++++++ Stage 1: Speaker Diarization ++++++++")
        num_speakers = 2  # 固定为2个说话人（主叫和被叫）
        # 初始化说话人切分管道
        sd_pipeline = pipeline('speaker-diarization', model=settings.DIARIZATION_MODEL_PATH, model_revision=settings.DIARIZATION_MODEL_REVISION, device=self.device)
        
        # 存储切分后的文件信息
        segmented_files_info = []
        temp_files_to_cleanup = []  # 用于跟踪需要清理的临时文件
        
        for file_path in tqdm(audio_files, desc="Diarization"):
            # 检查音频长度，如果不足20秒则扩展（仅在内存中处理，使用临时文件）
            processing_path, needs_cleanup, original_duration = extend_audio_if_needed(
                file_path, min_duration=20.0, temp_dir=None
            )
            if needs_cleanup:
                temp_files_to_cleanup.append(processing_path)
                print(f"[INFO] 音频 {file_path} 原始时长 {original_duration:.2f}秒，已扩展至至少20秒")
            
            with suppress_stdout_stderr():
                result = sd_pipeline(processing_path, oracle_num=num_speakers)
            
            if not result or 'text' not in result or not result['text']:
                print(f"[WARN] Diarization failed for {file_path}, skipping.")
                continue

            original_filename = os.path.basename(file_path)
            file_name_base, _ = os.path.splitext(original_filename)
            
            # 加载原始音频（用于提取说话人片段）
            audio, sr = librosa.load(file_path, sr=None)
            original_duration_samples = len(audio)
            original_duration_seconds = original_duration_samples / sr
            
            # 如果音频被扩展过，需要过滤掉超出原始音频长度的切分结果
            filtered_result = []
            if needs_cleanup:
                # 只保留时间戳在原始音频长度内的切分结果
                for seg_start, seg_end, spk_id in result['text']:
                    if seg_start < original_duration_seconds:
                        # 限制结束时间不超过原始音频长度
                        seg_end = min(seg_end, original_duration_seconds)
                        if seg_end > seg_start:  # 确保还有有效时长
                            filtered_result.append((seg_start, seg_end, spk_id))
            else:
                filtered_result = result['text']
            
            # 提取两个说话人的音频
            # speaker 0 -> Calling, speaker 1 -> Called
            speaker_map = {0: "calling", 1: "called"}
            for speaker_id, role in speaker_map.items():
                segment_filename = f"{file_name_base}_{role.capitalize()}.wav"
                output_path = os.path.join(self.audio_segmentation_dir, segment_filename)
                
                audio_out = np.zeros_like(audio)
                for seg_start, seg_end, spk_id in filtered_result:
                    if spk_id == speaker_id:
                        start_sample, end_sample = int(seg_start * sr), int(seg_end * sr)
                        # 确保索引不超出范围
                        end_sample = min(end_sample, original_duration_samples)
                        if end_sample > start_sample:
                            audio_out[start_sample:end_sample] = audio[start_sample:end_sample]
                
                sf.write(output_path, audio_out, sr)
                
                segmented_files_info.append({
                    "origin_audio_file": original_filename,
                    "segment_audio_file": segment_filename,
                    "segment_audio_path": output_path,
                    "calling_called": role
                })
            
            # 清理本次循环产生的临时文件（如果有）
            if needs_cleanup and os.path.exists(processing_path):
                try:
                    os.remove(processing_path)
                except Exception as e:
                    print(f"[WARN] 清理临时文件失败 {processing_path}: {e}")
        
        return segmented_files_info

    def _run_vad_for_segment(self, wav_path: str) -> Optional[Dict]:
        """
        对单个音频片段执行VAD（语音活动检测）
        
        识别音频中的有效语音段，过滤掉静音部分。
        
        Args:
            wav_path: 音频文件的绝对路径
            
        Returns:
            Optional[Dict]: VAD检测结果字典，键为片段ID，值为包含文件路径、起始和结束时间的字典。
                          如果检测失败则返回None
        """
        # 提取文件名（不含扩展名）作为工作ID
        wid = os.path.basename(wav_path).rsplit('.', 1)[0]
        # VAD结果JSON文件路径
        vad_json_path = os.path.join(self.vad_dir, wid + '_vad.json')
        # 初始化VAD管道
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=settings.VAD_MODEL_PATH,
            model_revision=settings.VAD_MODEL_REVISION,
            device=self.device,
        )
        
        try:
            with suppress_stdout_stderr():
                vad_result = vad_pipeline(wav_path)
            
            # 处理VAD结果格式
            if vad_result and isinstance(vad_result, list):
                result_dict = vad_result[0]
            else:
                result_dict = vad_result
                
            segments = result_dict.get('text', result_dict.get('value'))
            if not segments:
                print(f"[WARNING] No speech segments found in {wav_path}")
                return None
                
            vad_segments = [[seg[0]/1000, seg[1]/1000] for seg in segments]
            vad_data = {}
            for i, (strt, end) in enumerate(vad_segments):
                vad_data[f"{wid}_{i}"] = {'file': wav_path, 'start': strt, 'stop': end}
            
            # 保存VAD结果
            with open(vad_json_path, 'w') as f:
                json.dump(vad_data, f, indent=2)
                
            return vad_data
        except Exception as e:
            print(f"[ERROR] VAD processing failed for {wav_path}: {e}")
            return None

    def _prepare_subsegments(self, wav_path: str, vad_data: Dict) -> Dict:
        """
        根据VAD结果生成子片段
        
        将VAD检测到的语音段切分为1秒的子片段，步长为0.5秒（有重叠），用于后续特征提取。
        
        Args:
            wav_path: 音频文件的绝对路径
            vad_data: VAD检测结果字典
            
        Returns:
            Dict: 子片段数据字典，键为子片段ID，值为包含文件路径、起始时间、结束时间和时长的字典
        """
        # 提取文件名（不含扩展名）作为工作ID
        wid = os.path.basename(wav_path).rsplit('.', 1)[0]
        # 子片段结果JSON文件路径
        subseg_json_path = os.path.join(self.vad_dir, wid + '_subseg.json')
        
        # 存储子片段数据
        subseg_data = {}
        for segid, data in vad_data.items():
            st, ed = float(data['start']), float(data['stop'])  # 起始和结束时间
            subseg_st = st  # 子片段起始时间
            while subseg_st < ed:
                # 计算子片段结束时间，最大不超过1秒或原段结束时间
                subseg_ed = min(subseg_st + 1.0, ed)
                # 如果子片段长度小于0.5秒，跳过
                if subseg_ed - subseg_st < 0.5:  # 小于0.5秒的片段跳过
                    break
                # 创建子片段数据项
                item = deepcopy(data)
                item.update({
                    'start': round(subseg_st, 2),
                    'stop': round(subseg_ed, 2),
                    'duration': round(subseg_ed - subseg_st, 2)
                })
                # 生成子片段ID
                subseg_id = f"{wid}_{round(subseg_st, 2)}_{round(subseg_ed, 2)}"
                subseg_data[subseg_id] = item
                subseg_st += 0.5  # 步长为0.5秒（50%重叠）
                
        with open(subseg_json_path, 'w') as f:
            json.dump(subseg_data, f, indent=2)
            
        return subseg_data

    def _extract_avg_embedding(self, wav_path: str, subseg_data: Dict, 
                              feature_extractor, embedding_model) -> Optional[np.ndarray]:
        """
        提取子片段嵌入并计算平均嵌入
        
        对音频的所有子片段提取声纹特征，然后计算平均嵌入向量。
        
        Args:
            wav_path: 音频文件的绝对路径
            subseg_data: 子片段数据字典
            feature_extractor: 特征提取器对象，用于提取音频特征
            embedding_model: 嵌入模型对象，用于生成声纹嵌入向量
            
        Returns:
            Optional[np.ndarray]: 192维的平均声纹嵌入向量，如果提取失败则返回None
        """
        try:
            wav = load_audio(wav_path, obj_fs=feature_extractor.sample_rate)
            sr = feature_extractor.sample_rate
            
            # 提取所有有效子片段
            wav_segments = []
            # 固定目标长度为1秒采样点
            target_length = int(1.0 * sr)
            for segid, seginfo in subseg_data.items():
                start_sample = int(float(seginfo['start']) * sr)
                end_sample = int(float(seginfo['stop']) * sr)
                segment = wav[0, start_sample:end_sample]
                
                # 跳过静音或太短的片段
                if segment.shape[0] < int(0.5 * sr):  # 小于0.5秒
                    continue
                if segment.shape[0] > target_length:
                    segment = segment[:target_length]  # 超长截断
                padded_segment = circle_pad(segment, target_length)  # 不足填充
                wav_segments.append(padded_segment)
            
            if not wav_segments:
                print(f"[WARNING] No valid subsegments found for {wav_path}")
                return None
                
            # 转换为张量
            wav_tensor = torch.stack(wav_segments).unsqueeze(1)
            
            # 提取所有子片段的嵌入
            with torch.no_grad():
                feats = torch.vmap(feature_extractor)(wav_tensor.to(self.device))
                embeddings = embedding_model(feats).cpu().numpy()
            
            # 计算平均嵌入
            avg_embedding = np.mean(embeddings, axis=0)
            
            # 保存嵌入结果
            emb_file = os.path.join(self.emb_dir, f"{os.path.basename(wav_path).rsplit('.', 1)[0]}.pkl")
            with open(emb_file, 'wb') as f:
                pickle.dump({'avg_embedding': avg_embedding}, f)
                
            return avg_embedding
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed for {wav_path}: {e}")
            return None

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个归一化向量的余弦相似度
        
        通过计算两个向量的点积（归一化后）来得到余弦相似度，范围在[-1, 1]之间。
        对于声纹特征向量，通常值在[0, 1]之间，值越大表示相似度越高。
        
        Args:
            emb1: 第一个嵌入向量
            emb2: 第二个嵌入向量
            
        Returns:
            float: 余弦相似度分数，范围通常在[0, 1]之间
        """
        # 归一化向量
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        # 计算点积（即余弦相似度）
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    async def run_pipeline(self, audio_files: List[str], collection_name: str) -> Dict:
        """
        执行完整的说话人切分及声纹比对流程
        
        处理流程：
        1. 主被叫切分：将音频切分为主叫和被叫两个片段
        2. 加载声纹库：从Milvus加载目标声纹库
        3. 准备模型：初始化特征提取器和嵌入模型
        4. 提取声纹：为所有切分片段提取声纹特征
        5. 聚类分析：使用t-SNE降维和HDBSCAN聚类
        6. 声纹比对：将切分片段与声纹库进行相似度比对
        
        Args:
            audio_files: 待处理的音频文件绝对路径列表
            collection_name: 用于比对的目标声纹库集合名称
            
        Returns:
            Dict: 包含以下键的字典：
                - collection_name: 参与比较的目标说话人声纹库集合名称
                - comparison_results: 所有音频片段的切分和比对结果列表
                - cluster_results: 所有分割音频的聚类结果列表
                
        Raises:
            RuntimeError: 当无法切分说话人片段或提取声纹特征时抛出
        """
        # 存储比对结果
        comparison_results = []
        
        try:
            # 1. 主被叫切分
            segmented_files = self._run_diarization(audio_files)
            if not segmented_files:
                raise RuntimeError("未能从任何音频文件中成功切分出说话人片段。")
            print(f"[INFO] Diarization completed. Segmented into {len(segmented_files)} files.")

            # 2. 加载声纹库
            print("++++++++ Stage 2: Loading Voiceprint Library from Milvus ++++++++")
            # 创建Milvus客户端
            mc = MilvusClient(config={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT})
            # 获取所有人员的平均嵌入向量
            voiceprint_library = mc.get_all_person_avg_embeddings(collection_name)
            if not voiceprint_library:
                print(f"[WARN] Voiceprint library '{collection_name}' is empty. All segments will be 'unknown'.")

            # 3. 准备模型
            print("++++++++ Stage 3: Preparing Models ++++++++")
            # 配置文件路径
            conf_path = os.path.join(self.workspace, 'diar.yaml')
            # 写入配置文件
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

            # 4. 提取所有分割片段的声纹
            print("++++++++ Stage 4: Extracting Embeddings for All Segments ++++++++")
            # 存储所有有效的声纹嵌入向量
            all_segment_embeddings = []
            # 存储对应的片段信息
            valid_segments_info = []
            for segment_info in tqdm(segmented_files, desc="Extracting Embeddings"):
                segment_path = segment_info["segment_audio_path"]
                vad_data = self._run_vad_for_segment(segment_path)
                if not vad_data:
                    print(f"[WARN] VAD failed for {segment_path}, skipping.")
                    continue
                subseg_data = self._prepare_subsegments(segment_path, vad_data)
                if not subseg_data:
                    print(f"[WARN] No valid subsegments for {segment_path}, skipping.")
                    continue
                segment_embedding = self._extract_avg_embedding(
                    segment_path, subseg_data, feature_extractor, embedding_model
                )
                if segment_embedding is not None:
                    all_segment_embeddings.append(segment_embedding)
                    valid_segments_info.append(segment_info)
                else:
                    print(f"[WARN] Could not extract embedding for {segment_path}, skipping.")

            if not all_segment_embeddings:
                raise RuntimeError("未能从任何分割片段中提取有效的声纹特征。")      
            
            # 5. t-SNE降维和HDBSCAN聚类
            print("++++++++ Stage 5: Clustering Segments ++++++++")
            # 将嵌入向量列表转换为numpy数组
            embeddings_array = np.array(all_segment_embeddings)

            # 使用t-SNE将高维嵌入向量降维到2维，用于可视化
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings_array)-1))
            embeddings_2d = tsne.fit_transform(embeddings_array)

            # 使用HDBSCAN进行聚类分析
            cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=2,  # 最小聚类大小为2
                cluster_selection_method='leaf'  # 使用叶子节点选择方法
            )
            # 执行聚类，返回每个样本的聚类标签（-1表示噪声点）
            cluster_labels = cluster_model.fit_predict(embeddings_2d)

            # 重新映射聚类标签：将-1（噪声点）转换为新的独立类别
            # 步骤1: 找到所有非噪声标签的最大值
            max_label = np.max(cluster_labels[cluster_labels >= 0]) if np.any(cluster_labels >= 0) else -1
            # 步骤2: 为每个噪声点分配新的唯一标签
            new_labels = []
            next_label = max_label + 1  # 新标签起始值
            for label in cluster_labels:
                if label == -1:  # 噪声点
                    new_labels.append(next_label)
                    next_label += 1
                else:
                    new_labels.append(label)

            # 构建聚类结果列表
            cluster_results = [
                {
                    "segment_audio_file": info["segment_audio_file"],
                    "x_coordinate": float(coords[0]),  # t-SNE降维后的X坐标
                    "y_coordinate": float(coords[1]),  # t-SNE降维后的Y坐标
                    "cluster_id": int(label)  # 聚类ID
                }
                for info, coords, label in zip(valid_segments_info, embeddings_2d, new_labels)
            ]
            
            # 6. 声纹比对与结果组装
            print("++++++++ Stage 6: Comparing Embeddings with Voiceprint Library ++++++++")
            for i, segment_embedding in enumerate(tqdm(all_segment_embeddings, desc="Comparing Segments")):
                segment_info = valid_segments_info[i]
                full_compare_result = []

                if voiceprint_library:
                    similarities = {
                        person_id: self._compute_cosine_similarity(segment_embedding, avg_emb)
                        for person_id, avg_emb in voiceprint_library.items()
                    }
                    full_compare_result = [{"person_id": p, "similarity": round(s, 4)} for p, s in similarities.items()]
                    if similarities:
                        top_match_speaker = max(similarities, key=similarities.get)
                        top_match_similarity = similarities[top_match_speaker]
                    else:
                        top_match_speaker = None
                        top_match_similarity = None
                else:
                    top_match_speaker = None
                    top_match_similarity = None

                result_item = {
                    "origin_audio_file": segment_info["origin_audio_file"],
                    "segment_audio_file": segment_info["segment_audio_file"],
                    "calling_called": segment_info["calling_called"],
                    "cluster_id": int(new_labels[i]),  # 修改这里，使用new_labels
                    "top_match_speaker": top_match_speaker,
                    "top_match_similarity": round(top_match_similarity, 4) if top_match_similarity is not None else None,
                    "compare_result": sorted(full_compare_result, key=lambda x: x['similarity'], reverse=True)
                }
                comparison_results.append(result_item)

            return {
                "collection_name": collection_name,
                "comparison_results": comparison_results,
                "cluster_results": cluster_results
            }

        finally:
            # 清理临时文件
            print("[INFO] Cleaning up temporary files...")
            conf_path = os.path.join(self.workspace, 'diar.yaml')
            if os.path.exists(conf_path):
                os.remove(conf_path)
            
            if os.path.exists(self.emb_dir):
                shutil.rmtree(self.emb_dir)
            
            if os.path.exists(self.vad_dir):
                shutil.rmtree(self.vad_dir)
            
            print("[INFO] Cleanup complete. Segmented audios are kept.")