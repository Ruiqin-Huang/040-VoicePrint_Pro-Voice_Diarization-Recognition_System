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

class DiarizationComparisonService:
    def __init__(self):
        self.workspace = os.path.join(settings.OUTPUT_DIR, "diarization_comparison")
        self.audio_segmentation_dir = os.path.join(self.workspace, "audio_segmentation")
        self.vad_dir = os.path.join(self.workspace, "vad")
        self.emb_dir = os.path.join(self.workspace, "emb")
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.audio_segmentation_dir, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        print(f"[INFO] Created temporary workspace for comparison: {self.workspace}, Device: {self.device}")

    def _run_diarization(self, audio_files: List[str]) -> List[Dict]:
        """对每个音频文件进行主被叫切分，返回切分后的文件信息列表"""
        print("++++++++ Stage 1: Speaker Diarization ++++++++")
        num_speakers = 2  # 固定为2个说话人
        sd_pipeline = pipeline('speaker-diarization', model=settings.DIARIZATION_MODEL_PATH, model_revision=settings.DIARIZATION_MODEL_REVISION, device=self.device)
        
        segmented_files_info = []
        for file_path in tqdm(audio_files, desc="Diarization"):
            with suppress_stdout_stderr():
                result = sd_pipeline(file_path, oracle_num=num_speakers)
            
            if not result or 'text' not in result or not result['text']:
                print(f"[WARN] Diarization failed for {file_path}, skipping.")
                continue

            original_filename = os.path.basename(file_path)
            file_name_base, _ = os.path.splitext(original_filename)
            
            # 提取两个说话人的音频
            # speaker 0 -> Calling, speaker 1 -> Called
            speaker_map = {0: "calling", 1: "called"}
            for speaker_id, role in speaker_map.items():
                segment_filename = f"{file_name_base}_{role.capitalize()}.wav"
                output_path = os.path.join(self.audio_segmentation_dir, segment_filename)
                
                audio, sr = librosa.load(file_path, sr=None)
                audio_out = np.zeros_like(audio)
                for seg_start, seg_end, spk_id in result['text']:
                    if spk_id == speaker_id:
                        start_sample, end_sample = int(seg_start * sr), int(seg_end * sr)
                        audio_out[start_sample:end_sample] = audio[start_sample:end_sample]
                
                sf.write(output_path, audio_out, sr)
                
                segmented_files_info.append({
                    "origin_audio_file": original_filename,
                    "segment_audio_file": segment_filename,
                    "segment_audio_path": output_path,
                    "calling_called": role
                })
        return segmented_files_info

    def _run_vad_for_segment(self, wav_path: str) -> Optional[Dict]:
        """对单个音频片段执行VAD检测"""
        wid = os.path.basename(wav_path).rsplit('.', 1)[0]
        vad_json_path = os.path.join(self.vad_dir, wid + '_vad.json')
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
        """根据VAD结果生成子片段"""
        wid = os.path.basename(wav_path).rsplit('.', 1)[0]
        subseg_json_path = os.path.join(self.vad_dir, wid + '_subseg.json')
        
        subseg_data = {}
        for segid, data in vad_data.items():
            st, ed = float(data['start']), float(data['stop'])
            subseg_st = st
            while subseg_st < ed:
                subseg_ed = min(subseg_st + 1.0, ed)
                if subseg_ed - subseg_st < 0.5:  # 小于0.5秒的片段跳过
                    break
                item = deepcopy(data)
                item.update({
                    'start': round(subseg_st, 2),
                    'stop': round(subseg_ed, 2),
                    'duration': round(subseg_ed - subseg_st, 2)
                })
                subseg_id = f"{wid}_{round(subseg_st, 2)}_{round(subseg_ed, 2)}"
                subseg_data[subseg_id] = item
                subseg_st += 0.5  # 步长为0.5秒
                
        with open(subseg_json_path, 'w') as f:
            json.dump(subseg_data, f, indent=2)
            
        return subseg_data

    def _extract_avg_embedding(self, wav_path: str, subseg_data: Dict, 
                              feature_extractor, embedding_model) -> Optional[np.ndarray]:
        """提取子片段嵌入并计算平均嵌入"""
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
        """计算两个归一化向量的余弦相似度"""
        # 归一化
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        # 计算点积（即余弦相似度）
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    async def run_pipeline(self, audio_files: List[str], collection_name: str) -> Dict:
        comparison_results = []
        
        try:
            # 1. 主被叫切分
            segmented_files = self._run_diarization(audio_files)
            if not segmented_files:
                raise RuntimeError("未能从任何音频文件中成功切分出说话人片段。")
            print(f"[INFO] Diarization completed. Segmented into {len(segmented_files)} files.")

            # 2. 加载声纹库
            print("++++++++ Stage 2: Loading Voiceprint Library from Milvus ++++++++")
            mc = MilvusClient(config={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT})
            voiceprint_library = mc.get_all_person_avg_embeddings(collection_name)
            if not voiceprint_library:
                print(f"[WARN] Voiceprint library '{collection_name}' is empty. All segments will be 'unknown'.")

            # 3. 准备模型
            print("++++++++ Stage 3: Preparing Models ++++++++")
            conf_path = os.path.join(self.workspace, 'diar.yaml')
            with open(conf_path, 'w', encoding='utf-8') as f:
                f.write(settings.DIAR_CLUSTER_CONFIG_CONTENT)
            conf = build_config(conf_path)
            feature_extractor = build('feature_extractor', conf)
            embedding_model = build('embedding_model', conf)
            model_path = os.path.join(settings.SPEAKER_EMBEDDING_MODEL_PATH, settings.SPEAKER_EMBEDDING_MODEL_FILE)
            embedding_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            embedding_model.eval().to(self.device)

            # 4. 提取所有分割片段的声纹
            print("++++++++ Stage 4: Extracting Embeddings for All Segments ++++++++")
            all_segment_embeddings = []
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
            
            # 5. t-SNE降维和谱聚类
            print("++++++++ Stage 5: Clustering Segments ++++++++")
            embeddings_array = np.array(all_segment_embeddings)

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings_array)-1))
            embeddings_2d = tsne.fit_transform(embeddings_array)

            cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=2,
                cluster_selection_method='leaf'
            )
            cluster_labels = cluster_model.fit_predict(embeddings_2d)

            # 重新映射聚类标签：将-1转换为新的独立类别
            # 步骤1: 找到所有非噪声标签的最大值
            max_label = np.max(cluster_labels[cluster_labels >= 0]) if np.any(cluster_labels >= 0) else -1
            # 步骤2: 为每个噪声点分配新的唯一标签
            new_labels = []
            next_label = max_label + 1  # 新标签起始值
            for label in cluster_labels:
                if label == -1:
                    new_labels.append(next_label)
                    next_label += 1
                else:
                    new_labels.append(label)

            cluster_results = [
                {
                    "segment_audio_file": info["segment_audio_file"],
                    "x_coordinate": float(coords[0]),
                    "y_coordinate": float(coords[1]),
                    "cluster_id": int(label)
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
                    "cluster_id": int(cluster_labels[i]),
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