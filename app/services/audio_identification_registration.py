import os
import sys
import json
import pickle
import shutil
import csv
from copy import deepcopy
from tqdm import tqdm
import warnings
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import circle_pad

from app.config.settings import settings
from utils.io_suppressor import suppress_stdout_stderr

warnings.filterwarnings("ignore", category=FutureWarning)

class IdentificationRegistrationService:
    def __init__(self, workspace: str = "./workspace"):
        self.workspace = os.path.abspath(workspace)
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        self.voiceprintlib_dir = os.path.join(self.workspace, 'result', 'voiceprintlib')
        self.emb_dir = os.path.join(self.workspace, 'emb')
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.voiceprintlib_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        print(f"[INFO] Workspace: {self.workspace}, Device: {self.device}")

    # --- 声纹预处理 (与聚类服务类似) ---
    def _preprocess_audios(self, audio_files: List[str], num_speakers: int) -> Optional[List[str]]:
        """执行分割、VAD、提取嵌入等预处理步骤，返回分割后的音频路径列表"""
        # 此处复用DiarizationClusterService中的大部分逻辑
        # 为保持独立性，这里重写关键部分
        print("++++++++ Stage 1: Preprocessing Audios ++++++++")
        # 1. Diarization
        separated_audios = self._run_diarization(audio_files, num_speakers)
        if not separated_audios: raise RuntimeError("Diarization failed.")
        # 2. VAD
        if not self._run_vad(separated_audios): raise RuntimeError("VAD failed.")
        # 3. Sub-segmentation
        if not self._run_prepare_subseg(separated_audios): raise RuntimeError("Sub-segmentation failed.")
        # 4. Embedding Extraction
        conf_path = os.path.join(self.workspace, 'diar.yaml')
        with open(conf_path, 'w', encoding='utf-8') as f:
            f.write(settings.DIAR_CLUSTER_CONFIG_CONTENT)
        conf = build_config(conf_path)
        if not self._run_extract_embeddings(separated_audios, conf): raise RuntimeError("Embedding Extraction failed.")
        
        return separated_audios

    # --- 声纹识别核心逻辑 ---
    def _load_voiceprint_library(self) -> Tuple[Dict, Dict]:
        speaker_voiceprints, speaker_audio_files = {}, {}
        if not os.path.exists(self.voiceprintlib_dir): return speaker_voiceprints, speaker_audio_files
        
        for speaker_id in os.listdir(self.voiceprintlib_dir):
            speaker_path = os.path.join(self.voiceprintlib_dir, speaker_id)
            if not os.path.isdir(speaker_path): continue
            
            voiceprint_file = os.path.join(speaker_path, f"{speaker_id}_voiceprint.pkl")
            if os.path.exists(voiceprint_file):
                with open(voiceprint_file, 'rb') as f:
                    data = pickle.load(f)
                    speaker_voiceprints[speaker_id] = data['avg_voiceprint']
                    speaker_audio_files[speaker_id] = data.get('audio', [])
        return speaker_voiceprints, speaker_audio_files

    def _compute_cosine_distance(self, emb1, emb2):
        return 1.0 - (np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _identify_speakers(self, segments_to_identify: List[str], speaker_voiceprints: Dict, threshold: float) -> List[Dict]:
        results = []
        for segment_path in segments_to_identify:
            rec_id = os.path.basename(segment_path).rsplit('.', 1)[0]
            emb_file = os.path.join(self.emb_dir, rec_id + '.pkl')
            if not os.path.exists(emb_file): continue

            with open(emb_file, 'rb') as f:
                embedding = pickle.load(f)['avg_embedding']

            distances = {spk_id: self._compute_cosine_distance(embedding, vp) for spk_id, vp in speaker_voiceprints.items()}
            
            res = {
                "source_segment": os.path.relpath(segment_path, self.workspace),
                "identified_speaker": "unknown",
                "is_new_speaker": True,
                "min_distance": None,
                "distances": {k: round(v, 4) for k, v in distances.items()}
            }

            if distances:
                min_speaker = min(distances, key=distances.get)
                min_dist = distances[min_speaker]
                res["min_distance"] = round(min_dist, 4)
                if min_dist <= threshold:
                    res["identified_speaker"] = min_speaker
                    res["is_new_speaker"] = False
            
            results.append(res)
        return results

    # --- 声纹库更新与注册 ---
    def _update_and_register(self, identification_results: List[Dict]):
        print("++++++++ Updating Voiceprint Library ++++++++")
        speaker_voiceprints, speaker_audio_files = self._load_voiceprint_library()
        
        # 找出所有新说话人的片段
        new_speaker_segments = defaultdict(list)
        updated_speakers = set()

        for res in identification_results:
            segment_rel_path = res['source_segment']
            segment_abs_path = os.path.join(self.workspace, segment_rel_path)
            rec_id = os.path.basename(segment_rel_path).rsplit('.', 1)[0]
            emb_file = os.path.join(self.emb_dir, rec_id + '.pkl')
            with open(emb_file, 'rb') as f:
                embedding = pickle.load(f)['avg_embedding']

            if res['is_new_speaker']:
                new_speaker_segments[segment_rel_path].append(embedding)
            else: # 更新已有说话人
                speaker_id = res['identified_speaker']
                updated_speakers.add(speaker_id)
                # 将新音频片段加入列表
                if segment_rel_path not in speaker_audio_files.get(speaker_id, []):
                    speaker_audio_files.setdefault(speaker_id, []).append(segment_rel_path)

        # 注册新说话人
        newly_registered = self._register_new_speakers(new_speaker_segments, speaker_voiceprints)

        # 更新已有说话人的平均声纹
        for speaker_id in updated_speakers:
            self._update_speaker_voiceprint(speaker_id, speaker_audio_files[speaker_id])
            
        return newly_registered, list(updated_speakers)

    def _get_next_speaker_id(self, existing_ids: List[str]) -> str:
        if not existing_ids: return "speaker_0"
        max_id = -1
        for sid in existing_ids:
            if sid.startswith("speaker_") and sid.split('_')[1].isdigit():
                max_id = max(max_id, int(sid.split('_')[1]))
        return f"speaker_{max_id + 1}"

    def _register_new_speakers(self, new_speaker_segments: Dict, existing_voiceprints: Dict) -> Dict:
        # 简单策略：每个未识别的片段都注册为一个新说话人
        # 高级策略：可以对这些新片段再进行一次聚类
        newly_registered = {}
        if not new_speaker_segments: return newly_registered

        print(f"Registering {len(new_speaker_segments)} new speaker(s)...")
        for segment_rel_path, embeddings in new_speaker_segments.items():
            new_speaker_id = self._get_next_speaker_id(list(existing_voiceprints.keys()) + list(newly_registered.keys()))
            
            # 创建新说话人目录和音频子目录
            speaker_dir = os.path.join(self.voiceprintlib_dir, new_speaker_id)
            audio_dir = os.path.join(speaker_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            
            # 复制音频文件
            shutil.copy2(os.path.join(self.workspace, segment_rel_path), audio_dir)
            
            # 计算并保存声纹
            avg_voiceprint = np.mean(embeddings, axis=0)
            voiceprint_data = {
                'audio': [os.path.basename(segment_rel_path)],
                'avg_voiceprint': avg_voiceprint
            }
            with open(os.path.join(speaker_dir, f"{new_speaker_id}_voiceprint.pkl"), 'wb') as f:
                pickle.dump(voiceprint_data, f)
            
            newly_registered[new_speaker_id] = [segment_rel_path]
        return newly_registered

    def _update_speaker_voiceprint(self, speaker_id: str, audio_rel_paths: List[str]):
        """根据一个说话人的所有音频片段，重新计算其平均声纹"""
        all_embeddings = []
        for rel_path in audio_rel_paths:
            rec_id = os.path.basename(rel_path).rsplit('.', 1)[0]
            emb_file = os.path.join(self.emb_dir, rec_id + '.pkl')
            if os.path.exists(emb_file):
                with open(emb_file, 'rb') as f:
                    all_embeddings.append(pickle.load(f)['avg_embedding'])
        
        if all_embeddings:
            new_avg_voiceprint = np.mean(all_embeddings, axis=0)
            speaker_dir = os.path.join(self.voiceprintlib_dir, speaker_id)
            audio_dir = os.path.join(speaker_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)

            # 更新音频文件
            for rel_path in audio_rel_paths:
                dest_path = os.path.join(audio_dir, os.path.basename(rel_path))
                if not os.path.exists(dest_path):
                    shutil.copy2(os.path.join(self.workspace, rel_path), dest_path)

            # 保存更新后的声纹
            voiceprint_data = {
                'audio': [os.path.basename(p) for p in audio_rel_paths],
                'avg_voiceprint': new_avg_voiceprint
            }
            with open(os.path.join(speaker_dir, f"{speaker_id}_voiceprint.pkl"), 'wb') as f:
                pickle.dump(voiceprint_data, f)
            print(f"Updated voiceprint for {speaker_id}.")

    # --- 主流程 ---
    async def run_pipeline(self, audio_files: List[str], num_speakers: int, update_library: bool, threshold: float) -> Dict:
        # 1. 预处理：分割、VAD、提取嵌入
        processed_segments = self._preprocess_audios(audio_files, num_speakers)
        if not processed_segments:
            raise RuntimeError("Audio preprocessing failed, no segments generated.")

        # 2. 加载现有声纹库
        speaker_voiceprints, _ = self._load_voiceprint_library()
        print(f"Loaded {len(speaker_voiceprints)} speakers from library.")

        # 3. 执行识别
        identification_results = self._identify_speakers(processed_segments, speaker_voiceprints, threshold)
        
        # 4. 根据需要更新和注册
        newly_registered, updated_speakers = None, None
        if update_library:
            newly_registered, updated_speakers = self._update_and_register(identification_results)

        return {
            "identification_results": identification_results,
            "newly_registered_speakers": newly_registered,
            "updated_speakers": updated_speakers,
            "library_updated": update_library
        }

    # --- 复用的辅助方法 ---
    def _run_diarization(self, input_files: List[str], num_speakers: int) -> Optional[List[str]]:
        # (此方法与DiarizationClusterService中的实现基本相同)
        print("... Running Diarization")
        audio_output_path = os.path.join(self.workspace, "dataset", "audio")
        os.makedirs(audio_output_path, exist_ok=True)
        separated_audio_files = []
        sd_pipeline = pipeline('speaker-diarization', model=settings.DIARIZATION_MODEL_PATH, model_revision=settings.DIARIZATION_MODEL_REVISION, device=self.device)
        for file_path in tqdm(input_files, desc="Diarization"):
            with suppress_stdout_stderr():
                result = sd_pipeline(file_path, oracle_num=num_speakers)
            if not result or 'text' not in result or not result['text']: continue
            file_name, _ = os.path.splitext(os.path.basename(file_path))
            for i in range(num_speakers):
                filename = f"{file_name}_speaker{i}.wav"
                output_audio_path_full = os.path.join(audio_output_path, filename)
                self._extract_speaker_audio(file_path, result['text'], i, output_audio_path_full)
                separated_audio_files.append(output_audio_path_full)
        return separated_audio_files if separated_audio_files else None

    def _extract_speaker_audio(self, wav_path, results, target_speaker, save_path):
        # (此方法与DiarizationClusterService中的实现相同)
        audio, sr = librosa.load(wav_path, sr=None)
        audio_out = np.zeros_like(audio)
        for seg in results:
            start_time, end_time, speaker_id = seg
            if speaker_id == target_speaker:
                start_sample, end_sample = int(start_time * sr), int(end_time * sr)
                audio_out[start_sample:end_sample] = audio[start_sample:end_sample]
        sf.write(save_path, audio_out, sr)

    def _run_vad(self, all_wavs: List[str]) -> bool:
        # (此方法与DiarizationClusterService中的实现相同)
        print("... Running VAD")
        vad_dir = os.path.join(self.workspace, 'vad')
        os.makedirs(vad_dir, exist_ok=True)
        vad_pipeline = pipeline(Tasks.voice_activity_detection, model=settings.VAD_MODEL_PATH, model_revision=settings.VAD_MODEL_REVISION, device=self.device)
        for wpath in tqdm(all_wavs, desc="VAD"):
            wid = os.path.basename(wpath).rsplit('.', 1)[0]
            output_file = os.path.join(vad_dir, wid + '_vad.json')
            json_dict = {}
            with suppress_stdout_stderr():
                vad_result = vad_pipeline(wpath)
            if vad_result and isinstance(vad_result, list): result_dict = vad_result[0]
            else: result_dict = vad_result
            segments = result_dict.get('text', result_dict.get('value'))
            if segments:
                vad_time = [[seg[0]/1000, seg[1]/1000] for seg in segments]
                for strt, end in vad_time:
                    json_dict[f"{wid}_{strt}_{end}"] = {'file': wpath, 'start': strt, 'stop': end}
            with open(output_file, 'w') as f: json.dump(json_dict, f, indent=2)
        return True

    def _run_prepare_subseg(self, all_wavs: List[str]) -> bool:
        # (此方法与DiarizationClusterService中的实现相同)
        print("... Running Sub-segmentation")
        for wpath in tqdm(all_wavs, desc="Sub-segmentation"):
            wid = os.path.basename(wpath).rsplit('.', 1)[0]
            vad_json_path = os.path.join(self.workspace, 'vad', wid + '_vad.json')
            seg_json_path = os.path.join(self.workspace, 'vad', wid + '_subseg.json')
            if not os.path.exists(vad_json_path): continue
            with open(vad_json_path, 'r') as f: vad_json_data = json.load(f)
            subseg_json = {}
            for segid, data in vad_json_data.items():
                st, ed = data['start'], data['stop']
                subseg_st = st
                while subseg_st < ed:
                    subseg_ed = min(subseg_st + 1.0, ed)
                    if subseg_ed - subseg_st < 0.5: break
                    item = deepcopy(data)
                    item.update({'start': round(subseg_st, 2), 'stop': round(subseg_ed, 2)})
                    subseg_json[f"{wid}_{round(subseg_st, 2)}_{round(subseg_ed, 2)}"] = item
                    subseg_st += 0.5
            with open(seg_json_path, 'w') as f: json.dump(subseg_json, f, indent=2)
        return True

    def _run_extract_embeddings(self, all_wavs: List[str], conf) -> bool:
        # (此方法与DiarizationClusterService中的实现基本相同)
        print("... Running Embedding Extraction")
        feature_extractor = build('feature_extractor', conf)
        embedding_model = build('embedding_model', conf)
        model_path = os.path.join(settings.SPEAKER_EMBEDDING_MODEL_PATH, settings.SPEAKER_EMBEDDING_MODEL_FILE)
        embedding_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        embedding_model.eval().to(self.device)
        for wpath in tqdm(all_wavs, desc="Embedding Extraction"):
            wid = os.path.basename(wpath).rsplit('.', 1)[0]
            subseg_json_path = os.path.join(self.workspace, 'vad', wid + '_subseg.json')
            if not os.path.exists(subseg_json_path): continue
            with open(subseg_json_path, "r") as f: meta = json.load(f)
            if not meta: continue
            wav = load_audio(wpath, obj_fs=feature_extractor.sample_rate)
            wavs_segments = [wav[0, int(meta[i]['start'] * 16000):int(meta[i]['stop'] * 16000)] for i in meta]
            valid_wavs = [s for s in wavs_segments if s.shape[0] > 0]
            if not valid_wavs: continue
            wavs_padded = [circle_pad(x, max(s.shape[0] for s in valid_wavs)) for x in valid_wavs]
            wavs_tensor = torch.stack(wavs_padded).unsqueeze(1)
            with torch.no_grad():
                feats = torch.vmap(feature_extractor)(wavs_tensor.to(self.device))
                embeddings = embedding_model(feats).cpu().numpy()
            stat_obj = {'avg_embedding': np.mean(embeddings, axis=0)}
            with open(os.path.join(self.emb_dir, wid + ".pkl"), 'wb') as f: pickle.dump(stat_obj, f)
        return True