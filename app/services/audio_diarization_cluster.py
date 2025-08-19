import os
import sys
import json
import pickle
import shutil
import csv
from copy import deepcopy
from tqdm import tqdm
import warnings
from typing import List, Dict, Optional

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

class DiarizationClusterService:
    def __init__(self, workspace: str = "./workspace"):
        self.workspace = os.path.abspath(workspace)
        self.device = f'cuda:{settings.GPU_ID}' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        os.makedirs(self.workspace, exist_ok=True)
        print(f"[INFO] Workspace: {self.workspace}, Device: {self.device}")

    def _extract_speaker_audio(self, wav_path, results, target_speaker, save_path):
        audio, sr = librosa.load(wav_path, sr=None)
        audio_out = np.zeros_like(audio)
        for seg in results:
            start_time, end_time, speaker_id = seg
            if speaker_id == target_speaker:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_out[start_sample:end_sample] = audio[start_sample:end_sample]
        sf.write(save_path, audio_out, sr)

    def _run_diarization(self, input_files: List[str], num_speakers: int) -> Optional[List[str]]:
        print("++++++++ Stage 1: Speaker Diarization ++++++++")
        audio_output_path = os.path.join(self.workspace, "dataset", "audio")
        os.makedirs(audio_output_path, exist_ok=True)
        
        separated_audio_files = []
        sd_pipeline = pipeline(
            task='speaker-diarization',
            model=settings.DIARIZATION_MODEL_PATH,
            model_revision=settings.DIARIZATION_MODEL_REVISION,
            device=self.device
        )

        for file_path in tqdm(input_files, desc="Stage 1: Diarization"):
            try:
                with suppress_stdout_stderr():
                    result = sd_pipeline(file_path, oracle_num=num_speakers)
                if not result or 'text' not in result or not result['text']:
                    print(f"[WARNING] Diarization failed for {file_path}. Skipping.")
                    continue

                file_name, _ = os.path.splitext(os.path.basename(file_path))
                for i in range(num_speakers):
                    filename = f"{file_name}_speaker{i}.wav"
                    output_audio_path_full = os.path.join(audio_output_path, filename)
                    self._extract_speaker_audio(file_path, result['text'], i, output_audio_path_full)
                    separated_audio_files.append(output_audio_path_full)
            except Exception as e:
                print(f"[ERROR] Error processing {file_path} in Stage 1: {e}")
                continue
        
        if not separated_audio_files: return None
        print(f"[INFO] Stage 1 completed. {len(separated_audio_files)} speaker audios saved.")
        return separated_audio_files

    def _run_vad(self, all_wavs: List[str]) -> bool:
        print("++++++++ Stage 2: Voice Activity Detection (VAD) ++++++++")
        vad_dir = os.path.join(self.workspace, 'vad')
        os.makedirs(vad_dir, exist_ok=True)
        
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=settings.VAD_MODEL_PATH,
            model_revision=settings.VAD_MODEL_REVISION,
            device=self.device,
        )

        for wpath in tqdm(all_wavs, desc="Stage 2: VAD"):
            wid = os.path.basename(wpath).rsplit('.', 1)[0]
            output_file = os.path.join(vad_dir, wid + '_vad.json')
            json_dict = {}
            try:
                with suppress_stdout_stderr():
                    vad_result = vad_pipeline(wpath)
                # ModelScope VAD的返回结果可能是一个包含字典的列表
                if vad_result and isinstance(vad_result, list):
                    result_dict = vad_result[0]
                else:
                    result_dict = vad_result

                segments = result_dict.get('text', result_dict.get('value'))
                if segments:
                    vad_time = [[seg[0]/1000, seg[1]/1000] for seg in segments]
                    for strt, end in vad_time:
                        json_dict[f"{wid}_{strt}_{end}"] = {'file': wpath, 'start': strt, 'stop': end}
            except Exception as e:
                print(f"[WARNING] VAD failed for {wpath}: {e}")
            with open(output_file, 'w') as f: json.dump(json_dict, f, indent=2)
        print("[INFO] Stage 2 completed.")
        return True

    def _run_prepare_subseg(self, all_wavs: List[str]) -> bool:
        print("++++++++ Stage 3.1: Prepare Sub-segments ++++++++")
        for wpath in tqdm(all_wavs, desc="Stage 3.1: Sub-segmentation"):
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
        print("[INFO] Stage 3.1 completed.")
        return True

    def _run_extract_embeddings(self, all_wavs: List[str], conf) -> bool:
        print("++++++++ Stage 3.2: Extract Embeddings ++++++++")
        emb_dir = os.path.join(self.workspace, 'emb')
        os.makedirs(emb_dir, exist_ok=True)
        
        feature_extractor = build('feature_extractor', conf)
        embedding_model = build('embedding_model', conf)
        model_path = os.path.join(settings.SPEAKER_EMBEDDING_MODEL_PATH, settings.SPEAKER_EMBEDDING_MODEL_FILE)
        embedding_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        embedding_model.eval().to(self.device)

        for wpath in tqdm(all_wavs, desc="Stage 3.2: Embedding Extraction"):
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
            with open(os.path.join(emb_dir, wid + ".pkl"), 'wb') as f: pickle.dump(stat_obj, f)
        print("[INFO] Stage 3.2 completed.")
        return True

    def _run_clustering_and_postprocess(self, all_wavs: List[str], conf) -> Optional[Dict]:
        print("++++++++ Stage 4: Cluster Embeddings ++++++++")
        result_dir = os.path.join(self.workspace, 'result')
        os.makedirs(result_dir, exist_ok=True)
        emb_dir = os.path.join(self.workspace, 'emb')
        
        all_embeddings, all_paths = [], []
        for wpath in all_wavs:
            rec_id = os.path.basename(wpath).rsplit('.', 1)[0]
            embs_file = os.path.join(emb_dir, rec_id + '.pkl')
            if os.path.exists(embs_file):
                with open(embs_file, 'rb') as pf:
                    stat_obj = pickle.load(pf)
                    all_embeddings.append(stat_obj['avg_embedding'])
                    all_paths.append(wpath)

        if not all_embeddings: return None

        cluster_model = build('cluster', conf)
        labels = cluster_model(np.array(all_embeddings))
        
        clusters = {}
        for i, label in enumerate(labels):
            speaker_id = f"speaker_{label}"
            if speaker_id not in clusters: clusters[speaker_id] = []
            # Store relative path
            relative_path = os.path.relpath(all_paths[i], self.workspace)
            clusters[speaker_id].append(relative_path)
        
        print(f"[INFO] Clustering completed. Found {len(clusters)} speakers.")
        # Create voiceprint library
        self._create_voiceprint_library(clusters, emb_dir)
        return clusters

    def _create_voiceprint_library(self, clusters: Dict, emb_dir: str):
        voiceprintlib_dir = os.path.join(self.workspace, 'result', 'voiceprintlib')
        if os.path.exists(voiceprintlib_dir): shutil.rmtree(voiceprintlib_dir)
        os.makedirs(voiceprintlib_dir)

        for speaker_id, files in clusters.items():
            speaker_dir = os.path.join(voiceprintlib_dir, speaker_id)
            audio_dir = os.path.join(speaker_dir, 'audio')
            os.makedirs(audio_dir)
            
            speaker_embeddings = []
            for relative_path in files:
                shutil.copy2(os.path.join(self.workspace, relative_path), audio_dir)
                rec_id = os.path.basename(relative_path).rsplit('.', 1)[0]
                embs_file = os.path.join(emb_dir, rec_id + '.pkl')
                with open(embs_file, 'rb') as f:
                    stat_obj = pickle.load(f)
                    speaker_embeddings.append(stat_obj['avg_embedding'])
            
            if speaker_embeddings:
                voiceprint = np.mean(speaker_embeddings, axis=0)
                with open(os.path.join(speaker_dir, f"{speaker_id}_voiceprint.pkl"), 'wb') as f:
                    pickle.dump({'avg_voiceprint': voiceprint}, f)
        print(f"[INFO] Voiceprint library created at {voiceprintlib_dir}")

    async def run_pipeline(self, audio_files: List[str], num_speakers: int) -> Optional[Dict]:
        for f in audio_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Input audio file not found: {f}")

        conf_path = os.path.join(self.workspace, 'diar.yaml')
        with open(conf_path, 'w', encoding='utf-8') as f:
            f.write(settings.DIAR_CLUSTER_CONFIG_CONTENT)
        conf = build_config(conf_path)

        separated_audios = self._run_diarization(audio_files, num_speakers)
        if not separated_audios: raise RuntimeError("Stage 1 (Diarization) failed.")

        if not self._run_vad(separated_audios): raise RuntimeError("Stage 2 (VAD) failed.")
        
        if not self._run_prepare_subseg(separated_audios): raise RuntimeError("Stage 3.1 (Sub-segmentation) failed.")
        
        if not self._run_extract_embeddings(separated_audios, conf): raise RuntimeError("Stage 3.2 (Embedding Extraction) failed.")
        
        final_clusters = self._run_clustering_and_postprocess(separated_audios, conf)
        if not final_clusters: raise RuntimeError("Stage 4 (Clustering) failed.")

        return {
            "total_clusters": len(final_clusters),
            "clusters": [{"speaker_id": k, "audio_files": v} for k, v in sorted(final_clusters.items())],
            "workspace": self.workspace
        }