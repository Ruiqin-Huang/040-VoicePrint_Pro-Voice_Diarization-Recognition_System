"""
使用方法:
# 使用 CPU
python local/audio_diarization_cluster.py --audio_files ./example/chinese/zh_1.wav ./example/chinese/zh_2.wav ./example/chinese/zh_3.wav

# 使用指定的 GPU (例如 GPU 0)
 python local/audio_diarization_cluster.py --audio_files ./example/chinese/zh_1.wav ./example/chinese/zh_2.wav ./example/chinese/zh_3.wav --gpu 0
"""

import os
import sys
import argparse
import json
import pickle
import shutil
import csv
from copy import deepcopy
from tqdm import tqdm
import warnings

import numpy as np
import librosa
import soundfile as sf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入必要的库，如果失败则提供安装提示
try:
    import torch
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from speakerlab.utils.config import build_config
    from speakerlab.utils.builder import build
    from speakerlab.utils.fileio import load_audio
    from speakerlab.utils.utils import circle_pad
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必需的依赖项。")
    print("您可以尝试运行: pip install torch torchaudio modelscope speakerlab librosa soundfile tqdm")
    sys.exit(1)

# 屏蔽不必要的警告和输出
warnings.filterwarnings("ignore", category=FutureWarning)
class SuppressStdoutStderr:
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# --- 从 audio_diarization.py 提取的函数 ---
def extract_speaker_audio(wav_path, results, target_speaker, save_path):
    audio, sr = librosa.load(wav_path, sr=None)
    audio_out = np.zeros_like(audio)
    for seg in results:
        start_time, end_time, speaker_id = seg
        if speaker_id == target_speaker:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_out[start_sample:end_sample] = audio[start_sample:end_sample]
    sf.write(save_path, audio_out, sr)

def run_diarization(input_files, workspace, num_speakers, gpu):
    print("++++++++ Stage 1: Speaker Diarization ++++++++")
    audio_output_path = os.path.join(workspace, "dataset", "audio")
    audio_source_path = os.path.join(workspace, "dataset", "audio_source")
    os.makedirs(audio_output_path, exist_ok=True)
    os.makedirs(audio_source_path, exist_ok=True)
    
    output_csv_path = os.path.join(workspace, "dataset", "metadata.csv")
    csv_data = []
    separated_audio_files = []

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print("[INFO] Loading speaker diarization model...")
    with SuppressStdoutStderr():
        sd_pipeline = pipeline(
            task='speaker-diarization',
            model='./pretrained_models/iic/speech_campplus_speaker-diarization_common',
            model_revision='v1.0.0'
        )
    print("[INFO] Speaker diarization model loaded.")

    for file_path in tqdm(input_files, desc="Stage 1: Speaker Diarization"):
        try:
            file_name_ext = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(audio_source_path, file_name_ext))

            with SuppressStdoutStderr():
                result = sd_pipeline(file_path, oracle_num=num_speakers)

            if not result or 'text' not in result or not result['text']:
                print(f"[WARNING] Diarization failed for {file_path}, no speaker segments found. Skipping.")
                continue

            file_name, _ = os.path.splitext(file_name_ext)
            for i in range(num_speakers):
                filename = f"{file_name}_speaker{i}.wav"
                output_audio_path_full = os.path.join(audio_output_path, filename)
                extract_speaker_audio(file_path, result['text'], i, output_audio_path_full)
                csv_data.append([filename, file_name_ext, i, 'unknown'])
                separated_audio_files.append(output_audio_path_full)
        except Exception as e:
            print(f"[ERROR] Error processing {file_path} in Stage 1: {str(e)}")
            continue

    if not separated_audio_files:
        print("[ERROR] Stage 1 failed. No audio files were separated.")
        return None

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['wav_name', 'source_name', 'speaker_id', 'language'])
        writer.writerows(csv_data)
    
    print(f"[INFO] Stage 1 completed. {len(separated_audio_files)} speaker audio(s) saved in {audio_output_path}")
    return separated_audio_files

# --- 从 voice_activity_detection.py 提取的函数 ---
def run_vad(workspace, all_wavs, device):
    print("++++++++ Stage 2: Voice Activity Detection (VAD) ++++++++")
    vad_dir = os.path.join(workspace, 'vad')
    os.makedirs(vad_dir, exist_ok=True)
    
    try:
        with SuppressStdoutStderr():
            vad_pipeline = pipeline(
                task=Tasks.voice_activity_detection,
                model='./pretrained_models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                model_revision='v2.0.4',
                device=device,
            )
    except Exception as e:
        print(f"[ERROR] Failed to load VAD model: {e}")
        return False

    for wpath in tqdm(all_wavs, desc="Stage 2: VAD"):
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        output_file = os.path.join(vad_dir, wid + '_vad.json')
        
        json_dict = {}
        try:
            with SuppressStdoutStderr():
                vad_result = vad_pipeline(wpath)
            
            # ModelScope VAD result key can be 'text' or 'value'
            # The pipeline returns a list containing the result dict
            if vad_result and isinstance(vad_result, list):
                result_dict = vad_result[0]
            else:
                result_dict = vad_result

            segments = result_dict.get('text', result_dict.get('value'))

            if segments:
                # The segments are inside another list
                vad_time = [[seg[0]/1000, seg[1]/1000] for seg in segments]
                vad_time = [[round(s, 3), round(e, 3)] for s, e in vad_time]
                for strt, end in vad_time:
                    subsegmentid = f"{wid}_{strt}_{end}"
                    json_dict[subsegmentid] = {'file': wpath, 'start': strt, 'stop': end}
        except Exception as e:
            print(f"[WARNING] VAD processing failed for {wpath}: {e}")

        with open(output_file, 'w') as f:
            json.dump(json_dict, f, indent=2)

    print("[INFO] Stage 2 completed.")
    return True

# --- 从 prepare_subseg_json.py 提取的函数 ---
def run_prepare_subseg(workspace, all_wavs, dur, shift, min_seg_len, max_seg_num):
    print("++++++++ Stage 3.1: Prepare Sub-segments ++++++++")
    total_subsegments = 0
    for wpath in tqdm(all_wavs, desc="Stage 3.1: Sub-segmentation"):
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        vad_json_path = os.path.join(workspace, 'vad', wid + '_vad.json')
        seg_json_path = os.path.join(workspace, 'vad', wid + '_subseg.json')

        if not os.path.exists(vad_json_path):
            continue

        with open(vad_json_path, 'r') as f:
            try:
                vad_json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not decode VAD JSON for {wpath}, skipping.")
                vad_json_data = {}
        
        subseg_json = {}
        for segid in vad_json_data:
            wavid = segid.rsplit('_', 2)[0]
            st, ed = vad_json_data[segid]['start'], vad_json_data[segid]['stop']
            
            if ed - st < min_seg_len:
                continue

            subseg_st = st
            segments_created = 0
            while subseg_st < ed and segments_created < max_seg_num:
                subseg_ed = min(subseg_st + dur, ed)
                if subseg_ed - subseg_st < min_seg_len:
                    if segments_created > 0: break
                
                item = deepcopy(vad_json_data[segid])
                item.update({'start': round(subseg_st, 2), 'stop': round(subseg_ed, 2)})
                subsegid = f"{wavid}_{round(subseg_st, 2)}_{round(subseg_ed, 2)}"
                subseg_json[subsegid] = item
                segments_created += 1
                subseg_st += shift
        
        total_subsegments += len(subseg_json)
        with open(seg_json_path, 'w') as f:
            json.dump(subseg_json, f, indent=2)

    if total_subsegments == 0:
        print("[ERROR] Stage 3.1 failed. No sub-segments were created from VAD results.")
        return False

    print(f"[INFO] Stage 3.1 completed. Created a total of {total_subsegments} sub-segments.")
    return True

# --- 从 extract_diar_embeddings.py 提取的函数 ---
def run_extract_embeddings(workspace, all_wavs, device, conf, batchsize):
    print("++++++++ Stage 3.2: Extract Embeddings ++++++++")
    emb_dir = os.path.join(workspace, 'emb')
    os.makedirs(emb_dir, exist_ok=True)

    try:
        feature_extractor = build('feature_extractor', conf)
        embedding_model = build('embedding_model', conf)
        pretrained_state = torch.load(conf.pretrained_model, map_location='cpu')
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval().to(device)
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return False

    embeddings_created = 0
    for wpath in tqdm(all_wavs, desc="Stage 3.2: Embedding Extraction"):
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        subseg_json_path = os.path.join(workspace, 'vad', wid + '_subseg.json')
        stat_emb_file = os.path.join(emb_dir, wid + ".pkl")

        if not os.path.exists(subseg_json_path):
            continue

        with open(subseg_json_path, "r") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                meta = {}
        
        if not meta: continue

        try:
            wav_path = list(meta.values())[0]['file']
            obj_fs = feature_extractor.sample_rate
            wav = load_audio(wav_path, obj_fs=obj_fs)
            
            wavs_segments = [wav[0, int(meta[i]['start'] * obj_fs):int(meta[i]['stop'] * obj_fs)] for i in meta]
            
            valid_indices = [idx for idx, x in enumerate(wavs_segments) if x.shape[0] > 0]
            if not valid_indices: continue
            
            valid_wavs = [wavs_segments[i] for i in valid_indices]
            max_len = max(x.shape[0] for x in valid_wavs)
            wavs_padded = [circle_pad(x, max_len) for x in valid_wavs]
            wavs_tensor = torch.stack(wavs_padded).unsqueeze(1)

            embeddings = []
            with torch.no_grad():
                for i in range(0, wavs_tensor.shape[0], batchsize):
                    wavs_batch = wavs_tensor[i:i+batchsize].to(device)
                    feats_batch = torch.vmap(feature_extractor)(wavs_batch)
                    embeddings_batch = embedding_model(feats_batch).cpu()
                    embeddings.append(embeddings_batch)
            
            if not embeddings: continue
            
            embeddings = torch.cat(embeddings, dim=0).numpy()
            avg_embedding = np.mean(embeddings, axis=0)
            
            valid_meta_keys = [list(meta.keys())[i] for i in valid_indices]
            stat_obj = {
                'embeddings': embeddings,
                'times': [[meta[key]['start'], meta[key]['stop']] for key in valid_meta_keys],
                'avg_embedding': avg_embedding
            }
            with open(stat_emb_file, 'wb') as f:
                pickle.dump(stat_obj, f)
            embeddings_created += 1
        except Exception as e:
            print(f"[WARNING] Embedding extraction failed for {wpath}: {e}")

    if embeddings_created == 0:
        print("[ERROR] Stage 3.2 failed. No embedding files were created.")
        return False

    print(f"[INFO] Stage 3.2 completed. Created {embeddings_created} embedding file(s).")
    return True

# --- 从 cluster_and_postprocess.py 提取的函数 ---
def run_clustering_and_postprocess(workspace, conf_path):
    print("++++++++ Stage 4: Cluster Embeddings and Create Voiceprint Library ++++++++")
    result_dir = os.path.join(workspace, 'result')
    os.makedirs(result_dir, exist_ok=True)
    audio_embs_dir = os.path.join(workspace, 'emb')
    
    all_embeddings, all_paths = [], []
    metadata_path = os.path.join(workspace, 'dataset', 'metadata.csv')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_name = row['wav_name']
            rec_id = wav_name.rsplit('.', 1)[0]
            embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
            if os.path.exists(embs_file):
                with open(embs_file, 'rb') as pf:
                    stat_obj = pickle.load(pf)
                    if 'avg_embedding' in stat_obj:
                        all_embeddings.append(stat_obj['avg_embedding'])
                        full_wav_path = os.path.join(workspace, 'dataset', 'audio', wav_name)
                        all_paths.append(full_wav_path)

    if not all_embeddings:
        print("[ERROR] No embeddings found to cluster. Please check previous stages for errors.")
        return None

    config = build_config(conf_path)
    cluster_model = build('cluster', config)
    labels = cluster_model(np.array(all_embeddings))
    
    unique_labels = np.unique(labels)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    new_labels = [label_map[l] for l in labels]

    clusters = {}
    for i, label in enumerate(new_labels):
        speaker_id = f"speaker_{label}"
        if speaker_id not in clusters:
            clusters[speaker_id] = []
        clusters[speaker_id].append(all_paths[i])
    
    print(f"[INFO] Clustering completed. Found {len(clusters)} speakers.")

    with open(os.path.join(result_dir, 'cluster_result.txt'), 'w') as f:
        for speaker_id, audio_files in sorted(clusters.items()):
            f.write(f"{speaker_id}:\n")
            for audio_file in audio_files:
                f.write(f"    {os.path.basename(audio_file)}\n")
            f.write("\n")
    
    with open(os.path.join(result_dir, 'cluster_result.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["wav_name", "speaker_id_pred", "language"])
        for speaker_id, audio_files in sorted(clusters.items()):
            speaker_num = speaker_id.split('_')[1]
            for audio_file in audio_files:
                writer.writerow([os.path.basename(audio_file), speaker_num, 'zh-cn'])

    voiceprintlib_dir = os.path.join(result_dir, 'voiceprintlib')
    if os.path.exists(voiceprintlib_dir): shutil.rmtree(voiceprintlib_dir)
    os.makedirs(voiceprintlib_dir, exist_ok=True)

    for speaker_id, audio_files in sorted(clusters.items()):
        speaker_dir = os.path.join(voiceprintlib_dir, speaker_id)
        audio_dir = os.path.join(speaker_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_filenames, speaker_avg_embeddings = [], []
        for audio_file in audio_files:
            audio_filename = os.path.basename(audio_file)
            shutil.copy2(audio_file, os.path.join(audio_dir, audio_filename))
            audio_filenames.append(audio_filename)
            
            rec_id = audio_filename.rsplit('.', 1)[0]
            embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
            with open(embs_file, 'rb') as f:
                stat_obj = pickle.load(f)
                speaker_avg_embeddings.append(stat_obj['avg_embedding'])
        
        speaker_voiceprint = np.mean(speaker_avg_embeddings, axis=0)
        voiceprint_data = {
            'audio': audio_filenames,
            'avg_embeddings': speaker_avg_embeddings,
            'avg_voiceprint': speaker_voiceprint
        }
        with open(os.path.join(speaker_dir, f"{speaker_id}_voiceprint.pkl"), 'wb') as f:
            pickle.dump(voiceprint_data, f)
    
    print(f"[INFO] Voiceprint library created at {voiceprintlib_dir}")
    return clusters

# --- Main Function ---
def main(args):
    for f in args.audio_files:
        if not os.path.exists(f):
            print(f"[ERROR] Input audio file not found: {f}")
            sys.exit(1)

    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print(f"[INFO] Using GPU: {device}")
    else:
        device = 'cpu'
        print("[INFO] Using CPU.")

    os.makedirs(args.workspace, exist_ok=True)
    print(f"[INFO] Workspace is set to: {args.workspace}")

    # --- Stage 1: 说话人分割 ---
    separated_audios = run_diarization(
        args.audio_files, args.workspace, args.num_speakers_per_audio, args.gpu
    )
    if not separated_audios:
        print("[FAILURE] Pipeline stopped at Stage 1.")
        return

    # --- Stage 2: VAD ---
    if not run_vad(args.workspace, separated_audios, device):
        print("[FAILURE] Pipeline stopped at Stage 2.")
        return

    # --- Stage 3.1: 准备子片段 ---
    if not run_prepare_subseg(args.workspace, separated_audios, 1.0, 0.5, 0.5, 150):
        print("[FAILURE] Pipeline stopped at Stage 3.1.")
        return

    # --- Stage 3.2: 提取声纹嵌入 ---
    conf_path = os.path.join(args.workspace, 'diar.yaml')
    conf_content = """
fbank_dim: 80
embedding_size: 192

feature_extractor:
  obj: speakerlab.process.processor.FBank
  args:
    n_mels: <fbank_dim>
    sample_rate: 16000
    mean_nor: True

embedding_model:
  obj: speakerlab.models.campplus.DTDNN.CAMPPlus
  args:
    feat_dim: <fbank_dim>
    embedding_size: <embedding_size>

cluster:
  obj: speakerlab.process.cluster.CommonClustering 
  args:
    cluster_type: spectral # 指定使用谱聚类算法
    mer_cos: 0.85 # 余弦相似度阈值，合并余弦相似度阈值。当两个聚类中心的余弦相似度超过0.9时，会被考虑合并为同一个聚类。值越高，合并条件越严格；值越低，更容易合并不同聚类
    min_num_spks: 1 # 最小说话人数量，聚类结果至少会有1个类别
    max_num_spks: 200 # 最大说话人数量限制，防止过度细分
    min_cluster_size: 1 # 最小聚类大小，每个聚类至少需要包含1个样本
    oracle_num: null # 预设的聚类数量，设为null表示系统将自动确定最佳聚类数量，如果已知说话人数量，可以在这里指定具体数字
    pval: 0.012 # p值阈值，用于确定聚类边界，控制聚类精细度：值越小，聚类越精细(产生更多聚类)；值越大，聚类越粗略(产生更少聚类)
"""
    with open(conf_path, 'w', encoding='utf-8') as f:
        f.write(conf_content)
    
    conf = build_config(conf_path)
    conf.pretrained_model = './pretrained_models/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt'
    
    if not run_extract_embeddings(args.workspace, separated_audios, device, conf, batchsize=64):
        print("[FAILURE] Pipeline stopped at Stage 3.2.")
        return

    # --- Stage 4: 聚类和后处理 ---
    final_clusters = run_clustering_and_postprocess(args.workspace, conf_path)

    if final_clusters:
        print("\n\n++++++++++++++ FINAL CLUSTERING RESULT ++++++++++++++")
        print(f"Successfully clustered into {len(final_clusters)} speakers.")
        for speaker_id, files in sorted(final_clusters.items()):
            print(f"\n--- {speaker_id} ---")
            for f in files:
                print(f"  - {os.path.basename(f)}")
        print("\nFull results, including the voiceprint library, are saved in:")
        print(f"  - {os.path.join(args.workspace, 'result')}")
        print("\n[SUCCESS] Pipeline completed successfully.")
    else:
        print("\n[FAILURE] Clustering process failed to produce results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-in-one Speaker Diarization and Clustering Pipeline")
    parser.add_argument('--audio_files', required=True, nargs='+', help="A list of input audio file paths.")
    parser.add_argument('--workspace', type=str, default='./workspace', help="Directory to save all intermediate and final results.")
    parser.add_argument('--num_speakers_per_audio', type=int, default=2, help="Number of speakers to separate from each audio file.")
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID to use (e.g., '0'). If not specified or not available, use CPU.")
    
    args = parser.parse_args()

    main(args)