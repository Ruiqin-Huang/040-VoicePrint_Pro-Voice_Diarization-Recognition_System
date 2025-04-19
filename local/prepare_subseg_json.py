import os
import json
import csv
import argparse
from copy import deepcopy
import sys

import torch
import torchaudio
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Cut out the sub-segments.')
parser.add_argument('--workspace', default='.', type=str, help='workspace path')
parser.add_argument('--dur', default=1.5, type=float, help='Duration of sub-segments') 
parser.add_argument('--shift', default=0.75, type=float, help='Shift of sub-segments') 
parser.add_argument('--min_seg_len', default=0.75, type=float, help='Min Length of sub-segments') 
parser.add_argument('--max_seg_num', default=100, type=int, help='Maximum number of sub-segments per audio file')

def main():
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK']) # 获取当前进程在本节点（服务器）的排名
    threads_num = int(os.environ['WORLD_SIZE']) # 获取当前节点的进程总数
    dist.init_process_group(backend='gloo') # 初始化分布式进程组，使用gloo作为后端
    # TODO: 虽然初始化了分布式环境，但仅作为基础设施，没有实际使用同步功能，此行可被注释
    
    wavs = []
    try:
        # 从 metadata.csv 读取 wav_name 列
        metadata_path = os.path.join(args.workspace, 'dataset', 'metadata.csv')
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_path = row['wav_name']
                full_wav_path = os.path.join(args.workspace, 'dataset', 'audio', wav_path)
                wavs.append(full_wav_path)
        
        if not wavs:
            raise Exception('[ERROR]: No wav files found in metadata.csv')
    except Exception as e:
        raise Exception(f'[ERROR]: Error reading metadata.csv: {str(e)}')

    if len(wavs) <= rank:
        print("[WARNING]: The number of threads exceeds the number of wavs.")
        sys.exit()
    
    # print(f'[INFO]: Start segmentation...')
    # 子进程工作分配
    local_wavs = wavs[rank::threads_num]
    
    for wpath in local_wavs:
        # 获取音频文件的名称
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        vad_json = os.path.join(args.workspace, 'vad', wid + '_vad.json')
        seg_json = os.path.join(args.workspace, 'vad', wid + '_subseg.json')
        
        # 如果${filename}_vad.json文件不存在，程序会发出警告并跳过该音频文件
        if not os.path.exists(vad_json):
            print(f"[WARNING]: VAD json file {vad_json} not found, skipping.")
            continue
        
        with open(vad_json, mode='r') as f:
            # 读取${filename}_vad.json文件
            vad_json = json.load(f)
        subseg_json = {}
        # 设置最小可接受的音频段长度 - 用于判断音频是否过短
        min_segment_length = max(args.min_seg_len, 0.5) # 最短音频段长度不得小于0.5秒，过短seg影响模型性能说话人声纹嵌入提取效果
        # print(f'[INFO]: Generate sub-segments...')
        # 统计跳过的音频段（vad.json中）数量和处理的音频段（vad.json中）数量
        skipped_segments = 0
        processed_segments = 0
        for segid in vad_json:
            wavid = segid.rsplit('_', 2)[0] # 分离出wav文件名
            # .rsplit从字符串右侧开始，按照下划线_分割，最多分割2次，[0]：获取分割后的第一个部分
            # "2speakers_example_0.08_23.84" → ["2speakers_example", "0.08", "23.84"] → "2speakers_example"
            st = vad_json[segid]['start']
            ed = vad_json[segid]['stop']
            # 检查音频段长度是否足够
            segment_length = ed - st
            if segment_length < min_segment_length:
                print(f"[WARNING]: Segment {segid} is too short ({segment_length:.2f}s < {min_segment_length}s), skipping.")
                skipped_segments += 1
                continue
            processed_segments += 1
            # 生成子段
            subseg_st = st
            segments_created = 0
            
            # 为什么不是while subseg_st + subseg_dur < ed:？
            # 是为了解决边界处理问题，确保音频尾部不被遗漏。使用 < ed + args.shift 而不是 < ed 是为了确保能处理接近音频末尾的片段，即使这些片段可能部分超出原始语音段
            # 即使subseg_st + subseg_dur > ed（由于上一轮满足(subseg_st - args.shift) + subseg_dur < ed + args.shift），即subseg_st < ed(当subseg_dur=1.5，args.shift=0.75)
            # subseg_ed = min(subseg_st + subseg_dur, ed) = ed，生成的子段区间为[subseg_st, ed](subseg_st < ed),也符合要求
            # TODO: 1. 但是若--dur和--shift不设置为1.5和0.75，或不满足2:1（相邻区间50％覆盖率），还是建议修改为while subseg_st + subseg_dur < ed:
            # TODO: 2. 若输入音频过短，初始条件都不满足，while循环不会执行，subseg_json为空，这种情况又怎么处理？
            # while subseg_st + subseg_dur < ed + args.shift:
            #     subseg_ed = min(subseg_st + subseg_dur, ed)
            #     item = deepcopy(vad_json[segid])
            #     # vad.json[segid]中的id中的时间/start/stop/中精确到三位小数（毫秒），但是subseg.json中精确到两位小数（秒）
            #     item.update({
            #         'start': round(subseg_st, 2),
            #         'stop': round(subseg_ed, 2)
            #     })
            #     subsegid = wavid + '_' + str(round(subseg_st, 2)) + '_' + str(round(subseg_ed, 2))
            #     subseg_json[subsegid] = item
            #     subseg_st += args.shift
        
            # 使用更清晰的边界条件，并在while循环中添加早停检查
            while subseg_st < ed:
                # 检查是否达到最大分段数
                if segments_created >= args.max_seg_num:
                    print(f"[WARNING]: Reached maximum segment limit ({args.max_seg_num}) for segment {segid}.")
                    break
                    
                # 确保子段不会过短
                remaining_length = ed - subseg_st
                if remaining_length < min_segment_length and segments_created > 0:
                    # 如果剩余长度太短且已经生成了其他子段，则跳过
                    break
                subseg_ed = min(subseg_st + args.dur, ed)
                item = deepcopy(vad_json[segid])
                item.update({
                    'start': round(subseg_st, 2),
                    'stop': round(subseg_ed, 2)
                })
                subsegid = wavid + '_' + str(round(subseg_st, 2)) + '_' + str(round(subseg_ed, 2))
                subseg_json[subsegid] = item
                segments_created += 1
                # 更新起始位置
                subseg_st += args.shift
                # 防止生成无意义的极短尾段
                if subseg_st + min_segment_length > ed:
                    break
        # 统计处理的音频段数量
        # print(f"[INFO]: Processed {processed_segments} segments, skipped {skipped_segments} short segments.")
        # print(f"[INFO]: Generated {len(subseg_json)} sub-segments for file {wid}.")
        # 如果没有生成任何子段，发出警告
        if len(subseg_json) == 0:
            print(f"[WARNING]: No valid sub-segments were generated. All input segments may be too short for json: {args.vad}.")

        with open(seg_json, 'w') as f:
            # 将该音频文件的子段结果写入到对应的json文件中
            json.dump(subseg_json, f, indent=2)

    # print(f'[INFO]: Segmentation finished for {len(local_wavs)} wavs.')
    

if __name__ == '__main__':
    main()