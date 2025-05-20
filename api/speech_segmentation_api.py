import os
import uuid
from typing import List, Dict, Any, Optional
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, validator
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import traceback

from modelscope.pipelines import pipeline
from utils.io_suppressor import suppress_stdout_stderr

# 响应码常量
class ResponseCode:
    SUCCESS = 200000          # 请求成功
    INVALID_PARAM = 100000    # 请求参数错误
    OPERATION_ERROR = 905000  # 操作错误
    UNKNOWN_ERROR = 999999    # 未知异常错误

# 创建FastAPI应用
app = FastAPI()

# 定义请求模型
class SpeechSegmentationRequest(BaseModel):
    files: List[str]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 定义响应模型 - 修改后不包含file_type字段
class FileInfo(BaseModel):
    file_id: str
    source_url: str
    file_url: str

# 修改ResponseData，增加file_type字段
class ResponseData(BaseModel):
    file_type: List[str] = []  # 新增字段，按输入文件顺序存储每个文件的语音类别
    files: List[FileInfo] = []

class ResponseResult(BaseModel):
    retcode: int
    msg: str
    data: Optional[ResponseData] = None

# 提取目标说话人音频
def extract_speaker_audio(wav_path, results, target_speaker, save_path):
    """
    从原始音频中提取目标说话人的语音，其他人语音置为静音
    """
    # 读取音频
    audio, sr = librosa.load(wav_path, sr=None)
    audio_out = np.zeros_like(audio)  # 初始化输出音频为零（静音）

    # 遍历所有说话人的语音段落
    for seg in results:
        start_time, end_time, speaker_id = seg

        if speaker_id == target_speaker:
            # 获取音频的起始和结束位置
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # 将目标说话人的语音段复制到输出音频中
            if end_sample <= len(audio):  # 确保索引不超出范围
                audio_out[start_sample:end_sample] = audio[start_sample:end_sample]

    # 保存输出音频
    sf.write(save_path, audio_out, sr)
    return save_path

# 根据说话人数量确定语音类别
def get_file_type(speaker_count):
    if speaker_count == 1:
        return "单人"
    elif speaker_count == 2:
        return "双人"
    else:
        return "多人"

# 处理语音分离
async def process_audio_files(file_paths):
    # 创建输出目录
    output_dir = "./data/audio_segmentation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化说话人分离模型
    try:
        with suppress_stdout_stderr():
            sd_pipeline = pipeline(
                task='speaker-diarization',
                model='./pretrained_models/iic/speech_campplus_speaker-diarization_common',
                model_revision='v1.0.0'
            )
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")
    
    results = []
    invalid_files = []
    file_types = []  # 存储每个输入文件的类型
    
    for file_path in tqdm(file_paths, desc="Processing audio files"):
        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                invalid_files.append(f"文件不存在: {file_path}")
                file_types.append("未知")  # 文件不存在，添加未知类型
                continue
                
            # 分割说话人
            with suppress_stdout_stderr():
                result = sd_pipeline(file_path)

            # 获取实际的说话人数量
            speaker_ids = set()
            for segment in result['text']:
                speaker_ids.add(segment[2])
            actual_speakers = len(speaker_ids)
            
            file_name = os.path.basename(file_path)
            base_name, ext = os.path.splitext(file_name)
            
            file_type = get_file_type(actual_speakers)
            file_types.append(file_type)  # 添加到文件类型列表
            
            # 处理每个说话人
            for i in range(actual_speakers):
                file_id = str(uuid.uuid4())
                output_filename = f"{base_name}_speaker{i}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 提取说话人音频
                extract_speaker_audio(file_path, result['text'], i, output_path)
                
                # 添加到结果列表 - 移除file_type字段
                results.append({
                    "file_id": file_id,
                    "source_url": file_path,
                    "file_url": output_path
                })
                
        except Exception as e:
            # 记录错误但继续处理其他文件
            print(f"Error processing {file_path}: {str(e)}")
            invalid_files.append(f"{file_path}: {str(e)}")
            file_types.append("错误")  # 处理出错，添加错误类型
    
    if invalid_files and not results:
        # 如果所有文件都处理失败
        raise Exception(f"所有文件处理失败: {'; '.join(invalid_files)}")
    
    return results, invalid_files, file_types

@app.post("/api/speech_segmentation", response_model=ResponseResult)
async def speech_segmentation(request: SpeechSegmentationRequest):
    try:
        # 参数验证
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表为空",
                data=ResponseData(file_type=[], files=[])
            )
            
        file_results, invalid_files, file_types = await process_audio_files(request.files)
        
        # 如果有部分文件处理失败，但有成功的结果，返回警告信息和成功结果
        if invalid_files and file_results:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=ResponseData(
                    file_type=file_types,
                    files=[FileInfo(**item) for item in file_results]
                )
            )
        
        # 全部成功
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=ResponseData(
                file_type=file_types,
                files=[FileInfo(**item) for item in file_results]
            )
        )
        
    except ValueError as e:
        # 参数验证错误
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except FileNotFoundError as e:
        # 文件不存在错误
        return ResponseResult(
            retcode=ResponseCode.OPERATION_ERROR,
            msg=f"文件不存在: {str(e)}",
            data=None
        )
    except Exception as e:
        # 记录详细错误信息，便于调试
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        # 根据错误类型返回不同响应码
        if "模型加载失败" in str(e):
            return ResponseResult(
                retcode=ResponseCode.OPERATION_ERROR,
                msg=f"模型加载失败: {str(e)}",
                data=None
            )
        elif "处理失败" in str(e):
            return ResponseResult(
                retcode=ResponseCode.OPERATION_ERROR,
                msg=str(e),
                data=None
            )
        else:
            # 未知异常
            return ResponseResult(
                retcode=ResponseCode.UNKNOWN_ERROR,
                msg=f"未知错误: {str(e)}",
                data=None
            )

# 如果需要直接运行文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)