import os
import sys
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import traceback

# 第三方库导入
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, validator
import torch
import whisper
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# 项目内导入
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from local.speech_recognition import transcribe_audio_file
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

# 创建路由器
router = APIRouter(tags=["Speech Processing"])

# ==========================================================================
# 共同的工具函数
# ==========================================================================
def format_datetime():
    """返回符合yyyy-MM-dd HH:mm:ss格式的当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==========================================================================
# 语音分割（说话人分离）API 相关模型和函数
# ==========================================================================
# 语音分割请求模型
class SpeechSegmentationRequest(BaseModel):
    files: List[str]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 语音分割响应模型
class FileInfo(BaseModel):
    file_id: str
    source_url: str
    file_url: str

class ResponseData(BaseModel):
    file_type: List[str] = []  # 按输入文件顺序存储每个文件的语音类别
    files: List[FileInfo] = []

# 提取目标说话人音频
def extract_speaker_audio(wav_path, results, target_speaker, save_path):
    """从原始音频中提取目标说话人的语音，其他人语音置为静音"""
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
                file_types.append("未知")
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
            file_types.append(file_type)
            
            # 处理每个说话人
            for i in range(actual_speakers):
                file_id = str(uuid.uuid4())
                output_filename = f"{base_name}_speaker{i}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 提取说话人音频
                extract_speaker_audio(file_path, result['text'], i, output_path)
                
                # 添加到结果列表
                results.append({
                    "file_id": file_id,
                    "source_url": file_path,
                    "file_url": output_path
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            invalid_files.append(f"{file_path}: {str(e)}")
            file_types.append("错误")
    
    if invalid_files and not results:
        raise Exception(f"所有文件处理失败: {'; '.join(invalid_files)}")
    
    return results, invalid_files, file_types

# ==========================================================================
# 语音识别API 相关模型和函数
# ==========================================================================
# 语音识别请求模型
class SpeechRecognitionRequest(BaseModel):
    files: List[str]
    language: str = "zh"  # 默认中文
    use_gpu: bool = True
    gpu: int = 0
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 语音识别响应模型
class RecognizedFile(BaseModel):
    file_id: str
    phone_number: str
    identity: str
    call_record: str
    create_time: str
    file_url: str

class SpeechRecognitionResponseData(BaseModel):
    calling_party_number: str
    called_party_number: str
    keywords: List[str]
    labels: List[str]
    call_original: str
    call_translation: str
    files: List[RecognizedFile]

# 统一响应模型（适用于两个API）
class ResponseResult(BaseModel):
    retcode: int
    msg: str
    data: Optional[Any] = None  # 使用Any类型以兼容两种不同的响应数据类型

# 加载Whisper模型（全局加载，避免重复加载）
os.environ["WHISPER_CACHE_DIR"] = "pretrained_models/whisper"
whisper_model = whisper.load_model("medium")

# 语音识别相关工具函数
def generate_phone_number():
    """生成电话号码"""
    return f"138{datetime.now().strftime('%H%M%S')}"

def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词"""
    return list(set(text.split()[:3])) if text else []

def translate_text(text: str, src_lang: str, tgt_lang: str = "zh") -> str:
    """文本翻译"""
    return f"Translated({src_lang}->{tgt_lang}): {text}"

def save_segments_to_file(segments: List[Dict], source_file_path: str, file_id: str) -> str:
    """将转录segments保存为JSON文件到源文件所在目录"""
    output_dir = os.path.dirname(source_file_path)
    json_filename = f"{file_id}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return str(json_path)

# ==========================================================================
# API端点定义
# ==========================================================================
@app.post("/api/speech_segmentation", response_model=ResponseResult)
async def speech_segmentation(request: SpeechSegmentationRequest):
    """语音分割API - 将多人语音分离为单个说话人"""
    try:
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表为空",
                data=ResponseData(file_type=[], files=[])
            )
            
        file_results, invalid_files, file_types = await process_audio_files(request.files)
        
        if invalid_files and file_results:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=ResponseData(
                    file_type=file_types,
                    files=[FileInfo(**item) for item in file_results]
                )
            )
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=ResponseData(
                file_type=file_types,
                files=[FileInfo(**item) for item in file_results]
            )
        )
        
    except ValueError as e:
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except FileNotFoundError as e:
        return ResponseResult(
            retcode=ResponseCode.OPERATION_ERROR,
            msg=f"文件不存在: {str(e)}",
            data=None
        )
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
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
            return ResponseResult(
                retcode=ResponseCode.UNKNOWN_ERROR,
                msg=f"未知错误: {str(e)}",
                data=None
            )

@app.post("/api/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest):
    """语音识别API - 将语音转换为文本"""
    try:
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )
        
        processed_files = []
        invalid_files = []
        
        for file_path in request.files:
            if not os.path.exists(file_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            try:
                result = transcribe_audio_file(whisper_model, file_path, request.language)
                
                file_id = str(uuid.uuid4())
                phone_number = generate_phone_number()
                file_url = save_segments_to_file(result["segments"], file_path, file_id)
                
                processed_files.append({
                    "file_id": file_id,
                    "phone_number": phone_number,
                    "identity": "主叫" if len(processed_files) == 0 else "被叫",
                    "call_record": result["full_text"],
                    "create_time": format_datetime(),
                    "file_url": file_url
                })
            except Exception as e:
                invalid_files.append(f"{file_path}: {str(e)}")
        
        if not processed_files:
            if invalid_files:
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文件处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的音频文件",
                    data=None
                )
        
        # 合并所有文本用于生成摘要和关键词
        all_text = " ".join([f["call_record"] for f in processed_files])
        
        # 构建响应数据
        response_data = SpeechRecognitionResponseData(
            calling_party_number=processed_files[0]["phone_number"] if len(processed_files) > 0 else "",
            called_party_number=processed_files[1]["phone_number"] if len(processed_files) > 1 else "",
            keywords=extract_keywords(all_text),
            labels=["自动识别"],
            call_original=all_text,
            call_translation=translate_text(all_text, request.language, "en"),
            files=[RecognizedFile(**f) for f in processed_files]
        )
        
        if invalid_files:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
            )
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
    
    except ValueError as e:
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        return ResponseResult(
            retcode=ResponseCode.UNKNOWN_ERROR,
            msg=f"未知错误: {str(e)}",
            data=None
        )

# ==========================================================================
# 路由器API端点
# ==========================================================================
@router.post("/speech_segmentation")
async def router_speech_segmentation(request: SpeechSegmentationRequest):
    """语音分割API - 路由器版本"""
    return await speech_segmentation(request)

@router.post("/speech_recognition")
async def router_speech_recognition(request: SpeechRecognitionRequest):
    """语音识别API - 路由器版本"""
    return await speech_recognition(request)

# 注册路由器
app.include_router(router, prefix="/api")

# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)