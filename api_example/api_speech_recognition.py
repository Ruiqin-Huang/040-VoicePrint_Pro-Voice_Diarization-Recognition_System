from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any
import os
import sys
import uuid
from datetime import datetime
import json
import whisper
import torch
import traceback

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from local.speech_recognition import transcribe_audio_file

# 响应码常量
class ResponseCode:
    SUCCESS = 200000          # 请求成功
    INVALID_PARAM = 100000    # 请求参数错误
    OPERATION_ERROR = 905000  # 操作错误
    UNKNOWN_ERROR = 999999    # 未知异常错误

app = FastAPI()

router = APIRouter(tags=["Speech Recognition"])

# 加载Whisper模型（全局加载，避免重复加载）
os.environ["WHISPER_CACHE_DIR"] = "./pretrained_models/whisper"
whisper_model = whisper.load_model("medium")

# 请求和响应模型
class SpeechFile(BaseModel):
    file_path: str

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

class ResponseResult(BaseModel):
    retcode: int
    msg: str
    data: Optional[SpeechRecognitionResponseData] = None

# 辅助函数
def generate_phone_number():
    """生成电话号码"""
    return f"138{datetime.now().strftime('%H%M%S')}"

def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词（示例实现）"""
    # 待实现
    return list(set(text.split()[:3])) if text else []

def translate_text(text: str, src_lang: str, tgt_lang: str = "zh") -> str:
    """文本翻译（示例实现）"""
    # 待实现，whisper强行设置语言有自动翻译，但错误率高且容易出现幻觉
    return f"Translated({src_lang}->{tgt_lang}): {text}"

def save_segments_to_file(segments: List[Dict], source_file_path: str, file_id: str) -> str:
    """
    将转录segments保存为JSON文件到源文件所在目录
    :param segments: 转录段落列表
    :param source_file_path: 原始音频文件路径
    :return: 保存的JSON文件路径
    """
    # 获取源文件所在目录和文件名
    output_dir = os.path.dirname(source_file_path)
    json_filename = f"{file_id}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入JSON文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return str(json_path)

def format_datetime():
    """返回符合yyyy-MM-dd HH:mm:ss格式的当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.post("/api/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest):
    """
    语音识别API
    - 接收音频文件路径数组
    - 返回识别结果和元数据
    """
    try:
        # 参数验证
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )
        
        # 处理所有音频文件
        processed_files = []
        invalid_files = []
        
        for file_path in request.files:
            if not os.path.exists(file_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            try:
                result = transcribe_audio_file(whisper_model, file_path, request.language)
                
                # 生成文件元数据
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
            labels=["自动识别"],  # 示例标签
            call_original=all_text,
            call_translation=translate_text(all_text, request.language, "en"),
            files=[RecognizedFile(**f) for f in processed_files]
        )
        
        # 如果有部分文件处理失败，返回警告信息
        if invalid_files:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
            )
        
        # 全部成功
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
    
    except ValueError as e:
        # 参数验证错误
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except Exception as e:
        # 记录详细错误信息，便于调试
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        # 未知异常
        return ResponseResult(
            retcode=ResponseCode.UNKNOWN_ERROR,
            msg=f"未知错误: {str(e)}",
            data=None
        )

@router.post("/speech_recognition")
async def recognize(request: SpeechRecognitionRequest):
    """
    语音识别API - 路由版本
    调用主API函数处理
    """
    return await speech_recognition(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)