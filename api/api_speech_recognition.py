from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
import uuid
from datetime import datetime
import json
import whisper
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from local.speech_recognition import transcribe_audio_file

app = FastAPI()

router = APIRouter(tags=["Speech Recognition"])

# 加载Whisper模型（全局加载，避免重复加载）
os.environ["WHISPER_CACHE_DIR"] = "pretrained_models\\whisper"
whisper_model = whisper.load_model("medium")

# 请求和响应模型
class SpeechFile(BaseModel):
    file_path: str

class SpeechRecognitionRequest(BaseModel):
    files: List[str]
    language: str = "zh"  # 默认中文
    use_gpu: bool = True
    gpu: int = 0

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
    retcode: str
    msg: str
    data: SpeechRecognitionResponseData

# 辅助函数
def generate_phone_number():
    """生成电话号码"""
    return f"13815902186"

def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词（示例实现）"""
    # 待实现
    return list(set(text.split()[:3])) if text else []

def translate_text(text: str, src_lang: str, tgt_lang: str = "zh") -> str:
    """文本翻译（示例实现）"""
    # 待实现，whisper强行设置语言有自动翻译，但错误率高且容易出现幻觉
    return f"Translated({src_lang}->{tgt_lang}): {text}"

def generate_phone_number():
    return f"138{datetime.now().strftime('%H%M%S')}"

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

@app.post("/api/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest):
    """
    语音识别API
    - 接收音频文件路径数组
    - 返回识别结果和元数据
    """
    if not request.files:
        raise HTTPException(
            status_code=400, 
            detail="必须提供至少一个音频文件路径"
        )
    
    try:
        # 处理所有音频文件
        processed_files = []
        file_urls = []
        for file_path in request.files:
            if not os.path.exists(file_path):
                continue
                
            try:
                result = transcribe_audio_file(whisper_model, file_path, request.language)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"处理音频文件 {file_path} 时出错: {str(e)}"
                )
            
            # 生成文件元数据
            file_id = str(uuid.uuid4())
            phone_number = generate_phone_number()
            file_url = save_segments_to_file(result["segments"], file_path, file_id)
            
            processed_files.append({
                "file_id": file_id,
                "phone_number": phone_number,
                "identity": "主叫" if len(processed_files) == 0 else "被叫",
                "call_record": result["full_text"],
                "create_time": datetime.now().isoformat(),
                "file_url": file_url
            })
        
        if not processed_files:
            raise HTTPException(
                status_code=400, 
                detail="没有找到可处理的音频文件"
            )
        
        # 合并所有文本用于生成摘要和关键词
        all_text = " ".join([f["call_record"] for f in processed_files])
        
        # 构建响应数据
        response_data = {
            "calling_party_number": processed_files[0]["phone_number"] if len(processed_files) > 0 else "",
            "called_party_number": processed_files[1]["phone_number"] if len(processed_files) > 1 else "",
            "keywords": extract_keywords(all_text),
            "labels": ["自动识别"],  # 示例标签
            "call_original": all_text,
            "call_translation": translate_text(all_text, request.language, "en"),
            "files": processed_files
        }
        
        return {
            "retcode": "200000",
            "msg": "success",
            "data": response_data
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"服务器错误: {str(e)}"
        )

@router.post("/speech_recognition")
async def recognize(request: SpeechRecognitionRequest):
    """
    语音识别API
    - 接收音频文件路径数组
    - 返回识别结果和元数据
    """
    if not request.files:
        raise HTTPException(
            status_code=400, 
            detail="必须提供至少一个音频文件路径"
        )
    
    try:
        # 处理所有音频文件
        processed_files = []
        file_urls = []
        for file_path in request.files:
            if not os.path.exists(file_path):
                continue
                
            try:
                result = transcribe_audio_file(whisper_model, file_path, request.language)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"处理音频文件 {file_path} 时出错: {str(e)}"
                )
            
            # 生成文件元数据
            file_id = str(uuid.uuid4())
            phone_number = generate_phone_number()
            file_url = save_segments_to_file(result["segments"], file_path, file_id)
            
            processed_files.append({
                "file_id": file_id,
                "phone_number": phone_number,
                "identity": "主叫" if len(processed_files) == 0 else "被叫",
                "call_record": result["full_text"],
                "create_time": datetime.now().isoformat(),
                "file_url": file_url
            })
        
        if not processed_files:
            raise HTTPException(
                status_code=400, 
                detail="没有找到可处理的音频文件"
            )
        
        # 合并所有文本用于生成摘要和关键词
        all_text = " ".join([f["call_record"] for f in processed_files])
        
        # 构建响应数据
        response_data = {
            "calling_party_number": processed_files[0]["phone_number"] if len(processed_files) > 0 else "",
            "called_party_number": processed_files[1]["phone_number"] if len(processed_files) > 1 else "",
            "keywords": extract_keywords(all_text),
            "labels": ["自动识别"],  # 示例标签
            "call_original": all_text,
            "call_translation": translate_text(all_text, request.language, "en"),
            "files": processed_files
        }
        
        return {
            "retcode": "200000",
            "msg": "success",
            "data": response_data
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"服务器错误: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)