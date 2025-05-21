from fastapi import APIRouter, HTTPException
import traceback
from typing import List

from app.models.common import ResponseResult
from app.models.speech_recognition import (
    SpeechRecognitionRequest, 
    SpeechRecognitionResponseData, 
    RecognizedFile
)
from app.services.speech_recognition import process_speech_files
from app.core.error_codes import ResponseCode
from utils.helpers import extract_keywords, translate_text

router = APIRouter(prefix="/api")

@router.post("/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest):
    """语音识别API - 将语音转换为文本"""
    try:
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )
        
        processed_files, invalid_files = await process_speech_files(
            request.files, 
            language=request.language
        )
        
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