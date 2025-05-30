from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

from app.config.path_mapper import PathMapper
from app.dependencies import get_path_mapper
from app.models.common import ResponseResult
from app.models.speech_recognition import (
    RecognizedDetails,
    SpeechRecognitionRequest, 
    SpeechRecognitionResponseData, 
    RecognizedFile
)
from app.services.speech_recognition import process_speech_files
from app.core.error_codes import ResponseCode
from utils.helpers import extract_keywords, translate_text

router = APIRouter(prefix="/api")

@router.post("/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest, path_mapper: PathMapper = Depends(get_path_mapper)):
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
            path_mapper
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
        
        
        
        # 将服务层返回的结果转换为响应格式
        response_data = []
        for result in processed_files:
            print("debug: ", result["recognitions"])
            recognized_files = [
                RecognizedFile(
                    identity=recognition["identity"],
                    call_records=recognition["text"],
                    call_records_details=RecognizedDetails(
                        start=recognition["start_time"],
                        end=recognition["end_time"],
                        text=translate_text(recognition["text"], "zh"),
                        no_speech_prob=recognition["no_speech_prob"]
                    )
                )
                for recognition in result["recognitions"]
            ]
            print("debug: ", recognized_files)

            response_data.append(
                SpeechRecognitionResponseData(
                    file_id=result["file_id"],
                    call_original=result["call_original"],
                    call_translation=translate_text(result["call_original"], "zh"),
                    call_records_collections=recognized_files
                )
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