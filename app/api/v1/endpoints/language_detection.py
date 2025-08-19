from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

from app.models.common import ResponseResult
from app.models.language_detection import LanguageDetectionRequest, LanguageDetectionResponseData
from app.services.language_detection import process_text_files
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/language_detection", 
             response_model=ResponseResult)
async def language_detection(requests: LanguageDetectionRequest):
    """语种检测API - 检测文本文件的语种"""
    try:
        if not requests.text:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文本列表不能为空",
                data=None
            )
        
        processed_files, invalid_files = await process_text_files(requests.text)
        
        if not processed_files:
            if invalid_files:
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文本处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的文本文件",
                    data=None
                )
        
        # 将服务层返回的结果转换为响应格式
        response_data = []
        for result in processed_files:
            response_data.append(
                LanguageDetectionResponseData(
                    language=result["language"],
                    language_name=result["language_name"],
                    confidence=result["confidence"]
                )
            )
        
        if invalid_files:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文本处理成功，{len(invalid_files)}个文本失败: {'; '.join(invalid_files)}",
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