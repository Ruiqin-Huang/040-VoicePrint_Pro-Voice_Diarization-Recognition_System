"""
语音识别API端点模块

提供RESTful API接口用于语音转文本服务。
通过FastAPI框架实现，支持批量音频文件处理和错误处理。

主要功能：
- 接收语音识别请求
- 调用语音识别服务进行处理
- 返回结构化的识别结果
- 支持说话人分离结果的整合

依赖：
- FastAPI: Web框架
- app.services.speech_recognition: 语音识别服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

# 导入路径映射和依赖（已注释）
# from app.config.path_mapper import PathMapper
# from app.dependencies import get_path_mapper
from app.models.common import ResponseResult
from app.models.speech_recognition import (
    RecognizedDetails,
    SpeechRecognitionRequest,
    SpeechRecognitionResponseData,
    RecognizedFile
)
from app.services.speech_recognition import process_speech_files
from app.core.error_codes import ResponseCode
# 导入工具函数（已注释）
# from utils.helpers import extract_keywords, translate_text

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/speech_recognition", response_model=ResponseResult)
async def speech_recognition(request: SpeechRecognitionRequest):
    """
    语音识别API端点

    接收音频文件列表，调用语音识别服务将语音转换为文本。
    支持说话人分离结果的整合，返回按说话人分组的识别结果。

    Args:
        request: 包含文件列表的语音识别请求对象

    Returns:
        ResponseResult: 包含处理结果的响应对象

    Raises:
        HTTPException: 当发生内部错误时抛出
    """
    try:
        # 验证请求参数：文件列表不能为空
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )

        # 调用语音识别服务处理文件
        processed_files, invalid_files = await process_speech_files(
            request.files
        )

        # 检查处理结果
        if not processed_files:
            if invalid_files:
                # 所有文件都处理失败
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文件处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                # 没有找到可处理的音频文件
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的音频文件",
                    data=None
                )

        # 将服务层返回的结果转换为API响应格式
        response_data = []
        for result in processed_files:
            # 将识别结果转换为RecognizedFile对象列表
            recognized_files = [
                RecognizedFile(
                    identity=recognition["identity"],
                    call_records=recognition["text"],
                    call_records_details=RecognizedDetails(
                        start=recognition["start_time"],
                        end=recognition["end_time"],
                        text="",  # 详细信息中的文本暂时为空
                        no_speech_prob=recognition["no_speech_prob"]
                    )
                )
                for recognition in result["recognitions"]
            ]

            # 构建单个文件的响应数据
            response_data.append(
                SpeechRecognitionResponseData(
                    file_id=result["file_id"],
                    call_original=result["call_original"],
                    call_translation="",  # 翻译字段暂时为空
                    call_records_collections=recognized_files
                )
            )

        # 根据处理结果返回相应的响应
        if invalid_files:
            # 部分成功，部分失败
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
        # 未知错误，记录详细错误信息
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")

        return ResponseResult(
            retcode=ResponseCode.UNKNOWN_ERROR,
            msg=f"未知错误: {str(e)}",
            data=None
        )