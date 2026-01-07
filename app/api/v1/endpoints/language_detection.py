"""
语种检测API端点模块

提供RESTful API接口用于文本语种检测服务。
通过FastAPI框架实现，支持批量文本处理和错误处理。

主要功能：
- 接收语种检测请求
- 调用语种检测服务进行处理
- 返回结构化的检测结果

依赖：
- FastAPI: Web框架
- app.services.language_detection: 语种检测服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

from app.models.common import ResponseResult
from app.models.language_detection import LanguageDetectionRequest, LanguageDetectionResponseData
from app.services.language_detection import process_text_files
from app.core.error_codes import ResponseCode

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/language_detection",
             response_model=ResponseResult)
async def language_detection(requests: LanguageDetectionRequest):
    """
    语种检测API端点

    接收文本列表，调用语种检测服务进行语言识别，并返回检测结果。
    支持批量处理多个文本。

    Args:
        requests: 包含文本列表的检测请求对象

    Returns:
        ResponseResult: 包含处理结果的响应对象

    Raises:
        HTTPException: 当发生内部错误时抛出
    """
    try:
        # 验证请求参数：文本列表不能为空
        if not requests.text:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文本列表不能为空",
                data=None
            )

        # 调用语种检测服务处理文本
        processed_files, invalid_files = await process_text_files(requests.text)

        # 检查处理结果
        if not processed_files:
            if invalid_files:
                # 所有文本都处理失败
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文本处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                # 没有找到可处理的文本
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的文本文件",
                    data=None
                )

        # 将服务层返回的结果转换为API响应格式
        response_data = []
        for result in processed_files:
            response_data.append(
                LanguageDetectionResponseData(
                    language=result["language"],
                    language_name=result["language_name"],
                    confidence=result["confidence"]
                )
            )

        # 根据处理结果返回相应的响应
        if invalid_files:
            # 部分成功，部分失败
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文本处理成功，{len(invalid_files)}个文本失败: {'; '.join(invalid_files)}",
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