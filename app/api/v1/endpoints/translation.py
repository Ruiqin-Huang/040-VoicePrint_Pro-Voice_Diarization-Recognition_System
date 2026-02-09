"""
机器翻译API端点模块

提供RESTful API接口用于机器翻译服务。
通过FastAPI框架实现，支持多种语言的文本翻译。

主要功能：
- 接收翻译请求
- 调用翻译服务进行处理
- 返回结构化的翻译结果
- 支持M2M100、Small100、混元MT1.5模型选择

依赖：
- FastAPI: Web框架
- app.services.translation: 翻译服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

from app.models.common import ResponseResult
from app.models.translation import TranslationRequest, TranslationResponseData
from app.services.translation import process_translation
from app.core.error_codes import ResponseCode

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/translation",
             response_model=ResponseResult)
async def translation(requests: TranslationRequest):
    """
    机器翻译API端点

    接收文本列表和翻译参数，调用翻译服务进行多语言翻译。
    支持指定源语言、目标语言和翻译模型类型。

    Args:
        requests: 包含文本列表和翻译参数的请求对象

    Returns:
        ResponseResult: 包含翻译结果的响应对象

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

        # 调用翻译服务处理文本
        response_results = await process_translation(
            requests.text,
            requests.source_lang,
            requests.target_lang,
            requests.model_type
        )

        if requests.model_type == "hy_mt1.5":
            return response_results
        
        processed_files, invalid_files = response_results

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
                    msg="没有找到可处理的文本",
                    data=None
                )

        # 将服务层返回的结果转换为API响应格式
        response_data = []
        for result in processed_files:
            response_data.append(
                TranslationResponseData(
                    source_lang=result["source_lang"],
                    source_lang_name=result["source_lang_name"],
                    source_text=result["source_text"],
                    target_lang=result["target_lang"],
                    target_lang_name=result["target_lang_name"],
                    translated_text=result["translated_text"],
                    model_name=result["model_name"]
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