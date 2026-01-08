"""
实体抽取API端点模块

提供RESTful API接口用于实体抽取服务。
通过FastAPI框架实现，支持从文本中抽取指定类型的实体。

主要功能：
- 接收实体抽取请求
- 调用实体抽取服务进行并行处理
- 支持超时控制和错误处理
- 返回结构化的实体抽取结果

依赖：
- FastAPI: Web框架
- app.services.entity_extraction: 实体抽取服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, HTTPException
import traceback
import logging
import asyncio

from app.models.common import ResponseResult
from app.models.entity_extraction import EntityExtractionRequest, EntityResult
from app.services.entity_extraction import process_entity_extraction
from app.core.error_codes import ResponseCode
from app.config.settings import settings

# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/entity_extraction", response_model=ResponseResult)
async def entity_extraction(request: EntityExtractionRequest):
    """
    实体抽取API端点
    
    从给定文本中抽取指定类型的实体。支持并行处理多个实体类型，提高处理效率。
    
    Args:
        request: 包含以下字段的请求对象：
            - text: 待抽取的文本内容
            - entity_types: 要抽取的实体类型列表，如果为None则使用默认实体类型列表
            - model_info: 指定用于抽取的大模型信息
        
    Returns:
        ResponseResult: 包含处理结果的响应对象，成功时包含：
            - retcode: 返回码
            - msg: 响应消息
            - data: 实体抽取结果列表，每个元素包含实体类型和名称
        
    Raises:
        HTTPException: 当发生参数错误、超时或处理失败时返回相应的错误响应
    """
    try:
        # 添加整体超时控制（所有任务的总时间）
        # 超时时间为单个任务超时时间的2倍，给足够的时间处理所有任务
        overall_timeout = settings.ENTITY_EXTRACTION_TIMEOUT * 2  # 给足够的时间处理所有任务
        
        try:
            # 调用实体抽取服务，并设置整体超时
            result_list = await asyncio.wait_for(
                process_entity_extraction(
                    text=request.text, 
                    model_info=request.model_info,
                    entity_types=request.entity_types
                ),
                timeout=overall_timeout
            )
        except asyncio.TimeoutError:
            # 处理超时异常
            logger.error(f"实体抽取整体超时（超过{overall_timeout}秒）")
            raise HTTPException(
                status_code=504, 
                detail=f"请求处理超时，文本可能过长或实体类型过多。请减少文本长度或实体类型数量。"
            )
        
        # 返回成功响应
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=result_list  # 直接将列表放入data字段
        )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except ValueError as e:
        # 捕获Pydantic验证错误和逻辑中的ValueError
        logger.warning(f"参数验证错误: {str(e)}")
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        # 未知错误，记录详细错误信息
        error_detail = traceback.format_exc()
        logger.error(f"实体抽取接口异常: {error_detail}")
        raise HTTPException(
            status_code=500, 
            detail=f"服务器内部错误: {str(e)}"
        )