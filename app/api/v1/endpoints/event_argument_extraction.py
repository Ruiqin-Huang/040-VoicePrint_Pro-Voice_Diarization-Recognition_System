"""
事件论元抽取API端点模块

提供RESTful API接口用于事件论元抽取服务。
通过FastAPI框架实现，支持从文本中抽取一个或多个指定事件的触发词和论元。

主要功能：
- 接收事件论元抽取请求
- 调用事件论元抽取服务进行并行处理
- 支持多个事件的批量抽取
- 返回结构化的事件论元抽取结果

依赖：
- FastAPI: Web框架
- app.services.event_argument_extraction: 事件论元抽取服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, HTTPException
from app.models.event_argument_extraction import EventArgumentExtractionRequest, EventArgumentExtractionResponse, EventArgumentExtractionData
from app.services import event_argument_extraction as service

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/event_argument_extraction", response_model=EventArgumentExtractionResponse)
async def event_argument_extraction(request: EventArgumentExtractionRequest):
    """
    事件论元抽取API端点
    
    从给定文本中抽取一个或多个指定事件的触发词和论元。支持并行处理多个事件，提高处理效率。
    
    Args:
        request: 包含以下字段的请求对象：
            - text: 待抽取的文本内容
            - events_info: 事件列表，每个事件包含event_type和可选的argument_types
            - model_info: 指定用于抽取的大模型信息
        
    Returns:
        EventArgumentExtractionResponse: 包含处理结果的响应对象，成功时包含：
            - retcode: 返回码，200000表示成功
            - msg: 响应消息，通常为"success"
            - data: 响应数据，包含原始文本和所有事件的抽取结果
        
    Raises:
        HTTPException: 当发生参数错误或处理失败时返回相应的错误响应
    """
    try:
        # 使用别名 'service' 来调用函数
        # 调用事件论元抽取服务，并行处理多个事件
        results = await service.process_multi_event_argument_extraction(
            request.text, request.events_info, request.model_info
        )
        
        # 构建响应数据对象
        response_data = EventArgumentExtractionData(
            text=request.text,
            events=results
        )
        
        # 返回成功响应
        return EventArgumentExtractionResponse(
            retcode=200000,
            msg="success",
            data=response_data
        )
    except ValueError as e:
        # 参数验证错误
        raise HTTPException(status_code=400, detail=f"参数错误: {e}")
    except Exception as e:
        # 未知错误，记录错误信息
        print(f"An unexpected error occurred in event argument extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during event argument extraction.")