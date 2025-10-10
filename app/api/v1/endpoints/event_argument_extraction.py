from fastapi import APIRouter, HTTPException
from app.models.event_argument_extraction import EventArgumentExtractionRequest, EventArgumentExtractionResponse, EventArgumentExtractionData
from app.services import event_argument_extraction as service

router = APIRouter(prefix="/api")

@router.post("/event_argument_extraction", response_model=EventArgumentExtractionResponse)
async def event_argument_extraction(request: EventArgumentExtractionRequest):
    """
    从给定文本中抽取一个或多个指定事件的论元。

    - **text**: 待抽取的文本内容。
    - **events_info**: 事件列表，每个事件包含 `event_type` 和可选的 `argument_types`。
    - **model_info**: 指定用于抽取的大模型信息。
    """
    try:
        # 使用别名 'service' 来调用函数
        results = await service.process_multi_event_argument_extraction(
            request.text, request.events_info, request.model_info
        )
        
        response_data = EventArgumentExtractionData(
            text=request.text,
            events=results
        )
        
        return EventArgumentExtractionResponse(
            retcode=200000,
            msg="success",
            data=response_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in event argument extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during event argument extraction.")