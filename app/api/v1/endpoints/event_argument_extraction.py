from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.event_argument_extraction import EventArgumentExtractionRequest, EventArgumentExtractionResponseData
from app.services.event_argument_extraction import process_event_argument_extraction
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/event_argument_extraction", response_model=ResponseResult)
async def event_argument_extraction(request: EventArgumentExtractionRequest):
    """
    事件论元抽取API - 从给定文本中抽取指定事件的论元
    """
    try:
        result_dict = await process_event_argument_extraction(
            request.text, request.event_type, request.argument_types
        )
        
        response_data = EventArgumentExtractionResponseData(
            trigger=result_dict["trigger"],
            arguments=result_dict["arguments"],
            event_type=result_dict["event_type"]
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