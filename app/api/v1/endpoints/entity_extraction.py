from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.entity_extraction import EntityExtractionRequest, EntityExtractionResponseData, EntityResult
from app.services.entity_extraction import process_entity_extraction
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/entity_extraction", response_model=ResponseResult)
async def entity_extraction(request: EntityExtractionRequest):
    """
    实体抽取API - 从给定文本中抽取指定类型的实体
    """
    try:
        result_list = await process_entity_extraction(request.text, request.entity_types)
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=result_list  # 直接将列表放入data字段
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