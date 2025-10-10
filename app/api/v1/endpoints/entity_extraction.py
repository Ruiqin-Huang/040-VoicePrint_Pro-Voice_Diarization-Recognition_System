from fastapi import APIRouter, HTTPException
import traceback

from app.models.common import ResponseResult
from app.models.entity_extraction import EntityExtractionRequest, EntityResult
from app.services.entity_extraction import process_entity_extraction
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/entity_extraction", response_model=ResponseResult)
async def entity_extraction(request: EntityExtractionRequest):
    """
    实体抽取API - 从给定文本中抽取指定类型的实体
    """
    try:
        result_list = await process_entity_extraction(
            text=request.text, 
            model_info=request.model_info,
            entity_types=request.entity_types
        )
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=result_list  # 直接将列表放入data字段
        )
        
    except ValueError as e:
        # 捕获Pydantic验证错误和逻辑中的ValueError
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"未知错误: {str(e)}")