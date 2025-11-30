from fastapi import APIRouter, HTTPException
import traceback
import logging
import asyncio

from app.models.common import ResponseResult
from app.models.entity_extraction import EntityExtractionRequest, EntityResult
from app.services.entity_extraction import process_entity_extraction
from app.core.error_codes import ResponseCode
from app.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

@router.post("/entity_extraction", response_model=ResponseResult)
async def entity_extraction(request: EntityExtractionRequest):
    """
    实体抽取API - 从给定文本中抽取指定类型的实体
    """
    try:
        # 添加整体超时控制（所有任务的总时间）
        overall_timeout = settings.ENTITY_EXTRACTION_TIMEOUT * 2  # 给足够的时间处理所有任务
        
        try:
            result_list = await asyncio.wait_for(
                process_entity_extraction(
                    text=request.text, 
                    model_info=request.model_info,
                    entity_types=request.entity_types
                ),
                timeout=overall_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"实体抽取整体超时（超过{overall_timeout}秒）")
            raise HTTPException(
                status_code=504, 
                detail=f"请求处理超时，文本可能过长或实体类型过多。请减少文本长度或实体类型数量。"
            )
        
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
        error_detail = traceback.format_exc()
        logger.error(f"实体抽取接口异常: {error_detail}")
        raise HTTPException(
            status_code=500, 
            detail=f"服务器内部错误: {str(e)}"
        )