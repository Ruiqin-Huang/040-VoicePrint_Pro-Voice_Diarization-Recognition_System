from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.diarization_comparison import DiarizationComparisonRequest, DiarizationComparisonResponseData
from app.services.diarization_comparison import DiarizationComparisonService
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/diarization_comparison", response_model=ResponseResult, summary="说话人切分及声纹比对接口")
async def diarization_comparison(request: DiarizationComparisonRequest):
    """
    对输入的音频文件进行主被叫切分，将切分后的声纹与目标库进行比对，并根据结果选择性地将新声纹入库。
    - **audio_files**: 一个包含音频文件绝对路径的列表。
    - **collection_name**: 用于比对的目标声纹库集合名称。
    - **accept_threshold**: (可选) 相似度阈值，高于此值则认为匹配成功并可入库，默认为0.85。
    """
    try:
        service = DiarizationComparisonService()
        result = await service.run_pipeline(
            audio_files=request.audio_files,
            collection_name=request.collection_name
        )
        response_data = DiarizationComparisonResponseData(**result)
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
        
    except ValueError as e:
        return ResponseResult(retcode=ResponseCode.INVALID_PARAM, msg=str(e))
    except FileNotFoundError as e:
        return ResponseResult(retcode=ResponseCode.INVALID_PARAM, msg=str(e))
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return ResponseResult(retcode=ResponseCode.OPERATION_ERROR, msg=f"处理失败: {e}")