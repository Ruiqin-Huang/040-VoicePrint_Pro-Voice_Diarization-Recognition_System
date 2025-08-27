from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.audio_registration import AudioRegistrationRequest, AudioRegistrationResponseData
from app.services.audio_registration import AudioRegistrationService
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/audio_registration", response_model=ResponseResult, summary="说话人声纹注册")
async def audio_registration(request: AudioRegistrationRequest):
    """
    接收一一对应的人员ID和音频文件列表，为每个音频提取声纹特征并存入Milvus声纹库。
    - **person_ids**: 人员ID列表。
    - **audio_files**: 音频文件绝对路径列表，每个文件应只包含一个说话人。
    - **collection_name**: (可选) 指定要存入的Milvus集合名称，如果未提供，则使用系统默认配置。
    """
    try:
        service = AudioRegistrationService()
        
        result = await service.run_pipeline(
            person_ids=request.person_ids,
            audio_files=request.audio_files,
            collection_name=request.collection_name
        )
        
        response_data = AudioRegistrationResponseData(**result)
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