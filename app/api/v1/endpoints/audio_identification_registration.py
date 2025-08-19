from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.audio_identification_registration import AudioIdentificationRequest, IdentificationResponseData
from app.services.audio_identification_registration import IdentificationRegistrationService
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/audio_identification_registration", response_model=ResponseResult, summary="说话人声纹识别与注册")
async def audio_identification_registration(request: AudioIdentificationRequest):
    """
    对输入的音频文件进行声纹识别，并根据参数选择性地注册新声纹或更新已有声纹。
    - **audio_files**: 一个包含音频文件绝对路径的列表。
    - **num_speakers_per_audio**: (可选) 每个音频文件中预期的说话人数量，默认为2。
    - **update_voiceprintlib**: (可选) 是否更新声纹库，默认为False。
    - **threshold**: (可选) 识别阈值，默认为0.65。
    """
    try:
        workspace = "./workspace"
        service = IdentificationRegistrationService(workspace=workspace)
        
        result = await service.run_pipeline(
            audio_files=request.audio_files,
            num_speakers=request.num_speakers_per_audio,
            update_library=request.update_voiceprintlib,
            threshold=request.threshold
        )
        
        response_data = IdentificationResponseData(**result)
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )

    except FileNotFoundError as e:
        return ResponseResult(retcode=ResponseCode.INVALID_PARAM, msg=str(e))
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return ResponseResult(retcode=ResponseCode.OPERATION_ERROR, msg=f"处理失败: {e}")