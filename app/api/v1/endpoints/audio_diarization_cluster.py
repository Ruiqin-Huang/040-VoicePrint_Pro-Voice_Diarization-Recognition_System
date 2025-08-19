from fastapi import APIRouter, HTTPException
import traceback
from datetime import datetime

from app.models.common import ResponseResult
from app.models.audio_diarization_cluster import AudioDiarizationClusterRequest, AudioDiarizationClusterResponse
from app.services.audio_diarization_cluster import DiarizationClusterService
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/audio_diarization_cluster", response_model=ResponseResult, summary="说话人分割与声纹聚类")
async def audio_diarization_cluster(request: AudioDiarizationClusterRequest):
    """
    对输入的音频文件列表进行说话人分割及声纹聚类。
    - **audio_files**: 一个包含音频文件绝对路径的列表。
    - **num_speakers_per_audio**: (可选) 每个音频文件中预期的说话人数量，默认为2。
    """
    try:
        # 使用固定的共享工作区
        workspace = "./workspace"
        
        service = DiarizationClusterService(workspace=workspace)
        result = await service.run_pipeline(request.audio_files, request.num_speakers_per_audio)
        
        response_data = AudioDiarizationClusterResponse(**result)
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