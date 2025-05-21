from fastapi import APIRouter, HTTPException
import traceback

from app.models.common import ResponseResult
from app.models.speech_segmentation import SpeechSegmentationRequest, ResponseData, FileInfo
from app.services.speech_segmentation import process_audio_files
from app.core.error_codes import ResponseCode

router = APIRouter(prefix="/api")

@router.post("/speech_segmentation", response_model=ResponseResult)
async def speech_segmentation(request: SpeechSegmentationRequest):
    """语音分割API - 将多人语音分离为单个说话人"""
    try:
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表为空",
                data=ResponseData(file_type=[], files=[])
            )
            
        file_results, invalid_files, file_types = await process_audio_files(request.files)
        
        if invalid_files and file_results:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=ResponseData(
                    file_type=file_types,
                    files=[FileInfo(**item) for item in file_results]
                )
            )
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=ResponseData(
                file_type=file_types,
                files=[FileInfo(**item) for item in file_results]
            )
        )
        
    except ValueError as e:
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except FileNotFoundError as e:
        return ResponseResult(
            retcode=ResponseCode.OPERATION_ERROR,
            msg=f"文件不存在: {str(e)}",
            data=None
        )
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        if "模型加载失败" in str(e):
            return ResponseResult(
                retcode=ResponseCode.OPERATION_ERROR,
                msg=f"模型加载失败: {str(e)}",
                data=None
            )
        elif "处理失败" in str(e):
            return ResponseResult(
                retcode=ResponseCode.OPERATION_ERROR,
                msg=str(e),
                data=None
            )
        else:
            return ResponseResult(
                retcode=ResponseCode.UNKNOWN_ERROR,
                msg=f"未知错误: {str(e)}",
                data=None
            )