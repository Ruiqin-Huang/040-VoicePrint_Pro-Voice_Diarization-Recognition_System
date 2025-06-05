from fastapi import APIRouter, Depends, HTTPException
import traceback

# from app.config.path_mapper import PathMapper
# from app.dependencies import get_path_mapper
from app.models.common import ResponseResult
from app.models.speech_segmentation import SpeechSegmentationRequest, FileResult, SegmentFile
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
                data=[]
            )
            
        file_results, invalid_files = await process_audio_files(request.files)
        
        # 将服务层返回的结果转换为响应格式
        response_data = []
        for result in file_results:
            segment_files = [
                SegmentFile(id=segment["id"], file_url=segment["file_url"])
                for segment in result["segment_files"]
            ]
            
            response_data.append(
                FileResult(
                    file_id=result["file_id"],
                    file_type=result["file_type"],
                    segment_files=segment_files
                )
            )
        
        if invalid_files and any(len(r.segment_files) > 0 for r in response_data):
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
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