"""
语音分割API端点模块

提供RESTful API接口用于语音分割服务。
通过FastAPI框架实现，支持批量音频文件处理和说话人分割。

主要功能：
- 接收语音分割请求
- 调用语音分割服务进行说话人分离
- 将多人语音分离为单个说话人的音频片段
- 返回结构化的分割结果

依赖：
- FastAPI: Web框架
- app.services.speech_segmentation: 语音分割服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter, Depends, HTTPException
import traceback

# from app.config.path_mapper import PathMapper
# from app.dependencies import get_path_mapper
from app.models.common import ResponseResult
from app.models.speech_segmentation import SpeechSegmentationRequest, FileResult, SegmentFile
from app.services.speech_segmentation import process_audio_files
from app.core.error_codes import ResponseCode

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/speech_segmentation", response_model=ResponseResult)
async def speech_segmentation(request: SpeechSegmentationRequest):
    """
    语音分割API端点
    
    将多人语音分离为单个说话人的音频片段。
    支持本地文件和URL，支持批量处理。
    
    Args:
        request: 包含文件列表的语音分割请求对象，每个文件请求包含文件ID和文件路径
        
    Returns:
        ResponseResult: 包含处理结果的响应对象，成功时包含：
            - file_id: 文件的唯一标识符
            - file_type: 文件类型（如"单人"、"双人"、"多人"等）
            - segment_files: 分割后的文件列表，每个包含id和file_url
        
    Raises:
        HTTPException: 当发生参数错误、文件不存在或处理失败时返回相应的错误响应
    """
    try:
        # 验证请求参数：文件列表不能为空
        if not request.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表为空",
                data=[]
            )
            
        # 调用语音分割服务处理文件
        file_results, invalid_files = await process_audio_files(request.files)
        
        # 将服务层返回的结果转换为响应格式
        response_data = []
        for result in file_results:
            # 构建分割文件列表
            segment_files = [
                SegmentFile(id=segment["id"], file_url=segment["file_url"])
                for segment in result["segment_files"]
            ]
            
            # 构建文件结果对象
            response_data.append(
                FileResult(
                    file_id=result["file_id"],
                    file_type=result["file_type"],
                    segment_files=segment_files,
                    metadata=result["metadata"]
                )
            )
        
        # 检查处理结果
        if invalid_files and any(len(r.segment_files) > 0 for r in response_data):
            # 部分成功，部分失败
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
            )
        
        # 全部成功
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
        
    except ValueError as e:
        # 参数验证错误
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except FileNotFoundError as e:
        # 文件不存在错误
        return ResponseResult(
            retcode=ResponseCode.OPERATION_ERROR,
            msg=f"文件不存在: {str(e)}",
            data=None
        )
    except Exception as e:
        # 未知错误，记录详细错误信息
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        # 根据错误类型返回相应的错误响应
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