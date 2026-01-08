"""
说话人切分及声纹比对API端点模块

提供RESTful API接口用于说话人切分和声纹比对服务。
通过FastAPI框架实现，支持批量音频文件处理和声纹比对。

主要功能：
- 接收说话人切分及声纹比对请求
- 调用服务进行主被叫切分
- 提取声纹特征并进行聚类分析
- 与Milvus声纹库进行相似度比对
- 返回结构化的切分和比对结果

依赖：
- FastAPI: Web框架
- app.services.diarization_comparison: 说话人切分及声纹比对服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.diarization_comparison import DiarizationComparisonRequest, DiarizationComparisonResponseData
from app.services.diarization_comparison import DiarizationComparisonService
from app.core.error_codes import ResponseCode

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/diarization_comparison", response_model=ResponseResult, summary="说话人切分及声纹比对接口")
async def diarization_comparison(request: DiarizationComparisonRequest):
    """
    说话人切分及声纹比对API端点
    
    对输入的音频文件进行主被叫切分，将切分后的声纹与目标库进行比对，并进行聚类分析。
    
    Args:
        request: 包含以下字段的请求对象：
            - audio_files: 一个包含音频文件绝对路径的列表
            - collection_name: 用于比对的目标声纹库集合名称
            - accept_threshold: (已注释) 相似度阈值，高于此值则认为匹配成功并可入库，默认为0.85
        
    Returns:
        ResponseResult: 包含处理结果的响应对象，成功时包含：
            - collection_name: 参与比较的目标说话人声纹库集合名称
            - comparison_results: 所有音频片段的切分和比对结果列表
            - cluster_results: 所有分割音频的聚类结果列表
        
    Raises:
        HTTPException: 当发生参数错误或处理失败时返回相应的错误响应
    """
    try:
        # 创建说话人切分及声纹比对服务实例
        service = DiarizationComparisonService()
        # 执行切分和比对流程
        result = await service.run_pipeline(
            audio_files=request.audio_files,
            collection_name=request.collection_name
        )
        # 构建响应数据对象
        response_data = DiarizationComparisonResponseData(**result)
        # 返回成功响应
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
        
    except ValueError as e:
        # 参数验证错误
        return ResponseResult(retcode=ResponseCode.INVALID_PARAM, msg=str(e))
    except FileNotFoundError as e:
        # 文件不存在错误
        return ResponseResult(retcode=ResponseCode.INVALID_PARAM, msg=str(e))
    except Exception as e:
        # 未知错误，记录详细错误信息
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return ResponseResult(retcode=ResponseCode.OPERATION_ERROR, msg=f"处理失败: {e}")