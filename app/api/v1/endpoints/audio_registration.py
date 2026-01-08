"""
音频声纹注册API端点模块

提供RESTful API接口用于音频声纹注册服务。
通过FastAPI框架实现，支持批量音频文件处理和声纹特征提取。

主要功能：
- 接收音频声纹注册请求
- 调用声纹注册服务进行特征提取
- 将声纹特征存储到Milvus向量数据库
- 返回结构化的注册结果

依赖：
- FastAPI: Web框架
- app.services.audio_registration: 音频声纹注册服务
- app.models: 数据模型定义
"""

from fastapi import APIRouter
import traceback

from app.models.common import ResponseResult
from app.models.audio_registration import AudioRegistrationRequest, AudioRegistrationResponseData
from app.services.audio_registration import AudioRegistrationService
from app.core.error_codes import ResponseCode

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/audio_registration", response_model=ResponseResult, summary="说话人声纹注册")
async def audio_registration(request: AudioRegistrationRequest):
    """
    音频声纹注册API端点
    
    接收一一对应的人员ID和音频文件列表，为每个音频提取声纹特征并存入Milvus声纹库。
    支持批量处理多个音频文件，每个文件对应一个人员ID。
    
    Args:
        request: 包含以下字段的请求对象：
            - person_ids: 人员ID列表，需与音频文件列表一一对应
            - audio_files: 音频文件绝对路径列表，每个文件应只包含一个说话人
            - collection_name: (可选) 指定要存入的Milvus集合名称，如果未提供，则使用系统默认配置
        
    Returns:
        ResponseResult: 包含处理结果的响应对象，成功时包含：
            - collection_name: 数据插入的目标集合名称
            - inserted_count: 成功插入的记录数量
            - inserted_result: 成功插入的记录详情列表
        
    Raises:
        HTTPException: 当发生参数错误或处理失败时返回相应的错误响应
    """
    try:
        # 创建声纹注册服务实例
        service = AudioRegistrationService()
        
        # 执行声纹注册流程
        result = await service.run_pipeline(
            person_ids=request.person_ids,
            audio_files=request.audio_files,
            collection_name=request.collection_name
        )
        
        # 构建响应数据对象
        response_data = AudioRegistrationResponseData(**result)
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