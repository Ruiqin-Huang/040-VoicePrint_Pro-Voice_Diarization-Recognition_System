"""
混元翻译API接口端点

提供混元MT1.5翻译模型的HTTP接口。

此模块定义：
- 健康检查接口 (GET /health)
- 文本翻译接口 (POST /translate)

该路由被挂载到app/hy_main.py中：
  app.include_router(hy_translation.router, prefix="/api", tags=["translation"])

因此实际端点为：
- GET /api/health
- POST /api/translate
"""

from typing import List
from fastapi import APIRouter, HTTPException

from app.models.translation import TranslationRequest, TranslationResponseData
from app.services.hy_translation import process_translation
from app.config.settings import settings

# 创建路由器
router = APIRouter(
    prefix="/api",
    tags=["hy_translation"],
    responses={500: {"description": "Server error"}}
)


@router.get("/health")
async def health_check():
    """
    健康检查端点
    
    检查服务是否正常运行，模型是否已加载
    
    Returns:
        dict: 包含状态、模型加载情况、设备信息
    """
    return {
        "status": "healthy",
        "model": "Tencent-Hunyuan/HY-MT1.5-1.8B",
        "device": f"cuda:{settings.GPU_ID}"
    }

@router.post("/translation", response_model=List[TranslationResponseData])
async def translate(request: TranslationRequest):
    """
    批量翻译文本接口
    
    接收一批待翻译的文本，返回翻译结果。
    
    Args:
        request: TranslationRequest
            - text: List[str] - 待翻译文本列表（不为空）
            - source_lang: str - 源语言代码（如 "zh", "en"）
            - target_lang: str - 目标语言代码
            - model_type: Optional[str] - 模型类型（默认为 "m2m100"，此服务强制使用 "hy_mt1.5"）
    
    Returns:
        List[TranslationResponseData]: 翻译结果列表，每条包含：
            - source_lang: 源语言代码
            - source_lang_name: 源语言中文名称
            - source_text: 原文本
            - target_lang: 目标语言代码
            - target_lang_name: 目标语言中文名称
            - translated_text: 翻译后的文本
            - model_name: 使用的模型名称
    
    Raises:
        HTTPException: 400 - 请求参数有效性错误
        HTTPException: 500 - 翻译过程错误
    
    Example:
        Request:
        {
            "text": ["你好", "谢谢"],
            "source_lang": "zh",
            "target_lang": "en",
            "model_type": "hy_mt1.5"
        }
        
        Response:
        [
            {
                "source_lang": "zh",
                "source_lang_name": "汉语",
                "source_text": "你好",
                "target_lang": "en",
                "target_lang_name": "英语",
                "translated_text": "Hello",
                "model_name": "Tencent-Hunyuan/HY-MT1.5-1.8B"
            },
            ...
        ]
    """
    try:
        # 验证请求
        if not request.text or len(request.text) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text list cannot be empty"
            )
        
        # 调用服务层进行批量翻译
        results = await process_translation(
            request.text,
            request.source_lang,
            request.target_lang
        )

        # 构建响应对象
        return [
            TranslationResponseData(
                source_lang=item["source_lang"],
                source_lang_name=item["source_lang_name"],
                source_text=item["source_text"],
                target_lang=item["target_lang"],
                target_lang_name=item["target_lang_name"],
                translated_text=item["translated_text"],
                model_name=item["model_name"]
            )
            for item in results
        ]
    
    except HTTPException:
        # 直接传递HTTPException
        raise
    except Exception as e:
        # 捕获其他异常并转换为HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )
