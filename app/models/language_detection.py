"""
语种检测数据模型模块

定义语种检测相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- LanguageDetectionRequest: 语种检测请求模型
- LanguageDetectionResponseData: 语种检测响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
- app.models.file_request: 文件请求模型
"""

from typing import List
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

# 请求模型
class LanguageDetectionRequest(BaseModel):
    """
    语种检测请求模型

    Attributes:
        text: 待检测语种的文本列表
    """
    text: List[str]

    @validator('text')
    def files_must_not_be_empty(cls, v):
        """
        验证器：确保文本列表不为空

        Args:
            v: 文本列表值

        Returns:
            List[str]: 验证后的文本列表

        Raises:
            ValueError: 当文本列表为空时抛出
        """
        if not v:
            raise ValueError("文本列表不能为空")
        return v

class LanguageDetectionResponseData(BaseModel):
    """
    语种检测响应数据模型

    Attributes:
        language: 检测到的语言代码（如'en', 'zh', 'ja'等）
        language_name: 语言的中文名称
        confidence: 检测置信度，范围0-1
    """
    language: str
    language_name: str
    confidence: float
