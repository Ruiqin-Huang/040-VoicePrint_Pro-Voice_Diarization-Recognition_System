"""
机器翻译数据模型模块

定义机器翻译相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- TranslationRequest: 翻译请求模型
- TranslationResponseData: 翻译响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
- app.models.file_request: 文件请求模型
"""

from typing import List, Optional
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

class TranslationRequest(BaseModel):
    """
    翻译请求参数模型

    Attributes:
        text: 待翻译的文本列表
        source_lang: 源语言代码（如'en', 'zh'）
        target_lang: 目标语言代码（如'en', 'zh'）
        model_type: 使用的翻译模型类型，默认为'm2m100'
    """
    text: List[str]
    source_lang: str
    target_lang: str
    model_type: Optional[str] = "m2m100"

    @validator('text')
    def text_must_not_be_empty(cls, v):
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
            raise ValueError("文件列表不能为空")
        return v

class TranslationResponseData(BaseModel):
    """
    翻译结果响应数据模型

    Attributes:
        source_lang: 源语言代码
        source_lang_name: 源语言中文名称
        source_text: 原文文本
        target_lang: 目标语言代码
        target_lang_name: 目标语言中文名称
        translated_text: 翻译后的文本
        model_name: 使用的翻译模型名称
    """
    source_lang: str
    source_lang_name: str
    source_text: str
    target_lang: str
    target_lang_name: str
    translated_text: str
    model_name: str
