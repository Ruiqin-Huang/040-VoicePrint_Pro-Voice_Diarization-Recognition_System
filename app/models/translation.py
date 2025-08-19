from typing import List, Optional
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

class TranslationRequest(BaseModel):
    """
    翻译请求参数模型（必须指定语种）
    :param file_requests: 待翻译文本列表
    :param source_lang: 源语言代码 (如 'en')
    :param target_lang: 目标语言代码 (如 'zh')
    :param target_lang: 模型选项 ('m2m100' 或 'small100')
    """
    text: List[str]
    source_lang: str
    target_lang: str
    model_type: Optional[str] = "m2m100"

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

class TranslationResponseData(BaseModel):
    """
    翻译结果模型（移除语种检测字段）
    :param source_lang: 源语言代码 (如 'en')
    :param source_lang_name: 源语言名称
    :param source_text: 原文
    :param source_lang: 目标语言代码 (如 'en')
    :param source_lang_name: 目标语言名称
    :param translated_text: 翻译结果全文
    :param model_name: 使用的模型标识
    """
    source_lang: str
    source_lang_name: str
    source_text: str
    target_lang: str
    target_lang_name: str
    translated_text: str
    model_name: str
