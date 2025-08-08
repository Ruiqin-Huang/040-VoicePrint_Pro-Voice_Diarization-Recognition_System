from typing import List
from pydantic import BaseModel
from app.models.file_request import FileRequest

class TranslationRequest(BaseModel):
    """
    翻译请求参数模型（必须指定语种）
    :param file_requests: 待翻译文件列表
    :param source_lang: 源语言代码 (如 'en')
    :param target_lang: 目标语言代码 (如 'zh')
    """
    file_requests: List[FileRequest]
    source_lang: str
    target_lang: str

class TranslationResult(BaseModel):
    """
    翻译结果模型（移除语种检测字段）
    :param file_id: 文件ID
    :param file_path: 文件路径
    :param source_text: 原文
    :param translated_text: 翻译结果全文
    :param model_name: 使用的模型标识
    """
    file_id: str
    file_path: str
    source_lang: str
    source_lang_name: str
    source_text: str  # 保留原文用于对照
    target_lang: str
    target_lang_name: str
    translated_text: str
    model_name: str  # 示例: "opus-mt-en-zh"

class TranslationError(BaseModel):
    """
    错误信息模型（保持不变）
    """
    file_id: str
    file_path: str
    error: str

class TranslationResponse(BaseModel):
    """
    响应模型（结构不变）
    """
    processed_files: List[TranslationResult]
    invalid_files: List[TranslationError]