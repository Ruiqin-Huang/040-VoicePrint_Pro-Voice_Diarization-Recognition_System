from typing import List
from pydantic import BaseModel

from app.models.file_request import FileRequest

class LanguageDetectionResult(BaseModel):
    """
    语种检测结果模型
    :param file_id: 文件ID
    :param file_path: 文件原始路径
    :param language: 检测到的语言代码
    :param language_name: 语言中文名称
    :param confidence: 检测置信度(0-1)
    """
    file_id: str
    file_path: str
    language: str
    language_name: str
    confidence: float

class LanguageDetectionResponse(BaseModel):
    """
    语种检测响应模型
    :param processed_files: 成功处理的文件列表
    :param invalid_files: 处理失败的文件列表
    """
    processed_files: List[LanguageDetectionResult]
    invalid_files: List[str]