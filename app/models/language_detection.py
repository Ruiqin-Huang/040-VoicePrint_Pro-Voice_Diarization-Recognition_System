from typing import List
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

# 请求模型
class LanguageDetectionRequest(BaseModel):
    text: List[str]
    
    @validator('text')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文本列表不能为空")
        return v

class LanguageDetectionResponseData(BaseModel):
    """
    语种检测结果模型
    :param language: 检测到的语言代码
    :param language_name: 语言中文名称
    :param confidence: 检测置信度(0-1)
    """
    language: str
    language_name: str
    confidence: float
