"""
语音识别数据模型模块

定义语音识别相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- SpeechRecognitionRequest: 语音识别请求模型
- RecognizedDetails: 识别细节模型
- RecognizedFile: 单段识别结果模型
- SpeechRecognitionResponseData: 语音识别响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
- app.models.file_request: 文件请求模型
"""

from typing import List, Optional
from pydantic import BaseModel, validator

from app.models.file_request import FileRequest

# 请求模型
class SpeechRecognitionRequest(BaseModel):
    """
    语音识别请求模型

    Attributes:
        files: 要进行语音识别的文件列表
    """
    files: List[FileRequest]

    @validator('files')
    def files_must_not_be_empty(cls, v):
        """
        验证器：确保文件列表不为空

        Args:
            v: 文件列表值

        Returns:
            List[FileRequest]: 验证后的文件列表

        Raises:
            ValueError: 当文件列表为空时抛出
        """
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 通话记录细节
class RecognizedDetails(BaseModel):
    """
    语音识别细节模型

    Attributes:
        start: 语音段开始时间（秒）
        end: 语音段结束时间（秒）
        text: 识别出的文本内容
        no_speech_prob: 无语音概率（0-1，越低表示越有语音）
    """
    start: float
    end: float
    text: str
    no_speech_prob: float

# 一段通话记录
class RecognizedFile(BaseModel):
    """
    单段语音识别结果模型

    Attributes:
        identity: 说话人身份标识
        call_records: 识别出的通话记录文本
        call_records_details: 详细的识别信息
    """
    identity: str
    call_records: str
    call_records_details: RecognizedDetails

# 单个通话记录结果（多个语音段集合）
class SpeechRecognitionResponseData(BaseModel):
    """
    语音识别响应数据模型

    Attributes:
        file_id: 文件唯一标识符
        call_original: 原始完整通话文本
        call_translation: 通话翻译文本（预留字段）
        call_records_collections: 按说话人分组的识别结果集合
    """
    file_id: str
    call_original: str
    call_translation: str
    call_records_collections: List[RecognizedFile]
