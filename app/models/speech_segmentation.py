"""
语音分割数据模型模块

定义语音分割相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- SpeechSegmentationRequest: 语音分割请求模型
- SegmentFile: 单个分割后的文件信息模型
- FileResult: 单个输入文件的处理结果模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from typing import List, Optional
from pydantic import BaseModel, validator

from app.models.file_request import FileRequest

# 请求模型
class SpeechSegmentationRequest(BaseModel):
    """
    语音分割请求模型
    
    用于接收文件列表，进行说话人分割处理。
    
    Attributes:
        files: 待处理的文件请求列表，每个文件请求包含文件ID和文件路径
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

# 单个分割后的文件信息
class SegmentFile(BaseModel):
    """
    单个分割后的文件信息模型
    
    表示分割后的单个说话人音频文件信息。
    
    Attributes:
        id: 分割文件的唯一标识符
        file_url: 分割后的音频文件路径或URL
    """
    id: str
    file_url: str

# 单个输入文件的处理结果
class FileResult(BaseModel):
    """
    单个输入文件的处理结果模型
    
    包含原始文件的处理结果，包括文件类型和分割后的文件列表。
    
    Attributes:
        file_id: 原始文件的唯一标识符
        file_type: 文件类型，如"单人"、"双人"、"多人"等
        segment_files: 分割后的文件列表
        metadata: 分割结果元数据路径
    """
    file_id: str
    file_type: str  # "单人"、"双人"、"多人"等
    segment_files: List[SegmentFile]
    metadata: str