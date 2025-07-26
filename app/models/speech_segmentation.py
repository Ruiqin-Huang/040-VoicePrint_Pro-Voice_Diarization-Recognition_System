from typing import List, Optional
from pydantic import BaseModel, validator

from app.models.file_request import FileRequest

# 请求模型
class SpeechSegmentationRequest(BaseModel):
    files: List[FileRequest]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 单个分割后的文件信息
class SegmentFile(BaseModel):
    id: str
    file_url: str

# 单个输入文件的处理结果
class FileResult(BaseModel):
    file_id: str
    file_type: str  # "单人"、"双人"、"多人"等
    segment_files: List[SegmentFile]