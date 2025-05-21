from typing import List, Optional
from pydantic import BaseModel, validator

class SpeechSegmentationRequest(BaseModel):
    files: List[str]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

class FileInfo(BaseModel):
    file_id: str
    source_url: str
    file_url: str

class ResponseData(BaseModel):
    file_type: List[str] = []  # 按输入文件顺序存储每个文件的语音类别
    files: List[FileInfo] = []