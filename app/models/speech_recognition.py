from typing import List, Optional
from pydantic import BaseModel, validator

from app.models.file_request import FileRequest

# 请求模型
class SpeechRecognitionRequest(BaseModel):
    files: List[FileRequest]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

# 通话记录细节
class RecognizedDetails(BaseModel):
    start: str
    end: str
    text: str
    no_speech_prob: str

# 一段通话记录
class RecognizedFile(BaseModel):
    identity: str
    call_records: str
    call_records_details: RecognizedDetails

# 单个通话记录结果（多个语音段集合）
class SpeechRecognitionResponseData(BaseModel):
    file_id: str
    call_original: str
    call_translation: str
    call_records_collections: List[RecognizedFile]
