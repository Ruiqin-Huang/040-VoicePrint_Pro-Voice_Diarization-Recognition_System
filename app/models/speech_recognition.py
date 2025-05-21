from typing import List, Optional
from pydantic import BaseModel, validator

class SpeechRecognitionRequest(BaseModel):
    files: List[str]
    language: str = "zh"  # 默认中文
    use_gpu: bool = True
    gpu: int = 0
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

class RecognizedFile(BaseModel):
    file_id: str
    phone_number: str
    identity: str
    call_record: str
    create_time: str
    file_url: str

class SpeechRecognitionResponseData(BaseModel):
    calling_party_number: str
    called_party_number: str
    keywords: List[str]
    labels: List[str]
    call_original: str
    call_translation: str
    files: List[RecognizedFile]