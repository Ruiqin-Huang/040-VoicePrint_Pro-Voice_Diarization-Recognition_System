from typing import Dict, List, Optional
from pydantic import BaseModel, validator

class EventArgumentExtractionRequest(BaseModel):
    text: str
    event_type: str
    argument_types: Optional[List[str]] = None

    @validator('text', 'event_type')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text 和 event_type 不能为空")
        return v
    
class EventArgumentExtractionResponseData(BaseModel):
    trigger: str
    arguments: List[Dict[str, str]]
    event_type: str