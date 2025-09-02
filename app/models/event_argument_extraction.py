from typing import Dict, List, Optional
from pydantic import BaseModel, validator, Field

class EventInfo(BaseModel):
    event_type: str
    argument_types: Optional[List[str]] = None

    @validator('event_type')
    def event_type_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("event_type 不能为空")
        return v

class EventArgumentExtractionRequest(BaseModel):
    text: str
    events_info: List[EventInfo] = Field(..., min_items=1)

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        return v

class SingleEventResult(BaseModel):
    event_type: str
    argument_types: Optional[List[str]]
    trigger: str
    arguments: List[Dict[str, str]]

class EventArgumentExtractionResponseData(BaseModel):
    text: str
    events: List[SingleEventResult]