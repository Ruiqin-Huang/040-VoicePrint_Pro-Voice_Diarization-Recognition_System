from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict

class EventInfo(BaseModel):
    event_type: str = Field(..., description="要抽取的事件类型")
    argument_types: Optional[List[str]] = Field(None, description="要抽取的论元类型列表，如果为None则使用默认值")

class EventArgumentExtractionRequest(BaseModel):
    text: str = Field(..., description="待抽取的原始文本")
    events_info: List[EventInfo] = Field(..., description="包含一个或多个事件信息的列表", min_items=1)

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        return v

class Argument(BaseModel):
    name: str
    value: str

class SingleEventResult(BaseModel):
    event_type: str
    argument_types: Optional[List[str]]
    trigger: str
    arguments: List[Argument]

class EventArgumentExtractionData(BaseModel):
    text: str
    events: List[SingleEventResult]

class EventArgumentExtractionResponse(BaseModel):
    retcode: int = 200000
    msg: str = "success"
    data: EventArgumentExtractionData