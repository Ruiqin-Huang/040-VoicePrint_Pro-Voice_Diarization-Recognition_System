from typing import List, Optional
from pydantic import BaseModel, validator

class EntityExtractionRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("输入文本不能为空")
        return v