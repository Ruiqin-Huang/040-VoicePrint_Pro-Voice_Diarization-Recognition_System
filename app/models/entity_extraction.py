from typing import List, Optional
from pydantic import BaseModel, validator
from app.models.common import ModelInfo

class EntityExtractionRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None
    model_info: ModelInfo

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("输入文本不能为空")
        return v
    
class EntityResult(BaseModel):
    """单个实体抽取结果"""
    type: str
    name: str
    
class EntityExtractionResponseData(BaseModel):
    """实体抽取响应数据模型"""
    data: List[EntityResult]