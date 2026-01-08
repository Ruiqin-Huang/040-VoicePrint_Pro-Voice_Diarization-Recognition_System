"""
实体抽取数据模型模块

定义实体抽取相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- EntityExtractionRequest: 实体抽取请求模型
- EntityResult: 单个实体抽取结果模型
- EntityExtractionResponseData: 实体抽取响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from typing import List, Optional
from pydantic import BaseModel, validator
from app.models.common import ModelInfo

class EntityExtractionRequest(BaseModel):
    """
    实体抽取请求模型
    
    用于接收文本和实体类型列表，进行实体抽取处理。
    
    Attributes:
        text: 待抽取的文本内容
        entity_types: 要抽取的实体类型列表，如果为None则使用默认实体类型列表
        model_info: 指定用于抽取的大模型信息
    """
    text: str
    entity_types: Optional[List[str]] = None
    model_info: ModelInfo

    @validator('text')
    def text_must_not_be_empty(cls, v):
        """
        验证器：确保输入文本不为空
        
        Args:
            v: 文本值
            
        Returns:
            str: 验证后的文本
            
        Raises:
            ValueError: 当文本为空或只包含空白字符时抛出
        """
        if not v or not v.strip():
            raise ValueError("输入文本不能为空")
        return v
    
class EntityResult(BaseModel):
    """
    单个实体抽取结果模型
    
    表示从文本中抽取出的单个实体信息。
    
    Attributes:
        type: 实体类型（如"人名"、"地名"、"时间"等）
        name: 实体名称或值
    """
    type: str
    name: str
    
class EntityExtractionResponseData(BaseModel):
    """
    实体抽取响应数据模型
    
    包含所有抽取出的实体结果列表。
    
    Attributes:
        data: 实体抽取结果列表，每个元素包含实体类型和名称
    """
    data: List[EntityResult]