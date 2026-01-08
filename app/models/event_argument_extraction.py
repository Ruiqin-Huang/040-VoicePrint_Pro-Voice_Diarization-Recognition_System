"""
事件论元抽取数据模型模块

定义事件论元抽取相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- EventInfo: 事件信息模型
- EventArgumentExtractionRequest: 事件论元抽取请求模型
- Argument: 单个论元模型
- SingleEventResult: 单个事件的抽取结果模型
- EventArgumentExtractionData: 事件论元抽取响应数据模型
- EventArgumentExtractionResponse: 事件论元抽取响应模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from app.models.common import ModelInfo

class EventInfo(BaseModel):
    """
    事件信息模型
    
    表示要抽取的事件类型和对应的论元类型列表。
    
    Attributes:
        event_type: 要抽取的事件类型
        argument_types: 要抽取的论元类型列表，如果为None则使用默认值（主体、客体、时间、地点）
    """
    event_type: str = Field(..., description="要抽取的事件类型")
    argument_types: Optional[List[str]] = Field(None, description="要抽取的论元类型列表，如果为None则使用默认值")

class EventArgumentExtractionRequest(BaseModel):
    """
    事件论元抽取请求模型
    
    用于接收文本和事件信息列表，进行事件论元抽取处理。
    
    Attributes:
        text: 待抽取的原始文本
        events_info: 包含一个或多个事件信息的列表，至少包含一个事件
        model_info: 指定用于抽取的大模型信息
    """
    text: str = Field(..., description="待抽取的原始文本")
    events_info: List[EventInfo] = Field(..., description="包含一个或多个事件信息的列表", min_items=1)
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
            raise ValueError("text 不能为空")
        return v

class Argument(BaseModel):
    """
    单个论元模型
    
    表示事件的一个论元信息。
    
    Attributes:
        name: 论元名称（如"主体"、"客体"、"时间"、"地点"等）
        value: 论元的值
    """
    name: str
    value: str

class SingleEventResult(BaseModel):
    """
    单个事件的抽取结果模型
    
    包含一个事件的完整抽取结果，包括触发词和所有论元。
    
    Attributes:
        event_type: 事件类型
        argument_types: 请求的论元类型列表
        trigger: 事件的触发词
        arguments: 抽取出的论元列表
    """
    event_type: str
    argument_types: Optional[List[str]]
    trigger: str
    arguments: List[Argument]

class EventArgumentExtractionData(BaseModel):
    """
    事件论元抽取响应数据模型
    
    包含原始文本和所有事件的抽取结果。
    
    Attributes:
        text: 原始输入文本
        events: 所有事件的抽取结果列表
    """
    text: str
    events: List[SingleEventResult]

class EventArgumentExtractionResponse(BaseModel):
    """
    事件论元抽取响应模型
    
    标准的API响应格式，包含返回码、消息和数据。
    
    Attributes:
        retcode: 返回码，200000表示成功
        msg: 响应消息，通常为"success"
        data: 响应数据，包含原始文本和事件抽取结果
    """
    retcode: int = 200000
    msg: str = "success"
    data: EventArgumentExtractionData