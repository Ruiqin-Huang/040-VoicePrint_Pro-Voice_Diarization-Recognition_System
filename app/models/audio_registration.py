"""
音频声纹注册数据模型模块

定义音频声纹注册相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- AudioRegistrationRequest: 音频声纹注册请求模型
- AudioRegistrationResponseData: 音频声纹注册响应数据模型

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict

class AudioRegistrationRequest(BaseModel):
    """
    音频声纹注册请求模型
    
    用于接收音频文件列表和对应的人员ID，进行声纹特征提取并注册到Milvus向量数据库。
    
    Attributes:
        person_ids: 人员ID列表，需与音频文件列表一一对应
        audio_files: 待注册的音频文件绝对路径列表，每个文件只包含一个说话人
        collection_name: 目标集合名称，默认为配置文件中的'voiceprint_db'
    """
    person_ids: List[str] = Field(..., description="人员ID列表，需与音频文件列表一一对应")
    audio_files: List[str] = Field(..., description="待注册的音频文件绝对路径列表，每个文件只包含一个说话人")
    collection_name: Optional[str] = Field(None, description="目标集合名称，默认为配置文件中的'voiceprint_db'")

    @model_validator(mode='after')
    def check_lists_length(self) -> 'AudioRegistrationRequest':
        """
        验证器：确保人员ID列表和音频文件列表长度一致
        
        Returns:
            AudioRegistrationRequest: 验证后的请求对象
            
        Raises:
            ValueError: 当两个列表长度不一致时抛出
        """
        ids = self.person_ids
        files = self.audio_files
        if ids is not None and files is not None and len(ids) != len(files):
            raise ValueError('人员ID列表和音频文件列表的长度必须相同')
        return self

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        """
        验证器：确保音频文件列表不为空
        
        Args:
            v: 音频文件列表值
            
        Returns:
            List[str]: 验证后的音频文件列表
            
        Raises:
            ValueError: 当音频文件列表为空时抛出
        """
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class AudioRegistrationResponseData(BaseModel):
    """
    音频声纹注册响应数据模型
    
    包含声纹注册操作的结果信息，包括成功插入的记录数量和详细信息。
    
    Attributes:
        collection_name: 数据插入的目标集合名称
        inserted_count: 成功插入的记录数量
        inserted_result: 成功插入的记录详情列表，每条记录包含音频文件路径、人员ID和生成的主键ID
    """
    collection_name: str = Field(..., description="数据插入的目标集合名称")
    inserted_count: int = Field(..., description="成功插入的记录数量")
    inserted_result: List[Dict[str, str]] = Field(
        ..., 
        description="成功插入的记录详情列表，每条记录包含音频文件路径、人员ID和生成的主键ID"
    )