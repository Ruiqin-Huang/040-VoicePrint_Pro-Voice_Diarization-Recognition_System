from typing import List, Optional
from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict

class AudioRegistrationRequest(BaseModel):
    person_ids: List[str] = Field(..., description="人员ID列表，需与音频文件列表一一对应")
    audio_files: List[str] = Field(..., description="待注册的音频文件绝对路径列表，每个文件只包含一个说话人")
    collection_name: Optional[str] = Field(None, description="目标集合名称，默认为配置文件中的'voiceprint_db'")

    @model_validator(mode='after')
    def check_lists_length(self) -> 'AudioRegistrationRequest':
        ids = self.person_ids
        files = self.audio_files
        if ids is not None and files is not None and len(ids) != len(files):
            raise ValueError('人员ID列表和音频文件列表的长度必须相同')
        return self

    @validator('audio_files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("音频文件列表不能为空")
        return v

class AudioRegistrationResponseData(BaseModel):
    collection_name: str = Field(..., description="数据插入的目标集合名称")
    inserted_count: int = Field(..., description="成功插入的记录数量")
    inserted_result: List[Dict[str, str]] = Field(
        ..., 
        description="成功插入的记录详情列表，每条记录包含音频文件路径、人员ID和生成的主键ID"
    )