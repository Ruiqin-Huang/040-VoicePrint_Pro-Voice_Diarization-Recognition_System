from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Literal

class ResponseResult(BaseModel):
    retcode: int
    msg: str
    data: Optional[Any] = None
    
class ModelInfo(BaseModel):
    model_call_type: Literal['local_hf', 'ollama_api'] = Field(..., description="'local_hf' 或 'ollama_api'")
    model_name: Optional[str] = Field(None, description="当 model_call_type 为 'ollama_api' 时必须提供")
    api_address: Optional[str] = Field(None, description="当 model_call_type 为 'ollama_api' 时必须提供")
    model_dir: Optional[str] = Field(None, description="当 model_call_type 为 'local_hf' 时必须提供")

    @validator('model_name', 'api_address', always=True)
    def check_ollama_fields(cls, v, values):
        if values.get('model_call_type') == 'ollama_api' and not v:
            raise ValueError("当 model_call_type 为 'ollama_api' 时, model_name 和 api_address 不能为空")
        return v

    @validator('model_dir', always=True)
    def check_local_hf_fields(cls, v, values):
        if values.get('model_call_type') == 'local_hf' and not v:
            raise ValueError("当 model_call_type 为 'local_hf' 时, model_dir 不能为空")
        return v