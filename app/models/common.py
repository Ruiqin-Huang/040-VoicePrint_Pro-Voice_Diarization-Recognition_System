from typing import Optional, Any, List, Dict
from pydantic import BaseModel, validator

class ResponseResult(BaseModel):
    retcode: int
    msg: str
    data: Optional[Any] = None