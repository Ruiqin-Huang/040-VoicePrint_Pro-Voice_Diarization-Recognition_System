from typing import Optional
from pydantic import BaseModel

# 语音识别请求中的文件对象
class RecognitionFileRequest(BaseModel):
    id: str
    file_path: str
    seg_file_path: Optional[str] = None