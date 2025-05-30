from pydantic import BaseModel

# 请求中的文件对象
class FileRequest(BaseModel):
    id: str
    file_path: str