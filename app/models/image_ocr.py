from typing import List
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

# 请求模型
class ImageOCRRequest(BaseModel):
    files: List[FileRequest]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("文件列表不能为空")
        return v

class OCRTextBox(BaseModel):
    """
    单个文本框的OCR识别结果
    :param text: 识别出的内容类别标签
    :param text: 识别出的文字内容
    :param int: 文本框在文件中的顺序
    :param box: 文本框的矩形边界坐标
    """
    label: str
    text: str
    box: List[int]

class OCRPage(BaseModel):
    """
    单个页面的OCR识别结果
    :param page: 页码
    :param content: 多个文本框识别结果
    :param total_text: 拼接后的总文本
    """
    page: str
    content: List[OCRTextBox]
    total_text: str

class OCResponseData(BaseModel):
    """
    单个文件的OCR识别结果
    :param file_id: 文件ID
    :param file_path: 文件路径
    :param ocr_path: 识别结果文件路径
    """
    file_id: str
    file_path: str
    ocr_path: str
