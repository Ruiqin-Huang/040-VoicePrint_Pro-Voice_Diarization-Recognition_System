"""
图像OCR数据模型模块

定义图像OCR识别相关的Pydantic数据模型，用于API请求和响应的数据验证和序列化。

包含以下模型：
- ImageOCRRequest: OCR请求模型
- OCRTextBox: 单个文本框识别结果
- OCRPage: 单页OCR识别结果
- OCResponseData: OCR响应数据

依赖：
- pydantic: 数据验证和序列化
- typing: 类型注解
"""

from typing import List
from pydantic import BaseModel, validator
from app.models.file_request import FileRequest

# 请求模型
class ImageOCRRequest(BaseModel):
    """
    图像OCR识别请求模型
    
    Attributes:
        files: 要进行OCR识别的文件列表
    """
    files: List[FileRequest]
    
    @validator('files')
    def files_must_not_be_empty(cls, v):
        """
        验证器：确保文件列表不为空
        
        Args:
            v: 文件列表值
            
        Returns:
            List[FileRequest]: 验证后的文件列表
            
        Raises:
            ValueError: 当文件列表为空时抛出
        """
        if not v:
            raise ValueError("文件列表不能为空")
        return v

class OCRTextBox(BaseModel):
    """
    单个文本框的OCR识别结果模型
    
    Attributes:
        label: 识别出的内容类别标签（如'text', 'title', 'table'等）
        text: 识别出的文字内容
        box: 文本框的矩形边界坐标 [x1, y1, x2, y2]
    """
    label: str
    text: str
    box: List[int]

class OCRPage(BaseModel):
    """
    单个页面的OCR识别结果模型
    
    Attributes:
        page: 页码（字符串格式）
        content: 该页包含的多个文本框识别结果列表
        total_text: 该页所有文本拼接后的完整文本
    """
    page: str
    content: List[OCRTextBox]
    total_text: str

class OCResponseData(BaseModel):
    """
    单个文件的OCR识别结果响应模型
    
    Attributes:
        file_id: 文件的唯一标识符
        file_path: 原始文件路径
        ocr_path: OCR识别结果JSON文件的保存路径
    """
    file_id: str
    file_path: str
    ocr_path: str
