from typing import List
from pydantic import BaseModel
from app.models.file_request import FileRequest

class OCRTextBox(BaseModel):
    """
    单个文本框的OCR识别结果
    :param text: 识别出的文字内容
    :param confidence: 置信度（0-1之间的浮点数）
    :param position: 文字区域坐标（多边形点的列表）
    :param language: 使用的语言模型（如 'ch', 'en' 等）
    :param box: 文本框的矩形边界（四个点）
    """
    text: str
    confidence: float
    position: List[List[float]]
    language: str
    box: List[List[float]]


class OCRResult(BaseModel):
    """
    单个文件的OCR识别结果
    :param file_id: 文件ID
    :param file_path: 文件路径
    :param ocr_results: 多个文本框识别结果
    :param total_text: 拼接后的总文本
    """
    file_id: str
    file_path: str
    ocr_results: List[OCRTextBox]
    total_text: str


class OCRResponse(BaseModel):
    """
    OCR处理响应结果
    :param processed_files: 成功识别的文件列表
    :param invalid_files: 无法处理的文件路径
    """
    processed_files: List[OCRResult]
    invalid_files: List[str]
