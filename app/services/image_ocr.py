import os
import uuid
import json
import requests
import tempfile
from tqdm import tqdm
from urllib.parse import urlparse
from typing import List, Dict, Tuple
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

from app.config.settings import settings
from utils.io_suppressor import suppress_stdout_stderr

with suppress_stdout_stderr():
    # 全局加载PaddleOCR模型（建议单例）
    OCR_ENGINE = PaddleOCR(
        text_detection_model_dir='./pretrained_models/paddleocr/PP-OCRv5_server_det',
        text_recognition_model_dir='./pretrained_models/paddleocr/PP-OCRv5_server_rec',
        # textline_orientation_model_dir='./pretrained_models/paddleocr/PP-LCNet_x1_0_textline_ori',
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device="gpu" if settings.USE_GPU else "cpu"
        # ocr_version="PP-OCRv5"  # 明确指定版本
    )

async def is_url(path: str) -> bool:
    """检查路径是否为URL（与原架构一致）"""
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """下载文件到临时目录（支持图片格式）"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return local_path
    except Exception as e:
        raise RuntimeError(f"文件下载失败: {str(e)}")

def preprocess_image(image_path: str):
    """图像预处理（增强OCR精度）"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 灰度化 + 自适应阈值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

async def recognize_text(image_path: str):
    """
    执行OCR识别
    :param image_path: 图片路径
    :param lang: 语言类型 ('ch', 'en', 'japan', 'ru'等)
    :return: 结构化识别结果
    """
    try:
        # 动态切换语言模型
        global OCR_ENGINE
        
        result = OCR_ENGINE.predict(input=image_path)
        # for res in result:
        #     res.print()

        # 结构化输出
        formatted_results = []
        for i, page in enumerate(result, start=1):  # 支持多页文档
            formatted_page = []
            # page.print()

            # if hasattr(page, '__dict__'):
            #     print("Yes, 对象属性:", vars(page))  # 显示所有实例变量
            # else:
            #     print("No, 对象属性:", dir(page))  # 显示所有方法和属性

            for item in page['rec_texts']:
                # 获取对应文本的索引
                idx = page['rec_texts'].index(item)
                
                formatted_page.append({
                    "text": item,
                    "confidence": page['rec_scores'][idx],
                    "position": page['rec_polys'][idx].tolist(),
                    "box": page['rec_boxes'][idx].tolist()
                })
            
            formatted_results.append({
                "page": i,
                "content": formatted_page,
                "total_text": " ".join([r["text"] for r in formatted_page])
            })
        
        return formatted_results
    
    except Exception as e:
        raise RuntimeError(f"OCR处理失败: {str(e)}")

async def process_ocr_files(file_requests: List[Dict]):
    """
    处理OCR文件的主函数
    :param file_requests: 文件请求列表，每个元素需包含id和file_path
    :return: (成功结果列表, 失败文件列表)
    """
    processed_files = []
    invalid_files = []
    temp_files = []

    for file_request in tqdm(file_requests, desc="Processing OCR files"):
        try:
            file_id = file_request.id
            file_path = file_request.file_path
            
            # 处理URL或本地路径
            local_path = file_path
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)

            # 验证文件存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            # 执行OCR
            ocr_results = await recognize_text(local_path)
            
            processed_files.append({
                "file_id": file_id,
                "file_path": file_path,
                "ocr_results": ocr_results
            })
            # print(processed_files)
            
        except Exception as e:
            invalid_files.append(f"{file_path}: {str(e)}")
    
    # 清理临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
    return processed_files, invalid_files

# def visualize_results(image_path: str, result: List[Dict], output_path: str):
#     """可视化OCR结果（调试用）"""
#     image = Image.open(image_path).convert('RGB')
#     boxes = [np.array(item['position'], dtype=np.int32) for item in result]
#     texts = [item['text'] for item in result]
#     scores = [item['confidence'] for item in result]
    
#     vis = draw_ocr(
#         image, boxes, texts, scores,
#         font_path='./fonts/simfang.ttf'  # 中文字体路径
#     )
#     vis.save(output_path)

# # 使用示例
# async def main():
#     test_requests = [
#          FileRequest(id=1, file_path='./readme_assets/说话人分割系统.png')
#     ]
    
#     results, errors = await process_ocr_files(test_requests)
#     print(f"成功: {len(results)} 个, 失败: {len(errors)} 个")
    
#     # # 可视化第一个结果
#     # if results:
#     #     visualize_results(
#     #         results[0]['file_path'],
#     #         results[0]['ocr_results'],
#     #         "ocr_result_visualization.jpg"
#     #     )

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())