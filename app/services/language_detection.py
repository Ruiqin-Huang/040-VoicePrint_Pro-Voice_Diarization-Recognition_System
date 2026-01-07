"""
语种检测服务模块

提供基于FastText的文本语种检测功能，支持中英文等多语言识别。
通过预加载的FastText模型实现高效的语种分类。

主要功能：
- 文本语种检测
- URL文件下载支持
- 批量文本处理
- 结果格式化和验证

依赖：
- fasttext: 语种检测模型
- requests: HTTP请求处理
- tqdm: 进度条显示
- app.config.settings: 配置管理
"""

import os
import uuid
import json
import requests
import tempfile
from tqdm import tqdm  # 添加tqdm导入
from urllib.parse import urlparse
from typing import Any, List, Dict
import fasttext

from app.models.language_detection import FileRequest
from app.config.path_mapper import PathMapper
from app.config.settings import settings

# 全局加载FastText语种检测模型（建议在应用启动时加载）
LANG_DETECTOR = fasttext.load_model(settings.FASTTEXT_CACHE_DIR)
MIN_LEN = 20  # 最小文本长度要求

async def is_url(path: str) -> bool:
    """
    检查给定的路径是否为有效的URL

    Args:
        path: 要检查的路径字符串

    Returns:
        bool: 如果是URL返回True，否则False
    """
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """
    从URL异步下载文件到临时目录

    Args:
        url: 文件的URL地址

    Returns:
        str: 下载后的本地文件路径

    Raises:
        Exception: 下载失败时抛出
    """
    try:
        # 发送HTTP请求下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功

        # 创建临时文件路径
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

        # 以二进制模式写入文件
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_filename
    except Exception as e:
        raise Exception(f"下载文件失败: {str(e)}")

def detect_language(text: str):
    """
    使用FastText进行语种检测的核心函数

    Args:
        text: 输入文本（建议至少20个字符以获得最佳效果）

    Returns:
        Tuple[str, float]: (语言代码, 置信度)
    """
    # 清理文本：移除换行符并限制检测长度
    cleaned_text = ' '.join(text.splitlines())[:1000]  # 限制检测长度为1000字符

    # 确保文本长度足够
    if len(cleaned_text) < MIN_LEN:
        repeats = (MIN_LEN // len(cleaned_text)) + 1
        cleaned_text = (cleaned_text + ' ') * repeats

    # 使用全局FastText模型进行预测
    global LANG_DETECTOR
    predictions = LANG_DETECTOR.predict(cleaned_text, k=1)
    # print(predictions)

    # 提取语言代码和置信度
    lang_code = predictions[0][0].replace('__label__', '')
    confidence = float(predictions[1][0])

    return lang_code, confidence

async def process_text_files(file_requests: List[FileRequest]):
    """
    批量处理文本文件的语种检测

    对多个文本进行语种检测，返回检测结果和失败列表。

    Args:
        file_requests: 文件请求列表（实际为文本列表）

    Returns:
        Tuple[List[Dict], List[str]]:
            - 成功检测的结果列表，每个包含language、language_name、confidence
            - 处理失败的文本列表，包含错误信息
    """
    processed_files = []
    invalid_files = []
    temp_files = []  # 存储临时下载的文件（如果有URL）

    # 使用进度条显示处理进度
    for text in tqdm(file_requests, desc="Detecting languages"):
        try:
            # 执行语种检测
            lang, confidence = detect_language(text)

            # 添加检测结果
            processed_files.append({
                "language": lang,
                "language_name": settings.LANG_DICT.get(lang, "其他语言"),
                "confidence": float(confidence)  # 确保JSON可序列化
                # "sample_text": text[:200]  # 返回前200字符用于验证（已注释）
            })

        except Exception as e:
            # 记录处理失败的文本及错误信息
            invalid_files.append(f"{text}: {str(e)}")

    # 清理临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass

    return processed_files, invalid_files

# 示例代码（已注释）
# file_request = [FileRequest(id=1, file_path='../test.txt')]
# process_text_files(file_request)

# async def main():
#     processed_files, invalid_files = await process_text_files(file_request)
#     print(processed_files)

# # 运行异步主函数
# import asyncio
# asyncio.run(main())