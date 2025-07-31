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
from utils.helpers import format_datetime, generate_phone_number

# 全局加载FastText语种检测模型（建议在应用启动时加载）
LANG_DETECTOR = fasttext.load_model(settings.FASTTEXT_CACHE_DIR)
LANG_DICT = {
    'zh': "中文",
    'en': "英文",
    'ru': "俄文",
    'ja': "日文"
}

async def is_url(path: str) -> bool:
    """检查路径是否为URL"""
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """从URL下载文件到本地临时目录"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # 写入文件
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_filename
    except Exception as e:
        raise Exception(f"下载文件失败: {str(e)}")

def detect_language(text: str):
    """
    FastText语种检测核心函数
    :param text: 输入文本（至少20个字符时效果最佳）
    :return: (语言代码, 置信度)
    """
    # 移除换行符并确保最小长度
    cleaned_text = ' '.join(text.splitlines())[:1000]  # 限制检测长度
    
    if len(cleaned_text) < 10:
        return "unknown", 0.0
    
    # 执行预测
    predictions = LANG_DETECTOR.predict(cleaned_text, k=1)
    # print(predictions)
    lang_code = predictions[0][0].replace('__label__', '')
    # lang_code = predictions[0][0]
    confidence = float(predictions[1][0])
    
    return lang_code, confidence

async def process_text_files(file_requests: List[FileRequest]):
    """
    核心处理函数 - 文本文件语种检测
    :param file_requests: 文件请求列表
    :return: (成功结果列表, 失败文件列表)
    """
    processed_files = []
    invalid_files = []
    temp_files = []

    for file_request in tqdm(file_requests, desc="Processing text files"):
        file_id = file_request.id
        file_path = file_request.file_path
        
        try:
            # 处理URL或本地路径
            local_path = file_path
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)

            # 验证文件存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            # 读取文本内容（自动处理编码）
            with open(local_path, 'rb') as f:
                raw_data = f.read()
                text = raw_data.decode('utf-8', errors='replace').strip()
            
            if not text:
                invalid_files.append(f"空文件: {file_path}")
                continue
            
            # 执行语种检测
            lang, confidence = detect_language(text)
            
            processed_files.append({
                "file_id": file_id,
                "file_path": file_path,
                "language": lang,
                "language_name": LANG_DICT.get(lang, "其他语言"),
                "confidence": float(confidence)  # 确保JSON可序列化
                # "sample_text": text[:200]  # 返回前200字符用于验证
            })
            
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

# file_request = [FileRequest(id=1, file_path='../test.txt')]

# process_text_files(file_request)
# async def main():
#     processed_files, invalid_files = await process_text_files(file_request)
#     print(processed_files)

# # 运行异步主函数
# import asyncio
# asyncio.run(main())