"""
图像OCR服务模块

该模块提供基于PaddleOCR的图像文字识别功能，支持本地文件和URL图片的异步处理。
主要功能包括：
- OCR引擎初始化
- 图片预处理和文字识别
- 结果格式化和去重
- 异步文件下载和处理

依赖：
- PaddleOCR: 用于OCR识别
- OpenCV: 用于图像预处理
- asyncio: 用于异步处理
"""

import asyncio
import gc
import os
import re
import uuid
import json
import paddle
import requests
import tempfile
from tqdm import tqdm
from urllib.parse import urlparse
from typing import List, Dict, Tuple, Optional
from paddleocr import PaddleOCRVL
import numpy as np
import cv2
import sys
import time

# 导入工具模块
# from app.config.settings import settings
# from app.models.file_request import FileRequest
from utils.io_suppressor import suppress_stdout_stderr

# 定义子进程日志输出函数
def log(msg: str):
    """
    输出OCR工作进程的日志信息
    
    Args:
        msg: 日志消息内容
    """
    print(f"[OCR-Worker][GPU={os.getenv('OCR_WORKER_GPU_ID', '0')}][PID={os.getpid()}] {msg}", file=sys.stderr, flush=True)

# 初始化OCR引擎
try:
    with suppress_stdout_stderr():
        OCR_ENGINE = PaddleOCRVL(
            vl_rec_model_dir='./pretrained_models/paddleocr/PP-OCR-VL_rec', 
            vl_rec_backend='native',
            layout_detection_model_name="PP-DocLayoutV2",
            layout_detection_model_dir="./pretrained_models/paddleocr/PP-OCR-VL_rec/PP-DocLayoutV2",
            format_block_content=True
        )
except Exception as e:
    print(f"OCR Engine failed: {e}", flush=True)
    sys.exit(1)
print("OCR Engine started", flush=True)

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
        RuntimeError: 下载失败时抛出
    """
    try:
        # 获取当前事件循环
        loop = asyncio.get_event_loop()
        
        def _download():
            # 发送HTTP请求下载文件
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 从URL中提取文件扩展名，默认.jpg
            ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
            temp_dir = tempfile.gettempdir()
            # 生成唯一的临时文件路径
            local_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
            
            # 以二进制模式写入文件
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return local_path
        
        # 在线程池中执行阻塞的下载操作
        local_path = await loop.run_in_executor(None, _download)
        return local_path
    except Exception as e:
        raise RuntimeError(f"文件下载失败: {str(e)}")

def preprocess_image(image_path: str):
    """
    对图像进行预处理以提高OCR识别精度
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        numpy.ndarray: 预处理后的二值化图像
        
    Raises:
        ValueError: 无法读取图像时抛出
    """
    # 读取图像文件
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用自适应阈值二值化
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def dedup_repeated_substring(s: str, min_repeat=10, max_unit_len=10):
    """
    清除字符串中的重复子串
    
    通过正则表达式检测并移除重复出现的子串，用于清理OCR识别结果中的重复文本。
    
    Args:
        s: 输入字符串
        min_repeat: 最少重复次数，默认10
        max_unit_len: 最大子串长度，默认10
        
    Returns:
        str: 清理后的字符串
    """
    for k in range(1, max_unit_len + 1):
        # 构建正则表达式模式，匹配k长度的子串重复min_repeat次以上
        pattern = rf'(.{{{k}}})\1{{{min_repeat-1},}}'
        s = re.sub(pattern, r'\1', s)
    return s

async def recognize_text(image_path: str):
    """
    对图像执行OCR文字识别
    
    使用PaddleOCR引擎识别图像中的文字，并将结果结构化为标准格式。
    支持多页文档的处理。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        List[Dict]: 结构化的OCR识别结果，每页包含：
            - page: 页码
            - content: 识别出的文本块列表
            - total_text: 拼接后的完整文本
            
    Raises:
        RuntimeError: OCR处理失败时抛出
    """
    try:
        # 使用全局OCR引擎实例
        global OCR_ENGINE
        
        # 获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        # 在线程池中执行同步的OCR预测，避免阻塞异步事件循环
        result = await loop.run_in_executor(None, OCR_ENGINE.predict, image_path)

        # 初始化结果存储列表
        formatted_results = []
        lines = []

        # 遍历每一页的结果（支持多页文档）
        for i, page in enumerate(result, start=1):
            formatted_page = []
            # page.print()

            # if hasattr(page, '__dict__'):
            #     print("Yes, 对象属性:", vars(page))  # 显示所有实例变量
            # else:
            #     print("No, 对象属性:", dir(page))  # 显示所有方法和属性

            for item in page['parsing_res_list']:
                # 将识别结果格式化为字典
                formatted_page.append({
                    "label": item.label,  # 文本块类型标签
                    "text": item.content,  # 识别出的文字内容
                    "box": item.bbox  # 文本块的边界框坐标
                })

                # 根据标签类型处理文本拼接
                if item.label != 'text':
                    # 非文本块（如标题、表格等）单独成行
                    lines.append(item.content)
                    lines.append("")
                else:
                    # 文本块连续拼接
                    if lines:
                        lines[-1] += item.content + " "
                    else:
                        lines.append(item.content)
            
            # 清理重复文本
            filtered_text = dedup_repeated_substring("\n".join(lines))

            # 添加页面结果到总结果中
            formatted_results.append({
                "page": i,
                "content": formatted_page,
                "total_text": filtered_text
            })

        return formatted_results
    
    except Exception as e:
        raise RuntimeError(f"OCR处理失败: {str(e)}")

async def process_ocr_files(file_requests: List[Dict]):
    """
    批量处理OCR文件请求
    
    对多个文件执行OCR识别，支持本地文件和URL。处理结果保存为JSON文件，
    并返回成功和失败的文件列表。
    
    Args:
        file_requests: 文件请求列表，每个字典包含：
            - id: 文件唯一标识符
            - file_path: 文件路径或URL
            
    Returns:
        Tuple[List[Dict], List[str]]: 
            - 成功处理的文件列表，每个包含file_id, file_path, ocr_path
            - 处理失败的文件列表，包含错误信息
    """
    # 初始化结果列表
    processed_files = []
    invalid_files = []
    temp_files = []  # 存储临时下载的文件路径，用于后续清理

    # 确保输出目录存在
    output_dir = "./data/output/image_ocr"
    os.makedirs(output_dir, exist_ok=True)

    # 使用进度条显示处理进度
    pbar = tqdm(
        file_requests,
        desc=f"[OCR-Worker][PID={os.getpid()}] Processing",
        file=sys.stderr
    )

    # 遍历每个文件请求进行处理
    for file_request in pbar:
        file_id = file_request['id']
        file_path = file_request['file_path']

        # 更新进度条描述
        pbar.set_description(f"[OCR-Worker][PID={os.getpid()}] Processing: {os.path.basename(file_path)}")

        try:
            # 处理URL或本地路径：如果是URL，先下载到本地临时文件
            local_path = file_path
            if await is_url(file_path):
                local_path = await download_file(file_path)
                temp_files.append(local_path)  # 记录临时文件用于后续清理

            # 验证本地文件是否存在
            if not os.path.exists(local_path):
                invalid_files.append(f"文件不存在: {file_path}")
                continue
            
            # 执行OCR识别
            ocr_results = await recognize_text(local_path)

            # 生成输出文件名（保持原文件名，扩展名改为.json）
            filename = os.path.basename(file_path)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)

            # 准备输出数据结构
            output_data = {
                "file_id": file_id,
                "file_path": file_path,
                "ocr_results": ocr_results
            }
                        
            # 将结果保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # 添加到成功处理列表
            processed_files.append({
                "file_id": file_id,
                "file_path": file_path,
                "ocr_path": output_path
            })

            # 释放内存，避免内存泄漏
            del ocr_results, output_data
            
        except Exception as e:
            # 记录处理失败的文件及错误信息
            invalid_files.append(f"{file_path}: {str(e)}")

    # 清理下载的临时文件
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass  # 忽略清理失败的错误
            
    return processed_files, invalid_files

# 可视化OCR结果的调试函数（已注释）
# def visualize_results(image_path: str, result: List[Dict], output_path: str):
#     """可视化OCR结果（调试用）"""
#     image = Image.open(image_path).convert('RGB')
#     boxes = [np.array(item['position'], dtype=np.int32) for item in result]
#     texts = [item['text'] for item in result]
#     scores = [item['confidence'] for item in result]
#     
#     vis = draw_ocr(
#         image, boxes, texts, scores,
#         font_path='./fonts/simfang.ttf'  # 中文字体路径
#     )
#     vis.save(output_path)

async def read_stdin_line():
    """
    异步读取标准输入的一行
    
    使用线程池执行阻塞的stdin读取操作，避免阻塞异步事件循环。
    
    Returns:
        str: 读取的一行内容
    """
    loop = asyncio.get_event_loop()
    # 使用线程池执行阻塞的stdin读取
    return await loop.run_in_executor(None, sys.stdin.readline)

async def main():
    """
    主函数：OCR工作进程的主循环
    
    通过stdin接收文件处理请求，执行OCR处理，并通过stdout返回结果。
    支持作为独立工作进程运行，处理来自父进程的请求。
    """
    # 示例代码（已注释）
    # output = OCR_ENGINE.predict("./data/16372762f1fbface6e8b828ad56e89a9.jpg")
    # print(output)
    # for res in output:
    #     res.print() ## 打印预测的结构化输出
    #     res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    #     res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
    # test_requests = [
    #      FileRequest(id="1", file_path='./readme_assets/说话人分割系统.png')
    # ]
    
    # results, errors = await process_ocr_files(test_requests)
    # print(f"成功: {len(results)} 个, 失败: {len(errors)} 个")
    # print(results)
    
    # # 可视化第一个结果
    # if results:
    #     visualize_results(
    #         results[0]['file_path'],
    #         results[0]['ocr_results'],
    #         "ocr_result_visualization.jpg"
    #     )

    # 主处理循环：持续监听stdin输入
    while True:
        try:
            # 异步读取一行输入
            line = await read_stdin_line()
            if not line:
                # stdin关闭时退出循环
                log("stdin closed, exiting worker loop")
                break
            if not line.strip():
                continue  # 跳过空行
                
            log(f"Current OCR Files: {line}")
            # 解析JSON格式的文件请求
            files = json.loads(line.strip())
            # 执行OCR处理
            processed_files, invalid_files = await process_ocr_files(files)

            # 构建响应数据
            response = {
                "status": "ok",
                "processed_files": processed_files,
                "invalid_files": invalid_files
            }
            log(f"OCR Worker response: {response}")
        except Exception as e:
            log(f"Exception in main loop: {repr(e)}")
            # 构建错误响应
            response = {
                "status": "error",
                "msg": str(e)
            }
        
        finally:
            # 每次任务结束后强制垃圾回收
            gc.collect()
            try:
                # 清理GPU缓存（如果可用）
                paddle.device.cuda.empty_cache()
                paddle.device.cuda.synchronize()
            except Exception:
                pass  # 忽略清理失败

        # 将响应写入stdout，返回给父进程
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

# 程序入口点
if __name__ == "__main__":
    asyncio.run(main())