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

# from app.config.settings import settings
# from app.models.file_request import FileRequest
from utils.io_suppressor import suppress_stdout_stderr

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
    """检查路径是否为URL（与原架构一致）"""
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

async def download_file(url: str) -> str:
    """下载文件到临时目录（支持图片格式）"""
    try:
        # 将阻塞的requests调用放到线程池中执行，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        
        def _download():
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
        
        local_path = await loop.run_in_executor(None, _download)
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
    :return: 结构化识别结果
    """
    try:
        # 全局加载模型
        global OCR_ENGINE
        
        # 将阻塞的同步调用放到线程池中执行，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, OCR_ENGINE.predict, image_path)

        # 结构化输出
        formatted_results = []
        lines = []

        for i, page in enumerate(result, start=1):  # 支持多页文档
            formatted_page = []
            # page.print()

            # if hasattr(page, '__dict__'):
            #     print("Yes, 对象属性:", vars(page))  # 显示所有实例变量
            # else:
            #     print("No, 对象属性:", dir(page))  # 显示所有方法和属性

            for item in page['parsing_res_list']:
                
                formatted_page.append({
                    "label": item.label,
                    "text": item.content,
                    # "order": item['block_order'],
                    "box": item.bbox
                })

                if item.label == 'doc_title':
                    lines.append(item.content)
                    lines.append("")
                else:
                    if lines:
                        lines[-1] += item.content + " "
                    else:
                        lines.append(item.content)
                
            formatted_results.append({
                "page": i,
                "content": formatted_page,
                "total_text": "\n".join(lines)
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

    # 创建输出目录
    output_dir = "./data/output/image_ocr"
    os.makedirs(output_dir, exist_ok=True)

    for file_request in tqdm(file_requests, desc="Processing OCR files"):
        try:
            file_id = file_request['id']
            file_path = file_request['file_path']
            
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

            # 生成输出文件名
            filename = os.path.basename(file_path)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)

            output_data = {
                "file_id": file_id,
                "file_path": file_path,
                "ocr_results": ocr_results
            }
                        
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            processed_files.append({
                "file_id": file_id,
                "file_path": file_path,
                "ocr_path": output_path
            })
            # print(processed_files)

            # 显存释放
            del ocr_results, output_data
            paddle.device.cuda.synchronize()  # 等待GPU完成
            paddle.device.cuda.empty_cache()
            gc.collect()
            await asyncio.sleep(0.05)
            
        except Exception as e:
            invalid_files.append(f"{file_path}: {str(e)}")

        finally:
            # 确保不会在下一轮保留上次的Tensor
            paddle.device.cuda.synchronize()
            paddle.device.cuda.empty_cache()
            gc.collect()
    
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

async def read_stdin_line():
    """异步读取stdin的一行，避免阻塞事件循环"""
    loop = asyncio.get_event_loop()
    # 使用线程池执行阻塞的stdin读取
    return await loop.run_in_executor(None, sys.stdin.readline)

async def main():
    # # 使用示例
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

    # 使用异步循环替代同步的for循环，避免阻塞事件循环
    while True:
        try:
            line = await read_stdin_line()
            if not line:
                # stdin关闭时退出
                break
            if not line.strip():
                continue
                
            files = json.loads(line.strip())
            # print("Received OCR request for files:", files, flush=True)
            processed_files, invalid_files = await process_ocr_files(files)

            response = {
                "status": "ok",
                "processed_files": processed_files,
                "invalid_files": invalid_files
            }
        except Exception as e:
            response = {
                "status": "error",
                "msg": str(e)
            }
        
        finally:
            # 每次OCR任务结束后，强制释放显存
            gc.collect()
        try:
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()
        except Exception:
            pass

        # 写回主进程
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())