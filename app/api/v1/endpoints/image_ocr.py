"""
图像OCR API端点模块

提供RESTful API接口用于图像OCR文字识别服务。
通过FastAPI框架实现，支持批量文件处理和错误处理。

主要功能：
- 接收图像OCR请求
- 调用OCR工作进程池进行处理
- 返回结构化的识别结果

依赖：
- FastAPI: Web框架
- app.worker.ocr_worker: OCR工作进程池
- app.models: 数据模型定义
"""

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

# 导入应用模块
from app.models.common import ResponseResult
from app.models.image_ocr import ImageOCRRequest, OCRTextBox, OCRPage, OCResponseData
from app.core.error_codes import ResponseCode
from app.worker.ocr_worker import ocr_worker_pool

# 创建API路由器，设置前缀为/api
router = APIRouter(prefix="/api")

@router.post("/image_ocr",
             response_model=ResponseResult)
async def image_ocr(requests: ImageOCRRequest):
    """
    图片OCR识别API端点
    
    接收图像文件列表，调用OCR服务进行文字识别，并返回识别结果。
    支持本地文件路径和URL，支持批量处理。
    
    Args:
        requests: 包含文件列表的OCR请求对象
        
    Returns:
        ResponseResult: 包含处理结果的响应对象
        
    Raises:
        HTTPException: 当发生内部错误时抛出
    """
    try:
        # 验证请求参数：文件列表不能为空
        if not requests.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )
        
        # 调用OCR工作进程池处理文件
        processed_files, invalid_files = await ocr_worker_pool.send(requests.files)
        
        # 检查处理结果
        if not processed_files:
            if invalid_files:
                # 所有文件都处理失败
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文件处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                # 没有找到可处理的图像文件
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的图像文件",
                    data=None
                )
        
        # 将服务层返回的结果转换为API响应格式
        response_data = []
        for result in processed_files:
            # 注释掉的详细OCR页面结果转换（可选）
            # ocr_pages = []
            # 注释掉的详细OCR页面结果转换（可选）
            # ocr_pages = []
            # for page_result in result["ocr_results"]:
            #     ocr_pages.append(
            #         OCRPage(
            #             page=str(page_result["page"]),
            #             content=[
            #                 OCRTextBox(
            #                     label=box_result["label"],
            #                     text=box_result["text"],
            #                     box=box_result["box"]
            #                 )
            #                 for box_result in page_result["content"]
            #             ],
            #             total_text=page_result["total_text"]
            #         )
            #     )
            
            # 构建简化的响应数据
            response_data.append(
                OCResponseData(
                    file_id=result["file_id"],
                    file_path=result["file_path"],
                    ocr_path=result["ocr_path"]
                )
            )
        
        # 根据处理结果返回相应的响应
        if invalid_files:
            # 部分成功，部分失败
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
            )
        
        # 全部成功
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
    
    except ValueError as e:
        # 参数验证错误
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except Exception as e:
        # 未知错误，记录详细错误信息
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        return ResponseResult(
            retcode=ResponseCode.UNKNOWN_ERROR,
            msg=f"未知错误: {str(e)}",
            data=None
        )