import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
import traceback
from typing import List

from app.models.common import ResponseResult
from app.models.image_ocr import ImageOCRRequest, OCRTextBox, OCRPage, OCResponseData
from app.core.error_codes import ResponseCode
from app.worker.ocr_worker import ocr_worker_pool

router = APIRouter(prefix="/api")

@router.post("/image_ocr", 
             response_model=ResponseResult)
async def image_ocr(requests: ImageOCRRequest):
    """图片OCR识别API - 识别图片上的文本"""
    try:
        if not requests.files:
            return ResponseResult(
                retcode=ResponseCode.INVALID_PARAM,
                msg="文件列表不能为空",
                data=None
            )
        
        processed_files, invalid_files = await ocr_worker_pool.send(requests.files)
        
        if not processed_files:
            if invalid_files:
                return ResponseResult(
                    retcode=ResponseCode.OPERATION_ERROR,
                    msg=f"所有文件处理失败: {'; '.join(invalid_files)}",
                    data=None
                )
            else:
                return ResponseResult(
                    retcode=ResponseCode.INVALID_PARAM,
                    msg="没有找到可处理的图像文件",
                    data=None
                )
        
        # 将服务层返回的结果转换为响应格式
        response_data = []
        for result in processed_files:
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
            
            response_data.append(
                OCResponseData(
                    file_id=result["file_id"],
                    file_path=result["file_path"],
                    ocr_path=result["ocr_path"]
                )
            )
        
        if invalid_files:
            return ResponseResult(
                retcode=ResponseCode.SUCCESS,
                msg=f"部分文件处理成功，{len(invalid_files)}个文件失败: {'; '.join(invalid_files)}",
                data=response_data
            )
        
        return ResponseResult(
            retcode=ResponseCode.SUCCESS,
            msg="success",
            data=response_data
        )
    
    except ValueError as e:
        return ResponseResult(
            retcode=ResponseCode.INVALID_PARAM,
            msg=f"参数错误: {str(e)}",
            data=None
        )
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        
        return ResponseResult(
            retcode=ResponseCode.UNKNOWN_ERROR,
            msg=f"未知错误: {str(e)}",
            data=None
        )