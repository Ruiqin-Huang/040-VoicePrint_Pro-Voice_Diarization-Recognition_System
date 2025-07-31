import os
import asyncio
import argparse
import json
from typing import List
from pathlib import Path

from app.services.image_ocr import process_ocr_files
from app.models.image_ocr import FileRequest

async def detect_ocr_files(file_paths: List[str]):
    """封装多文件OCR识别逻辑"""
    file_requests = [
        FileRequest(
            id=f"{i}",
            file_path=path.strip()
        )
        for i, path in enumerate(file_paths, 1)
    ]
    
    processed, errors = await process_ocr_files(file_requests)
    
    if errors:
        raise RuntimeError("以下文件处理失败:\n" + "\n".join(errors))
    return processed

def print_results(results: List[dict]):
    """格式化打印OCR识别结果"""
    print("\nOCR识别结果:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] 文件: {result['file_path']}")
        print(f"   识别文本总数: {len(result['ocr_results'])}")
        print(f"   提取文本: {result['ocr_results']}")  # 截取预览

def main():
    parser = argparse.ArgumentParser(
        description="OCR 图像文字识别工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s -f image1.jpg,image2.png
  %(prog)s -f https://example.com/image.jpg
  %(prog)s -f "*.png" -j"""
    )
    parser.add_argument(
        "-f", "--files",
        required=True,
        help="要识别的图像路径列表，用逗号分隔（支持本地文件和URL）"
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="输出JSON格式结果"
    )

    args = parser.parse_args()

    try:
        input_files = [f for f in args.files.split(",") if f.strip()]
        if not input_files:
            raise ValueError("未提供有效的图像路径")

        results = asyncio.run(detect_ocr_files(input_files))

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print_results(results)

    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
