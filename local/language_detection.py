import os
import asyncio
import argparse
from typing import List
from pathlib import Path

from app.services.language_detection import process_text_files
from app.models.language_detection import FileRequest

async def detect_language_files(file_paths: List[str]):
    """封装多文件检测逻辑"""
    # 自动生成带序号的文件ID (file_001, file_002...)
    file_requests = [
        FileRequest(
            id=f"{i}",  # 按顺序生成ID
            file_path=path.strip()
        ) 
        for i, path in enumerate(file_paths, 1)
    ]
    
    processed, errors = await process_text_files(file_requests)
    
    if errors:
        raise RuntimeError("\n".join(errors))
    return processed

def print_results(results: List[dict]):
    """格式化打印检测结果"""
    print("\n语言检测结果:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] 文件: {result['file_path']}")
        print(f"   语言: {result['language_name']} ({result['language']})")
        print(f"   置信度: {result['confidence']:.1%}")
        # if result['language'] == 'unknown':
        #     print("注意: 检测结果置信度较低")

def main():
    parser = argparse.ArgumentParser(
        description="文本语种检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s -f text1.txt,text2.txt
  %(prog)s -f url1.txt,url2.txt
  %(prog)s -f "*.txt"  # 使用引号防止shell扩展"""
    )
    parser.add_argument(
        "-f", "--files", 
        required=True,
        help="要检测的文件路径列表，用逗号分隔（支持本地文件和URL）"
    )
    parser.add_argument(
        "-j", "--json", 
        action="store_true", 
        help="输出JSON格式结果"
    )
    
    args = parser.parse_args()

    try:
        # 分割输入的文件列表
        input_files = [f for f in args.files.split(",") if f.strip()]
        
        if not input_files:
            raise ValueError("未提供有效的文件路径")
            
        results = asyncio.run(detect_language_files(input_files))
        
        if args.json:
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print_results(results)
            
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()