import os
import asyncio
import argparse
from typing import List
from pathlib import Path

from app.services.translation import process_translation
from app.models.translation import FileRequest

async def translate_files(file_paths: List[str], source_lang: str, target_lang: str, model_type: str):
    """封装多文件翻译逻辑"""
    file_requests = [
        FileRequest(
            id=f"{i}",
            file_path=path.strip()
        )
        for i, path in enumerate(file_paths, 1)
    ]
    
    result = await process_translation(file_requests, source_lang, target_lang, model_type)
    
    if result["invalid_files"]:
        raise RuntimeError(f"{len(result['invalid_files'])}个文件翻译失败")
    return result["processed_files"]

def print_results(results: List[dict]):
    """格式化打印翻译结果"""
    print("\n翻译结果:")
    for i, item in enumerate(results, 1):
        print(f"\n[{i}] 文件: {item['file_path']}")
        print(f"   源语言: {item['source_lang_name']} ({item['source_lang']})")
        print(f"   目标语言: {item['target_lang_name']} ({item['target_lang']})")
        print(f"   模型: {item['model_name']}")
        print("原文:")
        print(f"   {item['source_text']}")
        print("译文:")
        print(f"   {item['translated_text']}")

def main():
    parser = argparse.ArgumentParser(
        description="多语言文件翻译工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s -f text1.txt,text2.txt -s en -t zh
  %(prog)s -f *.txt -s ja -t zh --json
  %(prog)s -f "http://example.com/doc.txt" -s ru -t en"""
    )
    
    # 必需参数
    parser.add_argument(
        "-f", "--files",
        required=True,
        help="要翻译的文件路径列表（逗号分隔，支持本地文件和URL）"
    )
    parser.add_argument(
        "-s", "--source-lang",
        required=True,
        choices=['zh', 'en', 'ru', 'ja', 'mn'],
        help="源语言代码 (zh:中文, en:英文, ru:俄文, ja:日文, mn:蒙文)"
    )
    parser.add_argument(
        "-t", "--target-lang",
        required=True,
        choices=['zh', 'en', 'ru', 'ja', 'mn'],
        help="目标语言代码"
    )

    # 可选参数
    parser.add_argument(
        "--model-type",
        choices=['m2m100', 'small100'],
        default='m2m100',
        help="使用的模型"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出JSON格式结果"
    )
    
    args = parser.parse_args()

    try:
        # 处理输入文件列表
        input_files = [f.strip() for f in args.files.split(",") if f.strip()]
        if not input_files:
            raise ValueError("必须提供至少一个有效文件路径")
            
        # 执行翻译
        results = asyncio.run(
            translate_files(input_files, args.source_lang, args.target_lang, args.model_type)
        )
        
        # 输出结果
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