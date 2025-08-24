"""
 python utils/milvus_query_all.py voiceprint_db_dev
"""

import os
import sys
import argparse

# 将项目根目录添加到Python路径中
# __file__ -> .../utils/milvus_query_all.py
# os.path.dirname(__file__) -> .../utils
# os.path.dirname(os.path.dirname(__file__)) -> .../ (项目根目录)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pymilvus import Collection
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

from utils.Milvus import MilvusClient
from app.config.settings import settings

def query_and_print(collection_name: str):
    """
    连接到Milvus，查询指定集合的所有条目，并以表格形式打印。
    """
    console = Console()

    try:
        # --- 1. 连接到 Milvus ---
        console.print(f"[cyan]正在连接到 Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})...[/cyan]")
        mc = MilvusClient(config={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT})

        if not mc.has_collection(collection_name):
            console.print(f"[bold red]错误: 集合 '{collection_name}' 不存在。[/bold red]")
            return

        collection = Collection(name=collection_name)
        
        # --- 2. 查询所有数据 ---
        # 使用一个始终为真的表达式来查询所有实体
        # 我们需要知道所有字段名才能查询
        primary_key_field = next((field.name for field in collection.schema.fields if field.is_primary), None)
        if not primary_key_field:
            console.print(f"[bold red]错误: 无法在集合 '{collection_name}' 中找到主键字段。[/bold red]")
            return
        
        # 使用一个始终为真的表达式来查询所有实体
        # 我们需要知道所有字段名才能查询
        schema_fields = [field.name for field in collection.schema.fields]
        
        # 排除向量字段以提高可读性，如果需要可以加入
        output_fields = [field for field in schema_fields if field != "embedding"]
        # output_fields = [field for field in schema_fields]

        console.print(f"[cyan]正在从集合 '{collection_name}' 查询所有条目 (使用主键: '{primary_key_field}')...[/cyan]")
        
        # 通过一个简单的表达式获取所有数据
        # 注意：这个表达式假设主键是数值类型。对于字符串类型的主键，可以用 "pk != ''"
        results = collection.query(
            expr=f"{primary_key_field} >= 0",
            output_fields=output_fields
        )
        
        if not results:
            console.print(f"[yellow]集合 '{collection_name}' 中没有找到任何数据。[/yellow]")
            return

        # --- 3. 以美观的表格打印结果 ---
        table = Table(
            title=f"Milvus 集合 '{collection_name}' 中的所有条目 ({len(results)} 条)",
            box=ROUNDED,
            header_style="bold magenta",
            show_lines=True
        )

        # 添加表头
        for field in output_fields:
            table.add_column(field, justify="left")

        # 添加数据行
        for item in results:
            # 将每行的数据转换为字符串
            row_data = [str(item.get(field, "N/A")) for field in output_fields]
            table.add_row(*row_data)

        console.print(table)
        console.print(f"[green]查询完成。共找到 {len(results)} 条记录。[/green]")

    except Exception as e:
        console.print(f"[bold red]发生错误: {e}[/bold red]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Milvus 集合中查询并打印所有条目。")
    parser.add_argument(
        "collection_name", 
        type=str, 
        help="要查询的 Milvus 集合的名称。"
    )
    args = parser.parse_args()
    
    query_and_print(args.collection_name)