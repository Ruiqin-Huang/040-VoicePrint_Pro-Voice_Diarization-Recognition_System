#!/usr/bin/env python3
"""
Qwen系列模型自动下载脚本（新版）
支持 Qwen2.5 和 Qwen3 系列最新模型
依赖: pip install huggingface-hub
使用方法:
    python download_hf.py
常见下载问题及解决：
    有时候可能出现下载路径相关的问题，如下：
     ❌ 下载失败: [Errno 2] No such file or directory: '\\\\?\\{path_to_your_project}/pretrained_models/Qwen_Qwen3-4B-Instruct-2507/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6.incomplete'
    可以通过更新huggingface-hub版本来解决
    pip install --upgrade huggingface-hub
"""

import os
from pathlib import Path
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# ================= 配置区域 =================

# 设置国内用户镜像（可选）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 国内镜像
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'    # 官方源

# 模型列表配置
MODELS = {
    # Qwen2.5 系列 Instruct
    # "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    # "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    # "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    # "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",

    # Qwen3 系列
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


# ================= 工具函数 =================

def download_model(model_id, local_dir, max_retries=5, sleep_time=10):
    """
    下载单个模型，支持断点续传和重试。
    """
    print(f"\n▶️ 开始下载模型: {model_id}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    force_download_next = False

    for attempt in range(max_retries):
        try:
            print(f"  第 {attempt + 1}/{max_retries} 次尝试...")
            if force_download_next:
                print("  ⚠️ 检测到问题，本次尝试将强制重新下载(force_download=True)。")

            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                force_download=force_download_next
            )
            print(f"✅ 模型 {model_id} 下载成功！保存到 {local_dir}")
            return True

        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            if attempt < max_retries - 1:
                print(f"  ⏳ {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
                force_download_next = True  # 下次强制下载

    print(f"❌ 模型 {model_id} 在尝试 {max_retries} 次后仍未下载成功。")
    return False


def main():
    print("=== Qwen系列模型下载工具（新版，无 CLI 依赖） ===\n")

    # 显示可用模型
    print("可用的Qwen模型列表:")
    for i, (key, model_id) in enumerate(MODELS.items(), 1):
        print(f"{i}. {key} -> {model_id}")

    # 用户选择
    try:
        choices = input("\n请选择要下载的模型编号(多个用空格隔开，全部下载输入all): ").strip()

        if choices.lower() == 'all':
            selected_models = list(MODELS.values())
        else:
            selected_indices = [int(x) for x in choices.split()]
            selected_models = []
            for idx in selected_indices:
                if 1 <= idx <= len(MODELS):
                    model_key = list(MODELS.keys())[idx - 1]
                    selected_models.append(MODELS[model_key])
                else:
                    print(f"警告: 无效的编号 {idx}，已跳过")

        if not selected_models:
            print("未选择任何模型，退出程序")
            return

        # 确认选择
        print(f"\n即将下载以下模型:")
        for model in selected_models:
            print(f"  - {model}")

        confirm = input("\n确认下载? (y/N): ").lower()
        if confirm != 'y':
            print("下载已取消")
            return

        # 开始下载
        base_dir = "./pretrained_models"
        Path(base_dir).mkdir(exist_ok=True)

        success_count = 0
        for model_id in selected_models:
            local_dir_name = model_id.replace('/', '_')
            local_path = os.path.join(base_dir, local_dir_name)

            if download_model(model_id, local_path, max_retries=10):
                success_count += 1

        print(f"\n=== 下载完成 ===")
        print(f"成功下载: {success_count}/{len(selected_models)} 个模型")
        print(f"模型保存路径: {os.path.abspath(base_dir)}")

    except KeyboardInterrupt:
        print("\n用户中断下载")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()