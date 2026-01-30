"""
机器翻译服务模块

提供基于Hugging Face Transformers的机器翻译功能，支持多种语言互译。
通过预加载的翻译模型实现高效的文本翻译。

主要功能：
- 多语言文本翻译
- 支持M2M100和Small100模型
- 模型缓存和重用
- 批量文本处理

依赖：
- transformers: Hugging Face模型库
- app.services.tokenization_small100: Small100分词器
- app.config.settings: 配置管理
"""

import gc
import os
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from app.services.tokenization_small100 import SMALL100Tokenizer

from app.models.translation import FileRequest
from app.config.settings import settings

# 模型配置字典，定义不同模型的配置信息
MODEL_CONFIG = {
    "m2m100": {
        "model_name": "facebook/m2m100_418M",  # M2M100模型名称
        "class": (M2M100ForConditionalGeneration, M2M100Tokenizer),  # M2M100模型和分词器类
        "model_dir": settings.TRANSLATION_M2M100_CACHE_DIR  # M2M100模型缓存目录
    },
    "small100": {
        "model_name": "alirezamsh/small100",  # Small100模型名称
        "class": (M2M100ForConditionalGeneration, SMALL100Tokenizer),  # 使用Small100自定义分词器
        "model_dir": settings.TRANSLATION_SMALL100_CACHE_DIR  # Small100模型缓存目录
    },
    "hy_mt1.5": {
        "model_name": "Tencent-Hunyuan/HY-MT1.5-1.8B",  # HY-MT1.5模型名称
        "model_dir": settings.TRANSLATION_HY_MT15_CACHE_DIR  # Hunyuan MT1.5模型缓存目录
    }
}

# 全局模型缓存，避免重复加载模型
_MODELS = {}
# _TOKENIZERS = {}  # 新增tokenizer缓存，用于small100（已注释）

# 强标点（句子级）
STRONG_PUNCT = set("。！？.!?")

# 弱标点（短语级）
WEAK_PUNCT = set("，,；;：:、")

def clear_models():
    """
    清除所有已加载的模型与 tokenizer，并释放 GPU 显存
    """
    for model, tokenizer in _MODELS.values():
        del model
        del tokenizer
    
    _MODELS.clear()

    # 强制垃圾回收
    gc.collect()

    # 强制垃圾回收 + GPU 显存回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model(model_type: str = "hy_mt1.5"):
    """
    加载指定类型的翻译模型和分词器

    使用缓存机制，如果模型已加载则直接返回，否则从本地加载。

    Args:
        model_type: 模型类型，"m2m100" 或 "small100" 或 "hy_mt1.5"

    Returns:
        Tuple: (model, tokenizer) 已加载的模型和分词器

    Raises:
        KeyError: 当model_type不在配置中时抛出
    """
    if model_type not in _MODELS:
        # 清理其它模型
        clear_models()

        # 获取模型配置
        config = MODEL_CONFIG[model_type]
        if model_type in ["m2m100", "small100"]:
            # 从本地缓存目录加载模型
            model = config["class"][0].from_pretrained(config["model_dir"], local_files_only=True)
            # 加载分词器，清理tokenization空格
            tokenizer = config["class"][1].from_pretrained(config["model_dir"], clean_up_tokenization_spaces=True, local_files_only=True)
        elif model_type == "hy_mt1.5":
            model = AutoModelForCausalLM.from_pretrained(config["model_dir"], device_map=f"cuda:{settings.GPU_ID}")  # You may want to use bfloat16 and/or move to GPU here
            tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
        else:
            raise KeyError(f"未知的模型类型: {model_type}")
        
        # 缓存模型和分词器
        _MODELS[model_type] = (model, tokenizer)
    return _MODELS[model_type]

def _split_text(
    text: str,
    max_len: int = 300
):
    """
    将文本按照最大长度分割成多个片段

    采用智能分割策略，优先在强标点（句子级）处分割，其次在弱标点（短语级）处分割，
    以保持文本的语义完整性。

    Args:
        text: 待分割的文本
        max_len: 单个片段的最大长度，默认为300

    Returns:
        List[str]: 分割后的文本片段列表
    """
    text = text.strip()
    if not text:
        return []

    segments = []

    start = 0
    last_strong = -1
    last_weak = -1

    for i, ch in enumerate(text):
        # 记录最近标点
        if ch in STRONG_PUNCT:
            last_strong = i
        elif ch in WEAK_PUNCT:
            last_weak = i

        # 是否超过长度
        if i - start + 1 >= max_len:
            # 优先强标点
            if last_strong >= start:
                cut = last_strong + 1
            elif last_weak >= start:
                cut = last_weak + 1
            else:
                cut = i + 1

            segments.append(text[start:cut].strip())
            start = cut

            # 重置标点位置（防止越界）
            last_strong = -1
            last_weak = -1

    # 处理尾部
    if start < len(text):
        segments.append(text[start:].strip())

    return segments

def translate_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model_type: str = "hy_mt1.5"
) -> str:
    """
    执行单条文本翻译的核心函数

    Args:
        text: 待翻译的文本
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
        model_type: 使用的模型类型

    Returns:
        str: 翻译后的文本
    """
    # 加载模型和分词器
    model_data = load_model(model_type)

    model, tokenizer = model_data
    # print(model_type, src_lang, tgt_lang, text)

    if model_type == "hy_mt1.5":
        # 混元MT1.5模型的翻译流程
        messages = [
            {"role": "user", "content": f"将以下文本翻译为{tgt_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"},
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )

        outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
        output_text = tokenizer.decode(outputs[0])
    else:
        # 设置源语言
        tokenizer.src_lang = src_lang

        if model_type == "m2m100":
            # M2M100模型的翻译流程
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)  # 强制目标语言开头token
            )
        else:
            # Small100模型的翻译流程
            tokenizer.tgt_lang = tgt_lang
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(
                **inputs
            )
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # 解码输出并返回翻译结果
    return output_text

async def process_translation(
    file_requests: List[FileRequest],
    source_lang: str,
    target_lang: str,
    model_type: str = "hy_mt1.5"
) -> Dict[str, Any]:
    """
    批量处理文本翻译

    对多个文本进行翻译，支持进度条显示。

    Args:
        file_requests: 待翻译的文本列表
        source_lang: 源语言代码
        target_lang: 目标语言代码
        model_type: 使用的模型类型

    Returns:
        Tuple[List[Dict], List[str]]:
            - 成功翻译的结果列表
            - 处理失败的文本列表，包含错误信息
    """
    processed_files = []
    invalid_files = []

    # 使用进度条显示翻译进度
    for text in tqdm(file_requests, desc="Translating text"):
        try:
            if model_type == "hy_mt1.5":
                # 混元MT1.5模型翻译
                translated_text = translate_text(text, source_lang, target_lang, model_type)
            else:
                # 文本切段
                segments = _split_text(text)
                translated_text = ""

                for segment in segments:
                    # 执行翻译
                    translated = translate_text(segment, source_lang, target_lang, model_type)
                    translated_text += translated

            # 构建翻译结果
            processed_files.append({
                "source_lang": source_lang,
                "source_lang_name": settings.LANG_DICT.get(source_lang, "其他语言"),
                "source_text": text,
                "target_lang": target_lang,
                "target_lang_name": settings.LANG_DICT.get(target_lang, "其他语言"),
                "translated_text": translated_text,
                "model_name": MODEL_CONFIG[model_type]["model_name"]
            })
        except Exception as e:
            # 记录翻译失败的文本及错误信息
            invalid_files.append(f"{text}: {str(e)}")

    return processed_files, invalid_files

# 使用示例代码（仅用于测试）
if __name__ == "__main__":
    # 测试m2m100模型
    model = "m2m100"
    print(model, "翻译结果:", translate_text("你好。", "zh", "en", model))

    # 测试small100模型 (需先安装ctranslate2: pip install ctranslate2)
    model = "small100"
    print(model, "翻译结果:", translate_text("Hello.", "en", "zh", model))