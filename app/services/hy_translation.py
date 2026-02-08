"""
混元大模型翻译服务

提供混元MT1.5翻译模型的核心业务逻辑。
- 模型加载和缓存管理
- 文本翻译功能
- GPU显存管理

启动接口定义在 app.api.v1.endpoints.hy_translation
启动服务定义在 app.hy_main
"""

import gc
from typing import List
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

from app.config.settings import settings

# 全局模型缓存
_MODEL = None
_TOKENIZER = None

def load_model():
    """
    加载混元MT1.5模型和分词器
    
    Returns:
        Tuple: (model, tokenizer)
    """
    global _MODEL, _TOKENIZER
    
    if _MODEL is None or _TOKENIZER is None:
        print("Loading Hunyuan MT1.5 model...")
        model_dir = settings.TRANSLATION_HY_MT15_CACHE_DIR
        
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map=f"cuda:{settings.GPU_ID}"
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
        print("Model loaded successfully!")
    
    return _MODEL, _TOKENIZER

def clear_models():
    """
    清除模型和分词器，释放GPU显存
    """
    global _MODEL, _TOKENIZER
    
    if _MODEL is not None:
        del _MODEL
        _MODEL = None
    
    if _TOKENIZER is not None:
        del _TOKENIZER
        _TOKENIZER = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

async def _translate_text(text: str) -> str:
    """
    翻译单条文本的内部函数
    
    Args:
        text: 待翻译文本
    
    Returns:
        str: 翻译后的文本
    """
    model, tokenizer = load_model()

    # 构建翻译提示
    messages = [
        {
            "role": "user",
            "content": f"将以下文本翻译为目标语言，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        },
    ]
    
    # 应用聊天模板并获取token
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    
    # 生成翻译
    outputs = model.generate(
        tokenized_chat.to(model.device),
        max_new_tokens=2048
    )
    
    # 解码输出
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

async def process_translation(
    texts: List[str],
    source_lang: str,
    target_lang: str
) -> List[dict]:
    """
    批量翻译文本（服务层）。

    Args:
        texts: 待翻译文本列表
        source_lang: 源语言代码
        target_lang: 目标语言代码

    Returns:
        List[dict]: 翻译结果列表（字典形式，供API层组装响应模型）
    """

    results: List[dict] = []
    for text in texts:
        translated_text = await _translate_text(text)
        results.append({
            "source_lang": source_lang,
            "source_lang_name": settings.LANG_DICT.get(source_lang, "其他语言"),
            "source_text": text,
            "target_lang": target_lang,
            "target_lang_name": settings.LANG_DICT.get(target_lang, "其他语言"),
            "translated_text": translated_text,
            "model_name": "Tencent-Hunyuan/HY-MT1.5-1.8B"
        })

    return results
