import os
from typing import List, Dict, Any
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from app.services.tokenization_small100 import SMALL100Tokenizer

from app.models.translation import FileRequest
from app.config.settings import settings

# 模型配置
MODEL_CONFIG = {
    "m2m100": {
        "model_name": "facebook/m2m100_418M",
        "class": (M2M100ForConditionalGeneration, M2M100Tokenizer),
        "model_dir": settings.TRANSLATION_M2M100_CACHE_DIR
    },
    "small100": {
        "model_name": "alirezamsh/small100",
        "class": (M2M100ForConditionalGeneration, SMALL100Tokenizer),
        "model_dir": settings.TRANSLATION_SMALL100_CACHE_DIR
    }
}

# 全局模型缓存
_MODELS = {}
# _TOKENIZERS = {}  # 新增tokenizer缓存，用于small100

def load_model(model_type: str = "m2m100"):
    """加载指定类型的模型和tokenizer"""
    if model_type not in _MODELS:
        config = MODEL_CONFIG[model_type]
        model = config["class"][0].from_pretrained(config["model_dir"])
        tokenizer = config["class"][1].from_pretrained(config["model_dir"], clean_up_tokenization_spaces=True)
        _MODELS[model_type] = (model, tokenizer)
    return _MODELS[model_type]

def translate_text(
    text: str, 
    src_lang: str, 
    tgt_lang: str, 
    model_type: str = "m2m100"
) -> str:
    """核心翻译函数"""
    model_data = load_model(model_type)
    
    model, tokenizer = model_data
    # print(model_type, src_lang, tgt_lang, text)
    
    tokenizer.src_lang = src_lang

    if model_type == "m2m100":
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
        )
    else:
        tokenizer.tgt_lang = tgt_lang
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

async def process_translation(
    file_requests: List[FileRequest],
    source_lang: str,
    target_lang: str,
    model_type: str = "m2m100"
) -> Dict[str, Any]:
    """
    文件翻译处理函数
    :param model_type: 可选 "m2m100" 或 "small100"
    """
    processed_files = []
    invalid_files = []

    for file_request in file_requests:
        try:
            with open(file_request.file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                translated = translate_text(text, source_lang, target_lang, model_type)
                
                processed_files.append({
                    "file_id": file_request.id,
                    "file_path": file_request.file_path,
                    "source_lang": source_lang,
                    "source_lang_name": settings.LANG_DICT.get(source_lang, "其他语言"),
                    "source_text": text,
                    "target_lang": target_lang,
                    "target_lang_name": settings.LANG_DICT.get(target_lang, "其他语言"),
                    "translated_text": translated,
                    "model_name": MODEL_CONFIG[model_type]["model_name"]
                })
        except Exception as e:
            invalid_files.append(f"{file_request.file_path}: {str(e)}")
    
    return {"processed_files": processed_files, "invalid_files": invalid_files}

# 使用示例
if __name__ == "__main__":
    # 测试m2m100
    model = "m2m100"
    print(model, "翻译结果:", translate_text("你好。", "zh", "en", model))
    
    # 测试small100 (需先安装ctranslate2: pip install ctranslate2)
    model = "small100"
    print(model, "翻译结果:", translate_text("Hello.", "en", "zh", model))