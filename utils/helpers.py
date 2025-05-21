from datetime import datetime
from typing import List

def format_datetime():
    """返回符合yyyy-MM-dd HH:mm:ss格式的当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_phone_number():
    """生成电话号码"""
    return f"138{datetime.now().strftime('%H%M%S')}"

def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词"""
    return list(set(text.split()[:3])) if text else []

def translate_text(text: str, src_lang: str, tgt_lang: str = "zh") -> str:
    """文本翻译"""
    return f"Translated({src_lang}->{tgt_lang}): {text}"

def get_file_type(speaker_count):
    """根据说话人数量确定语音类别"""
    if speaker_count == 1:
        return "单人"
    elif speaker_count == 2:
        return "双人"
    else:
        return "多人"