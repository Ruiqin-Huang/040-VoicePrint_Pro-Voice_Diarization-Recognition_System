"""
    事件论元抽取demo
    1. 仅文本和事件类型（用默认论元类型）：
    python local/event_argument_extraction.py --text "腾讯股价暴跌，市值蒸发千亿。" --event_type 财经

    2. 指定事件类型和论元类型（使用空格分隔）：
    python local/event_argument_extraction.py --text "腾讯股价暴跌，市值蒸发千亿。" --event_type 财经 --argument_types 主体 客体 时间

    text：待抽取的文本，不能为空
    event_type：事件类型，不能为空
    argument_types：事件论元类型列表（可选），含默认值，可自定义需要抽取的事件论元类型，例如--argument_types 主体 客体 触发词 时间 地点 时态
    text、event_type与argument_types参数说明均支持中文和英文输入

    example：
    1. 中文text
    python local/event_argument_extraction.py --text "腾讯股价暴跌，市值蒸发千亿。" --event_type 财经 --argument_types 主体 客体 时间

    2. 英文text
    python local/event_argument_extraction.py --text "A student submitted the assignment yesterday at the university library in the past tense." --event_type Education
"""

import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_ARGUMENT_TYPES = ['主体', '客体', '时间', '地点', '时态']
MODEL_PATH = './pretrained_models/iic/nlp_seqgpt-560m'
GEN_TOK = '[GEN]'

# 全局加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'
if torch.cuda.is_available():
    model = model.half().cuda()
model.eval()

def extract_event_arguments(text: str, event_type: str, argument_types: list = None) -> str:
    if not event_type:
        raise ValueError("event_type不能为空")
    if argument_types is None or len(argument_types) == 0:
        argument_types = DEFAULT_ARGUMENT_TYPES
    # 构造labels
    labels = [f"{event_type}事件"]
    labels += [f"事件{event_type}/{arg}" for arg in argument_types]
    labels_str = '，'.join(labels)
    prompt = f'输入: {text}\n抽取: {labels_str}\n输出: {GEN_TOK}'
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = input_ids.to(model.device)
    outputs = model.generate(**input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
    input_ids = input_ids.get('input_ids', input_ids)
    outputs = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    return response

def parse_result(result_str, event_type):
    """
    将模型输出的形如 '财经事件: 股价暴跌\n事件财经/主体: 腾讯\n事件财经/客体: 股价\n事件财经/时间: 2024年7月31日'
    解析为字典，将首行key改为“{event_type}事件/事件触发词”
    """
    result = {}
    lines = result_str.strip().split('\n')
    for idx, line in enumerate(lines):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if idx == 0 and key == f"{event_type}事件":
                key = f"{event_type}事件/事件触发词"
            result[key] = value
    return result

def main():
    parser = argparse.ArgumentParser(description="事件论元抽取模型本地demo")
    parser.add_argument('--text', type=str, required=True, help='待抽取文本，不能为空')
    parser.add_argument('--event_type', type=str, required=True, help='事件类型，不能为空')
    parser.add_argument('--argument_types', type=str, nargs='*', default=None, help='事件论元类型列表（可选），如: --argument_types 主体 客体 触发词 时间 地点 时态')
    args = parser.parse_args()

    result_str = extract_event_arguments(args.text, args.event_type, args.argument_types)
    result_dict = parse_result(result_str, args.event_type)
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()