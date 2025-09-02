import torch
from typing import List, Dict, Optional
import asyncio

# 复用实体抽取服务中已加载的模型和分词器
from app.services.entity_extraction import model, tokenizer
from app.models.event_argument_extraction import EventInfo

DEFAULT_ARGUMENT_TYPES = ['主体', '客体', '时间', '地点', '时态']
GEN_TOK = '[GEN]'

def _parse_result(result_str: str, event_type: str) -> Dict[str, any]:
    """
    将模型输出的字符串解析为字典，并格式化首个键。
    """
    result = {
        "trigger": "",
        "arguments": [],
        "event_type": event_type
    }
    lines = result_str.strip().split('\n')
    for idx, line in enumerate(lines):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            # 将首行的key（如“财经事件”）转换为“trigger”
            if idx == 0 and key == f"{event_type}事件":
                result["trigger"] = value
            else:
                # 将其他行转换为argument格式，移除"事件{event_type}/"前缀
                if key.startswith(f"事件{event_type}/"):
                    argument_name = key.replace(f"事件{event_type}/", "")
                else:
                    argument_name = key
                argument = {
                    "name": argument_name,
                    "value": value
                }
                result["arguments"].append(argument)
    return result

async def _extract_single_event(text: str, event_type: str, argument_types: Optional[List[str]] = None) -> Dict[str, any]:
    """
    执行单个事件论元抽取
    :param text: 待抽取的文本
    :param event_type: 目标事件类型
    :param argument_types: 自定义的论元类型列表
    :return: 包含抽取结果的字典
    """
    if not argument_types:
        argument_types = DEFAULT_ARGUMENT_TYPES
    
    # 构造prompt
    labels = [f"{event_type}事件"]
    labels += [f"事件{event_type}/{arg}" for arg in argument_types]
    labels_str = '，'.join(labels)
    prompt = f'输入: {text}\n抽取: {labels_str}\n输出: {GEN_TOK}'
    
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
    
    input_ids_len = input_ids.get('input_ids', input_ids).shape[1]
    response_ids = outputs[0][input_ids_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    parsed_result = _parse_result(response, event_type)
    parsed_result["argument_types"] = argument_types
    return parsed_result

async def process_multi_event_argument_extraction(text: str, events_info: List[EventInfo]) -> List[Dict[str, any]]:
    """
    并行执行多个事件论元抽取任务
    :param text: 待抽取的文本
    :param events_info: 包含多个事件类型和论元类型信息的列表
    :return: 包含所有事件抽取结果的列表
    """
    tasks = [
        _extract_single_event(text, event.event_type, event.argument_types)
        for event in events_info
    ]
    results = await asyncio.gather(*tasks)
    return results