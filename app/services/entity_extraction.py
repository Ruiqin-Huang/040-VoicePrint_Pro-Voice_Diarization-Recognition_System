import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

from app.config.settings import settings

# 默认实体类型
DEFAULT_ENTITY_TYPES = [
    '电话号码','邮箱地址','人名','地名','组织','军衔','呼号','时间','日期','武器类型','车辆类型','任务代号','部队名称','坐标','频率',
    '警报类型','通信指令','装备型号','加密等级','通播组','通信状态','身份验证码','通播等级','接力站点','呼号后缀','报文类型',
    '通播网号','认证口令','通信协议','信道编号','频谱波段','导航点','战位标识','敌我识别码','密语代号','通播站标识','信号强度',
    '干扰情况','时隙分配','网络节点'
]

GEN_TOK = '[GEN]'

# 全局加载模型以提高性能
try:
    tokenizer = AutoTokenizer.from_pretrained(settings.ENTITY_EXTRACTION_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(settings.ENTITY_EXTRACTION_MODEL_PATH)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if torch.cuda.is_available() and settings.USE_GPU:
        model = model.half().cuda(settings.GPU_ID)
    model.eval()
except Exception as e:
    # 如果模型加载失败，则在启动时抛出异常
    raise RuntimeError(f"实体抽取模型加载失败: {e}")

def _parse_result(result_str: str) -> Dict[str, str]:
    """
    将模型输出的字符串解析为字典
    """
    result = {}
    for line in result_str.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result

async def process_entity_extraction(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, str]:
    """
    执行实体抽取
    :param text: 待抽取的文本
    :param entity_types: 自定义的实体类型列表
    :return: 包含抽取结果的字典
    """
    if not entity_types:
        entity_types = DEFAULT_ENTITY_TYPES
    
    labels = '，'.join(entity_types)
    prompt = f'输入: {text}\n抽取: {labels}\n输出: {GEN_TOK}'
    
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
    
    input_ids_len = input_ids.get('input_ids', input_ids).shape[1]
    response_ids = outputs[0][input_ids_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return _parse_result(response)