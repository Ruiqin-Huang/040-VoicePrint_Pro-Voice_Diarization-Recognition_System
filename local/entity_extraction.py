"""
    实体抽取demo
    1. 仅文本（用默认实体类型）：
    python local/entity_extraction.py --text "张三出生于北京，任职于百度"
    2. 指定实体类型（使用空格分隔）：
    python local/entity_extraction.py --text "张三出生于北京，任职于百度" --entity_types 时间 胜者 败者 赛事名称

    text：待抽取的文本，不能为空
    entity_types：实体类型列表（可选），含默认值，可自定义需要抽取的实体类型，例如--entity_types 天气 人名 时间 组织 公司
    text与entity_types参数说明均支持中文和英文输入
    
    example：
    1. 中文text
    python local/entity_extraction.py --text "龙井茶，浙江省特产，中国国家地理标志产品。特级龙井茶扁平光滑挺直，色泽嫩绿光润，香气鲜嫩清高，滋味鲜爽甘醇，叶底细嫩呈朵。2001年，国家质监总局正式批准“龙井茶”为原产地域保护产品。" --entity_types 颜色 组织 产品名称
    2. 英文text
    python local/entity_extraction.py --text "Falcon One to Command Center, this is Lieutenant John Smith reporting from the Marine Corps, urgent request: we are under enemy ambush at Grid Point Three in Baghdad, time 13:30, date July 31, 2024, hostile forces with AK-47 fire damaging our Humvee convoy, Operation Thunderstorm ongoing, Command Center please authorize air strike support, over; Command Center to Falcon One, roger, confirming Grid Point Three Baghdad, time 13:35, Colonel Lee of the 10th Division dispatching AH-64 Apache support immediately, maintain comms link for successful Operation Thunderstorm completion."
    python local/entity_extraction.py --text "Falcon One to Command Center, this is Lieutenant John Smith reporting from the Marine Corps, urgent request: we are under enemy ambush at Grid Point Three in Baghdad, time 13:30, date July 31, 2024, hostile forces with AK-47 fire damaging our Humvee convoy, Operation Thunderstorm ongoing, Command Center please authorize air strike support, over; Command Center to Falcon One, roger, confirming Grid Point Three Baghdad, time 13:35, Colonel Lee of the 10th Division dispatching AH-64 Apache support immediately, maintain comms link for successful Operation Thunderstorm completion." --entity_types 人名 组织 军衔
    
""" 

import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_ENTITY_TYPES = [
    '电话号码','邮箱地址','人名','地名','组织','军衔','呼号','时间','日期','武器类型','车辆类型','任务代号','部队名称','坐标','频率',
    '警报类型','通信指令','装备型号','加密等级','通播组','通信状态','身份验证码','通播等级','接力站点','呼号后缀','报文类型',
    '通播网号','认证口令','通信协议','信道编号','频谱波段','导航点','战位标识','敌我识别码','密语代号','通播站标识','信号强度',
    '干扰情况','时隙分配','网络节点'
]

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

def extract_entities(text: str, entity_types: list = None) -> str:
    if entity_types is None or len(entity_types) == 0:
        entity_types = DEFAULT_ENTITY_TYPES
    labels = '，'.join(entity_types)
    prompt = f'输入: {text}\n抽取: {labels}\n输出: {GEN_TOK}'
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = input_ids.to(model.device)
    outputs = model.generate(**input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
    input_ids = input_ids.get('input_ids', input_ids)
    outputs = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    return response

def parse_result(result_str):
    """
    将模型输出的形如 '时间: None\n胜者: 天津天海  天津泰达\n败者: None\n赛事名称: 德比战'
    解析为字典
    """
    result = {}
    for line in result_str.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result

def main():
    parser = argparse.ArgumentParser(description="实体抽取模型本地demo")
    parser.add_argument('--text', type=str, required=True, help='待抽取文本，不能为空')
    parser.add_argument('--entity_types', type=str, nargs='*', default=None, help='实体类型列表（可选），如: --entity_types 时间 胜者 败者 赛事名称')
    args = parser.parse_args()

    result_str = extract_entities(args.text, args.entity_types)
    result_dict = parse_result(result_str)
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()