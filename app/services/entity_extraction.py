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

# 全局加载模型（避免重复加载）
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