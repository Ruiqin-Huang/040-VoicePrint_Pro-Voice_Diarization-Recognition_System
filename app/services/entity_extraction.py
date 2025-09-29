import json
import re
import asyncio
from typing import List, Dict, Optional

from app.llm import llm_client
from app.models.entity_extraction import EntityResult

# --- Prompts Definition ---

SYSTEM_PROMPT = """你是一个专门用于实体抽取的API。你的唯一任务是根据用户指令，从给定文本中抽取出指定类型的实体，并返回一个原始、干净的JSON对象。
你绝对禁止输出任何解释、注释、对话或Markdown标记（例如 ```json）。
不要在JSON中添加任何形式的注释（如//或/**/）。
你的全部响应必须是一个单一、有效的、可以直接被解析的JSON对象，除此之外别无他物。"""

ENTITY_EXTRACTION_PROMPT = """
请从以下文本中，抽取出所有类型为“{entity_type}”的实体。

**重要提示：** 文本中可能包含多种实体，但你的任务是**仅**关注并抽取出与实体类型“{entity_type}”严格相关的词语或短语。

**待抽取文本:**
---
{text}
---

**需要抽取的实体类型:**
- {entity_type}

**输出要求:**
请严格按照以下JSON格式返回结果。
- `entities` 字段必须是一个字符串列表，包含所有在文本中找到的“{entity_type}”实体。
- 如果在文本中找不到任何该类型的实体，请返回一个空列表 `[]`，禁止凭空捏造不存在的实体。

**JSON格式示例 (当 entity_type 为 "地名"):**
{{
  "entities": ["北京", "上海"]
}}

**JSON格式示例 (当找不到实体时):**
{{
  "entities": []
}}

请开始分析并返回JSON对象。
"""

# 默认实体类型
DEFAULT_ENTITY_TYPES = [
    '电话号码','邮箱地址','人名','地名','组织','军衔','呼号','时间','日期','武器类型','车辆类型','任务代号','部队名称','坐标','频率',
    '警报类型','通信指令','装备型号','加密等级','通播组','通信状态','身份验证码','通播等级','接力站点','呼号后缀','报文类型',
    '通播网号','认证口令','通信协议','信道编号','频谱波段','导航点','战位标识','敌我识别码','密语代号','通播站标识','信号强度',
    '干扰情况','时隙分配','网络节点'
]

# --- JSON Parsing Logic ---

def _clean_json_string(json_str: str) -> str:
    """清理JSON字符串，移除注释和可能导致解析错误的字符。"""
    json_str = re.sub(r'//.*?($|\n)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = json_str.replace('\n', ' ').replace('\r', '')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    return json_str

async def _extract_single_entity_type(text: str, entity_type: str) -> List[EntityResult]:
    """对单一实体类型进行抽取。"""
    user_prompt = ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        entity_type=entity_type
    )

    response_text = ""
    parsed_json = {}
    try:
        response_text = await llm_client.generate_text(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            dict_str = match.group(1)
        else:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                dict_str = response_text[start:end]
            else:
                raise json.JSONDecodeError("No JSON object found", response_text, 0)

        cleaned_dict_str = _clean_json_string(dict_str)
        parsed_json = json.loads(cleaned_dict_str)

    except Exception as e:
        print(f"Error processing entity type '{entity_type}': {e}")
        print(f"Original LLM response for entity type '{entity_type}':\n{response_text}")
        # 如果解析失败，返回空列表
        return []

    # --- 结果转换与校验 ---
    try:
        entity_names = parsed_json.get("entities", [])
        if not isinstance(entity_names, list):
            print(f"Warning: 'entities' field is not a list for type '{entity_type}'. Found: {type(entity_names)}")
            return []
        
        # 构建符合API响应格式的列表
        results = [EntityResult(type=entity_type, name=str(name)) for name in entity_names if isinstance(name, str) and name]
        return results

    except Exception as e:
        print(f"Error converting parsed JSON to model for entity type '{entity_type}': {e}")
        return []


async def process_entity_extraction(text: str, entity_types: Optional[List[str]] = None) -> List[EntityResult]:
    """
    并行执行多个实体类型的抽取任务
    :param text: 待抽取的文本
    :param entity_types: 自定义的实体类型列表
    :return: 包含所有抽取结果的字典列表
    """
    if not entity_types:
        entity_types = DEFAULT_ENTITY_TYPES
    
    # 为每个实体类型创建一个异步任务
    tasks = [_extract_single_entity_type(text, etype) for etype in entity_types]
    
    # 并行执行所有任务
    results_from_tasks = await asyncio.gather(*tasks)
    
    # 将所有任务返回的结果（它们都是列表）合并成一个大的列表
    final_results = []
    for result_list in results_from_tasks:
        final_results.extend(result_list)
        
    return final_results