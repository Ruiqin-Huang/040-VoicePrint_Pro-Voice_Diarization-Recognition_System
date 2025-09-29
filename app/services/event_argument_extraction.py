import json
import re
import asyncio
from typing import List, Dict, Any, Optional

from app.llm import llm_client
from app.models.event_argument_extraction import EventInfo, Argument, SingleEventResult

# --- Prompts Definition ---

SYSTEM_PROMPT = """你是一个专门用于事件论元抽取的API。你的唯一任务是根据用户指令，从给定文本中抽取出指定事件的触发词和论元，并返回一个原始、干净的JSON对象。
你绝对禁止输出任何解释、注释、对话或Markdown标记（例如 ```json）。
不要在JSON中添加任何形式的注释（如//或/**/）。
你的全部响应必须是一个单一、有效的、可以直接被解析的JSON对象，除此之外别无他物。"""

ARGUMENT_EXTRACTION_PROMPT = """
请从以下文本中，针对类型为“{event_type}”的事件，抽取指定的论元。

**重要提示：** 待抽取的文本中可能包含多个不同的事件。您的任务是**仅**关注与事件类型“{event_type}”严格相关的信息，并忽略文本中提及的任何其他事件。

**待抽取文本:**
---
{text}
---

**需要抽取的事件信息:**
- 事件类型: {event_type}
- 论元列表: {argument_list_str}
- 事件触发词 (trigger)

**输出要求:**
请严格按照以下JSON格式返回结果。
- `trigger` 字段必须是字符串，表示事件的触发词。
- `arguments` 字段必须是一个列表，其中每个元素都是一个包含 "name" 和 "value" 的对象。
- 如果在文本中找不到某个论元，请将该论元的 "value" 设置为 "None"，禁止凭空捏造不存在的论元值。

**JSON格式示例:**
{{
  "trigger": "一个具体的触发词",
  "arguments": [
    {{"name": "论元1", "value": "抽到的值1"}},
    {{"name": "论元2", "value": "None"}}
  ]
}}

请开始分析并返回JSON对象。
"""

DEFAULT_ARGUMENT_TYPES = ['主体', '客体', '时间', '地点']

# --- JSON Parsing Logic ---

def _clean_json_string(json_str: str) -> str:
    """清理JSON字符串，移除注释和可能导致解析错误的字符。"""
    json_str = re.sub(r'//.*?($|\n)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = json_str.replace('\n', ' ').replace('\r', '')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    return json_str

async def _extract_single_event(text: str, event_info: EventInfo) -> SingleEventResult:
    """对单个事件进行论元抽取。"""
    event_type = event_info.event_type
    argument_types = event_info.argument_types or DEFAULT_ARGUMENT_TYPES
    argument_list_str = ", ".join(argument_types)

    user_prompt = ARGUMENT_EXTRACTION_PROMPT.format(
        text=text,
        event_type=event_type,
        argument_list_str=argument_list_str
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
        print(f"Error processing event '{event_type}': {e}")
        print(f"Original LLM response for event '{event_type}':\n{response_text}")
        # 如果解析失败，返回一个包含错误信息的默认结构
        return SingleEventResult(
            event_type=event_type,
            argument_types=argument_types,
            trigger="解析失败",
            arguments=[Argument(name=arg, value="解析失败") for arg in argument_types]
        )

    # --- 结果转换与校验 ---
    try:
        trigger = parsed_json.get("trigger", "解析失败")
        if not isinstance(trigger, str): trigger = str(trigger)

        raw_arguments = parsed_json.get("arguments", [])
        if not isinstance(raw_arguments, list): raw_arguments = []

        # 创建一个字典以便快速查找抽到的论元值
        extracted_args_map = {arg.get("name"): arg.get("value", "None") for arg in raw_arguments if isinstance(arg, dict)}

        # 按照请求的论元顺序构建最终结果，确保所有请求的论元都在
        final_arguments = []
        for arg_type in argument_types:
            value = extracted_args_map.get(arg_type, "None")
            if not isinstance(value, str): value = str(value)
            final_arguments.append(Argument(name=arg_type, value=value))

        return SingleEventResult(
            event_type=event_type,
            argument_types=argument_types,
            trigger=trigger,
            arguments=final_arguments
        )
    except Exception as e:
        print(f"Error converting parsed JSON to model for event '{event_type}': {e}")
        return SingleEventResult(
            event_type=event_type,
            argument_types=argument_types,
            trigger="转换失败",
            arguments=[Argument(name=arg, value="转换失败") for arg in argument_types]
        )


async def process_multi_event_argument_extraction(text: str, events_info: List[EventInfo]) -> List[SingleEventResult]:
    """并行处理单个文本中的多个事件论元抽取任务。"""
    tasks = [_extract_single_event(text, event) for event in events_info]
    results = await asyncio.gather(*tasks)
    return results