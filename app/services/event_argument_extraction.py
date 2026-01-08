"""
事件论元抽取服务模块

该模块提供基于大语言模型的事件论元抽取功能，支持从文本中抽取事件的触发词和论元。
主要功能包括：
- 事件论元抽取Prompt构建
- 文本预处理和解析
- 并行事件抽取处理
- JSON结果解析和验证
- 时间/日期论元格式化处理

依赖：
- app.llm: 大语言模型客户端
- asyncio: 用于异步并行处理
- json: 用于JSON解析
- re: 用于正则表达式处理
"""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.llm import llm_client
from app.models.event_argument_extraction import EventInfo, Argument, SingleEventResult
from app.models.common import ModelInfo

# --- Prompts Definition ---

# 系统提示词：指导大语言模型如何输出结果
SYSTEM_PROMPT = """你是一个专门用于事件论元抽取的API。你的唯一任务是根据用户指令，从给定文本中抽取出指定事件的触发词和论元，并返回一个原始、干净的JSON对象。
你绝对禁止输出任何解释、注释、对话或Markdown标记（例如 ```json）。
不要在JSON中添加任何形式的注释（如//或/**/）。
你的全部响应必须是一个单一、有效的、可以直接被解析的JSON对象，除此之外别无他物。"""

# 事件论元抽取Prompt模板
# 用于指导大语言模型进行事件论元抽取的标准Prompt
ARGUMENT_EXTRACTION_PROMPT = """
请从以下文本中，针对类型为“{event_type}”的事件，抽取指定的论元。

**重要提示：** 待抽取的文本中可能包含多个不同的事件。您的任务是**仅**关注与事件类型“{event_type}”严格相关的信息，并忽略文本中提及的任何其他事件。

**待抽取文本:**
---
{text}
---

**需要抽取的事件信息:**
- 事件类型: {event_type}
- 事件触发词 (trigger)
- 论元列表: {argument_list_str}
{argument_definitions_str}
**输出要求:**
请严格按照以下JSON格式返回结果。
- `trigger` 字段必须是字符串，表示事件的触发词。
- `arguments` 字段必须是一个列表，其中每个元素都是一个包含 "name" 和 "value" 的对象。
- 如果在文本中找不到某个论元，请将该论元的 "value" 设置为 null，禁止凭空捏造不存在的论元值。
- 对于有特殊格式化要求的论元（如时间和日期），请在抽取时结合上下文，并尽量遵循格式要求。
- 使用占位符'X'替换未知的时间/日期信息。例如对于具体年份不确定的日期'7月8日'，仅输出"XXXX-07-08"即可。对于具体日期不确定的时间'下午3点'，仅输出"XXXX-XX-XX 15:00:00"即可。

**JSON格式示例:**
{{
  "trigger": "一个具体的触发词",
  "arguments": [
    {{"name": "论元1", "value": "抽到的值1"}},
    {{"name": "时间", "value": "2025-10-19 14:30:00"}},
  ]
}}

请开始分析并返回JSON对象。
"""

# 为关键、需规范输出格式的论元提供清晰的描述和格式
# 用于在Prompt中向大语言模型说明特定论元类型的含义和格式要求
ARGUMENT_DEFINITIONS = {
    "时间": "表示一天中特定时刻的时间点（如'下午3点'）或包含日期的完整时间（如'2025年10月9日15点'）。需结合上下文（如'当天'、'同年'）补全信息。最终值应尽可能格式化为'YYYY-MM-DD HH:MM:SS'或'HH:MM:SS'。",
    "日期": "表示特定日期的文本（如'2025年10月9日'），不包含具体时间点。需结合上下文（如'同年'）补全信息。最终值应尽可能格式化为'YYYY-MM-DD'。",
}

# 默认论元类型列表
# 当用户未指定论元类型时，使用此列表中的类型进行抽取
DEFAULT_ARGUMENT_TYPES = ['主体', '客体', '时间', '地点']

# --- JSON Parsing Logic ---

def _clean_json_string(json_str: str) -> str:
    """
    清理JSON字符串，移除注释和可能导致解析错误的字符
    
    移除JSON字符串中的单行注释（//）、多行注释（/* */）、
    多余的换行符、尾随逗号和Python的None值，确保JSON可以被正确解析。
    
    Args:
        json_str: 待清理的JSON字符串
        
    Returns:
        str: 清理后的JSON字符串
    """
    json_str = re.sub(r'//.*?($|\n)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = json_str.replace('\n', ' ').replace('\r', '')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    json_str = json_str.replace('None', 'null')
    return json_str

def _normalize_datetime_output(raw_text: str, entity_type: str) -> str:
    """
    将抽取的原始时间/日期文本规范化为指定格式
    
    此函数按优先级顺序尝试多种正则表达式来解析日期和时间，
    支持多种时间/日期格式，并将结果格式化为标准格式。
    
    Args:
        raw_text: 抽取出的原始时间/日期文本
        entity_type: 实体类型（"时间"或"日期"）
        
    Returns:
        str: 格式化后的时间/日期字符串，格式为：
            - 日期：YYYY-MM-DD 或 YYYY-MM-XX 或 XXXX-MM-DD 等
            - 时间：YYYY-MM-DD HH:MM:SS 或 HH:MM:SS 等
            如果无法解析则返回空字符串
    """
    # 初始化所有时间日期组件为占位符
    year, month, day = 'XXXX', 'XX', 'XX'  # 年月日占位符
    hour, minute, second = 'XX', 'XX', 'XX'  # 时分秒占位符

    # --- 1. 提取年月日 (按最长、最完整模式优先) ---

    # 模式 A: 完整格式 YYYY-MM-DD (或 YYYY年M月D日)
    date_match = re.search(r'(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?', raw_text)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2).zfill(2)
        day = date_match.group(3).zfill(2)
    else:
        # 模式 B: 年月格式 YYYY-MM (或 YYYY年M月)
        year_month_match = re.search(r'(\d{4})[年/-](\d{1,2})月?', raw_text)
        if year_month_match:
            year = year_month_match.group(1)
            month = year_month_match.group(2).zfill(2)
            # day 保持 'XX'
        else:
            # 模式 C: 仅年份 YYYY年
            year_match = re.search(r'(\d{4})年', raw_text)
            if year_match:
                year = year_match.group(1)
                # month 和 day 保持 'XX'
            else:
                # 模式 D: 月日格式 MM-DD (或 M月D日)
                month_day_match = re.search(r'(\d{1,2})[月/-](\d{1,2})[日号]?', raw_text)
                if month_day_match:
                    month = month_day_match.group(1).zfill(2)
                    day = month_day_match.group(2).zfill(2)
                    # year 保持 'XXXX'
                else:
                    # 模式 E: 仅日期 D日
                    day_match = re.search(r'(\d{1,2})[日号]', raw_text)
                    if day_match:
                        day = day_match.group(1).zfill(2)
                        # year 和 month 保持 'XXXX', 'XX'

    # --- 2. 提取时分秒 ---

    # 模式 A: 完整时间 HH:MM:SS (或 H点M分S秒)
    time_match = re.search(r'(\d{1,2})[:：时点](\d{1,2})[:：分]?(\d{1,2})?秒?', raw_text)
    if time_match:
        h, m, s = time_match.groups()
        if h: hour = h.zfill(2)
        if m: minute = m.zfill(2)
        if s: second = s.zfill(2)
        else: second = '00' # 如果没有秒，则补00
    else:
        # 模式 B: 小时+半点，例如 "3点半"
        hour_match = re.search(r'(\d{1,2})[:：时点]', raw_text)
        if hour_match:
            hour = hour_match.group(1)
            if '半' in raw_text:
                minute = '30'
            else:
                minute = '00' # 如果只有小时，分钟补00
            second = '00' # 秒补00

    # --- 3. 处理上午/下午/晚上 ---
    if hour != 'XX' and int(hour) <= 12:
        if '下午' in raw_text or '晚上' in raw_text:
            if int(hour) < 12: # 12点下午还是12点
                hour = str(int(hour) + 12)
        elif '上午' in raw_text:
             hour = hour.zfill(2) # 确保上午时间是两位数，如 09

    if hour != 'XX':
        hour = hour.zfill(2)

    # --- 4. 根据实体类型组合最终结果 ---
    has_date_info = not (year == 'XXXX' and month == 'XX' and day == 'XX')
    has_time_info = not (hour == 'XX' and minute == 'XX' and second == 'XX')

    # 如果没有任何有效的时间/日期信息被解析出来，则返回空
    if not has_date_info and not has_time_info:
        return ""

    if entity_type == '日期':
        if day != 'XX':
            return f"{year}-{month}-{day}"
        elif month != 'XX':
            return f"{year}-{month}-XX" # 允许返回 YYYY-MM-XX 或 XXXX-MM-XX
        elif year != 'XXXX':
            return f"{year}-XX-XX" # 允许返回 YYYY-XX-XX
        else:
            return "" # 如果连年/月都没有，则认为抽取失败

    if entity_type == '时间':
        date_part = ""
        if day != 'XX':
            date_part = f"{year}-{month}-{day}"
        elif month != 'XX':
            date_part = f"{year}-{month}-XX"
        elif year != 'XXXX':
            date_part = f"{year}-XX-XX"

        time_part = f"{hour}:{minute}:{second}" if has_time_info else ""

        if date_part and time_part:
            return f"{date_part} {time_part}"
        elif has_time_info:
            return time_part
        elif date_part: # 如果只有日期部分
            return f"{date_part} {hour}:{minute}:{second}"

    return raw_text # 对于其他类型或无法解析的情况，返回原始文本

async def _extract_single_event(text: str, event_info: EventInfo, model_info: ModelInfo) -> SingleEventResult:
    """
    对单个事件进行论元抽取
    
    使用大语言模型从文本中抽取指定事件的触发词和论元。
    对时间和日期论元进行特殊处理和格式化。
    
    Args:
        text: 待抽取的文本内容
        event_info: 事件信息，包含事件类型和论元类型列表
        model_info: 指定用于抽取的大模型信息
        
    Returns:
        SingleEventResult: 事件抽取结果，包含事件类型、触发词和论元列表
                          如果解析失败，返回包含错误信息的默认结构
    """
    event_type = event_info.event_type
    argument_types = event_info.argument_types or DEFAULT_ARGUMENT_TYPES
    argument_list_str = ", ".join(argument_types)

    # 构建论元定义字符串
    arg_defs = [f'- {arg}: {ARGUMENT_DEFINITIONS[arg]}' for arg in argument_types if arg in ARGUMENT_DEFINITIONS]
    argument_definitions_str = ""
    if arg_defs:
        argument_definitions_str = "\n**论元定义与格式:**\n" + "\n".join(arg_defs) + "\n"

    user_prompt = ARGUMENT_EXTRACTION_PROMPT.format(
        text=text,
        event_type=event_type,
        argument_list_str=argument_list_str,
        argument_definitions_str=argument_definitions_str
    )

    response_text = ""
    parsed_json = {}
    try:
        response_text = await llm_client.generate_text(
            system_prompt=SYSTEM_PROMPT, 
            user_prompt=user_prompt,
            model_info=model_info
        )
        
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

        # 按照请求的论元顺序构建最终结果，并过滤掉无效的论元
        final_arguments = []
        for arg_type in argument_types:
            value = extracted_args_map.get(arg_type, "None")
            if not isinstance(value, str): value = str(value)
            
            # 如果值是 "none" (不区分大小写) 或 "未知"，则跳过
            if value.lower() == ('none' or 'null') or value == '未知' or not value:
                continue

            # 对时间和日期论元进行后处理
            if arg_type in ["时间", "日期"]:
                normalized_value = _normalize_datetime_output(value, arg_type)
                if normalized_value: # 只有在规范化后非空才添加
                    final_arguments.append(Argument(name=arg_type, value=normalized_value))
            else:
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


async def process_multi_event_argument_extraction(text: str, events_info: List[EventInfo], model_info: ModelInfo) -> List[SingleEventResult]:
    """
    并行处理单个文本中的多个事件论元抽取任务
    
    对指定的多个事件进行并行抽取，使用asyncio.gather实现并行处理，
    提高处理效率。每个事件独立抽取，互不影响。
    
    Args:
        text: 待抽取的文本内容
        events_info: 事件信息列表，每个元素包含事件类型和论元类型列表
        model_info: 指定用于抽取的大模型信息
        
    Returns:
        List[SingleEventResult]: 所有事件的抽取结果列表，每个元素包含事件类型、触发词和论元列表
    """
    tasks = [_extract_single_event(text, event, model_info) for event in events_info]
    results = await asyncio.gather(*tasks)
    return results