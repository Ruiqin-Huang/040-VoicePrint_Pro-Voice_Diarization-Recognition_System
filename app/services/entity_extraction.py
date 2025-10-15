import json
import re
import asyncio
from typing import List, Dict, Optional

from app.llm import llm_client
from app.models.entity_extraction import EntityResult
from app.models.common import ModelInfo

# --- Prompts Definition ---

SYSTEM_PROMPT = """你是一个专门用于实体抽取的API。你的唯一任务是根据用户指令，从给定文本中抽取出指定类型的实体，并返回一个原始、干净的JSON对象。
你绝对禁止输出任何解释、注释、对话或Markdown标记（例如 ```json）。
不要在JSON中添加任何形式的注释（如//或/**/）。
你的全部响应必须是一个单一、有效的、可以直接被解析的JSON对象，除此之外别无他物。"""

# 实体类型定义字典，为关键、易混淆的实体提供清晰的描述和格式
ENTITY_DEFINITIONS = {
    "手机号": "由数字和特殊字符（如'+', '-', '()'）组成的联系号码，可能包含国家代码和区号。例如'13812345678', '+1 (555) 123-4567'",
    "电话号码": "由数字和特殊字符（如'+', '-', '()'）组成的联系号码，可能包含国家代码和区号。例如'13812345678', '+1 (555) 123-4567'",
    "邮箱": "包含'@'符号的标准电子邮件地址格式。例如'张三@示例.com', 'john.doe@example.com'",
    "邮箱地址": "包含'@'符号的标准电子邮件地址格式。例如'张三@示例.com', 'john.doe@example.com'",
    "人名": "指代人物的姓名，可以是全名、昵称或姓氏。例如'王伟', 'John Smith'",
    "地名": "城市、国家、街道、山脉等地理位置名称。例如'北京', 'New York'",
    "组织": "公司、政府机构、非政府组织等实体的名称。例如'联合国', 'Google'",
    "军衔": "军队或准军事组织中的等级称号。例如'上校', 'Captain'",
    "呼号": "用于无线电通信中识别身份的唯一代号，通常由字母和数字组成。例如'洞幺', 'Alpha Bravo 1'",
    "时间": "表示一天中特定时刻的时间点，可以是12小时制或24小时制。例如'下午3点15分', '15:15 UTC'",
    "日期": "表示特定日期的文本，格式多样。例如'2025年10月9日', 'October 9, 2025'",
    "事件": "描述特定活动或事件的名称或标题。例如'奥运会', 'World War II'",
    "战争": "指代历史或当前的军事冲突名称。例如'二战', 'Vietnam War'",
    "武器类型": "武器装备的具体类别或名称。例如'95式自动步枪', 'F-22 Raptor'",
    "车辆类型": "陆、海、空、天载具的具体类别或名称。例如'99A主战坦克', 'Toyota Camry'",
    "任务代号": "为特定行动或计划指定的名称。例如'长城行动', 'Operation Overlord'",
    "部队名称": "军队单位的正式番号或通称。例如'第75集团军', '82nd Airborne Division'",
    "坐标": "用于在地理空间中定位点的经纬度或其他坐标系统表示。例如'北纬39.9°, 东经116.4°', '40.7128° N, 74.0060° W'",
    "频率": "无线电通信中使用的特定频率，通常以赫兹（Hz）的倍数表示。例如'145.5兆赫', '462.5625 MHz'",
    "警报类型": "指示特定威胁或状态的警报信号名称。例如'红色警报', 'Air Raid Siren'",
    "通信指令": "在通信中下达的具体操作或命令。例如'立即执行', 'Roger that, proceeding to waypoint'",
    "装备型号": "设备或工具的具体型号标识。例如'AN/PRC-152', 'iPhone 17'",
    "加密等级": "描述信息加密强度的级别。例如'绝密', 'Top Secret'",
    "通播组": "在通信网络中预先设定的特定接收方群体。例如'指挥组', 'Command Net'",
    "通信状态": "描述通信链路当前状况的术语。例如'通信畅通', 'Comms clear'",
    "身份验证码": "用于验证用户或设备身份的一串字符或数字。例如'口令长城', 'Password Alpha-7'",
    "通播等级": "通信广播的优先级或重要性级别。例如'紧急通播', 'Flash Traffic'",
    "接力站点": "用于中继通信信号的站点名称或编号。例如'01号中继站', 'Relay Station Alpha'",
    "呼号后缀": "附加在主呼号后的补充标识符。例如'主控', 'Actual'",
    "报文类型": "消息的格式或类别。例如'加密报文', 'Encrypted Message'",
    "通播网号": "通信网络的唯一数字或字母数字标识符。例如'网络05', 'Net 05'",
    "认证口令": "用于访问系统或网络的密码或短语。例如'山鹰', 'Eagle Has Landed'",
    "通信协议": "通信双方必须遵守的规则和标准集。例如'TCP/IP', 'SINCGARS'",
    "信道编号": "分配给特定通信路径的数字标识。例如'3号信道', 'Channel 3'",
    "频谱波段": "电磁频谱中的一个特定频率范围。例如'甚高频段', 'UHF Band'",
    "导航点": "在导航路径上预先定义的地理位置点。例如'导航点A', 'Waypoint Alpha'",
    "战位标识": "特定战斗或操作位置的名称或代码。例如'狙击阵地', 'Sniper's Nest'",
    "敌我识别码": "用于在战场上区分友军和敌军的电子信号代码。例如'模式四', 'Mode 4'",
    "密语代号": "用于在通信中指代敏感信息的预设词语。例如'苹果', 'Apple' (意指目标)",
    "通播站标识": "广播站点的唯一名称或代码。例如'北京总站', 'Main Station Beijing'",
    "信号强度": "衡量接收到的无线电信号功率的指标。例如'信号满格', 'Signal strength five by five'",
    "干扰情况": "描述通信受到的干扰类型或程度。例如'受到强烈干扰', 'Heavy Jamming'",
    "时隙分配": "在时分多址（TDMA）系统中分配给特定用户的时间段。例如'3号时隙', 'Time Slot 3'",
    "网络节点": "网络中的一个连接点或设备。例如'服务器A', 'Node A'"
    # 可以根据需要为更多实体类型添加定义
    # 对于ENTITY_DEFINITIONS中不存在的实体类型，默认使用通用描述： "指代“{entity_type}”的词语或短语。"
}

# 更详细、包含“思维链”（Chain-of-Thought）引导的Prompt
ENTITY_EXTRACTION_PROMPT = """
**任务：** 从给定文本中抽取出所有类型为“{entity_type}”的实体。

**实体类型定义与示例 ({entity_type}):**
{entity_definition}

**执行步骤:**
1.  **识别:** 仔细阅读下面的“待抽取文本”，判断是否存在符合“{entity_type}”定义的实体。
2.  **决策:**
    - **如果存在**一个或多个实体，将它们收集到一个列表中。
    - **如果不存在**任何符合定义的实体，你**必须**生成一个空列表 `[]`。**绝对禁止**猜测或创造不存在的实体。
3.  **输出:** 严格按照指定的JSON格式返回结果。

**待抽取文本:**
---
{text}
---

**JSON输出格式:**
{{
  "entities": ["实体1", "实体2", ...]
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

async def _extract_single_entity_type(text: str, entity_type: str, model_info: ModelInfo) -> List[EntityResult]:
    """对单一实体类型进行抽取。"""
    # 从定义字典中获取实体定义，如果不存在则使用通用描述
    entity_definition = ENTITY_DEFINITIONS.get(entity_type, f"指代“{entity_type}”的词语或短语。")
    
    user_prompt = ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        entity_type=entity_type,
        entity_definition=entity_definition
    )

    response_text = ""
    parsed_json = {}
    try:
        response_text = await llm_client.generate_text(
            system_prompt=SYSTEM_PROMPT, 
            user_prompt=user_prompt,
            model_info=model_info
        )
        
        # 优先尝试从Markdown代码块中提取
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            content_str = match.group(1)
        else:
            # 否则，假设整个响应都是JSON内容
            content_str = response_text

        cleaned_str = _clean_json_string(content_str.strip())

        # 尝试直接解析清理后的字符串
        parsed_data = json.loads(cleaned_str)

        # 判断解析结果是列表还是字典
        if isinstance(parsed_data, list):
            # 如果直接是列表，手动构造成期望的字典格式
            parsed_json = {"entities": parsed_data}
        elif isinstance(parsed_data, dict):
            # 如果是字典，直接使用
            parsed_json = parsed_data
        else:
            # 如果是其他类型（数字、字符串等），视为无效
            raise json.JSONDecodeError("Parsed JSON is not a list or a dict", cleaned_str, 0)

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
        
        # 构建符合API响应格式的列表，并过滤掉无效的实体名称
        results = [
            EntityResult(type=entity_type, name=str(name)) 
            for name in entity_names 
            if isinstance(name, str) and name and str(name).lower() != 'none' and str(name) != '未知'
        ]
        return results

    except Exception as e:
        print(f"Error converting parsed JSON to model for entity type '{entity_type}': {e}")
        return []


async def process_entity_extraction(text: str, model_info: ModelInfo, entity_types: Optional[List[str]] = None) -> List[EntityResult]:
    """
    并行执行多个实体类型的抽取任务
    :param text: 待抽取的文本
    :param model_info: 模型调用信息
    :param entity_types: 自定义的实体类型列表
    :return: 包含所有抽取结果的字典列表
    """
    if not entity_types:
        entity_types = DEFAULT_ENTITY_TYPES
    
    # 为每个实体类型创建一个异步任务
    tasks = [_extract_single_entity_type(text, etype, model_info) for etype in entity_types]
    
    # 并行执行所有任务
    results_from_tasks = await asyncio.gather(*tasks)
    
    # 将所有任务返回的结果（它们都是列表）合并成一个大的列表
    final_results = []
    for result_list in results_from_tasks:
        final_results.extend(result_list)
        
    return final_results