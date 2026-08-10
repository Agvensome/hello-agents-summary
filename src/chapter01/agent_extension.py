# agent_extension.py
# 在 1.3 节智能旅行助手(agent.py)基础上扩展:
# 1. 记忆功能: 记住用户偏好(景点类型、预算)与已被拒绝的景点
# 2. 门票查询: 推荐景点后先查票, 售罄则让LLM自动提供备选(限制重试次数)
# 3. 拒绝反思: 用户连续拒绝3个推荐后, 自动修改系统提示词调整推荐策略

import os
import re
from config import *
from OpenAICompatibleClient import OpenAICompatibleClient
from wttrin import get_weather
from attraction import get_attraction

# --- 0. 记忆存储和状态变量 ---

memory = {
    "user_preferences": {
        "attraction_types": [],      # 用户喜欢的景点类型, 如 ["历史文化"]
        "budget_range": None,        # 预算范围, 如 "中等"
    },
    "rejected_attractions": [],      # 已被用户拒绝的景点, 避免重复推荐
    "rejection_count": 0,            # 连续拒绝次数
    "current_attraction": None,      # 当前待用户确认的景点
    "ticket_retry_count": 0,         # 门票售罄后的备选重试次数
}

MAX_TICKET_RETRY = 2               # 门票售罄最多重试2次备选
MAX_REJECT_BEFORE_REFLECT = 3      # 连续拒绝3次触发反思
MAX_LOOP = 10                      # 主循环安全上限(交互与重试会消耗轮次)


def get_ticket_status(attraction: str) -> str:
    """
    模拟门票查询: 预置售罄名单, 景点名称命中即视为售罄.

    :param attraction: 景点名称
    :return: 门票状态
    """
    sold_out_keywords = ["欢乐谷", "世界之窗", "迪士尼", "环球影城"]
    if any(keyword in attraction for keyword in sold_out_keywords):
        return f"门票状态: '{attraction}' 今日门票已售罄"
    return f"门票状态: '{attraction}' 今日有票"


def build_memory_str() -> str:
    """将记忆内容格式化为可注入Prompt的文本."""
    prefs = memory["user_preferences"]
    types_str = ",".join(prefs["attraction_types"]) if prefs["attraction_types"] else "未知"
    budget_str = prefs["budget_range"] or "未知"
    rejected_str = ",".join(memory["rejected_attractions"]) if memory["rejected_attractions"] else "无"
    return (
        f"[长期记忆] 用户偏好景点类型: {types_str}; 预算范围: {budget_str}; "
        f"已被拒绝的景点(禁止再次推荐): {rejected_str}"
    )


# --- 1. 配置LLM客户端 ---

os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---

# 2.1 收集初始用户偏好(可直接回车跳过)
print("=" * 40)
initial_pref = input("请告诉我您的偏好(如: 喜欢历史文化景点, 预算中等), 直接回车跳过: ").strip()
if initial_pref:
    memory["user_preferences"]["attraction_types"].append(initial_pref)
    print(f"已记住您的偏好: {initial_pref}")

# 2.2 基于基础提示词构建扩展系统提示词(后续反思时还会动态追加)
system_prompt = AGENT_SYSTEM_PROMPT + """
# 新增工具:
- `get_ticket_status(attraction: str)`: 查询指定景点当日门票状态(有票/已售罄)。

# 扩展规则:
- 向用户推荐任何景点前, 必须先调用 get_ticket_status 确认有票; 若Observation提示已售罄, 请改推其他景点
- 用户消息中的[长期记忆]记录了用户偏好与已被拒绝的景点, 推荐时必须参考偏好, 且禁止推荐已被拒绝的景点
"""

# 2.3 初始化对话历史
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "=" * 40)

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "get_ticket_status": get_ticket_status,
}

# --- 3. 运行主循环 ---
for i in range(MAX_LOOP):
    print(f"--- 循环 {i + 1} ---\n")

    # 3.1 构建Prompt: 每轮在最前面注入最新的记忆摘要
    full_prompt = build_memory_str() + "\n\n" + "\n".join(prompt_history)

    # 3.2 调用LLM思考
    llm_output = llm.generate(full_prompt, system_prompt=system_prompt)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        final_answer = finish_match.group(1) if finish_match else action_str
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str)
    if not tool_match or not args_match:
        observation = "错误: Action 格式无法解析。调用工具时请使用 function_name(arg_name=\"arg_value\") 格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue
    tool_name = tool_match.group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_match.group(1)))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义工具 {tool_name}"

    # 3.4. 门票查询结果的额外处理: 售罄重试 / 有票则进入用户确认
    if tool_name == "get_ticket_status":
        attraction_name = kwargs.get("attraction", "")
        if "已售罄" in observation:
            memory["ticket_retry_count"] += 1
            if memory["ticket_retry_count"] > MAX_TICKET_RETRY:
                observation += (
                    f"\n系统提示: 备选重试次数已用完(上限{MAX_TICKET_RETRY}次), "
                    f"请立即使用 Action: Finish[...] 告知用户今日景点门票紧张, 无法完成推荐。"
                )
            else:
                observation += (
                    f"\n系统提示: 该景点门票已售罄(第{memory['ticket_retry_count']}次售罄), "
                    f"请改推其他景点并先查询其门票状态。"
                )
        elif "有票" in observation:
            if attraction_name in memory["rejected_attractions"]:
                # 代码层面兜底: 已被拒绝的景点不再打扰用户
                observation += "\n系统提示: 该景点此前已被用户拒绝, 请勿重复推荐, 请更换其他景点。"
            else:
                # 门票有货, 先记录门票观察结果, 再暂停等待用户确认
                memory["current_attraction"] = attraction_name
                memory["ticket_retry_count"] = 0
                observation_str = f"Observation: {observation}"
                print(f"{observation_str}\n" + "=" * 40)
                prompt_history.append(observation_str)

                print(f"\n>>> 智能体向您推荐景点: {attraction_name}")
                choice = input(">>> 是否接受该推荐? (接受/拒绝): ").strip()
                if choice == "接受":
                    pref = input(">>> 您喜欢它的哪类特点? (如: 历史文化/自然风光, 可回车跳过): ").strip()
                    if pref:
                        memory["user_preferences"]["attraction_types"].append(pref)
                        print(f">>> 已记住您的偏好: {pref}")
                    memory["rejection_count"] = 0
                    observation = (
                        f"系统消息: 用户已接受景点 '{attraction_name}' 的推荐。"
                        f"请使用 Action: Finish[...] 输出包含天气、推荐景点及门票情况的最终旅行建议。"
                    )
                else:
                    memory["rejected_attractions"].append(attraction_name)
                    memory["rejection_count"] += 1
                    print(f">>> 已记录: 拒绝 '{attraction_name}' (连续拒绝 {memory['rejection_count']} 次)")
                    if memory["rejection_count"] >= MAX_REJECT_BEFORE_REFLECT:
                        # 连续拒绝达到3次: 反思并修改系统提示词, 强制调整推荐策略
                        system_prompt += """
# 反思调整(用户已连续拒绝多个推荐):
- 你此前的推荐策略未能满足用户, 请彻底调整思路: 更换景点类型、考虑风格完全不同的备选(如室内/室外切换、免费场馆等)
- 优先参考[长期记忆]中的用户偏好重新匹配, 严禁重复推荐任何已被拒绝的景点
"""
                        memory["rejection_count"] = 0
                        observation = (
                            f"系统消息: 用户已连续拒绝 {MAX_REJECT_BEFORE_REFLECT} 个推荐, 推荐策略已强制调整。"
                            f"请按照新的系统提示词彻底更换思路, 给出风格明显不同的推荐。"
                        )
                    else:
                        observation = (
                            f"系统消息: 用户拒绝了景点 '{attraction_name}'。"
                            f"请重新推荐其他景点(禁止推荐: {','.join(memory['rejected_attractions'])}), 推荐前务必先查询门票状态。"
                        )

    # 3.5. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
else:
    print(f"已达最大循环次数({MAX_LOOP}), 任务终止。")
