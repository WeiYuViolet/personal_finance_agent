"""LangGraph ReAct 智能体编排层。

采用 langgraph.prebuilt.create_react_agent 构建标准 ReAct 循环：
    用户提问 → 模型 Thought（规划/判断）→ Tool Call（调用工具）
    → Observation（工具返回）→ 再次 Thought → …→ 给出最终答复。

多步规划示例（用户一句话触发工具串行）：
    "帮我统计本月开销，看看餐饮占比，再给下个月省钱建议"
    1) query_monthly_bills 读取账本 → 2) generate_spending_chart 出图
    → 3) search_finance_notes RAG 检索理财规则 → 4) 综合输出分析报告。
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from . import db, memory, tools
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, llm_configured

# ---------------------------------------------------------------- 系统提示词（完整版，设计文档同步收录）
SYSTEM_PROMPT = """你是「本地个人财务记账智能体」，面向大学生做日常收支记账与消费分析。
你不是通用闲聊机器人，所有账目数字都必须来自工具读取的本地账本，禁止凭空编造金额。

【可用工具与使用时机】
1. add_bill_record：用户说"记一笔/花了/收入"时调用，写入一条收支；金额、分类、日期要问清或按对话推断。分类处理遵循下方【记账分类规则】。
2. query_monthly_bills：用户要"查账/统计/看看某月开销/结余/某分类占比"时调用，先取数再回答。
3. generate_spending_chart：用户要看图、占比饼图、每日趋势时调用；分析月度开销时主动出图。
4. search_finance_notes：给省钱建议、预算方法、理财规则前调用，结合检索到的本地笔记作答，并注明依据；当检索返回多条笔记片段时，必须综合使用全部片段作答，不得只引用其中一条或忽略其余。
5. set_budget：用户说"我月预算XXX/这个月预算XXX/设置预算"时调用，更新当前用户的月预算金额。
6. query_year_bills：用户要"这一年/全年/某一年总开销、年度汇总、全年结余"时调用，返回全年收支、分类占比与按月分布。
7. delete_bill_record：用户要"删除/撤销/删掉某笔账单"时调用；该工具只"锁定"待删除账单并返回详情，不会真正删除，真正删除需用户在界面点击【确认删除】。调用后应把待删除详情转告用户，并提示其点击【确认删除】按钮。
8. add_category：用户要"新增/添加一个分类"（如"添加分类：宠物"）时调用，把自定义分类加入分类列表。
9. update_bill_category：用户要"修改/更改某笔账单的分类"时调用；先用 query_monthly_bills 列出明细拿到账本ID，再传入 bill_id 与新分类。

【工作规则】
- 先想清楚再调用工具（Thought）：判断该记账、查账、绘图还是查笔记；一步拿不到结论就分步串行调用。
- 用户说"本月/这个月"而未给年月时，使用系统提供的当前年月；说"上月"则减一个月。
- 支出分类：餐饮/交通/购物/娱乐/学习/住宿/医疗/通讯/其他；收入分类：生活费/奖学金/兼职/红包/退款/其他。
- 【记账分类规则】用户明确说出的分类，add_bill_record 的 user_specified_category 传 True 并原样传该分类（系统会自动创建不在列表中的新分类）；用户未说明分类时由你推断，user_specified_category 传 False，推断的分类若不在列表，系统会自动归入"其他"。
- 分析报告结构：①本月收入/支出/结余；②支出分类占比（点名最大项）；③结合检索到的全部笔记片段，综合给出 2-3 条可执行的省钱建议，并逐条注明依据来源。
- 综合全部笔记：search_finance_notes 返回的每条笔记片段都是有效依据，作答时必须全部纳入考虑并综合引用，禁止只选取相关度最高的一条而丢弃其余；每条建议都应标注其来自哪篇笔记。
- 数字一律以工具返回为准；账本为空时如实说明，并引导先记账。
- 删除账单不可逆：先 query_monthly_bills 列出明细与账本ID，再用 delete_bill_record 锁定要删的那一笔（该工具不真正删除，只是登记待删），把详情展示给用户并提示其在界面点击【确认删除】。
- 修改账单分类：先 query_monthly_bills 列出明细定位账本ID，再用 update_bill_category 修改；用户明确说出的新分类会自动创建。
- 定位账单时若用户描述匹配到多条（如同一天多笔同分类的账单），不要擅自选一条，先用 query_monthly_bills 把候选（日期/分类/金额/备注）列给用户，请用户确认是哪一笔后再操作。
- 账单的 ID 是系统内部标识，仅用于定位要删除/修改的记录；向用户回复时不要输出 ID 数字。
- 语气简洁、口语化，适合学生；不要输出工具内部字段。
"""


@lru_cache(maxsize=1)
def get_llm():
    """构造 OpenAI 兼容客户端（DeepSeek/OpenAI/Ollama 均可）。"""
    if not llm_configured():
        raise RuntimeError(
            "未检测到 LLM_API_KEY，请复制 .env.example 为 .env 并填入大模型密钥后重启。"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_agent():
    """构建并缓存 ReAct 智能体（绑定全部自定义工具）。"""
    return create_react_agent(model=get_llm(), tools=tools.TOOLS, prompt=SYSTEM_PROMPT)


def _current_period() -> tuple[int, int]:
    now = datetime.now()
    return now.year, now.month


def run(user_id: str, message: str, session_id: str = "default") -> tuple[str, list[str], dict | None]:
    """执行一轮对话，返回（最终答复文本, 本轮产物路径, 待删除账单信息或 None）。"""
    db.ensure_user(user_id)
    tools.set_context(user_id)

    # 短期记忆：取历史，追加本轮用户消息
    memory.short_term.add_user(user_id, session_id, message)
    history = memory.short_term.history(user_id, session_id)

    # 长期记忆：把用户偏好/最近账目注入系统消息
    y, m = _current_period()
    long_term = memory.recall_long_term(user_id)
    system = SystemMessage(
        content=(
            SYSTEM_PROMPT
            + f"\n\n【当前时间】{datetime.now():%Y-%m-%d}，当前年月 {y}年{m}月。\n"
            + "【长期记忆/用户画像】\n" + long_term
        )
    )

    agent = get_agent()
    result = agent.invoke({"messages": [system, *history]})

    ai_message = result["messages"][-1]
    reply = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    memory.short_term.add_ai(user_id, session_id, reply)

    return reply, tools.take_artifacts(), tools.take_pending_delete()


def reset_memory(user_id: str, session_id: str = "default") -> None:
    memory.short_term.clear(user_id, session_id)
