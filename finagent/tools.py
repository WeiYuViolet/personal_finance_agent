"""自定义工具（Tool / Function Calling）层。

注册给 ReAct 智能体的 9 个工具：
1. add_bill_record        新增收支记录（写 SQLite 长期账本）
2. query_monthly_bills    查询/汇总某月账单（读长期账本）
3. generate_spending_chart 生成消费统计图表（matplotlib，产物路径推入 artifacts）
4. search_finance_notes   RAG 检索本地理财笔记
5. set_budget             设置/更新当前用户月度预算
6. query_year_bills       年度收支汇总（分类占比、按月分布）
7. delete_bill_record     锁定待删除账单（需 UI 确认后才真正删除）
8. add_category           添加自定义分类（按用户隔离）
9. update_bill_category   修改账单分类（支持自动创建自定义分类）

用户身份通过 contextvars 在每次调用前注入，避免让大模型凭空猜测 user_id；
图表产物与待删除账单等"工具写回、调用方读取"的状态改用进程内全局变量 + 锁维护。
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from . import chart, db, rag_store

# 请求级上下文：本轮对话所属用户（只读，工具在子 context 中可正确读取）
_current_user: ContextVar[str] = ContextVar("current_user", default="default")

# 产物与待删除账单是"工具写回、调用方读取"的状态。工具经 LangChain/LangGraph 执行时
# 可能运行在独立 context，ContextVar 的 set 不会传播回调用方 context，因此改用
# 进程内全局状态 + 锁（本应用为同步单进程，同一时刻仅一个请求写入）。
_lock = threading.Lock()
_artifacts: list[str] = []
_pending_delete: Optional[dict] = None


def set_context(user_id: str) -> None:
    """在每次 invoke 前调用，绑定当前用户并清空产物/待删除状态。"""
    global _artifacts, _pending_delete
    _current_user.set(user_id)
    with _lock:
        _artifacts = []
        _pending_delete = None


def take_artifacts() -> list[str]:
    """调用结束后取回本轮生成的图表等文件路径。"""
    with _lock:
        return list(_artifacts)


def take_pending_delete() -> Optional[dict]:
    """调用结束后取回本轮锁定的待删除账单（未真正删除，需 UI 确认）。"""
    with _lock:
        return _pending_delete


# ---------------------------------------------------------------- 工具 1：记账
@tool
def add_bill_record(
    bill_type: str,
    amount: float,
    category: str,
    note: str = "",
    date: str = "",
    user_specified_category: bool = False,
) -> str:
    """新增一条收支记录并写入本地账本。

    Args:
        bill_type: 类型，'expense'（支出）或 'income'（收入）。
        amount: 金额，单位元，必须为正数。
        category: 分类。若用户明确说出分类则原样传入；否则按备注/金额推断。
        note: 备注，例如“食堂午餐”，可空。
        date: 发生日期 YYYY-MM-DD，可空（空则记为今天）。
        user_specified_category: 用户是否在原话中明确说出了分类。
            True  → 使用用户指定的分类；若不在列表则自动创建并加入列表。
            False → 该分类由你推断；若不在列表则自动归入“其他”。
    """
    user_id = _current_user.get()
    date = date.strip() or datetime.now().date().isoformat()
    try:
        bt = db.normalize_bill_type(bill_type)
        category = (category or "").strip() or "其他"
        if user_specified_category:
            category = db.ensure_category(bt, category, user_id)
        elif not db.valid_category(bt, category, user_id):
            category = "其他"
        rec = db.add_bill(user_id, bt, float(amount), category, note.strip(), date)
    except ValueError as e:
        return f"记账失败：{e}。请检查金额是否为正数、日期是否为 YYYY-MM-DD、收支类型是否正确。"
    sign = "支出" if rec["bill_type"] == "expense" else "收入"
    return (
        f"记账成功：已记录 {date} {sign} {rec['category']} {rec['amount']} 元"
        f"（备注：{rec['note'] or '无'}）。账本ID={rec['id']}。"
    )


# ---------------------------------------------------------------- 工具 2：查询/汇总
@tool
def query_monthly_bills(year: int, month: int, bill_type: str = "all",
                       category: str = "") -> str:
    """查询并汇总指定年月的账单，返回总额、结余与支出分类占比。

    Args:
        year: 四位年份，如 2026。
        month: 月份 1-12。
        bill_type: all=收支都看（默认），expense=仅支出，income=仅收入。
        category: 可选，只看某一分类，如“餐饮”；空则统计全部分类。
    """
    user_id = _current_user.get()
    bt = None if bill_type == "all" else bill_type
    rows = db.query_bills(user_id, year, month, bill_type=bt, category=category or None)
    if not rows:
        return f"{year}年{month}月没有查询到账单记录。"

    income = round(sum(r["amount"] for r in rows if r["bill_type"] == "income"), 2)
    expense_rows = [r for r in rows if r["bill_type"] == "expense"]
    expense = round(sum(r["amount"] for r in expense_rows), 2)

    lines = [f"=== {year}年{month}月账单（共{len(rows)}笔）==="]
    lines.append(f"总收入：{income:.2f} 元；总支出：{expense:.2f} 元；结余：{income - expense:.2f} 元。")
    if expense_rows:
        cat: dict[str, float] = {}
        for r in expense_rows:
            cat[r["category"]] = round(cat.get(r["category"], 0) + r["amount"], 2)
        lines.append("支出分类：")
        for c, amt in sorted(cat.items(), key=lambda x: -x[1]):
            share = amt / expense * 100
            lines.append(f"  - {c}：{amt:.2f} 元（占比 {share:.1f}%）")
    lines.append("明细：")
    for r in rows:
        sign = "支" if r["bill_type"] == "expense" else "收"
        lines.append(
            f"  [ID:{r['id']}] {r['bill_date']} [{sign}] {r['category']} {r['amount']:.2f}元 {r['note']}"
        )
    if len(rows) > 1:
        lines.append(
            f"（存在 {len(rows)} 笔匹配记录；若需对其中某一笔进行删除或修改，"
            "请结合日期/金额/备注进一步说明是哪一笔。）"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 工具 3：图表
@tool
def generate_spending_chart(year: int, month: int, chart_type: str = "pie") -> str:
    """为指定年月生成消费统计图（支出分类饼图/每日支出柱状图），并在界面展示。

    Args:
        year: 四位年份。
        month: 月份 1-12。
        chart_type: 'pie'=分类占比饼图（默认）；'bar'=每日支出柱状图；'both'=两张都生成。
    """
    user_id = _current_user.get()
    summary = db.summarize_month(user_id, year, month)
    if summary["count"] == 0:
        return f"{year}年{month}月暂无账单，无法绘图，请先记账。"

    produced: list[str] = []
    if chart_type in ("pie", "both") and summary["category_share"]:
        p = chart.category_pie(summary["category_share"], year, month)
        produced.append(str(p))
    if chart_type in ("bar", "both"):
        p = chart.daily_bar(summary["bills"], year, month)
        produced.append(str(p))

    with _lock:
        _artifacts.extend(produced)

    cat_text = "；".join(
        f"{c} {info['share']}%" for c, info in summary["category_share"].items()
    )
    return (
        f"已生成{year}年{month}月消费图。总支出 {summary['expense']:.2f} 元，"
        f"分类占比：{cat_text or '无'}。图表文件：{', '.join(produced)}"
    )


# ---------------------------------------------------------------- 工具 4：RAG
@tool
def search_finance_notes(query: str) -> str:
    """从用户本地"个人理财笔记"知识库中检索与问题相关的规则/经验（RAG）。

    Args:
        query: 要查询的理财问题，例如“大学生如何控制餐饮开销”“50/30/20 预算法”。
    """
    return rag_store.rag.search_text(query, k=3)


# ---------------------------------------------------------------- 工具 5：设置预算
@tool
def set_budget(monthly_budget: float) -> str:
    """设置或更新当前用户的月度预算（元）。

    Args:
        monthly_budget: 每月预算金额，单位元，必须为正数。例如 2000。
    """
    user_id = _current_user.get()
    if monthly_budget <= 0:
        return "设置失败：预算必须为正数。"
    db.set_monthly_budget(user_id, monthly_budget)
    return f"已设置用户「{user_id}」的月预算为 {monthly_budget:.0f} 元。"


# ---------------------------------------------------------------- 工具 6：年度汇总
@tool
def query_year_bills(year: int, bill_type: str = "all", category: str = "") -> str:
    """查询并汇总指定全年的账单，返回年度总收入/总支出/结余、支出分类占比与按月分布。

    Args:
        year: 四位年份，如 2026。
        bill_type: expense=支出，income=收入，'all'=两者都看（默认）。
        category: 可选，只看某一分类，如“餐饮”；空则统计全部分类。
    """
    user_id = _current_user.get()
    bt = None if bill_type == "all" else bill_type
    rows = db.query_bills(user_id, year, bill_type=bt, category=category or None)
    if not rows:
        return f"{year}年没有查询到账单记录。"

    income = round(sum(r["amount"] for r in rows if r["bill_type"] == "income"), 2)
    expense_rows = [r for r in rows if r["bill_type"] == "expense"]
    expense = round(sum(r["amount"] for r in expense_rows), 2)

    lines = [f"=== {year}年全年账单（共{len(rows)}笔）==="]
    lines.append(f"总收入：{income:.2f} 元；总支出：{expense:.2f} 元；结余：{income - expense:.2f} 元。")
    if expense_rows:
        cat: dict[str, float] = {}
        for r in expense_rows:
            cat[r["category"]] = round(cat.get(r["category"], 0.0) + r["amount"], 2)
        lines.append("全年支出分类：")
        for c, amt in sorted(cat.items(), key=lambda x: -x[1]):
            share = amt / expense * 100
            lines.append(f"  - {c}：{amt:.2f} 元（占比 {share:.1f}%）")
    monthly: dict[int, dict[str, float]] = {}
    for r in rows:
        m = int(r["bill_date"][5:7])
        slot = monthly.setdefault(m, {"income": 0.0, "expense": 0.0})
        slot[r["bill_type"]] = round(slot[r["bill_type"]] + r["amount"], 2)
    lines.append("按月分布：")
    for m in sorted(monthly):
        s = monthly[m]
        lines.append(f"  {m}月：收入 {s['income']:.2f} 元，支出 {s['expense']:.2f} 元，结余 {s['income'] - s['expense']:.2f} 元")
    if len(rows) > 1:
        lines.append(
            f"（存在 {len(rows)} 笔匹配记录；若需对其中某一笔进行删除或修改，"
            "请结合日期/金额/备注进一步说明是哪一笔。）"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 工具 7：删除账单（锁定，需 UI 确认）
@tool
def delete_bill_record(bill_id: int) -> str:
    """锁定一条待删除账单并返回其详情，不会真正删除；真正删除需用户在界面点击【确认删除】。

    Args:
        bill_id: 要删除的账单 ID（整数）。
    """
    user_id = _current_user.get()
    bill = db.get_bill(user_id, bill_id)
    if bill is None:
        return f"未找到账本ID={bill_id} 的记录（不存在，或不属于当前用户「{user_id}」）。"
    global _pending_delete
    with _lock:
        _pending_delete = bill
    sign = "收入" if bill["bill_type"] == "income" else "支出"
    return (
        f"已锁定待删除账单：{bill['bill_date']} {sign} {bill['category']} {bill['amount']}元"
        f"（{bill['note'] or '无'}）。该账单尚未删除，请在界面点击【确认删除】后才会真正删除。"
    )


# ---------------------------------------------------------------- 工具 8：添加/管理分类
@tool
def add_category(bill_type: str, category: str) -> str:
    """添加一个自定义分类到分类列表，之后记账时可直接使用该分类。

    Args:
        bill_type: 'expense'（支出）或 'income'（收入）。
        category: 分类名称，例如“宠物”“游戏”。
    """
    user_id = _current_user.get()
    try:
        created = db.add_category(bill_type, category, user_id)
    except ValueError as e:
        return f"添加分类失败：{e}"
    if not created:
        return f"分类「{category}」已存在，无需重复添加。"
    sign = "支出" if db.normalize_bill_type(bill_type) == "expense" else "收入"
    return f"已添加分类「{category}」到{sign}分类列表，之后记账可直接使用。"


# ---------------------------------------------------------------- 工具 9：修改账单分类
@tool
def update_bill_category(bill_id: int, category: str,
                         user_specified_category: bool = True) -> str:
    """修改指定账单的分类。

    Args:
        bill_id: 账单 ID（整数）。可先调用 query_monthly_bills 列出明细获取 ID。
        category: 新的分类名称。
        user_specified_category: 用户是否明确说出了新分类（默认 True）。
            True  → 新分类不在列表则自动创建并加入列表。
            False → 新分类不在列表则归入"其他"。
    """
    user_id = _current_user.get()
    try:
        bill = db.get_bill(user_id, bill_id)
        if bill is None:
            return f"未找到账本ID={bill_id} 的记录（不存在，或不属于当前用户「{user_id}」）。"
        bt = bill["bill_type"]
        category = (category or "").strip() or "其他"
        if user_specified_category:
            category = db.ensure_category(bt, category, user_id)
        elif not db.valid_category(bt, category, user_id):
            category = "其他"
        rec = db.update_bill_category(user_id, bill_id, category)
    except ValueError as e:
        return f"修改失败：{e}"
    sign = "支出" if rec["bill_type"] == "expense" else "收入"
    return (
        f"已把账单ID={bill_id}（{rec['bill_date']} {sign} {rec['note'] or '无备注'}）"
        f"的分类修改为「{rec['category']}」。"
    )


# 注册工具列表（供 LangGraph 绑定）
TOOLS = [add_bill_record, query_monthly_bills, query_year_bills, generate_spending_chart,
         search_finance_notes, set_budget, add_category, delete_bill_record,
         update_bill_category]


