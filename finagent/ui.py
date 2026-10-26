"""Gradio 前端交互界面：左侧对话，右侧账本面板 + 统计图表 + 笔记上传。"""
from __future__ import annotations

import datetime as dt

import gradio as gr

from . import agent, db, rag_store
from .config import APP_PORT, SEED_DEMO_DATA


# ---------------------------------------------------------------- 面板数据
def _profile_text(user_id: str) -> str:
    p = db.get_profile(user_id)
    budget = f"{p['monthly_budget']:.0f} 元" if p["monthly_budget"] else "未设置"
    return (
        f"**用户**：{p['user_id']}"
        + (f"（{p['nickname']}）" if p["nickname"] else "")
        + f"　|　**月预算**：{budget}"
    )


def _recent_table(user_id: str, days: int = 3) -> list[list]:
    rows = db.recent_bills_days(user_id, days=days)
    table = []
    for r in reversed(rows):
        table.append([
            r["bill_date"],
            "收入" if r["bill_type"] == "income" else "支出",
            r["category"],
            f"{r['amount']:.2f}",
            r["note"] or "",
        ])
    return table


# ---------------------------------------------------------------- 交互回调
def respond(message: str, user_id: str, history: list[dict]):
    """每轮对话：调用智能体，刷新账本面板、图表与待删除状态。"""
    history = history or []
    if message.strip():
        history.append({"role": "user", "content": message.strip()})
    try:
        reply, artifacts, pending = agent.run(user_id.strip() or "default", message.strip())
    except Exception as e:
        reply = f"⚠️ {type(e).__name__}：{e}"
        artifacts, pending = [], None
    history.append({"role": "assistant", "content": reply})

    user_id = user_id.strip() or "default"
    chart_path = artifacts[-1] if artifacts else None
    pending_id = pending["id"] if pending else None
    return (history, _recent_table(user_id), _profile_text(user_id), chart_path,
            _pending_text(pending), pending_id)


def clear_chat(history: list[dict], user_id: str) -> tuple[list[dict], str]:
    agent.reset_memory(user_id.strip() or "default")
    return [], "已清空本轮对话记忆（历史账单仍保留在账本中）。"


def refresh_panel(user_id: str):
    user_id = user_id.strip() or "default"
    db.ensure_user(user_id)
    p = db.get_profile(user_id)
    budget_val = p["monthly_budget"] or 0
    return _recent_table(user_id), _profile_text(user_id), budget_val, "暂无待删除账单", None


def update_budget(user_id: str, budget: float):
    """手动更新当前用户的月预算，20秒后提示自动消失。"""
    user_id = user_id.strip() or "default"
    try:
        budget = float(budget)
        if budget <= 0:
            return _profile_text(user_id), "⚠️ 预算必须为正数", gr.Timer(active=True)
        db.set_monthly_budget(user_id, budget)
        return _profile_text(user_id), f"✅ 已设置「{user_id}」月预算为 {budget:.0f} 元", gr.Timer(active=True)
    except (ValueError, TypeError):
        return _profile_text(user_id), "⚠️ 请输入有效数字", gr.Timer(active=True)


def clear_budget_status():
    """定时器触发：清空预算提示并停止定时器。"""
    return gr.update(value=""), gr.Timer(active=False)


def upload_notes(file_obj) -> str:
    """上传理财笔记（md/txt/pdf）并入 RAG 索引。"""
    if file_obj is None:
        return "未选择文件。"
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    n = rag_store.rag.add_document(path)
    return f"已并入知识库，当前共 {n} 个文本片段可检索。"


def _pending_text(pending: dict | None) -> str:
    """把待删除账单信息渲染成展示文本。"""
    if not pending:
        return "暂无待删除账单"
    sign = "收入" if pending["bill_type"] == "income" else "支出"
    return (
        f"**待删除**：{pending['bill_date']} {sign} "
        f"{pending['category']} {pending['amount']}元（{pending['note'] or '无'}）\n\n"
        "确认无误请点击下方【确认删除】按钮。"
    )


def confirm_delete(user_id: str, pending_id):
    """用户在界面点击【确认删除】后，真正删除被锁定的账单。"""
    user_id = user_id.strip() or "default"
    if pending_id is None:
        return "⚠️ 当前没有待删除账单。", _recent_table(user_id), None
    ok = db.delete_bill(user_id, int(pending_id))
    if ok:
        return f"✅ 已删除账本ID={int(pending_id)} 的账单。", _recent_table(user_id), None
    return f"⚠️ 删除失败：账本ID={int(pending_id)} 不存在或不属于当前用户。", _recent_table(user_id), None


# ---------------------------------------------------------------- 界面
def build_demo() -> gr.Blocks:
    db.init_db()
    with gr.Blocks(title="个人财务记账智能体") as demo:
        gr.Markdown(
            "# 💰 本地个人财务记账智能体\n"
            "ReAct 智能体 · 工具调用 · 长期账本 · RAG 理财笔记 · 消费可视化"
        )
        with gr.Row():
            # ---------------- 左：对话 ----------------
            with gr.Column(scale=3):
                user_id = gr.Textbox(label="用户ID（不同 ID 账本隔离，可填自己的昵称/学号）", value="default")
                chatbot = gr.Chatbot(height=460, label="与记账智能体对话")
                msg = gr.Textbox(
                    label="输入指令",
                    placeholder="示例：记一笔，今天午餐18元，餐饮；/ 统计本月开销，看看餐饮占比并出图，给下月省钱建议",
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话记忆")
                status = gr.Markdown("")

            # ---------------- 右：面板 ----------------
            with gr.Column(scale=2):
                profile = gr.Markdown(_profile_text("default"))
                gr.Markdown("### 设置月预算")
                with gr.Row():
                    budget_input = gr.Number(label="月预算（元）", value=0, scale=3)
                    budget_btn = gr.Button("更新预算", scale=1)
                budget_status = gr.Markdown("")
                bills_tbl = gr.Dataframe(
                    label="最近三天账单",
                    headers=["日期", "类型", "分类", "金额(元)", "备注"],
                    value=_recent_table("default"),
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("### 删除账单（对话发起，此处确认）")
                del_info = gr.Markdown(_pending_text(None))
                del_confirm_btn = gr.Button("确认删除", variant="stop")
                pending_state = gr.State(None)

                gr.Markdown("### 消费统计图")
                chart = gr.Image(label="图表（饼图/柱状图）", height=320)
                gr.Markdown("### 上传个人理财笔记（并入 RAG）")
                note_file = gr.File(label="支持 .md / .txt / .pdf", file_types=[".md", ".txt", ".pdf"])
                note_status = gr.Markdown("")

                # 预算提示自动消失定时器（20秒）
                budget_timer = gr.Timer(value=15, active=False)

        # ---------------- 事件绑定 ----------------
        msg.submit(respond, [msg, user_id, chatbot],
                   [chatbot, bills_tbl, profile, chart, del_info, pending_state])
        send_btn.click(respond, [msg, user_id, chatbot],
                       [chatbot, bills_tbl, profile, chart, del_info, pending_state])
        send_btn.click(lambda: "", None, msg)
        msg.submit(lambda: "", None, msg)
        clear_btn.click(clear_chat, [chatbot, user_id], [chatbot, status])
        user_id.change(refresh_panel, [user_id],
                       [bills_tbl, profile, budget_input, del_info, pending_state])
        budget_btn.click(update_budget, [user_id, budget_input], [profile, budget_status, budget_timer])
        budget_timer.tick(clear_budget_status, None, [budget_status, budget_timer])
        note_file.upload(upload_notes, [note_file], [note_status])
        del_confirm_btn.click(confirm_delete, [user_id, pending_state],
                              [del_info, bills_tbl, pending_state])

    return demo


def main() -> None:
    db.init_db()
    rag_store.rag.build()
    if SEED_DEMO_DATA:
        db.seed_demo_data("default")
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=APP_PORT)


if __name__ == "__main__":
    main()
