"""核心模块离线测试（不调用大模型）：数据库、多用户隔离、RAG、图表、工具调用。

运行：
    cd personal_finance_agent
    python tests/test_core.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import finagent.chart as chart  # noqa: E402
import finagent.db as db  # noqa: E402
import finagent.rag_store as rag_store  # noqa: E402
import finagent.tools as tools  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    # 重定向到临时库与临时图表目录，避免污染正式数据
    db.DB_PATH = tmp / "test.db"
    chart.CHART_DIR = tmp / "charts"
    chart.CHART_DIR.mkdir(exist_ok=True)

    # 1) 建库 + 记账 + 汇总
    db.init_db()
    db.ensure_user("alice", monthly_budget=2000)
    db.add_bill("alice", "expense", 18.5, "餐饮", "午餐", "2026-10-01")
    db.add_bill("alice", "expense", 32.0, "餐饮", "聚餐", "2026-10-03")
    db.add_bill("alice", "expense", 59.9, "购物", "生活用品", "2026-10-05")
    db.add_bill("alice", "income", 1500, "生活费", "转账", "2026-10-01")

    s = db.summarize_month("alice", 2026, 10)
    assert s["expense"] == 110.4, s["expense"]
    assert s["income"] == 1500
    assert s["balance"] == 1389.6
    assert abs(s["category_share"]["餐饮"]["share"] - (50.5 / 110.4 * 100)) < 0.1
    print("[PASS] 1. 建库/记账/月度汇总（收入、支出、结余、分类占比）正确")

    # 2) 多用户隔离
    db.add_bill("bob", "expense", 999, "娱乐", "游戏", "2026-10-02")
    assert db.summarize_month("bob", 2026, 10)["expense"] == 999
    assert db.summarize_month("alice", 2026, 10)["expense"] == 110.4
    print("[PASS] 2. 多用户账单隔离，互不串账")

    # 3) RAG 检索
    rag = rag_store.FinanceRAG(notes_dir=ROOT / "data" / "notes")
    n = rag.build()
    assert n > 5, n
    hits = rag.search("50/30/20 预算分配法", k=2)
    assert hits and "50" in hits[0].text, hits
    hits2 = rag.search("如何减少餐饮外卖奶茶开销", k=3)
    joined = " ".join(h.text for h in hits2)
    assert hits2 and ("餐饮" in joined or "外卖" in joined or "奶茶" in joined)
    print(f"[PASS] 3. RAG 离线检索（索引 {n} 片段，两条查询均命中）")

    # 4) 图表生成
    pie = chart.category_pie(s["category_share"], 2026, 10)
    bar = chart.daily_bar(s["bills"], 2026, 10)
    assert pie.exists() and pie.stat().st_size > 1000
    assert bar.exists() and bar.stat().st_size > 1000
    print("[PASS] 4. matplotlib 饼图/柱状图成功生成")

    # 5) 工具链（模拟 ReAct 调用，绑定 alice 上下文）
    tools.set_context("alice")
    r1 = tools.add_bill_record.invoke(
        {"bill_type": "expense", "amount": 12, "category": "餐饮", "note": "早餐", "date": "2026-10-08"}
    )
    assert "记账成功" in r1, r1
    r2 = tools.query_monthly_bills.invoke({"year": 2026, "month": 10, "bill_type": "expense"})
    assert "总支出" in r2 and "餐饮" in r2, r2
    # 5b) 默认（all）调用必须同时统计收入与支出，保证结余正确
    r2_all = tools.query_monthly_bills.invoke({"year": 2026, "month": 10})
    assert "总收入：1500.00" in r2_all and "总支出：122.40" in r2_all, r2_all
    r3 = tools.generate_spending_chart.invoke({"year": 2026, "month": 10, "chart_type": "both"})
    artifacts = tools.take_artifacts()
    assert len(artifacts) == 2, artifacts
    r4 = tools.search_finance_notes.invoke({"query": "大学生如何控制餐饮开销"})
    assert "笔记" in r4 or "检索" in r4
    print("[PASS] 5. 四个自定义工具串行调用成功，图表产物已收集")

    # 6) 删除闭环：工具锁定 → pending 回传 → 确认删除 → 越权防护
    tools.set_context("alice")
    rec = db.add_bill("alice", "expense", 25, "交通", "打车", "2026-10-20")
    tools.delete_bill_record.invoke({"bill_id": rec["id"]})
    pending = tools.take_pending_delete()
    assert pending is not None and pending["id"] == rec["id"], pending
    assert db.delete_bill("alice", rec["id"]) is True
    assert db.get_bill("alice", rec["id"]) is None
    # 越权：alice 无法删除 bob 的账单
    rec_bob = db.add_bill("bob", "expense", 50, "购物", "测试", "2026-10-20")
    assert db.delete_bill("alice", rec_bob["id"]) is False
    assert db.get_bill("bob", rec_bob["id"]) is not None
    print("[PASS] 6. 删除闭环（锁定/回传/确认删除/越权防护）正确")

    # 7) 分类管理与收支类型校验
    tools.set_context("alice")
    # 7a) 非法收支类型应报错而非静默归入 income
    r_bad = tools.add_bill_record.invoke(
        {"bill_type": "未知类型", "amount": 10, "category": "餐饮"}
    )
    assert "记账失败" in r_bad, r_bad
    # 7b) 用户明确指定的自定义分类 → 自动创建并加入列表
    r_new = tools.add_bill_record.invoke(
        {"bill_type": "expense", "amount": 66, "category": "宠物", "note": "猫粮",
         "date": "2026-10-20", "user_specified_category": True}
    )
    assert "记账成功" in r_new and "宠物" in r_new, r_new
    assert "宠物" in db.list_categories("expense", "alice")
    # 7c) 用户未指定、推断的分类不在列表 → 归入"其他"
    r_other = tools.add_bill_record.invoke(
        {"bill_type": "expense", "amount": 20, "category": "乱写的分类", "note": "测试",
         "date": "2026-10-20", "user_specified_category": False}
    )
    assert "记账成功" in r_other and "其他" in r_other, r_other
    # 7d) 手动添加分类工具（幂等）
    r_add = tools.add_category.invoke({"bill_type": "expense", "category": "游戏"})
    assert "已添加" in r_add, r_add
    assert "游戏" in db.list_categories("expense", "alice")
    r_add2 = tools.add_category.invoke({"bill_type": "expense", "category": "游戏"})
    assert "已存在" in r_add2, r_add2
    # 7e) 分类按用户隔离：alice 的自定义分类 bob 不可见
    assert "宠物" not in db.list_categories("expense", "bob")
    assert "游戏" not in db.list_categories("expense", "bob")
    print("[PASS] 7. 分类管理（非法类型报错/自定义分类创建/未指定归其他/手动添加/用户隔离）正确")

    # 8) 修改账单分类
    tools.set_context("alice")
    rec8 = db.add_bill("alice", "expense", 30, "餐饮", "修改测试", "2026-10-20")
    # 8a) 改为预设分类
    r_upd = tools.update_bill_category.invoke({"bill_id": rec8["id"], "category": "交通"})
    assert "修改为" in r_upd and "交通" in r_upd, r_upd
    assert db.get_bill("alice", rec8["id"])["category"] == "交通"
    # 8b) 改为用户明确的自定义分类（自动创建）
    r_upd2 = tools.update_bill_category.invoke(
        {"bill_id": rec8["id"], "category": "健身", "user_specified_category": True}
    )
    assert "健身" in r_upd2, r_upd2
    assert db.get_bill("alice", rec8["id"])["category"] == "健身"
    assert "健身" in db.list_categories("expense", "alice")
    # 8c) 越权：bob 无法修改 alice 的账单
    tools.set_context("bob")
    r_upd3 = tools.update_bill_category.invoke({"bill_id": rec8["id"], "category": "娱乐"})
    assert "未找到" in r_upd3, r_upd3
    tools.set_context("alice")
    # 8d) 修改不存在的账单
    r_upd4 = tools.update_bill_category.invoke({"bill_id": 999999, "category": "娱乐"})
    assert "未找到" in r_upd4, r_upd4
    print("[PASS] 8. 修改账单分类（预设/自定义/自动创建/越权防护/不存在）正确")

    print("\n全部测试通过 ✅")


if __name__ == "__main__":
    main()
