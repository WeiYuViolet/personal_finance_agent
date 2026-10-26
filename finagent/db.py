"""SQLite 持久化层：用户表 + 账单表，实现"多用户隔离 + 长期历史账单保存"。

设计说明：
- 所有账单都带 user_id，天然实现不同用户账单互不混淆；
- 账单永久落库，构成智能体的长期记忆（历史账本）；
- users 表保存用户偏好（如月预算、记账习惯），作为长期个性化记忆。
"""
from __future__ import annotations

import json
import sqlite3
from calendar import monthrange
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from .config import DB_PATH, EXPENSE_CATEGORIES, INCOME_CATEGORIES


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """建表（幂等）。"""
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id      TEXT PRIMARY KEY,
                nickname     TEXT,
                monthly_budget REAL,
                preferences  TEXT DEFAULT '{}',
                created_at   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bills (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                bill_type   TEXT NOT NULL CHECK (bill_type IN ('income','expense')),
                amount      REAL NOT NULL CHECK (amount > 0),
                category    TEXT NOT NULL,
                note        TEXT DEFAULT '',
                bill_date   TEXT NOT NULL,          -- YYYY-MM-DD
                created_at  TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bills_user ON bills(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bills_date ON bills(bill_date)")
        _ensure_categories_schema(conn)


def _ensure_categories_schema(conn: sqlite3.Connection) -> None:
    """确保 categories 表为最新结构（UNIQUE 含 created_by，实现多用户分类隔离）。

    该表为轻量自定义分类表；若检测到旧结构（UNIQUE 不含 created_by）则直接重建，
    因为自定义分类数据量小且可重新添加。
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='categories'"
    ).fetchone()
    sql = (row["sql"] if row else "") or ""
    if "UNIQUE (bill_type, category, created_by)" in sql:
        return  # 已是新结构
    if row is not None:
        conn.execute("DROP TABLE categories")  # 旧结构：重建
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_type   TEXT NOT NULL CHECK (bill_type IN ('income','expense')),
            category    TEXT NOT NULL,
            created_by  TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            UNIQUE (bill_type, category, created_by)
        )
        """
    )


# ---------------------------------------------------------------- 用户 / 长期偏好
def ensure_user(user_id: str, nickname: Optional[str] = None,
                monthly_budget: Optional[float] = None) -> None:
    """用户首次出现时建档；已存在则按传入值补齐（不覆盖已有非空值）。"""
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        now = datetime.now().isoformat(timespec="seconds")
        if row is None:
            conn.execute(
                "INSERT INTO users(user_id, nickname, monthly_budget, preferences, created_at)"
                " VALUES(?,?,?,?,?)",
                (user_id, nickname, monthly_budget, "{}", now),
            )
        else:
            if nickname and not row["nickname"]:
                conn.execute("UPDATE users SET nickname=? WHERE user_id=?", (nickname, user_id))
            if monthly_budget and not row["monthly_budget"]:
                conn.execute(
                    "UPDATE users SET monthly_budget=? WHERE user_id=?", (monthly_budget, user_id)
                )


def get_profile(user_id: str) -> dict[str, Any]:
    """读取用户长期偏好（供注入系统提示词，实现个性化记忆）。"""
    ensure_user(user_id)
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    prefs = {}
    if row and row["preferences"]:
        try:
            prefs = json.loads(row["preferences"])
        except json.JSONDecodeError:
            prefs = {}
    return {
        "user_id": user_id,
        "nickname": row["nickname"] if row else None,
        "monthly_budget": row["monthly_budget"] if row else None,
        "preferences": prefs,
    }


def set_preference(user_id: str, key: str, value: Any) -> None:
    """更新一项长期偏好（如月预算、常用分类）。"""
    ensure_user(user_id)
    profile = get_profile(user_id)
    profile["preferences"][key] = value
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET preferences=? WHERE user_id=?",
            (json.dumps(profile["preferences"], ensure_ascii=False), user_id),
        )

def set_monthly_budget(user_id: str, budget: float) -> None:
    """更新用户的月度预算（元）。"""
    ensure_user(user_id)
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET monthly_budget=? WHERE user_id=?",
            (float(budget), user_id),
        )


# ---------------------------------------------------------------- 分类管理
def normalize_bill_type(bill_type: str) -> str:
    """把中文/英文变体归一化为 'income'/'expense'；非法值抛 ValueError。"""
    bt = (bill_type or "").strip()
    if bt in ("income", "收入", "收", "inc"):
        return "income"
    if bt in ("expense", "支出", "消费", "exp"):
        return "expense"
    raise ValueError(f"未知的收支类型「{bill_type}」，应为 收入(income) 或 支出(expense)")


def list_categories(bill_type: str, user_id: str = "default") -> list[str]:
    """返回某类型下该用户可用的全部分类（预设 + 该用户自定义），去重保序。"""
    bill_type = normalize_bill_type(bill_type)
    base = EXPENSE_CATEGORIES if bill_type == "expense" else INCOME_CATEGORIES
    with _conn() as conn:
        rows = conn.execute(
            "SELECT category FROM categories WHERE bill_type=? AND created_by=?",
            (bill_type, user_id),
        ).fetchall()
    result = list(base)
    for r in rows:
        if r["category"] not in result:
            result.append(r["category"])
    return result


def add_category(bill_type: str, category: str, user_id: str = "default") -> bool:
    """为该用户新增一个自定义分类（幂等）。返回是否真正新增（False 表示已存在）。"""
    bill_type = normalize_bill_type(bill_type)
    category = (category or "").strip()
    if not category:
        raise ValueError("分类名称不能为空")
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM categories WHERE bill_type=? AND category=? AND created_by=?",
            (bill_type, category, user_id),
        ).fetchone()
        if exists:
            return False
        conn.execute(
            "INSERT INTO categories(bill_type, category, created_by, created_at)"
            " VALUES(?,?,?,?)",
            (bill_type, category, user_id, now),
        )
    return True


def ensure_category(bill_type: str, category: str, user_id: str = "default") -> str:
    """确保该用户可用此分类：预设或该用户已有自定义分类则直接返回；否则创建。"""
    bill_type = normalize_bill_type(bill_type)
    category = (category or "").strip()
    if valid_category(bill_type, category, user_id):
        return category
    add_category(bill_type, category, user_id)
    return category


# ---------------------------------------------------------------- 账单 CRUD
def add_bill(user_id: str, bill_type: str, amount: float, category: str,
             note: str = "", bill_date: Optional[str] = None) -> dict[str, Any]:
    """新增一条收支记录，返回落库后的记录。bill_date 缺省为今天。"""
    ensure_user(user_id)
    bill_type = normalize_bill_type(bill_type)
    bill_date = bill_date or datetime.now().date().isoformat()
    # 校验日期格式
    datetime.strptime(bill_date, "%Y-%m-%d")
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO bills(user_id, bill_type, amount, category, note, bill_date, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (user_id, bill_type, float(amount), category, note, bill_date, now),
        )
        rid = cur.lastrowid
        row = conn.execute("SELECT * FROM bills WHERE id=?", (rid,)).fetchone()
    return dict(row)


def query_bills(user_id: str, year: int, month: Optional[int] = None,
                bill_type: Optional[str] = None,
                category: Optional[str] = None) -> list[dict[str, Any]]:
    """按年（可选月、类型、分类）查询账单，按日期升序。month 为 None 表示全年。"""
    sql = "SELECT * FROM bills WHERE user_id=? AND strftime('%Y', bill_date)=?"
    args: list[Any] = [user_id, f"{year:04d}"]
    if month is not None:
        sql += " AND strftime('%m', bill_date)=?"
        args.append(f"{month:02d}")
    if bill_type:
        sql += " AND bill_type=?"
        args.append(bill_type)
    if category:
        sql += " AND category=?"
        args.append(category)
    sql += " ORDER BY bill_date, id"
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def summarize_month(user_id: str, year: int, month: int) -> dict[str, Any]:
    """月度汇总：总收入 / 总支出 / 结余 / 支出分类占比（供工具与图表使用）。"""
    bills = query_bills(user_id, year, month)
    income = round(sum(b["amount"] for b in bills if b["bill_type"] == "income"), 2)
    expense_rows = [b for b in bills if b["bill_type"] == "expense"]
    expense = round(sum(b["amount"] for b in expense_rows), 2)

    cat: dict[str, float] = {}
    for b in expense_rows:
        cat[b["category"]] = round(cat.get(b["category"], 0.0) + b["amount"], 2)
    cat_share = {
        k: {"amount": v, "share": round(v / expense * 100, 1) if expense else 0.0}
        for k, v in sorted(cat.items(), key=lambda x: -x[1])
    }
    return {
        "year": year,
        "month": month,
        "count": len(bills),
        "income": income,
        "expense": expense,
        "balance": round(income - expense, 2),
        "category_share": cat_share,
        "bills": bills,
    }


def recent_bills(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """最近 N 条账单（前端面板展示）。"""
    ensure_user(user_id)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bills WHERE user_id=? ORDER BY bill_date DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_bills_days(user_id: str, days: int = 3) -> list[dict[str, Any]]:
    """最近 N 天（含今天）的账单，按日期倒序（前端面板展示）。"""
    ensure_user(user_id)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bills WHERE user_id=? AND bill_date >= date('now', 'localtime', ?)"
            " ORDER BY bill_date DESC, id DESC",
            (user_id, f"-{days - 1} days"),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_bill(user_id: str, bill_id: int) -> bool:
    """删除指定用户名下的一条账单，返回是否删除成功。带 user_id 防止越权删除他人账单。"""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM bills WHERE id=? AND user_id=?", (bill_id, user_id))
        return cur.rowcount > 0


def get_bill(user_id: str, bill_id: int) -> Optional[dict[str, Any]]:
    """查询指定用户名下的一条账单，不存在或不属于该用户则返回 None。"""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM bills WHERE id=? AND user_id=?", (bill_id, user_id)
        ).fetchone()
    return dict(row) if row else None


def update_bill_category(user_id: str, bill_id: int, category: str) -> Optional[dict[str, Any]]:
    """修改指定用户名下一条账单的分类，返回更新后的记录；不存在或不属于该用户则返回 None。"""
    category = (category or "").strip()
    if not category:
        raise ValueError("分类名称不能为空")
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE bills SET category=? WHERE id=? AND user_id=?",
            (category, bill_id, user_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM bills WHERE id=?", (bill_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- 演示数据
def seed_demo_data(user_id: str = "default", force: bool = False) -> None:
    """为 default 用户写入近三个月演示账单（收入 + 多分类支出），便于评审快速体验。

    默认仅在用户尚无账单时写入；force=True 时先清空该用户账单再重新写入，用于重新生成
    干净演示数据。可在 .env 设 SEED_DEMO_DATA=0 关闭自动写入。
    """
    if not force and recent_bills(user_id, limit=1):
        return
    if force:
        with _conn() as conn:
            conn.execute("DELETE FROM bills WHERE user_id=?", (user_id,))
    ensure_user(user_id, nickname="演示用户", monthly_budget=2000.0)
    today = datetime.now().date()

    def d(day: int, y: int, m: int) -> str:
        return f"{y:04d}-{m:02d}-{day:02d}"

    # 每月固定收支模板：(日, 类型, 金额, 分类, 备注)
    monthly = [
        (1, "income", 1500.0, "生活费", "父母转账"),
        (2, "expense", 16.0, "餐饮", "食堂早餐+午餐"),
        (3, "expense", 45.0, "餐饮", "外卖+奶茶"),
        (4, "expense", 8.0, "交通", "地铁通勤"),
        (5, "expense", 59.9, "购物", "生活用品"),
        (6, "expense", 15.0, "娱乐", "视频会员"),
        (8, "expense", 25.0, "餐饮", "聚餐（同学）"),
        (10, "expense", 12.0, "交通", "共享单车"),
        (12, "expense", 39.0, "学习", "打印/教材"),
        (15, "income", 400.0, "兼职", "家教/勤工助学"),
        (16, "expense", 50.0, "通讯", "话费"),
        (18, "expense", 28.0, "餐饮", "咖啡+轻食"),
        (20, "expense", 35.0, "购物", "文具/日用品"),
        (22, "expense", 20.0, "娱乐", "电影"),
        (25, "expense", 18.0, "餐饮", "早餐"),
        (26, "expense", 60.0, "医疗", "感冒药"),
    ]

    y, m = today.year, today.month
    for offset in range(3):
        # 当前月只写到今天，前两个月写满整月
        last_day = today.day if offset == 0 else monthrange(y, m)[1]
        for day, bt, amt, cat, note in monthly:
            if day <= last_day:
                add_bill(user_id, bt, amt, cat, note, d(day, y, m))
        # 最旧一个月追加一笔奖学金，展示收入分类多样性
        if offset == 2:
            add_bill(user_id, "income", 800.0, "奖学金", "学术竞赛奖金", d(5, y, m))
        # 当前月：追加最近三天账单，保证右侧面板即时可见
        if offset == 0:
            recent = [
                (2, "expense", 9.9, "购物", "日用品"),
                (1, "expense", 28.0, "餐饮", "午餐"),
                (0, "expense", 15.0, "餐饮", "早餐"),
            ]
            for back, bt, amt, cat, note in recent:
                add_bill(user_id, bt, amt, cat, note, (today - timedelta(days=back)).isoformat())
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1


def valid_category(bill_type: str, category: str, user_id: str = "default") -> bool:
    """判断分类对该用户是否合法（预设分类 + 该用户自定义分类）。"""
    return category in list_categories(bill_type, user_id)
