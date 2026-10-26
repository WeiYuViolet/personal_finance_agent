"""记忆管理：短期对话记忆 + 长期记忆。

- 短期记忆：按 (user_id, session_id) 维护对话缓冲，记住"当前统计维度/刚聊的月份"等上下文；
- 长期记忆：历史账单在 SQLite（db.py）持久保存；用户偏好（预算、习惯）存 users 表，
  在每轮注入系统提示词，实现跨会话个性化。
"""
from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from . import db

# 每个会话最多保留的最近轮次（一对问答=2条消息），避免上下文无限膨胀
MAX_TURNS = 10


class ShortTermMemory:
    """线程安全的内存短期记忆，key = f"{user_id}::{session_id}"。"""

    def __init__(self, max_messages: int = MAX_TURNS * 2) -> None:
        self._store: dict[str, deque[BaseMessage]] = defaultdict(lambda: deque(maxlen=max_messages))
        self._lock = Lock()

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id}"

    def add_user(self, user_id: str, session_id: str, text: str) -> None:
        with self._lock:
            self._store[self._key(user_id, session_id)].append(HumanMessage(content=text))

    def add_ai(self, user_id: str, session_id: str, text: str) -> None:
        with self._lock:
            self._store[self._key(user_id, session_id)].append(AIMessage(content=text))

    def history(self, user_id: str, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._store[self._key(user_id, session_id)])

    def clear(self, user_id: str, session_id: str) -> None:
        with self._lock:
            self._store.pop(self._key(user_id, session_id), None)


# 全局单例
short_term = ShortTermMemory()


def recall_long_term(user_id: str) -> str:
    """把用户长期偏好拼成提示词片段，供每轮注入。"""
    profile = db.get_profile(user_id)
    parts: list[str] = [f"用户ID：{user_id}"]
    if profile["nickname"]:
        parts.append(f"昵称：{profile['nickname']}")
    if profile["monthly_budget"]:
        parts.append(f"月预算：{profile['monthly_budget']} 元")
    prefs = profile["preferences"]
    if prefs:
        for k, v in prefs.items():
            parts.append(f"偏好-{k}：{v}")
    recent = db.recent_bills(user_id, limit=3)
    if recent:
        tip = "最近记账：" + "；".join(
            f"{b['bill_date']} {b['bill_type']} {b['category']} {b['amount']}元" for b in recent
        )
        parts.append(tip)
    return "\n".join(f"- {p}" for p in parts)
