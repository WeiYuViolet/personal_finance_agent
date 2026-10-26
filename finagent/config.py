"""全局配置：从 .env 读取模型参数，集中管理项目路径。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（finagent 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------- 路径 ----------
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"              # 个人理财笔记（RAG 知识库目录）
DB_PATH = PROJECT_ROOT / "output" / "finance.db"
CHART_DIR = PROJECT_ROOT / "output" / "charts"

for _p in (DATA_DIR, NOTES_DIR, DB_PATH.parent, CHART_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------- 大模型配置（OpenAI 兼容协议）----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---------- 应用配置 ----------
SEED_DEMO_DATA = os.getenv("SEED_DEMO_DATA", "1") == "1"
APP_PORT = int(os.getenv("APP_PORT", "7860"))

# 支持的支出 / 收入分类（供智能体与工具校验，也可由用户自定义新增）
EXPENSE_CATEGORIES = ["餐饮", "交通", "购物", "娱乐", "学习", "住宿", "医疗", "通讯", "其他"]
INCOME_CATEGORIES = ["生活费", "奖学金", "兼职", "红包", "退款", "工资", "其他"]


def llm_configured() -> bool:
    """检查是否已配置可用的大模型密钥。"""
    return bool(LLM_API_KEY) and bool(LLM_MODEL)
