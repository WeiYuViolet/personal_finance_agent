"""matplotlib 消费统计图表：支出分类饼图 + 每日支出柱状图，保存为 PNG 供 Gradio 展示。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

from .config import CHART_DIR  # noqa: E402

# 中文字体按可用性回退，避免方块字
_CANDIDATE_FONTS = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei", "Microsoft YaHei",
    "PingFang SC", "Arial Unicode MS",
]


def _setup_font() -> None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATE_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_font()


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]", "_", s)


def category_pie(category_share: dict[str, dict], year: int, month: int) -> Path:
    """支出分类占比饼图。category_share: {分类: {amount, share}}。"""
    labels, values = [], []
    for cat, info in category_share.items():
        labels.append(f"{cat} {info['share']}%")
        values.append(info["amount"])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    colors = plt.cm.Set3.colors[: len(values)]
    ax.pie(values, labels=labels, autopct=lambda p: f"{p:.1f}%",
           colors=colors, startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"{year}年{month}月 支出分类占比", fontsize=13, pad=14)
    ax.axis("equal")
    out = CHART_DIR / f"pie_{year}{month:02d}_{datetime.now().strftime('%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def daily_bar(bills: list[dict], year: int, month: int) -> Path:
    """每日支出柱状图（仅统计 expense）。"""
    daily: dict[int, float] = {}
    for b in bills:
        if b["bill_type"] != "expense":
            continue
        day = int(b["bill_date"].split("-")[2])
        daily[day] = round(daily.get(day, 0.0) + b["amount"], 2)
    days = sorted(daily)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(days, [daily[d] for d in days], color="#4C78A8", width=0.7)
    ax.set_xlabel("日期（日）")
    ax.set_ylabel("支出金额（元）")
    ax.set_title(f"{year}年{month}月 每日支出趋势", fontsize=13)
    ax.set_xticks(days)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    out = CHART_DIR / f"bar_{year}{month:02d}_{datetime.now().strftime('%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
