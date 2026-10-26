# 本地个人财务记账智能体（Personal Finance Agent）

> 北京物资学院 · 人工智能综合实践 · 期末大作业
> 技术栈：Python 原生 + LangChain + LangGraph（ReAct）+ 自定义工具调用 + RAG + SQLite 长期记忆 + Gradio

一个面向大学生日常收支管理的本地记账智能体：你用自然语言说一句，智能体自主完成
**记账 → 查账汇总 → 生成消费图表 → 检索本地理财笔记 → 输出分析与省钱建议** 的多步任务。

---

## 一、核心功能（≥3 项，全部落地）

| # | 功能 | 实现方式 |
|---|------|----------|
| 1 | **自然语言记账** | 自定义工具 `add_bill_record`，写入 SQLite 长期账本，支持支出/收入、分类、备注、日期 |
| 2 | **月度消费查询与分析** | 自定义工具 `query_monthly_bills`，汇总收入/支出/结余、分类占比、明细 |
| 3 | **消费统计可视化** | 自定义工具 `generate_spending_chart`，matplotlib 生成分类饼图 / 每日支出柱状图，界面展示 |
| 4 | **RAG 理财知识检索** | `search_finance_notes` 检索本地 `data/notes` 理财笔记（md/txt/pdf），结合依据给建议 |
| 5 | **记忆管理** | 短期：按用户+会话保存对话上下文；长期：SQLite 永久保存历史账单与用户偏好（月预算等），多用户账本隔离 |
| 6 | **ReAct 任务规划** | LangGraph `create_react_agent` 驱动 Thought→Tool→Observation 循环，多工具串行调用 |
| 7 | **年度账单汇总** | 自定义工具 `query_year_bills`，汇总全年收入/支出/结余、分类占比、按月分布 |
| 8 | **删除账单** | 自定义工具 `delete_bill_record` 锁定待删账单 → 用户在界面点击【确认删除】后才真正删除，带 user_id 防越权 |

### 典型一句话指令（智能体自动分步执行）
> “帮我统计本月开销，看看餐饮占比，同时给我下个月省钱建议”

执行链路：`query_monthly_bills` 读账本 → `generate_spending_chart` 出饼图 →
`search_finance_notes` RAG 检索 → 综合输出分析报告。

---

## 二、环境配置与安装

### 1. Python 版本
Python 3.10+（开发验证 3.12）。建议新建虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置大模型密钥（必填）

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`，填入任意 **OpenAI 兼容接口** 的密钥：

```ini
# DeepSeek（推荐，学生友好，控制台 https://platform.deepseek.com 申请）
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的密钥
LLM_TEMPERATURE=0.3
```

也可切换 OpenAI / 本地 Ollama：

```ini
# OpenAI
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx

# 本地 Ollama（无需密钥，模型提前 ollama pull）
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

---

## 三、运行方式

```bash
python run.py
```

浏览器打开终端提示的地址（默认 `http://127.0.0.1:7860`）。

- 首次启动会自动建库，并为 `default` 用户写入演示账单（可在 `.env` 设 `SEED_DEMO_DATA=0` 关闭）；
- 左侧对话记账/分析，右侧实时刷新最近三天账单、月预算面板与统计图；
- 在右上角“用户ID”填不同昵称/学号，即可体验**多用户账本隔离**；
- 可上传自己的理财笔记（.md/.txt/.pdf）并入 RAG 知识库。

### 离线自检（不需要大模型密钥）

```bash
python tests/test_core.py
```

覆盖：建库记账、月度汇总、多用户隔离、RAG 检索、图表生成、四个工具串行调用。

---

## 四、项目结构

```
personal_finance_agent/
├── run.py                  # 入口：初始化 + 启动 Gradio
├── requirements.txt        # 依赖清单
├── .env.example           # 配置模板
├── data/
│   └── notes/              # RAG 知识库（个人理财笔记 md/txt/pdf）
│       └── 个人理财笔记.md
├── finagent/
│   ├── config.py           # .env 配置、路径、分类常量
│   ├── db.py              # SQLite：users/bills 表、CRUD、月度汇总、演示数据
│   ├── memory.py          # 短期对话记忆 + 长期偏好召回
│   ├── rag_store.py       # 文档加载/切分 + TF-IDF 离线 RAG 检索
│   ├── chart.py           # matplotlib 饼图/柱状图
│   ├── tools.py          # 7 个 LangChain 自定义工具（Function Calling）
│   ├── agent.py           # LangGraph ReAct 智能体编排、系统提示词
│   └── ui.py              # Gradio 界面
├── tests/
│   └── test_core.py       # 离线核心测试
└── docs/
    ├── 设计文档.md         # 选题/架构/实现/测试/提示词完整版
    ├── architecture.png    # 整体架构图（mermaid 导出）
    ├── sequence.png        # 核心调度流程图（mermaid 导出）
    └── 演示视频脚本.md      # 演示视频分镜与讲稿
```

---

## 五、常见问题

1. **图表中文显示方块？** 系统缺少中文字体。Linux 安装 `fonts-noto-cjk`，Windows/macOS 一般自带；
   `chart.py` 已按 Noto/文泉驿/黑体/雅黑顺序自动回退。
2. **提示未配置 LLM_API_KEY？** 检查 `.env` 是否存在、密钥是否填对、是否在项目根目录启动。
3. **RAG 想换更强向量库？** `rag_store.py` 已封装加载/切分/检索接口，可把 TF-IDF 替换为
   FAISS + OpenAI Embedding 或本地 sentence-transformers，工具层无需改动。
4. **数据存在哪？** 全部本地：账本 `output/finance.db`，图表 `output/charts/`，不上云。
