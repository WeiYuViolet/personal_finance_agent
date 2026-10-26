"""本地个人财务记账智能体（Personal Finance Agent）包。

模块划分：
- config:      全局配置（.env 读取、路径、模型参数）
- db:           SQLite 持久化（用户、账单、长期偏好）
- memory:       短期对话记忆 + 长期偏好记忆
- rag_store:    个人理财笔记 RAG 检索（离线 TF-IDF）
- chart:        matplotlib 消费统计图表
- tools:        LangChain 自定义工具（记账 / 查询 / 绘图 / RAG）
- agent:        LangGraph ReAct 智能体编排
- ui:           Gradio 前端交互界面
"""

__version__ = "1.0.0"
