"""项目入口：初始化数据库/RAG/演示数据并启动 Gradio 界面。

用法：
    python run.py
"""
from finagent import db, rag_store
from finagent.config import SEED_DEMO_DATA
from finagent.ui import build_demo
from finagent.config import APP_PORT


def main() -> None:
    db.init_db()
    n = rag_store.rag.build()
    print(f"[初始化] RAG 知识库索引完成，共 {n} 个文本片段。")
    if SEED_DEMO_DATA:
        db.seed_demo_data("default")
        print("[初始化] 已为 default 用户写入演示账单（可在 .env 关闭）。")
    demo = build_demo()
    demo.launch(server_port=APP_PORT)


if __name__ == "__main__":
    main()
