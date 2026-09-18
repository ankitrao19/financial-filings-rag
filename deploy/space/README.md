---
title: Financial Filings RAG
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: serve.py
pinned: false
short_description: RAG over SEC 10-K/10-Q filings with Langfuse tracing
---

# Financial Filings RAG

Question answering over SEC 10-K / 10-Q filings for 15 companies.

- UI: `/`
- API docs: `/rag/docs` (`POST /rag/ask`, `POST /rag/feedback`, `GET /rag/health`, `GET /rag/tickers`)

Retrieval: nomic-embed-text-v1.5 + ChromaDB, filing inferred from the question. Answers: gpt-4o-mini. Traced with Langfuse.
