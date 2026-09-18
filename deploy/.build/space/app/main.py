"""
main.py — single-process entrypoint for deployment: FastAPI at /, Gradio UI mounted at /ui.

    venv/bin/uvicorn app.main:app --port 7860
"""

import os

# the UI calls the API over HTTP; in one process that is this same server
os.environ.setdefault("RAG_API_URL", f"http://127.0.0.1:{os.getenv('PORT', '7860')}")

import gradio as gr  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from app.api import app  # noqa: E402
from app.ui import demo  # noqa: E402


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


app = gr.mount_gradio_app(app, demo, path="/ui")