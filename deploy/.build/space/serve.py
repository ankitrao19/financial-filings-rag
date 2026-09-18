"""
serve.py — Hugging Face Spaces entrypoint (Gradio SDK on ZeroGPU hardware).

ZeroGPU constraints this file works around:
  - `spaces` must be imported before torch, and the Space must start through gr.Blocks.launch()
    (ZeroGPU hooks launch to register the app), so we can't just run uvicorn on app.main.
    Instead: launch the Gradio UI, then mount the FastAPI app into Gradio's server at /rag.
  - ZeroGPU wants at least one @spaces.GPU function. Everything here runs on CPU (query
    embedding is cheap), so gpu_check is registered but never called — no GPU quota used.
  - it reports cuda as available everywhere, so the embedder is pinned to CPU (EMBED_DEVICE).

No build step on this SDK, so the vector store is fetched from the HF dataset repo at startup
(Space disk is ephemeral — each restart re-downloads it).

Routes: /  (UI) · /rag/docs · /rag/ask · /rag/feedback · /rag/health · /rag/tickers

    python serve.py        # locally: same layout on http://localhost:7860
"""

import os
from pathlib import Path

try:
    import spaces  # noqa: F401  — HF ZeroGPU; must come before torch
except ImportError:
    spaces = None

PORT = int(os.getenv("PORT", "7860"))
ROOT = Path(__file__).resolve().parent
INDEX_REPO = os.getenv("INDEX_REPO", "hereIcome/filings-rag-index")

os.environ.setdefault("EMBED_DEVICE", "cpu")
# the UI calls the API over HTTP; here that is this same server, under /rag
os.environ.setdefault("RAG_API_URL", f"http://127.0.0.1:{PORT}/rag")

if not (ROOT / "data" / "chromadb").exists():
    from huggingface_hub import snapshot_download

    print(f"downloading vector store from {INDEX_REPO} ...", flush=True)
    snapshot_download(INDEX_REPO, repo_type="dataset", local_dir=ROOT / "data", token=os.getenv("HF_TOKEN"))

from app import api  # noqa: E402
from app.ui import demo  # noqa: E402

if spaces is not None:
    @spaces.GPU(duration=10)
    def gpu_check():
        import torch
        return torch.cuda.is_available()

api.load_state()  # mounted sub-apps don't get their lifespan run

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, ssr_mode=False, prevent_thread_lock=True)
    demo.app.mount("/rag", api.app)
    print(f"API mounted at http://localhost:{PORT}/rag/docs", flush=True)
    try:
        demo.block_thread()
    finally:
        api.langfuse.flush()
