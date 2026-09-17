FROM python:3.13-slim

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    INDEX_REPO=hereIcome/filings-rag-index
WORKDIR /home/user/app

# CPU-only torch first, so requirements.txt doesn't pull the CUDA build
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# bake the embedding model into the image (no download on cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

# bake the vector store + period map -> data/chromadb, data/filing-periods.json (matches pipeline/paths.py)
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    python -c "import os, pathlib; from huggingface_hub import snapshot_download; \
t = pathlib.Path('/run/secrets/HF_TOKEN'); \
snapshot_download(os.environ['INDEX_REPO'], repo_type='dataset', local_dir='data', \
                  token=t.read_text().strip() if t.exists() else None)"

COPY --chown=user pipeline pipeline
COPY --chown=user app app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]