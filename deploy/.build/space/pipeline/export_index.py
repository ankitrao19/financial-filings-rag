"""
export_index.py — ship the vector store for deployment.

data/chromadb holds every collection (MiniLM + nomic, ~1 GB). The deployed API only reads one,
so this copies that collection (ids, documents, metadata, precomputed embeddings — no re-embedding)
into a fresh store, and optionally uploads it plus filing-periods.json to a Hugging Face dataset repo.
The Docker image downloads that repo at build time.

Usage:
    python pipeline/export_index.py                                   # -> data/chromadb-deploy
    python pipeline/export_index.py --upload <hf-user>/filings-rag-index   # also push to HF (needs HF_TOKEN)
    python pipeline/export_index.py --upload <hf-user>/filings-rag-index --private
"""

import argparse
import os
import shutil

import chromadb
from dotenv import load_dotenv

from build_chunks import EMBED_MODELS
from paths import CHROMA_PATH, DATA_DIR, ENV_FILE, PERIOD_MAP_PATH

DEPLOY_DIR = DATA_DIR / "chromadb-deploy"
PAGE = 2000


def export(embed_model):
    name = EMBED_MODELS[embed_model]["collection"]
    src = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(name)
    total = src.count()

    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    dst = chromadb.PersistentClient(path=str(DEPLOY_DIR)).create_collection(name, metadata=src.metadata)

    for offset in range(0, total, PAGE):
        page = src.get(offset=offset, limit=PAGE, include=["documents", "metadatas", "embeddings"])
        dst.add(ids=page["ids"], documents=page["documents"],
                metadatas=page["metadatas"], embeddings=page["embeddings"])
        print(f"  {min(offset + PAGE, total)}/{total}")

    assert dst.count() == total, f"copied {dst.count()} of {total}"
    print(f"exported {name} ({total} chunks) -> {DEPLOY_DIR}")


def upload(repo_id, private):
    from huggingface_hub import HfApi

    load_dotenv(ENV_FILE)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(DEPLOY_DIR),
                      path_in_repo="chromadb", commit_message="vector store")
    api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=str(PERIOD_MAP_PATH),
                    path_in_repo="filing-periods.json", commit_message="filing period map")
    print(f"uploaded -> https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-model", choices=sorted(EMBED_MODELS), default="nomic")
    ap.add_argument("--upload", metavar="REPO_ID", help="HF dataset repo, e.g. user/filings-rag-index")
    ap.add_argument("--private", action="store_true", help="create the dataset repo as private")
    ap.add_argument("--skip-export", action="store_true", help="upload an existing data/chromadb-deploy")
    args = ap.parse_args()

    if not args.skip_export:
        export(args.embed_model)
    if args.upload:
        upload(args.upload, args.private)


if __name__ == "__main__":
    main()
