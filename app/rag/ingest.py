from __future__ import annotations
from pathlib import Path
from typing import Dict

from git import Repo

from app.config import REPO_WORKDIR
from app.rag.chunk import iter_repo_files, chunk_file
from app.rag.milvus_store import MilvusStore

def clone_or_pull(repo_url: str) -> Path:
    Path(REPO_WORKDIR).mkdir(parents=True, exist_ok=True)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = Path(REPO_WORKDIR) / repo_name

    if repo_dir.exists():
        repo = Repo(str(repo_dir))
        repo.remotes.origin.pull()
    else:
        Repo.clone_from(repo_url, str(repo_dir))

    return repo_dir

def ingest_repo(repo_url: str, collection_name: str) -> Dict:
    repo_path = clone_or_pull(repo_url)
    store = MilvusStore(collection_name)

    scanned_files = 0
    chunks_added = 0

    for f in iter_repo_files(repo_path):
        scanned_files += 1
        chunks = chunk_file(f, repo_path)
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        ids = [f"{c.metadata['path']}::{c.metadata['chunk_id']}" for c in chunks]

        chunks_added += store.add_texts(texts, metas, ids)

    return {
        "repo_url": repo_url,
        "repo_path": str(repo_path),
        "scanned_files": scanned_files,
        "chunks_added": chunks_added,
        "collection": collection_name,
    }
