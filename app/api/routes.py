from fastapi import APIRouter, Query
from app.config import MILVUS_COLLECTION
from app.rag.ingest import ingest_repo

router = APIRouter()

@router.post("/ingest")
def ingest(
    repo_url: str = Query(..., description="GitHub repo URL"),
    collection: str = Query(MILVUS_COLLECTION, description="Milvus collection name"),
):
    return ingest_repo(repo_url=repo_url, collection_name=collection)
