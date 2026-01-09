import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "repo_chunks")
REPO_WORKDIR = os.getenv("REPO_WORKDIR", "./data/repo");
