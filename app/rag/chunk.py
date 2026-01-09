from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cs", ".go", ".rb", ".php",
    ".md", ".txt",
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".next", ".pytest_cache", ".mypy_cache",
}

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
)



@dataclass
class DocChunk:
    text: str
    metadata: dict

def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)

def iter_repo_files(repo_path: Path) -> Iterable[Path]:
    for p in repo_path.rglob("*"):
        if p.is_file() and not should_ignore(p) and p.suffix.lower() in CODE_EXTS:
            yield p

def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""

def chunk_file(file_path: Path, repo_root: Path) -> List[DocChunk]:
    content = read_text_file(file_path)
    if not content.strip():
        return []

    rel = str(file_path.relative_to(repo_root))
    splits = _SPLITTER.split_text(content)

    chunks: List[DocChunk] = []
    for i, s in enumerate(splits):
        chunks.append(
            DocChunk(
                text=s,
                metadata={"path": rel, "chunk_id": i, "ext": file_path.suffix.lower()},
            )
        )
    return chunks
