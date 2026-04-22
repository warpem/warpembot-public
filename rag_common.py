"""Shared utilities for RAG indexing and search."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "qwen3-embedding:8b"
THREADS_DIR = "threads"
INDEX_PATH = "embeddings.npz"
REPOS_DIR = "repos"
REPO_NAMES = ["warp", "warpylib", "torch-projectors", "warpem.github.io", "relion"]


def code_index_path(repo: str) -> str:
    """Return the npz index file path for a repo."""
    return f"code_index_{repo}.npz"


def code_meta_path(repo: str) -> str:
    """Return the metadata JSON file path for a repo."""
    return f"code_meta_{repo}.json"


def message_id_to_hash(message_id: str) -> str:
    """Convert an email message ID to a filesystem-safe SHA256 hash."""
    return hashlib.sha256(message_id.encode()).hexdigest()


def save_message(msg: dict, threads_dir: str = THREADS_DIR) -> str:
    """Save a message dict as a JSON file in threads_dir. Returns the file path."""
    os.makedirs(threads_dir, exist_ok=True)
    h = message_id_to_hash(msg["message_id"])
    path = os.path.join(threads_dir, f"{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(msg, f, ensure_ascii=False, indent=2)
    return path


def load_message(path: str) -> dict:
    """Load a message dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text: str, instruction: str | None = None) -> np.ndarray | None:
    """Get L2-normalized embedding vector from Ollama. Returns None on error.

    Args:
        text: The text to embed.
        instruction: Optional task instruction for query-side embeddings.
            Qwen3 embedding models are instruction-aware: queries benefit from
            a task description prefix, while documents should be embedded without one.
    """
    if instruction:
        text = f"Instruct: {instruction}\nQuery: {text}"
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        if resp.status_code != 200:
            print(f"Ollama error: {resp.status_code} {resp.text}")
            return None
        vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def get_embeddings_batch(
    texts: list[str],
    batch_size: int = 64,
) -> list[np.ndarray | None]:
    """Get L2-normalized embeddings for multiple texts in batched Ollama calls.

    Returns a list the same length as texts, with None for any that failed.
    """
    results: list[np.ndarray | None] = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": batch},
            )
            if resp.status_code != 200:
                print(f"Ollama batch error: {resp.status_code} {resp.text}")
                continue
            embeddings = resp.json()["embeddings"]
            for i, raw in enumerate(embeddings):
                vec = np.array(raw, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results[start + i] = vec
        except Exception as e:
            print(f"Batch embedding error: {e}")
    return results


def load_index(path: str = INDEX_PATH) -> tuple[np.ndarray | None, list[str]]:
    """Load the embedding index from disk. Returns (vectors, id_list) or (None, []) if missing."""
    if not os.path.exists(path):
        return None, []
    data = np.load(path, allow_pickle=True)
    vectors = data["vectors"]
    ids = [id_.replace("\\", "/") for id_ in data["ids"].tolist()]
    return vectors, ids


def save_index(vectors: np.ndarray | None, ids: list[str], path: str = INDEX_PATH) -> None:
    """Save the embedding index to disk. No-op if index is empty."""
    if vectors is None or len(ids) == 0:
        return
    np.savez(path, vectors=vectors, ids=np.array(ids))


def add_to_index(
    vectors: np.ndarray | None,
    ids: list[str],
    new_vec: np.ndarray,
    new_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Add a vector to the index, skipping if the ID already exists. Returns updated (vectors, ids)."""
    if new_id in ids:
        return vectors, ids
    new_vec = new_vec.reshape(1, -1)
    if vectors is None:
        return new_vec, [new_id]
    return np.vstack([vectors, new_vec]), ids + [new_id]


def load_code_meta(path: str) -> dict:
    """Load code metadata from JSON. Returns empty structure if missing.

    Normalizes backslashes to forward slashes in all path keys for
    cross-platform compatibility.
    """
    if not os.path.exists(path):
        return {"checksums": {}, "chunks": {}}
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["checksums"] = {k.replace("\\", "/"): v for k, v in meta["checksums"].items()}
    meta["chunks"] = {
        k.replace("\\", "/"): {**v, "file": v["file"].replace("\\", "/")}
        for k, v in meta["chunks"].items()
    }
    return meta


def save_code_meta(meta: dict, path: str) -> None:
    """Save code metadata to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file's contents, normalized to CRLF line endings.

    Reads in binary and normalizes all line endings to CRLF so checksums
    match across platforms regardless of git's autocrlf setting.
    """
    with open(filepath, "rb") as f:
        content = f.read()
    content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").replace(b"\n", b"\r\n")
    h = hashlib.sha256(content)
    return h.hexdigest()


def remove_from_index(
    vectors: np.ndarray | None,
    ids: list[str],
    ids_to_remove: set[str],
) -> tuple[np.ndarray | None, list[str]]:
    """Remove entries from the index by ID. Returns updated (vectors, ids)."""
    if vectors is None or len(ids) == 0 or not ids_to_remove:
        return vectors, ids
    keep = [i for i, id_ in enumerate(ids) if id_ not in ids_to_remove]
    if not keep:
        return None, []
    return vectors[keep], [ids[i] for i in keep]


def search(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    ids: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Search the index by dot product. Returns list of (id_hash, score) sorted by score desc."""
    if vectors is None or len(ids) == 0:
        return []
    scores = vectors @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in top_indices]
