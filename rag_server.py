# rag_server.py
"""Stdio MCP server for RAG search over Google Group threads."""

import json
import os
from mcp.server.fastmcp import FastMCP

from rag_common import (
    THREADS_DIR,
    INDEX_PATH,
    REPOS_DIR,
    REPO_NAMES,
    code_index_path,
    code_meta_path,
    message_id_to_hash,
    save_message,
    load_message,
    get_embedding,
    load_index,
    save_index,
    add_to_index,
    load_code_meta,
    search,
)

mcp_server = FastMCP("rag")

SEARCH_INSTRUCTION = "Given a user question about cryo-EM software, retrieve relevant email threads that answer the question"
CODE_SEARCH_INSTRUCTION = "Given a user question about cryo-EM software, retrieve relevant source code that answers the question"

# Load index into memory on startup
_vectors, _ids = load_index(INDEX_PATH)

# Load per-repo code indexes into memory on startup
_code_indexes: dict[str, tuple] = {}
for _repo in REPO_NAMES:
    _vecs, _code_ids = load_index(code_index_path(_repo))
    _meta = load_code_meta(code_meta_path(_repo))
    if _vecs is not None:
        _code_indexes[_repo] = (_vecs, _code_ids, _meta)


@mcp_server.tool()
def search_group_threads(query: str, top_k: int = 5) -> list[dict]:
    """Search historical Google Group threads by semantic similarity.

    Args:
        query: The search query text (use the full email body for best results).
        top_k: Number of results to return (default 5).

    Returns:
        List of matching messages with subject, sender, date, body, and similarity score.
    """
    global _vectors, _ids

    if _vectors is None or len(_ids) == 0:
        return []

    query_vec = get_embedding(query, instruction=SEARCH_INSTRUCTION)
    if query_vec is None:
        return [{"error": "Failed to compute embedding. Is Ollama running?"}]

    results = search(query_vec, _vectors, _ids, top_k=top_k)

    output = []
    for id_hash, score in results:
        msg_path = os.path.join(THREADS_DIR, f"{id_hash}.json")
        if os.path.exists(msg_path):
            msg = load_message(msg_path)
            output.append({
                "subject": msg.get("subject", ""),
                "sender": msg.get("sender", ""),
                "date": msg.get("date", ""),
                "body": msg.get("body", ""),
                "similarity_score": round(score, 4),
            })
    return output


@mcp_server.tool()
def index_message(
    message_id: str,
    subject: str,
    sender: str,
    date: str,
    body: str,
    thread_id: str = "",
) -> dict:
    """Index a new message for future RAG search.

    Saves the message as a JSON file and adds its embedding to the search index.
    Deduplicates by message ID — calling this with an already-indexed message is safe.

    Args:
        message_id: The email Message-ID header value.
        subject: Email subject line.
        sender: Sender email address.
        date: ISO 8601 date string.
        body: Plain text body of the message.
        thread_id: Optional thread identifier.

    Returns:
        Dict with message_id, status ("new" or "exists"), and id_hash.
    """
    global _vectors, _ids

    id_hash = message_id_to_hash(message_id)

    # Check dedup
    if id_hash in _ids:
        return {"message_id": message_id, "status": "exists", "id_hash": id_hash}

    # Save message file
    msg = {
        "message_id": message_id,
        "thread_id": thread_id,
        "subject": subject,
        "sender": sender,
        "date": date,
        "body": body,
    }
    save_message(msg, THREADS_DIR)

    # Embed and add to index
    vec = get_embedding(body)
    if vec is None:
        return {
            "message_id": message_id,
            "status": "saved_but_not_embedded",
            "id_hash": id_hash,
            "error": "Embedding failed. Message saved but not searchable.",
        }

    _vectors, _ids = add_to_index(_vectors, _ids, vec, id_hash)
    save_index(_vectors, _ids, INDEX_PATH)

    return {"message_id": message_id, "status": "new", "id_hash": id_hash}


@mcp_server.tool()
def search_repo_code(repo: str, query: str, top_k: int = 5) -> list[dict]:
    """Search source code in a repository by semantic similarity.

    Available repos: warp, warpylib, torch-projectors, warpem.github.io, relion

    Args:
        repo: Repository name to search.
        query: Search query (formulate a targeted query from the user's question).
        top_k: Number of results to return (default 5).

    Returns:
        List of matching code locations with file path, line range, and similarity score.
    """
    if repo not in REPO_NAMES:
        return [{"error": f"Unknown repo '{repo}'. Available: {', '.join(REPO_NAMES)}"}]
    if repo not in _code_indexes:
        return []

    vecs, ids, meta = _code_indexes[repo]

    query_vec = get_embedding(query, instruction=CODE_SEARCH_INSTRUCTION)
    if query_vec is None:
        return [{"error": "Failed to compute embedding. Is Ollama running?"}]

    results = search(query_vec, vecs, ids, top_k=top_k)

    output = []
    for chunk_id, score in results:
        chunk_info = meta["chunks"].get(chunk_id, {})
        output.append({
            "file": chunk_info.get("file", chunk_id),
            "start_line": chunk_info.get("start_line", 0),
            "end_line": chunk_info.get("end_line", 0),
            "similarity_score": round(score, 4),
        })
    return output


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
