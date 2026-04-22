#!/usr/bin/env python3
"""CLI script for bulk mbox import and index rebuild."""

import argparse
import email
import email.policy
import fnmatch
import mailbox
import os
import sys

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from rag_common import (
    OLLAMA_URL,
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
    get_embeddings_batch,
    load_index,
    save_index,
    add_to_index,
    remove_from_index,
    compute_file_checksum,
    load_code_meta,
    save_code_meta,
)

INCLUDE_EXTENSIONS = {".cs", ".xaml", ".py", ".cpp", ".cu", ".h", ".c", ".md", ".css", ".js"}
EXCLUDE_DIRS = {"bin", "obj", "build", "__pycache__", ".git", "node_modules"}


def chunk_file(
    filepath: str,
    repo_relative_path: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[tuple[str, str, int, int]]:
    """Split a file into overlapping chunks with file path context.

    Returns list of (chunk_id, chunk_text, start_line, end_line).
    chunk_id format: "{repo_relative_path}::{chunk_index}".
    Lines are 1-indexed.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    chunks = []
    stride = chunk_size - overlap
    chunk_idx = 0
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_text = f"# File: {repo_relative_path}\n" + "".join(lines[start:end])
        chunk_id = f"{repo_relative_path}::{chunk_idx}"
        chunks.append((chunk_id, chunk_text, start + 1, end))
        chunk_idx += 1
        if end == len(lines):
            break
        start += stride

    return chunks


def walk_repo_files(repo_dir: str) -> list[str]:
    """Walk a repo directory and return sorted relative paths of source files to index."""
    results = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if fnmatch.fnmatch(f, "*.generated.*"):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in INCLUDE_EXTENSIONS:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, repo_dir).replace("\\", "/")
                results.append(rel_path)
    return sorted(results)


def diff_checksums(
    current: dict[str, str],
    stored: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Compare current and stored checksums. Returns (new, changed, deleted) file lists."""
    current_set = set(current.keys())
    stored_set = set(stored.keys())
    new_files = sorted(current_set - stored_set)
    deleted_files = sorted(stored_set - current_set)
    changed_files = sorted(f for f in current_set & stored_set if current[f] != stored[f])
    return new_files, changed_files, deleted_files


def extract_body(msg: email.message.EmailMessage) -> str:
    """Extract plain text body from an email message.

    Prefers text/plain part. Falls back to stripping HTML via BeautifulSoup.
    """
    if msg.is_multipart():
        # Walk parts, prefer text/plain
        plain_parts = []
        html_parts = []
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    plain_parts.append(payload.decode(charset, errors="replace"))
            elif ct == "text/html":
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode(charset, errors="replace"))
        if plain_parts:
            return "\n".join(plain_parts).strip()
        if html_parts:
            return "\n".join(BeautifulSoup(h, "html.parser").get_text(separator="\n") for h in html_parts).strip()
        return ""
    else:
        ct = msg.get_content_type()
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload is None:
            return ""
        text = payload.decode(charset, errors="replace")
        if ct == "text/html":
            return BeautifulSoup(text, "html.parser").get_text(separator="\n").strip()
        return text.strip()


def parse_mbox_message(msg) -> dict | None:
    """Parse a mailbox message into our JSON format. Returns None if no message ID."""
    msg_id = msg.get("Message-ID") or msg.get("Message-Id")
    if not msg_id:
        return None

    subject = msg.get("Subject", "")
    sender = msg.get("From", "")
    date = msg.get("Date", "")
    in_reply_to = msg.get("In-Reply-To", "")
    body = extract_body(msg)

    return {
        "message_id": msg_id.strip(),
        "thread_id": in_reply_to.strip() if in_reply_to else "",
        "subject": subject,
        "sender": sender,
        "date": date,
        "body": body,
    }


def cmd_import_mbox(mbox_path: str) -> None:
    """Import all messages from an mbox file, save as JSON, embed, and build index."""
    if not os.path.exists(mbox_path):
        print(f"Error: {mbox_path} not found")
        sys.exit(1)

    mbox = mailbox.mbox(mbox_path)
    messages = []
    seen_ids = set()

    print(f"Parsing {mbox_path}...")
    for i, msg in enumerate(mbox):
        parsed = parse_mbox_message(msg)
        if parsed is None:
            continue
        if parsed["message_id"] in seen_ids:
            continue
        seen_ids.add(parsed["message_id"])
        messages.append(parsed)
        if (i + 1) % 100 == 0:
            print(f"  Parsed {i + 1} messages...", flush=True)

    print(f"Parsed {len(messages)} unique messages.")

    # Save all message JSON files
    print("Saving message files...")
    os.makedirs(THREADS_DIR, exist_ok=True)
    for msg in messages:
        save_message(msg, THREADS_DIR)

    # Embed all messages
    vectors, ids = load_index(INDEX_PATH)
    existing_ids = set(ids) if ids else set()
    to_embed = [msg for msg in messages if message_id_to_hash(msg["message_id"]) not in existing_ids]

    embedded = 0
    batch_size = 64
    for batch_start in tqdm(range(0, len(to_embed), batch_size), desc="Embedding messages", unit="batch"):
        batch = to_embed[batch_start : batch_start + batch_size]
        texts = [msg["body"] for msg in batch]
        vecs = get_embeddings_batch(texts, batch_size=batch_size)
        for msg, vec in zip(batch, vecs):
            if vec is not None:
                id_hash = message_id_to_hash(msg["message_id"])
                vectors, ids = add_to_index(vectors, ids, vec, id_hash)
                embedded += 1
        if embedded > 0:
            save_index(vectors, ids, INDEX_PATH)

    save_index(vectors, ids, INDEX_PATH)
    print(f"Done. Index contains {len(ids)} embeddings.")


def cmd_rebuild() -> None:
    """Rebuild the embedding index from all JSON files in threads/."""
    if not os.path.exists(THREADS_DIR):
        print(f"Error: {THREADS_DIR}/ not found")
        sys.exit(1)

    json_files = sorted(f for f in os.listdir(THREADS_DIR) if f.endswith(".json"))
    print(f"Found {len(json_files)} message files.")

    vectors = None
    ids = []

    batch_size = 64
    for batch_start in tqdm(range(0, len(json_files), batch_size), desc="Embedding messages", unit="batch"):
        batch_files = json_files[batch_start : batch_start + batch_size]
        batch_msgs = [load_message(os.path.join(THREADS_DIR, f)) for f in batch_files]
        batch_hashes = [f.replace(".json", "") for f in batch_files]
        texts = [msg["body"] for msg in batch_msgs]
        vecs = get_embeddings_batch(texts, batch_size=batch_size)
        for id_hash, vec in zip(batch_hashes, vecs):
            if vec is not None:
                vectors, ids = add_to_index(vectors, ids, vec, id_hash)
        save_index(vectors, ids, INDEX_PATH)

    save_index(vectors, ids, INDEX_PATH)
    print(f"Done. Index contains {len(ids)} embeddings.")


def cmd_update_repos() -> None:
    """Update code indexes for all repos. Only re-embeds changed files."""
    try:
        requests.get(OLLAMA_URL, timeout=5)
    except Exception:
        print("Error: Ollama is not running. Start it before running update-repos.")
        sys.exit(1)

    for repo_name in REPO_NAMES:
        repo_dir = os.path.join(REPOS_DIR, repo_name)
        if not os.path.isdir(repo_dir):
            print(f"  Skipping {repo_name} (not found)")
            continue

        print(f"Checking {repo_name}...")

        idx_path = code_index_path(repo_name)
        meta_path = code_meta_path(repo_name)

        # Load existing state
        meta = load_code_meta(meta_path)
        vectors, ids = load_index(idx_path)

        # Walk and checksum current files
        files = walk_repo_files(repo_dir)
        current_checksums = {}
        for f in files:
            current_checksums[f] = compute_file_checksum(os.path.join(repo_dir, f))

        # Diff
        new_files, changed_files, deleted_files = diff_checksums(
            current_checksums, meta["checksums"]
        )

        if not new_files and not changed_files and not deleted_files:
            print(f"  {repo_name}: no changes.")
            continue

        print(f"  {repo_name}: {len(new_files)} new, {len(changed_files)} changed, {len(deleted_files)} deleted files.")

        # Remove old chunks for changed + deleted files
        ids_to_remove = set()
        for f in changed_files + deleted_files:
            prefix = f + "::"
            ids_to_remove.update(id_ for id_ in (ids or []) if id_.startswith(prefix))
            meta["chunks"] = {k: v for k, v in meta["chunks"].items() if not k.startswith(prefix)}

        if ids_to_remove:
            vectors, ids = remove_from_index(vectors, ids, ids_to_remove)

        # Remove checksums for deleted files
        for f in deleted_files:
            meta["checksums"].pop(f, None)

        # Chunk and embed new + changed files
        files_to_embed = new_files + changed_files
        all_chunks = []
        for rel_path in files_to_embed:
            full_path = os.path.join(repo_dir, rel_path)
            for chunk in chunk_file(full_path, rel_path):
                all_chunks.append((rel_path, chunk))
        total_chunks = len(all_chunks)

        embed_count = 0
        embedded_files = set()
        batch_size = 64
        for batch_start in tqdm(
            range(0, len(all_chunks), batch_size), desc=f"  {repo_name}", unit="batch"
        ):
            batch = all_chunks[batch_start : batch_start + batch_size]
            texts = [chunk_text for _, (_, chunk_text, _, _) in batch]
            vecs = get_embeddings_batch(texts, batch_size=batch_size)
            for (rel_path, (chunk_id, _, start_line, end_line)), vec in zip(batch, vecs):
                if vec is not None:
                    vectors, ids = add_to_index(vectors, ids, vec, chunk_id)
                    meta["chunks"][chunk_id] = {
                        "file": rel_path,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                    embed_count += 1
                    embedded_files.add(rel_path)
                else:
                    tqdm.write(f"  Warning: embedding failed for {chunk_id}, skipping")
            save_index(vectors, ids, idx_path)
            save_code_meta(meta, meta_path)

        # Only persist checksums for files where at least one chunk was embedded.
        # If all chunks failed, omitting the checksum ensures the file is retried
        # on the next run rather than silently dropped from the index forever.
        for rel_path in embedded_files:
            meta["checksums"][rel_path] = current_checksums[rel_path]

        # Final save. save_index is a no-op for an empty index (vectors is None / ids is
        # empty), which would leave a stale index file on disk if all previously-indexed
        # files were deleted. Explicitly remove the file in that case so callers never
        # read stale data.
        if vectors is not None and ids:
            save_index(vectors, ids, idx_path)
        elif os.path.exists(idx_path):
            os.remove(idx_path)
        save_code_meta(meta, meta_path)
        total = len(ids) if ids else 0
        print(f"  {repo_name}: done. {embed_count} chunks embedded, {total} total in index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG indexing for Google Group threads")
    sub = parser.add_subparsers(dest="command")

    import_parser = sub.add_parser("import-mbox", help="Import messages from an mbox file")
    import_parser.add_argument("mbox_path", help="Path to the .mbox file")

    sub.add_parser("rebuild", help="Rebuild index from existing thread JSON files")

    sub.add_parser("update-repos", help="Update code indexes for all repos (incremental)")

    args = parser.parse_args()
    if args.command == "import-mbox":
        cmd_import_mbox(args.mbox_path)
    elif args.command == "rebuild":
        cmd_rebuild()
    elif args.command == "update-repos":
        cmd_update_repos()
    else:
        parser.print_help()
