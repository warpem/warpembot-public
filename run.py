#!/usr/bin/env python3
"""Warp Bot harness — triage, draft, review, send.

Usage:
    python run.py              # Full run with review
    python run.py --no-review  # Full run, send immediately
    python run.py --just-send  # Skip to stage 3, send existing drafts
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import agentmail_client as amc
import github_client as ghc
from rag_common import (
    INDEX_PATH,
    THREADS_DIR,
    message_id_to_hash,
    save_message,
    get_embedding,
    load_index,
    save_index,
    add_to_index,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPOS_DIR = SCRIPT_DIR / "repos"
DRAFTS_DIR = SCRIPT_DIR / "drafts"
LAST_CHECK_FILE = SCRIPT_DIR / "last_check.txt"
PENDING_FILE = SCRIPT_DIR / "pending_messages.json"

ollama_process = None


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def run_cmd(cmd, check=True, capture=False, **kwargs):
    """Run a shell command. Returns CompletedProcess."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        **kwargs,
    )


def pull_repos():
    print("Pulling latest code...")
    if not REPOS_DIR.is_dir():
        return
    for repo in sorted(REPOS_DIR.iterdir()):
        if not repo.is_dir():
            continue
        print(f"  Pulling {repo.name}...")
        result = run_cmd(["git", "-C", str(repo), "pull", "--ff-only"], check=False)
        if result.returncode != 0:
            print(f"  WARNING: pull failed for {repo.name}, continuing with stale code")


def ensure_ollama():
    """Start Ollama if not running. Returns True if we started it."""
    global ollama_process
    result = run_cmd(["curl", "-s", "http://localhost:11434"], check=False, capture=True)
    if result.returncode == 0:
        print("Ollama already running.")
        return False

    print("Starting Ollama...")
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    import time
    for i in range(30):
        time.sleep(1)
        r = run_cmd(["curl", "-s", "http://localhost:11434"], check=False, capture=True)
        if r.returncode == 0:
            print("  Ollama ready.")
            return True
    print("  Ollama timeout!")
    sys.exit(1)


def stop_ollama():
    global ollama_process
    if ollama_process is not None:
        print("Stopping Ollama...")
        ollama_process.terminate()
        ollama_process.wait()
        ollama_process = None


def update_code_indexes():
    print("Ensuring embedding model is available...")
    result = run_cmd(["ollama", "pull", "qwen3-embedding:8b"], check=False)
    if result.returncode != 0:
        print("  Warning: could not pull latest model (network issue?), using cached version.")
    print("Updating code indexes...")
    run_cmd(["python3", "rag_index.py", "update-repos"], cwd=str(SCRIPT_DIR))


def invoke_claude(prompt):
    """Invoke claude CLI. Returns True on success."""
    result = run_cmd(
        ["claude", "--verbose", "-p", prompt],
        check=False,
        cwd=str(SCRIPT_DIR),
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Stage 1: Triage
# ---------------------------------------------------------------------------

def load_last_check():
    if LAST_CHECK_FILE.exists():
        ts = LAST_CHECK_FILE.read_text().strip()
        print(f"  Last check: {ts}")
        return ts
    return None


def save_last_check():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LAST_CHECK_FILE.write_text(ts)


def triage(last_check):
    """Run email + GitHub triage. Returns list of pending items."""
    print("Checking for new messages...")

    print("  Email triage...")
    email_items = amc.triage_emails(after=last_check)
    print(f"  {len(email_items)} unprocessed email(s)")

    print("  GitHub issue triage...")
    issue_items = ghc.triage_issues(since=last_check)
    print(f"  {len(issue_items)} unprocessed issue(s)")

    pending = email_items + issue_items
    PENDING_FILE.write_text(json.dumps(pending, indent=2))
    return pending


# ---------------------------------------------------------------------------
# RAG indexing
# ---------------------------------------------------------------------------

_rag_vectors = None
_rag_ids = None
_rag_loaded = False


def _ensure_rag_index():
    """Lazy-load the RAG index on first use."""
    global _rag_vectors, _rag_ids, _rag_loaded
    if not _rag_loaded:
        _rag_vectors, _rag_ids = load_index(INDEX_PATH)
        _rag_loaded = True


def index_message(message_id: str, subject: str, sender: str,
                  date: str, body: str, thread_id: str = "") -> str:
    """Index a message for RAG search. Returns status string."""
    global _rag_vectors, _rag_ids
    _ensure_rag_index()

    id_hash = message_id_to_hash(message_id)

    if _rag_ids is not None and id_hash in _rag_ids:
        return "exists"

    msg = {
        "message_id": message_id,
        "thread_id": thread_id,
        "subject": subject,
        "sender": sender,
        "date": date,
        "body": body,
    }
    save_message(msg, THREADS_DIR)

    vec = get_embedding(body)
    if vec is None:
        return "saved_but_not_embedded"

    _rag_vectors, _rag_ids = add_to_index(_rag_vectors, _rag_ids, vec, id_hash)
    save_index(_rag_vectors, _rag_ids, INDEX_PATH)
    return "new"


def index_incoming(item: dict, content: dict, draft: Optional[dict] = None):
    """Index the incoming message(s) for an item, skipping service messages.

    Only indexes if the draft action is not 'skip' (i.e., Claude classified
    the message as substantive, not a service/spam/non-question message).
    """
    if draft and draft.get("action") == "skip":
        return

    if item["type"] == "email":
        # Find the specific message being processed
        target_id = item["message_id"]
        for msg in content.get("messages", []):
            if msg["message_id"] == target_id:
                status = index_message(
                    message_id=msg["message_id"],
                    subject=msg.get("subject", ""),
                    sender=msg.get("from", ""),
                    date=msg.get("date", ""),
                    body=msg.get("body", ""),
                    thread_id=item["thread_id"],
                )
                print(f"  Indexed email: {status}")
                break
    else:
        # Index the issue body
        issue_mid = f"github:{item['owner']}/{item['repo']}/issues/{item['issue_number']}"
        issue_tid = issue_mid
        status = index_message(
            message_id=issue_mid,
            subject=content.get("title", ""),
            sender=content.get("author", ""),
            date=content.get("created_at", ""),
            body=content.get("body", ""),
            thread_id=issue_tid,
        )
        print(f"  Indexed issue body: {status}")

        # Index each comment
        for c in content.get("comments", []):
            comment_mid = f"github:{item['owner']}/{item['repo']}/issues/{item['issue_number']}/comments/{c['id']}"
            status = index_message(
                message_id=comment_mid,
                subject=content.get("title", ""),
                sender=c.get("author", ""),
                date=c.get("created_at", ""),
                body=c.get("body", ""),
                thread_id=issue_tid,
            )
            print(f"  Indexed comment {c['id']}: {status}")


# ---------------------------------------------------------------------------
# Stage 2: Draft generation
# ---------------------------------------------------------------------------

def fetch_content(item):
    """Fetch content for an item. Returns (tag_name, content_dict)."""
    if item["type"] == "email":
        return "thread", amc.fetch_thread(item["thread_id"])
    else:
        return "issue", ghc.fetch_issue(item["owner"], item["repo"], item["issue_number"])


def generate_draft(item, draft_file, content=None):
    """Invoke Claude to generate a draft for one item.

    If content is not provided, it will be fetched. Returns (success, content_dict).
    """
    if content is None:
        _, content = fetch_content(item)

    tag = "thread" if item["type"] == "email" else "issue"
    content_json = json.dumps(content, indent=2, ensure_ascii=False)

    if item["type"] == "email":
        item_desc = f"email message with ID '{item['message_id']}' from thread '{item['thread_id']}'"
    else:
        item_desc = f"GitHub issue #{item['issue_number']} in {item['owner']}/{item['repo']}"

    prompt = f"""REVIEW MODE — Stage 2. Process the {item_desc}.

The content has been fetched for you — do NOT attempt to call any AgentMail or GitHub tools for reading. Use the content below:

<{tag}>
{content_json}
</{tag}>

Write draft to {draft_file}. Follow CLAUDE.md."""

    success = invoke_claude(prompt)
    return success, content


def generate_all_drafts(pending):
    """Generate drafts for all pending items, then index incoming messages."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    count = len(pending)
    print(f"\nGenerating draft replies ({count} messages, review mode)...")

    for i, item in enumerate(pending):
        draft_file = DRAFTS_DIR / f"{i + 1}.json"
        label = item["message_id"] if item["type"] == "email" else f"{item['owner']}/{item['repo']}#{item['issue_number']}"
        print(f"Drafting {i + 1}/{count} ({item['type']}: {label})...")

        success, content = generate_draft(item, str(draft_file))
        if not success:
            print(f"  WARNING: draft generation failed")

        draft = None
        if draft_file.exists():
            draft = json.loads(draft_file.read_text())

        index_incoming(item, content, draft)


# ---------------------------------------------------------------------------
# Stage 2.5: Consolidation
# ---------------------------------------------------------------------------

def consolidate_drafts():
    """Consolidate multiple drafts for the same thread/issue."""
    groups = defaultdict(list)
    for f in sorted(DRAFTS_DIR.glob("*.json")):
        draft = json.loads(f.read_text())
        if draft.get("action") == "skip":
            continue
        if draft.get("source") == "issue":
            key = f"issue:{draft['owner']}/{draft['repo']}#{draft['issue_number']}"
        else:
            key = f"thread:{draft['thread_id']}"
        groups[key].append(str(f))

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        print("No consolidation needed.")
        return

    print(f"Consolidating {len(multi)} thread(s) with multiple replies...")
    invoke_claude(f"""Stage 2.5: Draft consolidation.

Multiple independent agents drafted replies for the same conversation threads. Since they ran without knowledge of each other, their replies may contradict, repeat information, or not read as a coherent conversation.

TASK:
1. Read ALL files in {DRAFTS_DIR}/*.json.
2. Group non-skip drafts by thread_id (emails) or owner/repo/issue_number (issues).
3. For each group with multiple reply drafts:
   a. Find the original conversation messages by grepping the threads/ folder for the thread_id. Read those files to understand who said what and in what order.
   b. Decide what to keep: if a later reply covers the ground of earlier ones, change earlier drafts to action 'skip'. If they address genuinely different follow-up questions, keep both but ensure consistency.
   c. Revise kept replies to incorporate any important unique information from skipped drafts.
   d. Resolve contradictions (e.g. one draft recommends a workaround while another warns against it).
4. Write revised JSON back to the SAME file paths. Do not create new files.

RULES:
- Only modify action and reply fields. Never change message_id, thread_id, source, or other metadata.
- Never remove the github_issue field from a draft that has one -- fold it into whichever reply you keep.
- Preserve the signature block at the end of every reply.
- Email replies must be plain text (no markdown). GitHub issue replies may use markdown.
- One screen of text max per reply.""")


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

def get_actionable_drafts():
    """Return list of (path, draft_dict) for non-skip drafts."""
    drafts = []
    for f in sorted(DRAFTS_DIR.glob("*.json")):
        draft = json.loads(f.read_text())
        if draft.get("action") != "skip":
            drafts.append((f, draft))
    return drafts


def show_drafts():
    """Print draft summaries. Returns number of actionable drafts."""
    drafts = get_actionable_drafts()
    if not drafts:
        print("No actionable drafts (all messages were skipped).")
        return 0

    print(f"\n=== {len(drafts)} draft(s) ready for review ===")
    for path, d in drafts:
        print(f"  {path}")
        if d.get("source") == "issue":
            print(f"    Source: GitHub issue #{d.get('issue_number', '?')} ({d.get('owner', '?')}/{d.get('repo', '?')})")
        else:
            print(f"    Source: Email from {d.get('sender', '?')}")
        print(f"    Subject: {d.get('subject', '?')}")
        print(f"    Action: {d.get('action', '?')}")
        reply = d.get("reply", "")
        if reply:
            preview = reply[:120].replace("\n", " ")
            print(f"    Preview: {preview}...")
        print()

    return len(drafts)


def wait_for_review():
    """Show drafts and wait for user confirmation. Returns True if should proceed."""
    count = show_drafts()
    if count == 0:
        return False

    print("Review and edit files in drafts/. Delete files to skip sending.")
    print("Press Enter when ready to send, or Ctrl+C to abort.")
    try:
        input()
        return True
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return False


# ---------------------------------------------------------------------------
# Stage 3: Sending
# ---------------------------------------------------------------------------

def send_draft(draft_path):
    """Send a single draft via both clients as needed."""
    draft = json.loads(Path(draft_path).read_text())

    if draft.get("action") == "skip":
        return

    source = draft.get("source", "email")
    action = draft.get("action")
    print(f"  Sending: {Path(draft_path).name} (action={action}, source={source})")

    ghc.execute_draft(draft)

    if source == "email":
        amc.execute_draft(draft)


def send_all_drafts():
    """Send all non-skip drafts in the drafts directory."""
    draft_files = sorted(DRAFTS_DIR.glob("*.json"))
    if not draft_files:
        print("No drafts remaining. Nothing to send.")
        return

    print("Sending approved drafts...")
    for f in draft_files:
        send_draft(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_just_send():
    if not DRAFTS_DIR.is_dir() or not list(DRAFTS_DIR.glob("*.json")):
        print(f"Error: no drafts found in {DRAFTS_DIR}")
        sys.exit(1)

    if not wait_for_review():
        return

    send_all_drafts()
    shutil.rmtree(DRAFTS_DIR, ignore_errors=True)
    PENDING_FILE.unlink(missing_ok=True)


def run_full(review_mode):
    last_check = load_last_check()
    pending = triage(last_check)
    save_last_check()

    if not pending:
        print("No new messages.")
        PENDING_FILE.unlink(missing_ok=True)
        return

    if review_mode:
        # Clean slate for drafts
        if DRAFTS_DIR.exists():
            shutil.rmtree(DRAFTS_DIR)

        generate_all_drafts(pending)
        consolidate_drafts()

        if not wait_for_review():
            shutil.rmtree(DRAFTS_DIR, ignore_errors=True)
            PENDING_FILE.unlink(missing_ok=True)
            return

        send_all_drafts()
        shutil.rmtree(DRAFTS_DIR, ignore_errors=True)

    else:
        # No review: generate and send each item immediately
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(pending):
            draft_file = DRAFTS_DIR / f"{i + 1}.json"
            label = item["message_id"] if item["type"] == "email" else f"{item['owner']}/{item['repo']}#{item['issue_number']}"
            print(f"\nItem {i + 1}/{len(pending)} ({item['type']}: {label})")

            _, content = generate_draft(item, str(draft_file))

            draft = None
            if draft_file.exists():
                draft = json.loads(draft_file.read_text())
                send_draft(draft_file)

            index_incoming(item, content, draft)

        shutil.rmtree(DRAFTS_DIR, ignore_errors=True)

    PENDING_FILE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Warp Bot harness")
    parser.add_argument("--no-review", action="store_true", help="Send immediately without review")
    parser.add_argument("--just-send", action="store_true", help="Skip to stage 3, send existing drafts")
    args = parser.parse_args()

    pull_repos()
    we_started_ollama = ensure_ollama()

    try:
        update_code_indexes()

        if args.just_send:
            run_just_send()
        else:
            run_full(review_mode=not args.no_review)
    finally:
        if we_started_ollama:
            stop_ollama()


if __name__ == "__main__":
    main()
