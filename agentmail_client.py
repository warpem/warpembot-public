#!/usr/bin/env python3
"""AgentMail client for Warp Bot — replaces the AgentMail MCP server.

Importable API:
    get_client()                      → AgentMail SDK client
    triage_emails(after=None)         → list of pending email dicts
    fetch_thread(thread_id)           → thread dict with all messages
    execute_draft(draft)              → send email action from draft dict
    mark_processed(message_id)        → add "processed" label

CLI:
    agentmail_client.py triage [--after TIMESTAMP]
    agentmail_client.py fetch-thread THREAD_ID
    agentmail_client.py send-draft DRAFT_FILE
    agentmail_client.py mark-processed MESSAGE_ID
"""

import argparse
import json
import os
import sys
from datetime import datetime

from agentmail import AgentMail

API_KEY = os.environ.get("AGENTMAIL_API_KEY", "")
INBOX_ID = os.environ.get("AGENTMAIL_INBOX_ID", "")
BOT_ADDRESS = os.environ.get("AGENTMAIL_BOT_ADDRESS", "")
ESCALATION_TARGET = os.environ.get("AGENTMAIL_ESCALATION_TARGET", "")

if not API_KEY or not INBOX_ID or not BOT_ADDRESS:
    print("Error: set AGENTMAIL_API_KEY, AGENTMAIL_INBOX_ID, and AGENTMAIL_BOT_ADDRESS environment variables", file=sys.stderr)
    sys.exit(1)

_client = None


def get_client() -> AgentMail:
    global _client
    if _client is None:
        _client = AgentMail(api_key=API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def triage_emails(after: str | None = None) -> list[dict]:
    """Find unprocessed email messages. Returns list of pending item dicts."""
    client = get_client()

    after_dt = None
    if after:
        after_dt = datetime.fromisoformat(after.replace("Z", "+00:00"))

    pending = []
    page_token = None

    while True:
        kwargs = {"limit": 50}
        if after_dt:
            kwargs["after"] = after_dt
        if page_token:
            kwargs["page_token"] = page_token

        resp = client.inboxes.threads.list(INBOX_ID, **kwargs)

        for thread_summary in resp.threads:
            thread = client.inboxes.threads.get(INBOX_ID, thread_summary.thread_id)

            for msg in thread.messages:
                from_ = msg.from_ or ""
                if BOT_ADDRESS in from_:
                    continue
                labels = msg.labels or []
                if "processed" in labels:
                    continue
                pending.append({
                    "type": "email",
                    "thread_id": thread.thread_id,
                    "message_id": msg.message_id,
                })

        if resp.next_page_token:
            page_token = resp.next_page_token
        else:
            break

    return pending


def fetch_thread(thread_id: str) -> dict:
    """Fetch full thread as a dict with all messages."""
    client = get_client()
    thread = client.inboxes.threads.get(INBOX_ID, thread_id)

    messages = []
    for msg in thread.messages:
        messages.append({
            "message_id": msg.message_id,
            "from": msg.from_ or "",
            "to": msg.to or [],
            "subject": msg.subject or "",
            "date": msg.timestamp.isoformat() if msg.timestamp else "",
            "body": msg.text or msg.extracted_text or "",
            "labels": msg.labels or [],
        })

    return {
        "thread_id": thread.thread_id,
        "subject": thread.subject or "",
        "messages": messages,
    }


def execute_draft(draft: dict) -> None:
    """Execute an email action from a draft dict."""
    if draft.get("source") != "email":
        return

    action = draft.get("action", "skip")
    if action == "skip":
        return

    client = get_client()
    message_id = draft.get("message_id")

    if action in ("reply", "issue"):
        reply_text = draft.get("reply", "")
        if not reply_text:
            print("Error: no reply text in draft", file=sys.stderr)
            sys.exit(1)
        client.inboxes.messages.reply_all(INBOX_ID, message_id, text=reply_text)
        print(f"  Replied to {message_id}")

    elif action == "escalate":
        subject = f"Warp Bot escalation: {draft.get('subject', '(no subject)')}"
        reason = draft.get("escalation_reason", "")
        reply_text = draft.get("reply", "")
        body = f"Escalation reason: {reason}\n\n{reply_text}" if reply_text else f"Escalation reason: {reason}"
        client.inboxes.messages.send(
            INBOX_ID,
            to=[ESCALATION_TARGET],
            subject=subject,
            text=body,
        )
        print(f"  Escalation sent to {ESCALATION_TARGET}")

    else:
        print(f"  Unknown action: {action}", file=sys.stderr)
        return

    if message_id:
        client.inboxes.messages.update(INBOX_ID, message_id, add_labels=["processed"])
        print(f"  Marked {message_id} as processed")


def mark_processed(message_id: str) -> None:
    """Add 'processed' label to a message."""
    client = get_client()
    client.inboxes.messages.update(INBOX_ID, message_id, add_labels=["processed"])


# ---------------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------------

def cmd_triage(args):
    pending = triage_emails(after=args.after)
    with open("pending_messages.json", "w") as f:
        json.dump(pending, f, indent=2)
    print(f"{len(pending)} unprocessed email(s)")


def cmd_fetch_thread(args):
    result = fetch_thread(args.thread_id)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


def cmd_send_draft(args):
    with open(args.draft_file) as f:
        draft = json.load(f)
    execute_draft(draft)


def cmd_mark_processed(args):
    mark_processed(args.message_id)
    print(f"Marked {args.message_id} as processed")


def main():
    parser = argparse.ArgumentParser(description="AgentMail client for Warp Bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("triage")
    p.add_argument("--after")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("fetch-thread")
    p.add_argument("thread_id")
    p.set_defaults(func=cmd_fetch_thread)

    p = sub.add_parser("send-draft")
    p.add_argument("draft_file")
    p.set_defaults(func=cmd_send_draft)

    p = sub.add_parser("mark-processed")
    p.add_argument("message_id")
    p.set_defaults(func=cmd_mark_processed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
