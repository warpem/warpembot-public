#!/usr/bin/env python3
"""Fetch CCP-EM mailing list archives from JiscMail LISTSERV via AgentMail.

Usage:
    AGENTMAIL_API_KEY=... python fetch_ccpem.py [--start YYMM] [--end YYMM] [--batch N]

Uses SEARCH to discover post numbers for each month, then GETPOST in batches
to retrieve messages as RFC822 attachments. Each message is parsed, cleaned,
and saved as a JSON file in ccpem/ named by SHA256(Message-ID). Existing files
are skipped, so the script is safe to re-run.
"""

import argparse
import calendar
import email
import email.policy
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests as http_requests
from agentmail import AgentMail

INBOX_ID = "warpbot@agentmail.to"
CCPEM_DIR = "ccpem"
LISTSERV_ADDR = "listserv@jiscmail.ac.uk"

POLL_INTERVAL = 10   # seconds between inbox polls
POLL_TIMEOUT = 1200  # seconds before giving up on a response

MONTH_ABBREVS = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Month generation
# ---------------------------------------------------------------------------

def generate_months(start: str, end: str) -> list[tuple[int, int]]:
    """Generate (year, month) tuples for an inclusive YYMM range."""
    sy, sm = int(start[:2]) + 2000, int(start[2:])
    ey, em = int(end[:2]) + 2000, int(end[2:])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def month_label(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


# ---------------------------------------------------------------------------
# RFC822 email parsing
# ---------------------------------------------------------------------------

def parse_rfc822(raw: str) -> dict | None:
    """Parse a raw RFC822 email into a clean dict with text-only body."""
    msg = email.message_from_string(raw, policy=email.policy.default)

    msg_id = msg.get("Message-ID", "").strip()
    if not msg_id:
        return None

    # Extract text content, discarding attachments
    body = _extract_text(msg)
    if not body:
        return None

    return {
        "message_id": msg_id,
        "date": msg.get("Date", ""),
        "from": msg.get("From", ""),
        "subject": msg.get("Subject", ""),
        "body": clean_body(body),
    }


def _extract_text(msg: email.message.Message) -> str:
    """Recursively extract text/plain content from an email, ignoring binaries."""
    if msg.is_multipart():
        parts = []
        for part in msg.iter_parts():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    parts.append(payload)
            elif ct.startswith("multipart/"):
                parts.append(_extract_text(part))
            # else: skip images, pdfs, html (if plain is available), etc.

        # If no text/plain found, try text/html as fallback
        if not parts:
            for part in msg.iter_parts():
                if part.get_content_type() == "text/html":
                    payload = part.get_content()
                    if isinstance(payload, str):
                        # Strip HTML tags for a rough plain-text version
                        parts.append(re.sub(r"<[^>]+>", "", payload))

        return "\n\n".join(p for p in parts if p.strip())
    else:
        ct = msg.get_content_type()
        if ct.startswith("text/"):
            payload = msg.get_content()
            if isinstance(payload, str):
                if ct == "text/html":
                    return re.sub(r"<[^>]+>", "", payload)
                return payload
        return ""


# ---------------------------------------------------------------------------
# Body cleaning
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    # LISTSERV unsubscribe footer
    re.compile(
        r"#{10,}\s*\nTo unsubscribe from the CCPEM list.*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"To unsubscribe from the CCPEM list, click the following link:.*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"This message was issued to members of www\.jiscmail\.ac\.uk/CCPEM.*$",
        re.DOTALL | re.IGNORECASE,
    ),
    # Scanner/antivirus notices
    re.compile(r"--\s*\nScanned by iCritical\.?\s*$", re.DOTALL),
    re.compile(r"Scanned by iCritical\.?\s*$"),
    re.compile(r"--\s*\nThis email was Anti Virus checked.*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"This message has been scanned for viruses.*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"_{20,}\s*Information from ESET.*$", re.DOTALL),
    # Confidentiality disclaimers
    re.compile(
        r"This e-?mail and any attachments?\s+(?:are|is)\s+confidential.*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(r"DISCLAIMER[:\s].*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"If you are not the intended recipient.*$", re.DOTALL | re.IGNORECASE),
    # University/institution charity notices
    re.compile(
        r"The University of \w[\w\s]* is a charit(?:able|y).*$",
        re.DOTALL | re.IGNORECASE,
    ),
    # Common UK academic email footers
    re.compile(
        r"This email has been checked by [\w\s]+ antivirus.*$",
        re.DOTALL | re.IGNORECASE,
    ),
]


def clean_body(body: str) -> str:
    """Remove boilerplate noise from a message body."""
    for pat in _BOILERPLATE_PATTERNS:
        body = pat.sub("", body)

    # Collapse 3+ consecutive blank lines to 2
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body.strip()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def msg_id_to_filename(message_id: str) -> str:
    """SHA256 hash of the Message-ID (angle brackets stripped)."""
    clean = message_id.strip().strip("<>")
    return hashlib.sha256(clean.encode()).hexdigest()


def save_msg(msg: dict, label: str) -> bool:
    """Save a message to ccpem/ if it doesn't already exist. Returns True if new."""
    fname = msg_id_to_filename(msg["message_id"])
    path = os.path.join(CCPEM_DIR, f"{fname}.json")
    if os.path.exists(path):
        return False

    record = {
        "message_id": msg["message_id"],
        "date": msg["date"],
        "from": msg["from"],
        "subject": msg["subject"],
        "body": msg["body"],
        "source": "ccpem",
        "month": label,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True


# ---------------------------------------------------------------------------
# AgentMail interaction
# ---------------------------------------------------------------------------

def send_and_wait(client: AgentMail, body: str, subject_match: str):
    """Send a command to LISTSERV and wait for a reply matching the given subject.

    Returns the matching Thread object, or None on timeout.
    """
    send_time = datetime.now(timezone.utc)

    client.inboxes.messages.send(
        INBOX_ID,
        to=[LISTSERV_ADDR],
        text=body,
    )

    deadline = time.time() + POLL_TIMEOUT

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            resp = client.inboxes.threads.list(INBOX_ID, limit=10)
        except Exception as e:
            print(f"    Poll error: {e}")
            continue

        for t in resp.threads:
            if subject_match not in (t.subject or ""):
                continue
            ts = getattr(t, "received_timestamp", None) or getattr(t, "timestamp", None)
            if ts and ts >= send_time:
                return client.inboxes.threads.get(INBOX_ID, t.thread_id)

    return None


def search_month(client: AgentMail, year: int, month: int) -> tuple[int, int] | None:
    """SEARCH for all posts in a given month. Returns (first, last) post numbers, or None."""
    last_day = calendar.monthrange(year, month)[1]
    mon = MONTH_ABBREVS[month]

    cmd = f"Search * in CCPEM since 01 {mon} {year} until {last_day} {mon} {year}"

    thread = send_and_wait(client, cmd, "Re: SEARCH *")
    if thread is None:
        return None

    text = thread.messages[0].text or thread.messages[0].extracted_text or ""

    m = re.search(r"GETPOST CCPEM (\d+)-(\d+)", text)
    if not m:
        if "0 matches" in text:
            return (0, 0)
        return None

    return int(m.group(1)), int(m.group(2))


def fetch_batch(client: AgentMail, start: int, end: int) -> list[dict]:
    """GETPOST a batch of posts, download attachments, parse into messages."""
    send_time = datetime.now(timezone.utc)

    client.inboxes.messages.send(
        INBOX_ID,
        to=[LISTSERV_ADDR],
        text=f"GETPOST CCPEM {start}-{end}",
    )

    # LISTSERV sends two emails: a confirmation and the actual results (with
    # attachments). We need to find the thread that has attachments.
    deadline = time.time() + POLL_TIMEOUT
    att_thread_id = None
    att_msg = None

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            resp = client.inboxes.threads.list(INBOX_ID, limit=10)
        except Exception as e:
            print(f"    Poll error: {e}")
            continue

        for t in resp.threads:
            if "GETPOST" not in (t.subject or ""):
                continue
            ts = getattr(t, "received_timestamp", None) or getattr(t, "timestamp", None)
            if not (ts and ts >= send_time):
                continue
            # Check if this thread has attachments (it's the results, not confirmation)
            thread_atts = getattr(t, "attachments", None)
            if thread_atts:
                full = client.inboxes.threads.get(INBOX_ID, t.thread_id)
                for m in full.messages:
                    if getattr(m, "attachments", None):
                        att_thread_id = t.thread_id
                        att_msg = m
                        break
            if att_msg:
                break
        if att_msg:
            break

    if att_msg is None:
        return []

    messages = []
    for att in att_msg.attachments:
        if att.content_type != "message/rfc822":
            continue
        try:
            att_resp = client.inboxes.threads.get_attachment(
                INBOX_ID, att_thread_id, att.attachment_id
            )
            raw = http_requests.get(att_resp.download_url, timeout=30).text
            parsed = parse_rfc822(raw)
            if parsed:
                messages.append(parsed)
        except Exception as e:
            print(f"    Attachment error: {e}")
            continue

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch CCP-EM archives from JiscMail")
    parser.add_argument("--start", default="2103", help="Start month as YYMM (default: 2103)")
    parser.add_argument("--end", default="2603", help="End month as YYMM (default: 2603)")
    parser.add_argument("--batch", type=int, default=20, help="Posts per GETPOST batch (default: 20)")
    args = parser.parse_args()

    api_key = os.environ.get("AGENTMAIL_API_KEY")
    if not api_key:
        print("Error: set AGENTMAIL_API_KEY environment variable")
        sys.exit(1)

    client = AgentMail(api_key=api_key)
    os.makedirs(CCPEM_DIR, exist_ok=True)

    months = generate_months(args.start, args.end)
    months.reverse()
    print(f"Fetching {len(months)} months: {month_label(*months[0])} to {month_label(*months[-1])} (newest first)")

    total_new = 0
    total_skipped = 0
    failed: list[str] = []

    for i, (year, month) in enumerate(months):
        label = month_label(year, month)
        print(f"\n[{i + 1}/{len(months)}] {label}: searching...", end=" ", flush=True)

        result = search_month(client, year, month)
        if result is None:
            print("SEARCH TIMEOUT")
            failed.append(label)
            continue

        first, last = result
        if first == 0 and last == 0:
            print("0 posts")
            continue

        count = last - first + 1
        print(f"{count} posts ({first}-{last}), fetching...", flush=True)

        month_new = 0
        month_skipped = 0

        for batch_start in range(first, last + 1, args.batch):
            batch_end = min(batch_start + args.batch - 1, last)
            print(f"  GETPOST {batch_start}-{batch_end}...", end=" ", flush=True)

            messages = fetch_batch(client, batch_start, batch_end)
            if not messages:
                print("FAILED (no messages)")
                failed.append(f"{label}:{batch_start}-{batch_end}")
                continue

            new = sum(1 for m in messages if save_msg(m, label))
            skipped = len(messages) - new
            month_new += new
            month_skipped += skipped
            print(f"{len(messages)} parsed ({new} new, {skipped} existing)")

        total_new += month_new
        total_skipped += month_skipped

    print(f"\nDone: {total_new} new messages saved, {total_skipped} already existed")
    if failed:
        print(f"Failed (re-run to retry): {', '.join(failed)}")


if __name__ == "__main__":
    main()
