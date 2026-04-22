# tests/test_rag_index.py
import email
import mailbox
import os
import tempfile

import pytest

from rag_index import extract_body, parse_mbox_message


def _make_plain_email(body="Hello world", subject="Test", sender="a@b.com", msg_id="<1@test>"):
    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = "Mon, 15 Jan 2025 10:30:00 +0000"
    msg["Message-ID"] = msg_id
    msg.set_content(body)
    return msg


def _make_html_email(html="<p>Hello <b>world</b></p>", subject="Test", sender="a@b.com", msg_id="<2@test>"):
    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = "Mon, 15 Jan 2025 10:30:00 +0000"
    msg["Message-ID"] = msg_id
    msg.set_content(html, subtype="html")
    return msg


def _make_multipart_email(text="Plain text", html="<p>HTML</p>", subject="Test", sender="a@b.com", msg_id="<3@test>"):
    """Create a multipart/alternative email with both plain and HTML parts."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = "Mon, 15 Jan 2025 10:30:00 +0000"
    msg["Message-ID"] = msg_id
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def test_extract_body_plain():
    msg = _make_plain_email("Hello world")
    assert extract_body(msg) == "Hello world"


def test_extract_body_html_only():
    msg = _make_html_email("<p>Hello <b>world</b></p>")
    body = extract_body(msg)
    assert "Hello" in body
    assert "world" in body
    assert "<p>" not in body
    assert "<b>" not in body


def test_extract_body_multipart_prefers_plain():
    msg = _make_multipart_email(text="Plain version", html="<p>HTML version</p>")
    body = extract_body(msg)
    assert body == "Plain version"


def test_parse_mbox_message():
    msg = _make_plain_email(body="Test body", subject="Test subject", sender="user@test.com", msg_id="<msg1@test>")
    result = parse_mbox_message(msg)
    assert result["message_id"] == "<msg1@test>"
    assert result["subject"] == "Test subject"
    assert result["sender"] == "user@test.com"
    assert result["body"] == "Test body"
    assert result["date"] != ""


def test_parse_mbox_message_no_id():
    msg = _make_plain_email()
    del msg["Message-ID"]
    result = parse_mbox_message(msg)
    assert result is None
