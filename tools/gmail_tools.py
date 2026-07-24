"""Gmail read tools: fetch recent/unread emails in a compact form the LLM
can reason over cheaply. Also supports drafting and sending mail."""
import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from google_auth import gmail_service

MAX_BODY_CHARS = 1500


def _extract_body(payload) -> str:
    """Pull the plain-text body out of a Gmail message payload, walking
    into multipart messages if needed."""
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        raw = payload["body"]["data"]
        return base64.urlsafe_b64decode(raw).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []) or []:
        text = _extract_body(part)
        if text:
            return text
    return ""


def get_recent_emails(hours: int = 24, max_results: int = 25, unread_only: bool = False) -> list[dict]:
    """Fetch emails from the last `hours` hours (default 24h = 'today').

    Returns a list of {id, from, subject, date, snippet, body} dicts, with
    body truncated so we don't blow the model's context window.
    """
    query = f"newer_than:{hours}h" if hours < 24 else f"newer_than:{hours // 24}d"
    if unread_only:
        query += " is:unread"
    return _fetch_by_query(query, max_results)


def search_emails(query: str, max_results: int = 10) -> list[dict]:
    """Search Gmail directly using Gmail search syntax, e.g. subject:"exact phrase",
    from:someone@example.com, or plain keywords. Use this instead of
    get_recent_emails whenever the user names a specific email, sender, or
    subject - get_recent_emails only returns a generic recent window and
    will miss anything older or not in that window.
    """
    return _fetch_by_query(query, max_results)


def _fetch_by_query(query: str, max_results: int) -> list[dict]:
    service = gmail_service()

    results = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    message_ids = results.get("messages", [])

    emails = []
    for m in message_ids:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=m["id"], format="full")
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        body = _extract_body(msg["payload"])[:MAX_BODY_CHARS]

        emails.append(
            {
                "id": msg["id"],
                "from": headers.get("From", "unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
                "body": body,
            }
        )

    return emails


def _build_raw_message(to: list[str], subject: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    message = MIMEText(body)
    message["to"] = ", ".join(to) if isinstance(to, list) else to
    message["subject"] = subject
    if cc:
        message["cc"] = ", ".join(cc) if isinstance(cc, list) else cc
    if bcc:
        message["bcc"] = ", ".join(bcc) if isinstance(bcc, list) else bcc
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw}


def create_email_draft(to: list[str], subject: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    """Create a draft email in Gmail. Does NOT send anything - the user still
    has to review and send it themselves from their Gmail. This is the safe,
    reversible option; prefer this over send_email unless the user has
    explicitly confirmed they want it sent immediately."""
    service = gmail_service()
    message = _build_raw_message(to, subject, body, cc, bcc)
    draft = service.users().drafts().create(userId="me", body={"message": message}).execute()
    return {"draft_id": draft["id"], "to": to, "subject": subject}


def send_email(to: list[str], subject: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    """Send an email immediately. This is IRREVERSIBLE - the recipient gets it
    right away. Only call this after the user has explicitly confirmed the
    exact recipient, subject, and body."""
    service = gmail_service()
    message = _build_raw_message(to, subject, body, cc, bcc)
    sent = service.users().messages().send(userId="me", body=message).execute()
    return {"sent": True, "message_id": sent["id"], "to": to, "subject": subject}


if __name__ == "__main__":
    for e in get_recent_emails(hours=24, max_results=5):
        print(f"- [{e['from']}] {e['subject']}")