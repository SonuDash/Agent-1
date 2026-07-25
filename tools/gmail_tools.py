"""Gmail read tools: fetch recent/unread emails in a compact form the LLM
can reason over cheaply. Also supports drafting and sending mail."""
import base64
import re
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime

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


_EMBEDDED_PARAM_RE = re.compile(r"\b(max_results|limit)\s*[:=]\s*\d+\b", re.IGNORECASE)


def search_emails(query: str, max_results: int = 10) -> list[dict]:
    """Search Gmail directly using Gmail search syntax, e.g. subject:"exact phrase",
    from:someone@example.com, or plain keywords. Use this instead of
    get_recent_emails whenever the user names a specific email, sender, or
    subject - get_recent_emails only returns a generic recent window and
    will miss anything older or not in that window.
    """
    # Defensive: a model occasionally embeds "max_results=N" as literal text
    # inside the query string instead of passing it as its own parameter.
    # Gmail then searches for that literal text and finds nothing. Strip it
    # out rather than let a malformed query silently return zero results.
    cleaned_query = _EMBEDDED_PARAM_RE.sub("", query).strip()
    return _fetch_by_query(cleaned_query, max_results)


def find_emails_from(sender: str, days: int | None = None, max_results: int = 5) -> list[dict]:
    """Find emails from a specific sender, newest first. Pass whatever the user
    actually gave you literally - a full email address if they gave one, or
    just a name/company if that's all you have. Do NOT guess or construct a
    domain/email address yourself (e.g. don't turn "Skydo" into
    "skydo.com" or similar) - this function builds a correct, valid Gmail
    query from the raw text for you, so there's nothing to compose or guess.

    By default the search covers all available mail. Set days only when the
    user explicitly asks for a recent time period.
    """
    sender = sender.strip()
    date_filter = f" newer_than:{days}d" if days else ""
    if "@" in sender:
        # A real address was given - search it as the sender directly.
        query = f"from:{sender}{date_filter}"
    else:
        # Only a name/company was given, not an exact address - match it
        # both as a sender-field match and as a general keyword, since
        # Gmail's from: operator alone won't reliably match on a display
        # name or company name without the actual domain.
        query = f'(from:"{sender}" OR "{sender}"){date_filter}'
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

        raw_date = headers.get("Date", "")
        try:
            parsed_date = parsedate_to_datetime(raw_date)
            if parsed_date and parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            parsed_date = None

        emails.append(
            {
                "id": msg["id"],
                "from": headers.get("From", "unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                # ISO format sorts and compares correctly as plain text, unlike
                # the raw RFC 2822 header (e.g. "Wed, 22 Jul 2026 10:06:00 +0000")
                # which does NOT sort correctly lexicographically.
                "date": parsed_date.isoformat() if parsed_date else raw_date,
                "snippet": msg.get("snippet", ""),
                "body": body,
                "_sort_key": parsed_date,  # internal only, stripped before returning
            }
        )

    # Explicitly sort newest-first ourselves rather than trusting Gmail API's
    # undocumented default ordering. Entries with an unparseable date sort last.
    emails.sort(key=lambda e: e["_sort_key"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for e in emails:
        del e["_sort_key"]

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


def _build_reply_mime(message_id: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    """Builds a properly threaded reply: correct 'Re:' subject, In-Reply-To/
    References headers, and the original Gmail threadId, so it lands in the
    same conversation thread instead of as a new standalone message."""
    service = gmail_service()
    original = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=["From", "Message-ID", "Subject", "References"],
        )
        .execute()
    )
    headers = {h["name"]: h["value"] for h in original["payload"].get("headers", [])}

    to_addr = parseaddr(headers.get("From", ""))[1]
    if not to_addr:
        raise RuntimeError(f"Could not determine sender address for message {message_id}")

    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    orig_msg_id_header = headers.get("Message-ID", "")
    references = headers.get("References", "")
    references = f"{references} {orig_msg_id_header}".strip() if references else orig_msg_id_header

    message = MIMEText(body)
    message["to"] = to_addr
    message["subject"] = subject
    if orig_msg_id_header:
        message["In-Reply-To"] = orig_msg_id_header
    if references:
        message["References"] = references
    if cc:
        message["cc"] = ", ".join(cc) if isinstance(cc, list) else cc
    if bcc:
        message["bcc"] = ", ".join(bcc) if isinstance(bcc, list) else bcc

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw, "threadId": original["threadId"]}


def create_reply_draft(message_id: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    """Create a draft reply to an existing email, properly threaded (correct
    subject, headers, and Gmail thread). Does NOT send - the user reviews and
    sends it themselves. Get message_id from search_emails or get_recent_emails."""
    service = gmail_service()
    message = _build_reply_mime(message_id, body, cc, bcc)
    draft = service.users().drafts().create(userId="me", body={"message": message}).execute()
    return {"draft_id": draft["id"], "in_reply_to": message_id}


def send_reply(message_id: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None) -> dict:
    """Send a reply to an existing email immediately, properly threaded. This
    is IRREVERSIBLE. Only call after the user has explicitly confirmed the
    exact body. Get message_id from search_emails or get_recent_emails."""
    service = gmail_service()
    message = _build_reply_mime(message_id, body, cc, bcc)
    sent = service.users().messages().send(userId="me", body=message).execute()
    return {"sent": True, "message_id": sent["id"], "in_reply_to": message_id}


if __name__ == "__main__":
    for e in get_recent_emails(hours=24, max_results=5):
        print(f"- [{e['from']}] {e['subject']}")
