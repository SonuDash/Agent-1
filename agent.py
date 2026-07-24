"""Interactive local agent. Chats with qwen3.5:9b via Ollama, using
OpenAI-style tool calling to reach into Gmail, Calendar, and Notion.

Run: python agent.py
"""
import json
import re
import sys
from datetime import datetime

import requests

import config
from tools.gmail_tools import get_recent_emails, search_emails, create_email_draft, send_email, create_reply_draft, send_reply
from tools.calendar_tools import list_upcoming_events, create_calendar_event, delete_calendar_event
from tools.notion_tools import search_notion, get_page_content
from storage import log_interaction, search_log

TODAY = datetime.now().strftime("%Y-%m-%d %A")

SYSTEM_PROMPT = f"""You are a personal local assistant running on the user's own machine.
Today's date is {TODAY}. The user's timezone is {config.TIMEZONE}.
The user's name is {config.USER_NAME or "not set - ask them for their name if you need to sign an email"}.

You have tools to read their Gmail, read/create Google Calendar events, and
search/read their Notion workspace. Use tools whenever a question needs
real data instead of guessing. When creating calendar events, always confirm
the exact date/time you're about to book if the user's phrasing was
ambiguous (e.g. "next Tuesday").

When summarizing email into action items, be concise: one line per action
item, note who it's from and any deadline mentioned. Skip newsletters,
notifications, and anything that isn't actually actionable.

For Notion: if you need to see everything accessible, or a keyword search
comes up empty, call search_notion with no query instead of guessing more
keywords - that lists everything the integration can currently see in one
call.

Grounding rule (search) - follow this strictly: if the user asks about a
specific email, event, or page and your tool results do NOT contain
something that actually matches (matching subject, sender, or title), say
plainly that you couldn't find it and show what you searched for. NEVER
summarize or describe a different, unrelated item as if it were the one
requested. Getting this wrong is worse than saying "not found."

Grounding rule (actions) - follow this even more strictly: NEVER tell the
user something was created, sent, deleted, or updated unless you can see the
actual successful result of that exact tool call earlier in THIS
conversation. If you have not called send_email, create_calendar_event,
delete_calendar_event, etc. and gotten a success result back, you have not
done it - don't say you have, don't invent an ID or confirmation detail for
it, and don't describe what "would have" happened as if it happened. If a
tool call errored, say so plainly and show the error, don't paper over it
with a fabricated success message.

Deleting calendar events is destructive and cannot be undone. Before calling
delete_calendar_event, always state clearly which event (title + date/time)
you're about to delete and ask the user to confirm, unless they already gave
an unambiguous, explicit instruction naming that exact event. If more than
one event could match what they described, list the candidates and ask which
one instead of guessing.

Sending email is irreversible - the recipient gets it immediately and it
cannot be recalled. Default to create_email_draft, not send_email, whenever
the user asks you to write/compose/draft an email. Only call send_email if
the user has explicitly told you to send it, and even then, show the exact
recipient, subject, and body first and get a clear "yes, send it" before
calling send_email - unless they already confirmed that exact content
earlier in this same conversation.

When drafting or sending emails on the user's behalf, sign off using the
user's real name given above - NEVER leave a placeholder like "[Your Name]"
or "[Sender]" in the final body. If the user's name isn't set and the email
needs a sign-off, ask them for it rather than inventing or leaving a
placeholder.

When the user wants to respond to a specific email, use create_reply_draft /
send_reply (not create_email_draft / send_email) so it stays properly
threaded in Gmail. Find the message_id with search_emails first if you don't
already have it - but check earlier in THIS conversation first: if you
already called search_emails or get_recent_emails and found the email being
referred to (e.g. "reply to that", "reply to the one you found"), reuse the
id field from that earlier tool result. Do not ask the user for a message ID
you already have in context.

Notion access is READ-ONLY. You can search the workspace and read page
content, but you cannot create, edit, or delete anything in Notion. If the
user asks you to create, write, add to, or modify a Notion page, tell them
plainly that Notion is read-only in this agent and you can't make that
change - don't attempt it, don't pretend to, and don't ask clarifying
questions as if you were about to do it.

Stay on the exact task the user asked for. After any tool call, your
response must move that specific request forward - answer it, ask the one
question needed to proceed, or report the result. Do not pivot to a general
summary of unrelated data a tool happened to return, and do not end with a
generic "what would you like to do next" unless you've already completed
what was asked.

If you just asked the user a clarifying question and their next message is a
short answer, treat it as the answer to YOUR question, applied to the
original pending task - not as a new, separate request.

Keep responses direct and concise. Never think out loud in your final answer
(no "wait, let me reconsider", no narrating your own uncertainty about your
output) - if you're unsure, just ask the user a clear question instead.
"""

# --- Tool schemas (OpenAI-compatible function-calling format, which Ollama supports) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search Gmail directly using Gmail search syntax - use this whenever the user names a specific email, subject, or sender. Examples: subject:\"exact phrase\", from:someone@example.com, or plain keywords. Prefer this over get_recent_emails for anything specific; get_recent_emails only returns a generic recent time window and will miss older or unmatched emails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gmail search query, e.g. subject:\"Streamable HTTP server\""},
                    "max_results": {"type": "integer", "description": "Max results. Default 10."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_emails",
            "description": "Fetch recent emails from Gmail to find action items or check for messages from someone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "How many hours back to look. Default 24."},
                    "max_results": {"type": "integer", "description": "Max number of emails to fetch. Default 25."},
                    "unread_only": {"type": "boolean", "description": "Only fetch unread emails. Default false."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_email_draft",
            "description": "Create a draft email in Gmail. Does NOT send it - the user reviews and sends it themselves. This is the safe default for anything email-composition related; prefer it over send_email unless the user explicitly says to send immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email address(es)."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body text."},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC addresses."},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional BCC addresses."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email immediately - IRREVERSIBLE, the recipient gets it right away. Only call this after the user has explicitly confirmed the exact recipient, subject, and body in this conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email address(es)."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body text."},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC addresses."},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional BCC addresses."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reply_draft",
            "description": "Create a draft REPLY to an existing email, properly threaded (correct subject/headers/Gmail thread). Does NOT send. Use this instead of create_email_draft when the user wants to respond to a specific email - get message_id from search_emails or get_recent_emails first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "The Gmail message ID being replied to."},
                    "body": {"type": "string", "description": "Reply body text."},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC addresses."},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional BCC addresses."},
                },
                "required": ["message_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_reply",
            "description": "Send a REPLY to an existing email immediately, properly threaded - IRREVERSIBLE. Only call after the user has explicitly confirmed the exact body. Get message_id from search_emails or get_recent_emails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "The Gmail message ID being replied to."},
                    "body": {"type": "string", "description": "Reply body text."},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC addresses."},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional BCC addresses."},
                },
                "required": ["message_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upcoming_events",
            "description": "List upcoming Google Calendar events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "How many days ahead to look. Default 1."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new Google Calendar event, optionally recurring. To make it recurring, set recurrence_freq (DAILY/WEEKLY/MONTHLY/YEARLY); leave it unset for a one-off event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title."},
                    "start_iso": {"type": "string", "description": "Start time, ISO 8601, e.g. 2026-07-24T15:00:00"},
                    "end_iso": {"type": "string", "description": "End time, ISO 8601."},
                    "description": {"type": "string", "description": "Optional event description."},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Optional list of attendee emails."},
                    "location": {"type": "string", "description": "Optional location."},
                    "recurrence_freq": {"type": "string", "enum": ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"], "description": "Set to make the event recurring. Omit for a one-off event."},
                    "recurrence_interval": {"type": "integer", "description": "Repeat every N periods, e.g. 2 = every other week. Default 1."},
                    "recurrence_count": {"type": "integer", "description": "Stop after N occurrences."},
                    "recurrence_until": {"type": "string", "description": "Stop after this date, ISO format e.g. 2026-12-31. Use instead of recurrence_count, not both."},
                    "recurrence_days": {"type": "array", "items": {"type": "string", "enum": ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]}, "description": "For weekly recurrence, which days e.g. ['MO','WE','FR']."},
                },
                "required": ["summary", "start_iso", "end_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a calendar event by its event_id. If you only know the event's title or time, call list_upcoming_events first to find the matching event_id - never guess an ID or delete an event without a confirmed match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The Google Calendar event ID, from list_upcoming_events results."},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notion",
            "description": "Search the user's Notion workspace by keyword/title. Call with NO query (or an empty string) to list everything the integration can access - most commonly used to find a parent page/database ID before creating something, or to answer 'what can you see' questions. Results are minimal (id/type/title/url) on purpose: use them to pick or reference specific items, not as content to summarize or report back wholesale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text. Omit or leave empty to list everything accessible."},
                    "max_results": {"type": "integer", "description": "Max results to return. Default 15."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Fetch the text content of a specific Notion page, given its page_id (from search_notion results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "The Notion page ID."},
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_conversation_log",
            "description": "Search past conversations with this agent by keyword, or fetch the most recent ones with no keyword. Use this if the user asks what they discussed/asked before, or references a past conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for. Omit to get most recent entries."},
                    "limit": {"type": "integer", "description": "Max entries to return. Default 20."},
                },
            },
        },
    },
]

TOOL_IMPLS = {
    "get_recent_emails": get_recent_emails,
    "search_emails": search_emails,
    "create_email_draft": create_email_draft,
    "send_email": send_email,
    "create_reply_draft": create_reply_draft,
    "send_reply": send_reply,
    "list_upcoming_events": list_upcoming_events,
    "create_calendar_event": create_calendar_event,
    "delete_calendar_event": delete_calendar_event,
    "search_notion": search_notion,
    "get_page_content": get_page_content,
    "search_conversation_log": search_log,
}


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def call_ollama(messages: list[dict]) -> dict:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "think": False,  # avoid raw <think> reasoning leaking into output
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()

    # Belt-and-suspenders: strip any stray <think>...</think> block that
    # slips through regardless (some model/Ollama version combos still emit
    # it as literal text in content rather than a separate field).
    content = result.get("message", {}).get("content")
    if content:
        result["message"]["content"] = THINK_TAG_RE.sub("", content).strip()

    return result


def run_turn(messages: list[dict]) -> str:
    """Runs the tool-calling loop until the model produces a final text answer."""
    empty_retry_used = False

    while True:
        try:
            result = call_ollama(messages)
        except requests.exceptions.Timeout:
            print(
                "\n[timeout] The model didn't respond within 120s - likely memory "
                "pressure on this machine. Exiting so you can restart a fresh "
                "session (long conversation history compounds this).\n"
            )
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            print("\n[error] Couldn't reach Ollama - make sure it's running (`ollama serve`).\n")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"\n[error] Ollama request failed: {e}\n")
            sys.exit(1)

        message = result["message"]
        tool_calls = message.get("tool_calls")
        content = (message.get("content") or "").strip()

        # A response with neither a tool call nor any text is a model failure,
        # not a valid "nothing to say" - don't let it print as silence.
        if not tool_calls and not content:
            if not empty_retry_used:
                empty_retry_used = True
                messages.append(
                    {
                        "role": "user",
                        "content": "(Your last response was empty. Please answer the previous request, or ask a clear question if you need more information.)",
                    }
                )
                continue
            return (
                "[The model returned an empty response twice in a row - this can happen "
                "under memory pressure or after a long conversation. Try rephrasing your "
                "request, or restart with 'exit' and a fresh session.]"
            )

        messages.append(message)

        if not tool_calls:
            return content

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            print(f"  [tool] {name}({args})")
            try:
                fn = TOOL_IMPLS[name]
                output = fn(**args)
            except Exception as e:
                output = {"error": str(e)}

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(output, default=str),
                }
            )


MAX_CONTEXT_MESSAGES = 40  # keeps context bounded on limited-RAM machines


def _trim_messages(messages: list[dict]) -> list[dict]:
    """Keeps the system prompt plus only the most recent messages, so a long
    session doesn't let context (and memory use) grow without bound."""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    system = messages[0]
    recent = messages[-(MAX_CONTEXT_MESSAGES - 1):]
    return [system] + recent


def main():
    print(f"Local agent ready ({config.OLLAMA_MODEL}). Type 'exit' to quit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        messages[:] = _trim_messages(messages)
        reply = run_turn(messages)
        print(f"\nagent> {reply}\n")
        log_interaction(user_input, reply)


if __name__ == "__main__":
    main()