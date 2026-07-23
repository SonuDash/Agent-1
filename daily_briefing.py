"""Standalone script: pulls today's emails, asks the local model to extract
action items, and prints/logs the result. Designed to be run by launchd
every morning (see README section 6).
"""
from datetime import datetime

from agent import call_ollama
from tools.gmail_tools import get_recent_emails
from tools.calendar_tools import list_upcoming_events
import config

TODAY = datetime.now().strftime("%Y-%m-%d %A")


def build_briefing() -> str:
    emails = get_recent_emails(hours=24, max_results=30)
    events = list_upcoming_events(days=1)

    email_block = "\n\n".join(
        f"From: {e['from']}\nSubject: {e['subject']}\nSnippet: {e['snippet']}\nBody: {e['body']}"
        for e in emails
    ) or "(no emails in the last 24h)"

    events_block = "\n".join(f"- {e['start']}: {e['summary']}" for e in events) or "(no events today)"

    prompt = f"""Today is {TODAY}.

Here are today's calendar events:
{events_block}

Here are emails from the last 24 hours:
{email_block}

Write a short morning briefing with two sections:
1. "Action items" - a bullet list of things that actually need a response or
   action from me, who they're from, and any deadline. Skip newsletters and
   notifications entirely.
2. "Today's schedule" - a one-line summary of the calendar events above.

Keep it tight and skimmable."""

    result = call_ollama(
        [
            {"role": "system", "content": "You are a concise personal assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return result["message"]["content"]


if __name__ == "__main__":
    print(f"=== Daily Briefing - {TODAY} ===\n")
    print(build_briefing())

    # Optional: uncomment to fire a native macOS notification instead of/in
    # addition to printing.
    # import subprocess
    # subprocess.run([
    #     "osascript", "-e",
    #     'display notification "Your daily briefing is ready" with title "Local Agent"'
    # ])