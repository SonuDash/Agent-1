"""Google Calendar tools: list upcoming events and create new ones."""
from datetime import datetime, timedelta

from google_auth import calendar_service
import config

VALID_FREQ = {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}
VALID_DAYS = {"MO", "TU", "WE", "TH", "FR", "SA", "SU"}


def _build_rrule(
    freq: str,
    interval: int = 1,
    count: int | None = None,
    until: str | None = None,
    by_day: list[str] | None = None,
) -> str:
    """Builds an RFC 5545 RRULE string from simple structured inputs, so the
    caller never has to hand-write RRULE syntax."""
    freq = freq.upper()
    if freq not in VALID_FREQ:
        raise ValueError(f"recurrence_freq must be one of {sorted(VALID_FREQ)}, got '{freq}'")

    parts = [f"FREQ={freq}"]

    if interval and interval > 1:
        parts.append(f"INTERVAL={interval}")

    if by_day:
        days = [d.upper() for d in by_day]
        bad = [d for d in days if d not in VALID_DAYS]
        if bad:
            raise ValueError(f"Invalid day(s) in recurrence_days: {bad}. Use MO, TU, WE, TH, FR, SA, SU.")
        parts.append(f"BYDAY={','.join(days)}")

    # count and until are mutually exclusive per RFC 5545; count wins if both given
    if count:
        parts.append(f"COUNT={count}")
    elif until:
        # Accept a plain date ('2026-12-31') or full ISO datetime; normalize to
        # the UTC 'YYYYMMDDTHHMMSSZ' form RRULE UNTIL requires.
        dt = datetime.fromisoformat(until)
        parts.append(f"UNTIL={dt.strftime('%Y%m%dT%H%M%SZ')}")

    return f"RRULE:{';'.join(parts)}"


def list_upcoming_events(days: int = 1, max_results: int = 15) -> list[dict]:
    """List events between now and `days` days from now."""
    service = calendar_service()
    now = datetime.utcnow().isoformat() + "Z"
    end = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            timeMax=end,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = []
    for e in result.get("items", []):
        start = e["start"].get("dateTime", e["start"].get("date"))
        events.append(
            {
                "id": e["id"],
                "summary": e.get("summary", "(no title)"),
                "start": start,
                "location": e.get("location", ""),
                "attendees": [a.get("email") for a in e.get("attendees", [])],
            }
        )
    return events


def create_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    attendees: list[str] | None = None,
    location: str = "",
    recurrence_freq: str | None = None,
    recurrence_interval: int = 1,
    recurrence_count: int | None = None,
    recurrence_until: str | None = None,
    recurrence_days: list[str] | None = None,
) -> dict:
    """Create a calendar event, optionally recurring.

    start_iso / end_iso must be ISO 8601, e.g. '2026-07-24T15:00:00'.
    Interpreted in the timezone configured in .env (TIMEZONE).

    To make it recurring, set recurrence_freq to one of
    'DAILY' / 'WEEKLY' / 'MONTHLY' / 'YEARLY', plus optionally:
      - recurrence_interval: repeat every N periods (e.g. 2 = every 2 weeks). Default 1.
      - recurrence_count: stop after N occurrences.
      - recurrence_until: stop after this date (ISO date/datetime), if not using count.
      - recurrence_days: for weekly events, which days e.g. ['MO', 'WE', 'FR'].
    Leave recurrence_freq unset for a one-off event.
    """
    service = calendar_service()

    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso, "timeZone": config.TIMEZONE},
        "end": {"dateTime": end_iso, "timeZone": config.TIMEZONE},
    }
    if attendees:
        event_body["attendees"] = [{"email": a} for a in attendees]

    if recurrence_freq:
        event_body["recurrence"] = [
            _build_rrule(
                freq=recurrence_freq,
                interval=recurrence_interval,
                count=recurrence_count,
                until=recurrence_until,
                by_day=recurrence_days,
            )
        ]

    created = service.events().insert(calendarId="primary", body=event_body).execute()
    return {
        "id": created["id"],
        "htmlLink": created.get("htmlLink"),
        "recurring": bool(recurrence_freq),
    }


def delete_calendar_event(event_id: str) -> dict:
    """Delete a calendar event by its ID. Get the event_id from
    list_upcoming_events first if you only know the event's title/time.

    Note on recurring events: list_upcoming_events expands recurring events
    into individual instances (each with its own instance ID). Deleting with
    an instance ID cancels only that one occurrence, not the whole series -
    which is almost always what the user means when they say "cancel my
    Tuesday standup", not "delete the entire recurring series forever". If
    the user explicitly asks to delete the whole series, say so clearly in
    your response since this tool as-is only removes single instances."""
    service = calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return {"deleted": True, "event_id": event_id}


if __name__ == "__main__":
    for e in list_upcoming_events(days=2):
        print(f"- {e['start']}: {e['summary']}")