"""Google Calendar tools: list upcoming events and create new ones."""
from datetime import datetime, timedelta

from google_auth import calendar_service
import config


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
) -> dict:
    """Create a calendar event.

    start_iso / end_iso must be ISO 8601, e.g. '2026-07-24T15:00:00'.
    Interpreted in the timezone configured in .env (TIMEZONE).
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

    created = service.events().insert(calendarId="primary", body=event_body).execute()
    return {"id": created["id"], "htmlLink": created.get("htmlLink")}


def delete_calendar_event(event_id: str) -> dict:
    """Delete a calendar event by its ID. Get the event_id from
    list_upcoming_events first if you only know the event's title/time."""
    service = calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return {"deleted": True, "event_id": event_id}


if __name__ == "__main__":
    for e in list_upcoming_events(days=2):
        print(f"- {e['start']}: {e['summary']}")