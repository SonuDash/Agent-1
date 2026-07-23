"""Logs every user query + agent response to a local SQLite DB so you can
search/review past interactions later. Zero external dependencies \u2014 uses
Python's built-in sqlite3.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "agent_log.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_query TEXT NOT NULL,
            agent_response TEXT NOT NULL
        )
        """
    )
    return conn


def log_interaction(user_query: str, agent_response: str) -> None:
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO conversation_log (timestamp, user_query, agent_response) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), user_query, agent_response),
        )
    conn.close()


def search_log(keyword: str = "", limit: int = 20) -> list[dict]:
    """Search past interactions by keyword (matches query or response text).
    Pass no keyword to get the most recent entries."""
    conn = _get_conn()
    if keyword:
        rows = conn.execute(
            """
            SELECT timestamp, user_query, agent_response FROM conversation_log
            WHERE user_query LIKE ? OR agent_response LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT timestamp, user_query, agent_response FROM conversation_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [{"timestamp": r[0], "query": r[1], "response": r[2]} for r in rows]


if __name__ == "__main__":
    # quick manual check: python storage.py
    for entry in search_log(limit=5):
        print(f"[{entry['timestamp']}] {entry['query'][:60]}")