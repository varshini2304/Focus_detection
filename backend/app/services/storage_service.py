from datetime import datetime, timedelta
from typing import List, Tuple
import sqlite3
import asyncio
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "focus.db")
_DB_PATH = os.path.abspath(_DB_PATH)


async def init_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS focus_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            focus REAL NOT NULL,
            events TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


async def log_result(ts: datetime, events: List[str], focus: float):
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO focus_log (ts, focus, events) VALUES (?, ?, ?)",
        (int(ts.timestamp()), float(focus), ",".join(events)),
    )
    conn.commit()
    conn.close()


async def get_last_30min() -> List[Tuple[datetime, float]]:
    cutoff = int((datetime.utcnow() - timedelta(minutes=30)).timestamp())
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT ts, focus FROM focus_log WHERE ts >= ? ORDER BY ts ASC", (cutoff,))
    rows = c.fetchall()
    conn.close()
    return [(datetime.utcfromtimestamp(ts), float(focus)) for ts, focus in rows]


async def get_last_7days() -> List[Tuple[datetime, float]]:
    cutoff = int((datetime.utcnow() - timedelta(days=7)).timestamp())
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    # Use day's first timestamp as x-axis representative and average focus per day
    c.execute(
        """
        SELECT MIN(ts) as day_ts, AVG(focus) as avg_focus
        FROM focus_log
        WHERE ts >= ?
        GROUP BY strftime('%Y-%m-%d', ts, 'unixepoch')
        ORDER BY day_ts ASC
        """,
        (cutoff,),
    )
    rows = c.fetchall()
    conn.close()
    return [(datetime.utcfromtimestamp(ts), float(avg)) for ts, avg in rows]
