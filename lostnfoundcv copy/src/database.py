from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DATABASE_PATH", "lostnfound.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id         TEXT    UNIQUE NOT NULL,
    filename         TEXT    NOT NULL DEFAULT '',
    date_found       TEXT    NOT NULL,
    location         TEXT    NOT NULL DEFAULT '',
    primary_category TEXT    NOT NULL DEFAULT 'Other',
    primary_subcat   TEXT    NOT NULL DEFAULT 'Uncategorized',
    color            TEXT    NOT NULL DEFAULT 'unknown',
    description      TEXT    NOT NULL DEFAULT '',
    status           TEXT    NOT NULL DEFAULT 'unclaimed',
    staff_notes      TEXT    NOT NULL DEFAULT '',
    all_detections   TEXT    NOT NULL DEFAULT '[]'
)
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(_SCHEMA)


def insert_item(
    *,
    image_id: str,
    filename: str,
    location: str,
    primary_category: str,
    primary_subcat: str,
    color: str,
    description: str,
    staff_notes: str,
    all_detections: list,
) -> int:
    now = datetime.now(timezone.utc).strftime("%b %d, %Y")
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO items
               (image_id, filename, date_found, location, primary_category,
                primary_subcat, color, description, status, staff_notes, all_detections)
               VALUES (?,?,?,?,?,?,?,?,'unclaimed',?,?)""",
            (
                image_id, filename, now, location,
                primary_category, primary_subcat, color, description,
                staff_notes, json.dumps(all_detections),
            ),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]


def get_items(
    *,
    status: str | None = None,
    search: str | None = None,
    category: str | None = None,
    color: str | None = None,
) -> list[dict]:
    where: list[str] = ["1=1"]
    params: list = []

    if status:
        where.append("status = ?")
        params.append(status)
    if search:
        for token in search.split():
            term = f"%{token}%"
            where.append(
                "(description LIKE ? OR primary_subcat LIKE ? OR primary_category LIKE ?"
                " OR staff_notes LIKE ? OR location LIKE ? OR color LIKE ?)"
            )
            params += [term, term, term, term, term, term]
    if category and category != "All":
        where.append("primary_category = ?")
        params.append(category)
    if color and color != "All":
        where.append("(',' || color || ',') LIKE ?")
        params.append(f"%,{color},%")

    sql = f"SELECT * FROM items WHERE {' AND '.join(where)} ORDER BY id DESC"
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_item(item_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return dict(row) if row else None


def update_item(item_id: int, **kwargs: str) -> None:
    allowed = {
        "primary_category", "primary_subcat", "color",
        "description", "status", "staff_notes", "location",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    sql = f"UPDATE items SET {', '.join(f'{k}=?' for k in updates)} WHERE id = ?"
    with _conn() as conn:
        conn.execute(sql, list(updates.values()) + [item_id])


def delete_item(item_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))


def get_stats() -> dict:
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        unclaimed = conn.execute(
            "SELECT COUNT(*) FROM items WHERE status='unclaimed'"
        ).fetchone()[0]
        claimed = conn.execute(
            "SELECT COUNT(*) FROM items WHERE status='claimed'"
        ).fetchone()[0]
    return {"total": total, "unclaimed": unclaimed, "claimed": claimed}
