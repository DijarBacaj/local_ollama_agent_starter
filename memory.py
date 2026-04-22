from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            connection.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, role, content),
            )
            connection.commit()

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                '''
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                ''',
                (session_id, limit),
            ).fetchall()

        items = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        return items

    def add_note(self, note: str) -> dict[str, Any]:
        note = note.strip()
        if not note:
            raise ValueError("Note cannot be empty.")

        with self._connect() as connection:
            cursor = connection.execute(
                'INSERT INTO notes (note) VALUES (?)',
                (note,),
            )
            connection.commit()

        return {"id": cursor.lastrowid, "note": note}

    def get_latest_notes(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                '''
                SELECT id, note, created_at
                FROM notes
                ORDER BY id DESC
                LIMIT ?
                ''',
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def search_notes(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        tokens = [token for token in re.findall(r"[\w\-/]+", query.lower()) if len(token) >= 3][:6]
        if not tokens:
            return self.get_latest_notes(limit=limit)

        where_clause = " OR ".join(["lower(note) LIKE ?" for _ in tokens])
        params = [f"%{token}%" for token in tokens]
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f'''
                SELECT id, note, created_at
                FROM notes
                WHERE {where_clause}
                ORDER BY id DESC
                LIMIT ?
                ''',
                params,
            ).fetchall()

        return [dict(row) for row in rows]
