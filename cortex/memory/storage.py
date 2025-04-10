"""
ACID-compliant relational persistence repository.
SQLite engine configured with Write-Ahead Logging (WAL) and B-tree indexing on turn timestamps.
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from cortex.types import Message, Role


class SQLiteStorageRepository:
    def __init__(self, db_path: str = "./data/cortex_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session_time 
                ON conversation_turns(session_id, timestamp DESC);
            """)

    def store_turn(self, session_id: str, message: Message):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO conversation_turns (session_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?);
            """, (session_id, message.role.value, message.content, message.timestamp, json.dumps(message.metadata)))

    def query_recent(self, session_id: str, limit: int = 50) -> List[Message]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT role, content, timestamp, metadata_json
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?;
            """, (session_id, limit))
            rows = cursor.fetchall()
            
            messages = []
            for role_str, content, ts, meta_json in reversed(rows):
                messages.append(Message(
                    role=Role(role_str),
                    content=content,
                    timestamp=ts,
                    metadata=json.loads(meta_json) if meta_json else {}
                ))
            return messages
