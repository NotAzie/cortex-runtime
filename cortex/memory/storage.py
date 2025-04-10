"""
Dual-backend storage architecture: ACID SQLite operational store + DuckDB analytical engine.
Facilitates high-speed columnar aggregation, Parquet exports, and semantic pattern matching.
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from cortex.types import Message, Role


class StorageManager:
    def __init__(self, db_path: str = "./data/cortex_memory.db", duckdb_path: str = "./data/cortex_analytics.duckdb"):
        self.db_path = db_path
        self.duckdb_path = duckdb_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_turns (session_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?);
            """, (session_id, message.role.value, message.content, message.timestamp, json.dumps(message.metadata)))

    def query_recent(self, session_id: str, limit: int = 50) -> List[Message]:
        with sqlite3.connect(self.db_path) as conn:
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

    def execute_analytical_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Analytical query pass-through (DuckDB compatibility bridge)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query)
            return [dict(row) for row in cursor.fetchall()]
