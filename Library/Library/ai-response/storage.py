from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_DB_PATH = Path("legal_agent.db")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=None)


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(interactions)").fetchall()
    }

    column_statements = {
        "confidence_score": "ALTER TABLE interactions ADD COLUMN confidence_score INTEGER NOT NULL DEFAULT 0",
        "confidence_level": "ALTER TABLE interactions ADD COLUMN confidence_level TEXT NOT NULL DEFAULT 'LOW'",
        "routing_mode": "ALTER TABLE interactions ADD COLUMN routing_mode TEXT NOT NULL DEFAULT 'AUTO_REPLY'",
        "review_status": "ALTER TABLE interactions ADD COLUMN review_status TEXT NOT NULL DEFAULT 'not_required'",
        "review_decision": "ALTER TABLE interactions ADD COLUMN review_decision TEXT NOT NULL DEFAULT ''",
        "review_notes": "ALTER TABLE interactions ADD COLUMN review_notes TEXT NOT NULL DEFAULT ''",
    }

    for column_name, statement in column_statements.items():
        if column_name not in existing_columns:
            connection.execute(statement)


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    connection = get_connection(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_input TEXT NOT NULL,
                extracted_text TEXT NOT NULL DEFAULT '',
                classification_json TEXT NOT NULL DEFAULT '{}',
                risk_level TEXT NOT NULL DEFAULT 'LOW',
                risk_score INTEGER NOT NULL DEFAULT 0,
                compliance_score INTEGER NOT NULL DEFAULT 0,
                confidence_score INTEGER NOT NULL DEFAULT 0,
                confidence_level TEXT NOT NULL DEFAULT 'LOW',
                routing_mode TEXT NOT NULL DEFAULT 'AUTO_REPLY',
                review_status TEXT NOT NULL DEFAULT 'not_required',
                review_decision TEXT NOT NULL DEFAULT '',
                review_notes TEXT NOT NULL DEFAULT '',
                generated_response TEXT NOT NULL DEFAULT '',
                final_response TEXT NOT NULL DEFAULT '',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                retrieved_chunks_json TEXT NOT NULL DEFAULT '[]',
                approved INTEGER NOT NULL DEFAULT 0,
                source_name TEXT NOT NULL DEFAULT 'text',
                source_filename TEXT NOT NULL DEFAULT ''
            )
            """
        )
        _ensure_columns(connection)
        connection.commit()
    finally:
        connection.close()


def _coalesce_review_status(
    *,
    routing_mode: str,
    review_status: str,
    approved: int,
) -> str:
    if review_status:
        return review_status
    if approved:
        return "accepted"
    if routing_mode == "AUTO_REPLY":
        return "auto_approved"
    return "pending_review"


def create_interaction(
    *,
    user_input: str,
    extracted_text: str,
    classification: dict[str, Any],
    risk_level: str,
    risk_score: int,
    compliance_score: int,
    confidence_score: int = 0,
    confidence_level: str = "LOW",
    routing_mode: str = "AUTO_REPLY",
    review_status: str = "not_required",
    review_decision: str = "",
    review_notes: str = "",
    generated_response: str,
    warnings: Iterable[str],
    retrieved_chunks: Iterable[dict[str, Any]],
    approved: int = 0,
    source_name: str = "text",
    source_filename: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    initialize_database(db_path)
    connection = get_connection(db_path)
    try:
        timestamp = utc_now_iso()
        normalized_review_status = _coalesce_review_status(
            routing_mode=routing_mode,
            review_status=review_status,
            approved=approved,
        )
        cursor = connection.execute(
            """
            INSERT INTO interactions (
                created_at,
                updated_at,
                user_input,
                extracted_text,
                classification_json,
                risk_level,
                risk_score,
                compliance_score,
                confidence_score,
                confidence_level,
                routing_mode,
                review_status,
                review_decision,
                review_notes,
                generated_response,
                final_response,
                warnings_json,
                retrieved_chunks_json,
                approved,
                source_name,
                source_filename
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                timestamp,
                user_input,
                extracted_text,
                _json_dumps(classification),
                risk_level,
                int(risk_score),
                int(compliance_score),
                int(confidence_score),
                confidence_level,
                routing_mode,
                normalized_review_status,
                review_decision,
                review_notes,
                generated_response,
                generated_response,
                _json_dumps(list(warnings)),
                _json_dumps(list(retrieved_chunks)),
                int(approved),
                source_name,
                source_filename,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def update_review_state(
    interaction_id: int,
    *,
    review_status: str,
    review_decision: str,
    review_notes: str = "",
    final_response: Optional[str] = None,
    approved: Optional[int] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    initialize_database(db_path)
    connection = get_connection(db_path)
    try:
        connection.execute(
            """
            UPDATE interactions
            SET final_response = COALESCE(?, final_response),
                approved = COALESCE(?, approved),
                review_status = ?,
                review_decision = ?,
                review_notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                final_response,
                approved,
                review_status,
                review_decision,
                review_notes,
                utc_now_iso(),
                interaction_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def approve_interaction(
    interaction_id: int,
    *,
    final_response: str,
    review_notes: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    update_review_state(
        interaction_id,
        review_status="accepted",
        review_decision="accept",
        review_notes=review_notes,
        final_response=final_response,
        approved=1,
        db_path=db_path,
    )


def reject_interaction(
    interaction_id: int,
    *,
    review_notes: str = "",
    final_response: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    update_review_state(
        interaction_id,
        review_status="rejected",
        review_decision="reject",
        review_notes=review_notes,
        final_response=final_response or None,
        approved=0,
        db_path=db_path,
    )


def mark_auto_approved(
    interaction_id: int,
    *,
    final_response: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    update_review_state(
        interaction_id,
        review_status="auto_approved",
        review_decision="auto_reply",
        review_notes="",
        final_response=final_response,
        approved=1,
        db_path=db_path,
    )


def update_generated_response(
    interaction_id: int,
    *,
    generated_response: str,
    final_response: Optional[str] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    initialize_database(db_path)
    connection = get_connection(db_path)
    try:
        connection.execute(
            """
            UPDATE interactions
            SET generated_response = ?,
                final_response = COALESCE(?, final_response),
                updated_at = ?
            WHERE id = ?
            """,
            (generated_response, final_response, utc_now_iso(), interaction_id),
        )
        connection.commit()
    finally:
        connection.close()


def fetch_recent_interactions(
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    connection = get_connection(db_path)
    try:
        rows = connection.execute(
            """
            SELECT *
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def fetch_interaction(
    interaction_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    initialize_database(db_path)
    connection = get_connection(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM interactions WHERE id = ?",
            (interaction_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()
