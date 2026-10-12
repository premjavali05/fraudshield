
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_PATH = (
    PROJECT_ROOT / "data" / "app" / "fraudshield.db"
)


def initialize_database() -> None:
    """Create the local prediction audit database if needed."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
        connection.execute("PRAGMA journal_mode=WAL")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                decision_threshold REAL NOT NULL,
                flag_for_review INTEGER NOT NULL,
                model_version TEXT NOT NULL
            )
        """)

        connection.commit()


def record_prediction(
    transaction_type: str,
    risk_score: float,
    decision_threshold: float,
    flag_for_review: bool,
    model_version: str,
) -> int:
    """Persist prediction metadata and return its audit ID."""
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_audit (
                created_at,
                transaction_type,
                risk_score,
                decision_threshold,
                flag_for_review,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                transaction_type,
                risk_score,
                decision_threshold,
                int(flag_for_review),
                model_version,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Return recent audit entries, newest first."""
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                transaction_type,
                risk_score,
                decision_threshold,
                flag_for_review,
                model_version
            FROM prediction_audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]
