"""SQLite database integration for logging prediction audit history."""
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from config import DB_PATH


def init_db(db_path=None) -> None:
    """Initialize the SQLite schema if not already present."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_identifier TEXT,
                ssc_p REAL,
                hsc_p REAL,
                degree_p REAL,
                etest_p REAL,
                mba_p REAL,
                workex TEXT,
                specialisation TEXT,
                degree_t TEXT,
                gender TEXT,
                predicted_status TEXT,
                probability REAL,
                risk_level TEXT,
                model_used TEXT
            )
        """)
        conn.commit()


def log_prediction(
    profile: Dict[str, Any],
    result: Dict[str, Any],
    model_name: str,
    identifier: str = "Student",
    db_path=None,
) -> int:
    """Record an individual prediction into the SQLite database."""
    path = db_path or DB_PATH
    init_db(path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (
                timestamp, student_identifier, ssc_p, hsc_p, degree_p, etest_p, mba_p,
                workex, specialisation, degree_t, gender, predicted_status, probability,
                risk_level, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            str(identifier),
            float(profile.get("ssc_p", 0)),
            float(profile.get("hsc_p", 0)),
            float(profile.get("degree_p", 0)),
            float(profile.get("etest_p", 0)),
            float(profile.get("mba_p", 0)),
            str(profile.get("workex", "No")),
            str(profile.get("specialisation", "N/A")),
            str(profile.get("degree_t", "N/A")),
            str(profile.get("gender", "N/A")),
            str(result.get("status_text", "Placed")),
            float(result.get("probability", 0.0)),
            str(result.get("risk_level", "Low Risk")),
            str(model_name),
        ))
        conn.commit()
        return cursor.lastrowid


def get_prediction_history(limit: int = 100, db_path=None) -> pd.DataFrame:
    """Retrieve historical prediction audit logs."""
    path = db_path or DB_PATH
    init_db(path)
    with sqlite3.connect(path) as conn:
        query = f"""
            SELECT id, timestamp, student_identifier, ssc_p, hsc_p, degree_p,
                   etest_p, mba_p, workex, specialisation, predicted_status,
                   probability, risk_level, model_used
            FROM prediction_history
            ORDER BY id DESC
            LIMIT {int(limit)}
        """
        df = pd.read_sql_query(query, conn)
    return df


def clear_prediction_history(db_path=None) -> None:
    """Reset the audit history log."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        conn.cursor().execute("DELETE FROM prediction_history")
        conn.commit()

