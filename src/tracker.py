"""Placement Readiness Tracker: Student journeys, progress over time, and adaptive planning."""
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import DB_PATH
from src.predict import predict_student_placement


# ---------------------------------------------------------
# Database Schema Initialization
# ---------------------------------------------------------
def init_tracker_db(db_path=None) -> None:
    """Initialize students, checkins, and tasks tables if not present."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                gender TEXT,
                specialisation TEXT,
                degree_t TEXT,
                ssc_b TEXT,
                hsc_b TEXT,
                hsc_s TEXT,
                ssc_p REAL,
                hsc_p REAL,
                degree_p REAL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkins (
                checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                ssc_p REAL,
                hsc_p REAL,
                degree_p REAL,
                etest_p REAL,
                mba_p REAL,
                workex TEXT,
                probability REAL,
                readiness_score REAL,
                notes TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                task_text TEXT NOT NULL,
                category TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                updated_at TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


# ---------------------------------------------------------
# Readiness Score Formula (0 - 100)
# ---------------------------------------------------------
def calculate_readiness_score(
    prob: float,
    etest_p: float,
    workex: str,
    checkin_count: int = 1,
    days_since_last: int = 0,
) -> Tuple[float, str, str]:
    """Calculate the comprehensive placement readiness score (0 to 100).

    Components:
    1. Calibrated Placement Probability (60%): prob * 100 * 0.60
    2. Employability & Aptitude Test (20%): min(100, max(0, etest_p)) * 0.20
    3. Work Experience / Internships (10%): (100 if Yes else 0) * 0.10
    4. Check-in Consistency (10%):
       - Baseline from check-in count: 1 checkin = 50, 2 = 70, 3 = 85, 4+ = 100
       - Inactivity penalty: -2 points per week if inactive > 14 days (down to min 0)

    Bands:
    - < 40: Needs Support
    - 40 - 70: Developing
    - > 70: Placement Ready

    Returns:
        (readiness_score, band_name, color_hex)
    """
    prob_clean = max(0.0, min(1.0, float(prob)))
    etest_clean = max(0.0, min(100.0, float(etest_p)))
    workex_val = 100.0 if str(workex).strip().lower() == "yes" else 0.0

    # Consistency scoring (0 - 100)
    if checkin_count >= 4:
        consistency = 100.0
    elif checkin_count == 3:
        consistency = 85.0
    elif checkin_count == 2:
        consistency = 70.0
    else:
        consistency = 50.0

    if days_since_last > 14:
        penalty = min(consistency, ((days_since_last - 14) // 7) * 5.0)
        consistency = max(0.0, consistency - penalty)

    score = (
        (prob_clean * 100.0 * 0.60)
        + (etest_clean * 0.20)
        + (workex_val * 0.10)
        + (consistency * 0.10)
    )
    score = round(max(0.0, min(100.0, score)), 1)

    if score > 70.0:
        band = "Placement Ready"
        color = "#10B981"
    elif score >= 40.0:
        band = "Developing"
        color = "#F59E0B"
    else:
        band = "Needs Support"
        color = "#EF4444"

    return score, band, color


# ---------------------------------------------------------
# Student and Check-in Operations
# ---------------------------------------------------------
def register_or_get_student(
    student_id: str,
    name: str,
    profile: Dict[str, Any],
    db_path=None,
) -> Dict[str, Any]:
    """Register student on first check-in or retrieve existing demographic record."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    s_id = str(student_id).strip()

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id = ?", (s_id,))
        row = cursor.fetchone()

        if row is None:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO students (
                    student_id, name, gender, specialisation, degree_t,
                    ssc_b, hsc_b, hsc_s, ssc_p, hsc_p, degree_p, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s_id,
                name or f"Student {s_id}",
                str(profile.get("gender", "M")),
                str(profile.get("specialisation", "Mkt&Fin")),
                str(profile.get("degree_t", "Comm&Mgmt")),
                str(profile.get("ssc_b", "Central")),
                str(profile.get("hsc_b", "Others")),
                str(profile.get("hsc_s", "Commerce")),
                float(profile.get("ssc_p", 70.0)),
                float(profile.get("hsc_p", 70.0)),
                float(profile.get("degree_p", 68.0)),
                now_str,
            ))
            conn.commit()

        cursor.execute("SELECT * FROM students WHERE student_id = ?", (s_id,))
        cols = [c[0] for c in cursor.description]
        return dict(zip(cols, cursor.fetchone()))


def record_student_checkin(
    student_id: str,
    etest_p: float,
    mba_p: float,
    workex: str,
    active_pipeline: Any,
    threshold: float = 0.50,
    checkin_date: Optional[str] = None,
    notes: str = "",
    db_path=None,
) -> Dict[str, Any]:
    """Record a new progress check-in for a student, updating locked scores from student record."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    s_id = str(student_id).strip()

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id = ?", (s_id,))
        student_row = cursor.fetchone()
        if not student_row:
            raise ValueError(f"Student ID '{s_id}' not found. Please register student first.")
        s_cols = [c[0] for c in cursor.description]
        s_dict = dict(zip(s_cols, student_row))

        # Retrieve past checkins for consistency & delta
        cursor.execute("""
            SELECT date, probability, readiness_score FROM checkins
            WHERE student_id = ? ORDER BY date ASC
        """, (s_id,))
        past_checkins = cursor.fetchall()
        count_before = len(past_checkins)

        # Days since last checkin
        date_str = checkin_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        days_since = 0
        if past_checkins:
            last_date_str = past_checkins[-1][0]
            try:
                d1 = datetime.strptime(last_date_str[:10], "%Y-%m-%d")
                d2 = datetime.strptime(date_str[:10], "%Y-%m-%d")
                days_since = max(0, (d2 - d1).days)
            except Exception:
                days_since = 0

        # Construct full student profile with locked academic scores
        full_profile = {
            "gender": s_dict["gender"],
            "ssc_p": s_dict["ssc_p"],
            "ssc_b": s_dict["ssc_b"],
            "hsc_p": s_dict["hsc_p"],
            "hsc_b": s_dict["hsc_b"],
            "hsc_s": s_dict["hsc_s"],
            "degree_p": s_dict["degree_p"],
            "degree_t": s_dict["degree_t"],
            "workex": str(workex).strip(),
            "etest_p": float(etest_p),
            "specialisation": s_dict["specialisation"],
            "mba_p": float(mba_p),
        }

        # Predict probability using shared model logic
        pred_res = predict_student_placement(full_profile, active_pipeline, threshold=threshold)
        prob = pred_res["probability"]

        # Calculate readiness score
        score, band, color = calculate_readiness_score(
            prob, etest_p, workex, checkin_count=count_before + 1, days_since_last=days_since
        )

        cursor.execute("""
            INSERT INTO checkins (
                student_id, date, ssc_p, hsc_p, degree_p, etest_p, mba_p, workex,
                probability, readiness_score, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s_id, date_str, s_dict["ssc_p"], s_dict["hsc_p"], s_dict["degree_p"],
            float(etest_p), float(mba_p), str(workex), prob, score, notes,
        ))
        conn.commit()

        # Generate / update adaptive weekly tasks
        generate_and_save_weekly_plan(s_id, full_profile, prob, db_path=path)

        return {
            "student_id": s_id,
            "date": date_str,
            "probability": prob,
            "readiness_score": score,
            "band": band,
            "color": color,
            "total_checkins": count_before + 1,
        }


def get_student_checkin_history(student_id: str, db_path=None) -> pd.DataFrame:
    """Retrieve full chronological check-in progression for a student."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    with sqlite3.connect(path) as conn:
        query = """
            SELECT date, ssc_p, hsc_p, degree_p, etest_p, mba_p, workex,
                   probability, readiness_score, notes
            FROM checkins
            WHERE student_id = ?
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(str(student_id).strip(),))
    return df


def get_all_tracked_students(db_path=None) -> List[Dict[str, Any]]:
    """Return list of all registered students with their latest check-in metrics."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.student_id, s.name, s.specialisation,
                   COUNT(c.checkin_id) as total_checkins,
                   MAX(c.date) as last_checkin,
                   c_latest.probability, c_latest.readiness_score,
                   c_latest.etest_p, c_latest.workex
            FROM students s
            LEFT JOIN checkins c ON s.student_id = c.student_id
            LEFT JOIN checkins c_latest ON c_latest.checkin_id = (
                SELECT checkin_id FROM checkins WHERE student_id = s.student_id ORDER BY date DESC LIMIT 1
            )
            GROUP BY s.student_id
            ORDER BY s.student_id ASC
        """)
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


def delete_student_data(student_id: str, db_path=None) -> bool:
    """Completely wipe a student's profile, check-ins, and tasks (GDPR / Delete-my-data)."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    s_id = str(student_id).strip()
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_tasks WHERE student_id = ?", (s_id,))
        cursor.execute("DELETE FROM checkins WHERE student_id = ?", (s_id,))
        cursor.execute("DELETE FROM students WHERE student_id = ?", (s_id,))
        conn.commit()
        return True


# ---------------------------------------------------------
# Adaptive Weekly Action Plan
# ---------------------------------------------------------
def generate_and_save_weekly_plan(
    student_id: str,
    profile: Dict[str, Any],
    prob: float,
    db_path=None,
) -> List[Dict[str, Any]]:
    """Generate and store an adaptive 4-week action plan based on the weakest improvable factor."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    s_id = str(student_id).strip()

    etest_p = float(profile.get("etest_p", 70.0))
    workex = str(profile.get("workex", "No")).strip().lower()
    mba_p = float(profile.get("mba_p", 65.0))

    # Weakest improvable factor analysis
    tasks = []
    if workex == "no":
        category = "Practical Industry Experience"
        tasks = [
            (1, "Resume & Portfolio: Create a 1-page standard placement resume with 2 capstone projects.", category),
            (2, "Internship Applications: Apply to 5 verified summer internships or live business projects.", category),
            (3, "Industry Case Study: Solve and document a real-world business case related to your domain.", category),
            (4, "Mentor Review: Schedule a 30-minute resume walkthrough with a placement cell counselor.", category),
        ]
    elif etest_p < 70.0:
        category = "Aptitude & Problem Solving"
        tasks = [
            (1, "Diagnostic Test: Complete 2 full-length quantitative and data interpretation mock drills.", category),
            (2, "Logical Reasoning: Practice 30 syllogism, seating arrangement, and analytical puzzle sets.", category),
            (3, "Speed Math Drills: Master percentage, ratio, and algebra shortcuts (aim for <60s per problem).", category),
            (4, "Company Screening Simulation: Take a timed 60-minute mock aptitude test on the college portal.", category),
        ]
    elif mba_p < 65.0:
        category = "Domain Knowledge & Certifications"
        tasks = [
            (1, "Core Concepts Refresher: Review fundamental corporate finance, marketing, or HR frameworks.", category),
            (2, "Professional Certification: Enroll in an industry-recognized certificate course (Coursera/NPTEL).", category),
            (3, "Business Analysis: Prepare a 10-slide deck analyzing an industry sector (Fintech/FMCG/Edtech).", category),
            (4, "Technical Interview Prep: Practice 20 standard domain interview questions with a peer.", category),
        ]
    else:
        category = "Interview & Leadership Readiness"
        tasks = [
            (1, "Elevator Pitch: Refine a 2-minute personal pitch highlighting your strengths and projects.", category),
            (2, "Behavioral STAR Stories: Write out 4 behavioral stories (Leadership, Conflict, Teamwork, Challenge).", category),
            (3, "Mock Technical Interview: Participate in a faculty-led or alumni-led mock interview.", category),
            (4, "Company Deep Dive: Research target recruiter business models, quarterly reports, and campus culture.", category),
        ]

    # Save to database (preserve existing completions if task text matches)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT task_text, completed FROM student_tasks WHERE student_id = ?", (s_id,))
        existing_completed = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("DELETE FROM student_tasks WHERE student_id = ?", (s_id,))
        for week_num, text, cat in tasks:
            was_completed = existing_completed.get(text, 0)
            cursor.execute("""
                INSERT INTO student_tasks (student_id, week, task_text, category, completed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (s_id, week_num, text, cat, was_completed, now_str))
        conn.commit()

    return [{"week": w, "task_text": t, "category": c} for w, t, c in tasks]


def get_student_tasks(student_id: str, db_path=None) -> List[Dict[str, Any]]:
    """Retrieve the 4-week task checklist for a student."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT task_id, week, task_text, category, completed
            FROM student_tasks
            WHERE student_id = ?
            ORDER BY week ASC
        """, (str(student_id).strip(),))
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


def update_task_completion(task_id: int, completed: bool, db_path=None) -> None:
    """Toggle completion status for a specific student task."""
    path = db_path or DB_PATH
    init_tracker_db(path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE student_tasks
            SET completed = ?, updated_at = ?
            WHERE task_id = ?
        """, (1 if completed else 0, now_str, int(task_id)))
        conn.commit()


# ---------------------------------------------------------
# Early-Warning Alerts
# ---------------------------------------------------------
def get_early_warning_alerts(db_path=None) -> pd.DataFrame:
    """Identify students needing immediate placement cell attention:
    1. Probability dropped across the last two check-ins.
    2. Inactive (no check-in for 30+ days).
    """
    path = db_path or DB_PATH
    init_tracker_db(path)
    alerts = []

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name, specialisation FROM students")
        students = cursor.fetchall()

        now = datetime.now()

        for s_id, name, spec in students:
            cursor.execute("""
                SELECT date, probability, readiness_score FROM checkins
                WHERE student_id = ? ORDER BY date DESC LIMIT 2
            """, (s_id,))
            recent = cursor.fetchall()

            if not recent:
                continue

            latest_date_str, latest_prob, latest_score = recent[0]
            try:
                latest_dt = datetime.strptime(latest_date_str[:10], "%Y-%m-%d")
                days_inactive = (now - latest_dt).days
            except Exception:
                days_inactive = 0

            # Rule 1: Dropped probability across two checkins
            if len(recent) >= 2:
                prev_date_str, prev_prob, prev_score = recent[1]
                prob_diff = latest_prob - prev_prob
                score_diff = latest_score - prev_score
                if prob_diff < -0.01:
                    alerts.append({
                        "student_id": s_id,
                        "name": name,
                        "specialisation": spec,
                        "alert_type": "Placement Probability Drop",
                        "severity": "High" if prob_diff <= -0.10 else "Medium",
                        "latest_metric": f"{latest_prob:.1%}",
                        "change": f"{prob_diff:+.1%}",
                        "days_inactive": days_inactive,
                        "details": f"Probability dropped from {prev_prob:.1%} to {latest_prob:.1%}.",
                    })

            # Rule 2: Inactivity over 30 days
            if days_inactive >= 30:
                alerts.append({
                    "student_id": s_id,
                    "name": name,
                    "specialisation": spec,
                    "alert_type": "Check-in Inactivity (30+ days)",
                    "severity": "High" if days_inactive >= 60 else "Medium",
                    "latest_metric": f"{latest_score:.1f} pts",
                    "change": f"{days_inactive} days idle",
                    "days_inactive": days_inactive,
                    "details": f"No progress check-in logged for {days_inactive} days.",
                })

    df = pd.DataFrame(alerts)
    if not df.empty:
        df["sort_key"] = df["severity"].map({"High": 0, "Medium": 1})
        df = df.sort_values(by=["sort_key", "days_inactive"], ascending=[True, False]).drop(columns=["sort_key"])
    return df


def draft_supportive_message(alert_row: Dict[str, Any]) -> str:
    """Draft a supportive, non-intrusive email message for counselors to review (never auto-sent)."""
    name = alert_row.get("name", "Student")
    alert_type = alert_row.get("alert_type", "Check-in Inactivity")
    spec = alert_row.get("specialisation", "MBA")

    if "Drop" in alert_type:
        body = f"""Subject: Career Coaching & Placement Readiness Check-in | Training & Placement Cell

Dear {name},

We hope your {spec} semester is progressing smoothly! 

During our regular placement progress review, our system noted a slight dip in your recent screening assessment scores. We want to emphasize that fluctuations are completely normal as course loads intensify.

The Placement Cell is hosting targeted workshops this month on aptitude shortcuts and technical interview practice. Would you be available for a brief 15-minute 1-on-1 session this Thursday or Friday to review customized strategies and capstone projects?

We're here to help you secure the best possible role on campus!

Warm regards,  
Training & Placement Cell  
Career Services & Student Mentorship"""
    else:
        days = alert_row.get("days_inactive", 30)
        body = f"""Subject: Checking In: Placement Readiness Milestone | Training & Placement Cell

Dear {name},

We noticed it has been about {days} days since your last placement readiness check-in. Campus recruitment drives are approaching, and we want to ensure you have all the tools needed for upcoming interview rounds.

Please take 2 minutes to update your latest mock aptitude score and internship status on the student portal. If you need any assistance with resume critiques or mock interview scheduling, feel free to drop by our office.

Best of luck with your preparation!

Warm regards,  
Training & Placement Cell  
Career Services & Student Mentorship"""

    return body


# ---------------------------------------------------------
# Sample Data Loader for Demonstration
# ---------------------------------------------------------
def load_sample_tracker_data(active_pipeline: Any, threshold: float = 0.50, db_path=None) -> None:
    """Seed sample students and multi-date check-in journeys for immediate demonstration."""
    path = db_path or DB_PATH
    init_tracker_db(path)

    samples = [
        {
            "student_id": "STU-201",
            "name": "Aarav Sharma",
            "specialisation": "Mkt&Fin",
            "gender": "M", "degree_t": "Comm&Mgmt", "ssc_b": "Central", "hsc_b": "Others", "hsc_s": "Commerce",
            "ssc_p": 68.0, "hsc_p": 65.0, "degree_p": 66.0,
            "checkins": [
                ("2026-08-15", 58.0, 60.0, "No", "Initial diagnostic test"),
                ("2026-09-15", 68.0, 62.0, "No", "Completed aptitude drills"),
                ("2026-10-15", 76.0, 64.0, "Yes", "Completed 6-week summer internship"),
            ]
        },
        {
            "student_id": "STU-202",
            "name": "Priya Nair",
            "specialisation": "Mkt&HR",
            "gender": "F", "degree_t": "Sci&Tech", "ssc_b": "Others", "hsc_b": "Others", "hsc_s": "Science",
            "ssc_p": 55.0, "hsc_p": 58.0, "degree_p": 59.0,
            "checkins": [
                ("2026-08-10", 62.0, 56.0, "No", "Initial baseline checkin"),
                ("2026-09-12", 52.0, 55.0, "No", "Missed several mock test rounds"),
            ]
        },
        {
            "student_id": "STU-203",
            "name": "Rohan Verma",
            "specialisation": "Mkt&Fin",
            "gender": "M", "degree_t": "Comm&Mgmt", "ssc_b": "Central", "hsc_b": "Central", "hsc_s": "Commerce",
            "ssc_p": 82.0, "hsc_p": 80.0, "degree_p": 78.0,
            "checkins": [
                ("2026-07-20", 82.0, 72.0, "Yes", "Pre-placement offer tracking"),
            ]
        },
    ]

    for s in samples:
        profile = {k: s[k] for k in ["gender", "specialisation", "degree_t", "ssc_b", "hsc_b", "hsc_s", "ssc_p", "hsc_p", "degree_p"]}
        register_or_get_student(s["student_id"], s["name"], profile, db_path=path)

        for d_str, etest, mba, workex, notes in s["checkins"]:
            record_student_checkin(
                s["student_id"],
                etest_p=etest,
                mba_p=mba,
                workex=workex,
                active_pipeline=active_pipeline,
                threshold=threshold,
                checkin_date=d_str,
                notes=notes,
                db_path=path,
            )

