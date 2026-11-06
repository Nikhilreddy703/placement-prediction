"""TPO (Training & Placement Officer) Command Center.

Provides cohort-level analytics, risk heatmap generation, company shortlisting
with customisable criteria, and exportable Excel reports for the placement cell.
"""
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import (
    NUMERIC_FEATURES,
    THRESHOLD_HIGH_RISK,
    THRESHOLD_MODERATE_RISK,
)
from src.explain import categorize_risk


# ---------------------------------------------------------
# 1.1 Cohort Risk Heatmap
# ---------------------------------------------------------
def generate_cohort_risk_heatmap(
    students: List[Dict],
    dataset_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Categorise all students into Red / Amber / Green risk bands.

    Risk bands:
        - **Red**   (<40 % probability)  — high-priority intervention needed
        - **Amber** (40-65 % probability) — borderline, targeted support
        - **Green** (>65 % probability)  — on track for placement

    Args:
        students: List of student dicts, each containing at minimum
                  ``probability`` (float 0-1) and identifying fields
                  (``student_id``, ``name``, ``specialisation``, etc.).
        dataset_df: Reference dataset DataFrame (used for contextual stats
                    but can be empty if unavailable).

    Returns:
        Dictionary with keys:
            ``red_students``   — list of student dicts in the red band
            ``amber_students`` — list of student dicts in the amber band
            ``green_students`` — list of student dicts in the green band
            ``summary``        — dict with counts and percentages per band
            ``specialisation_breakdown`` — dict mapping specialisation to
                                           ``{red: int, amber: int, green: int}``
    """
    red: List[Dict] = []
    amber: List[Dict] = []
    green: List[Dict] = []

    for stu in students:
        prob = float(stu.get("probability", stu.get("Placement_Probability", 0.0)))
        enriched = {**stu, "probability": prob}

        # Assign risk band label for convenience
        if prob < THRESHOLD_HIGH_RISK:
            enriched["risk_band"] = "Red"
            red.append(enriched)
        elif prob < THRESHOLD_MODERATE_RISK:
            enriched["risk_band"] = "Amber"
            amber.append(enriched)
        else:
            enriched["risk_band"] = "Green"
            green.append(enriched)

    total = max(len(students), 1)
    summary = {
        "total": len(students),
        "red_count": len(red),
        "amber_count": len(amber),
        "green_count": len(green),
        "red_pct": round(len(red) / total * 100, 1),
        "amber_pct": round(len(amber) / total * 100, 1),
        "green_pct": round(len(green) / total * 100, 1),
    }

    # Specialisation breakdown
    spec_breakdown: Dict[str, Dict[str, int]] = {}
    for stu in red + amber + green:
        spec = str(stu.get("specialisation", "Unknown"))
        if spec not in spec_breakdown:
            spec_breakdown[spec] = {"red": 0, "amber": 0, "green": 0}
        band_key = stu.get("risk_band", "Amber").lower()
        spec_breakdown[spec][band_key] += 1

    return {
        "red_students": red,
        "amber_students": amber,
        "green_students": green,
        "summary": summary,
        "specialisation_breakdown": spec_breakdown,
    }


# ---------------------------------------------------------
# 1.2 Company Shortlist Generator
# ---------------------------------------------------------
def generate_company_shortlist(
    students_data: pd.DataFrame,
    criteria: Dict[str, Any],
) -> pd.DataFrame:
    """Filter students matching ALL company hiring criteria.

    Args:
        students_data: DataFrame with at least the columns
            ``degree_p``, ``etest_p``, ``Placement_Probability``,
            ``specialisation``, ``workex``, ``gender``, and
            ``Risk_Level`` (as produced by ``batch_predict_students``).
        criteria: Dictionary of filters. Supported keys:
            - ``min_degree_p``     — minimum undergraduate percentage
            - ``min_etest_p``      — minimum employability test score
            - ``min_probability``  — minimum placement probability (0-1)
            - ``specialisation``   — ``"Mkt&Fin"`` / ``"Mkt&HR"`` / ``"All"``
            - ``workex_required``  — bool, if True only students with ``workex == 'Yes'``
            - ``gender``           — ``"M"`` / ``"F"`` / ``"All"``

    Returns:
        Filtered DataFrame sorted by probability descending with columns:
        ``student_id`` (or index), key scores, probability, and risk_level.
    """
    df = students_data.copy()

    # Normalise probability column name
    prob_col = "Placement_Probability"
    if prob_col not in df.columns and "probability" in df.columns:
        df.rename(columns={"probability": prob_col}, inplace=True)

    # Apply numeric filters
    min_degree = criteria.get("min_degree_p")
    if min_degree is not None and "degree_p" in df.columns:
        df = df[df["degree_p"] >= float(min_degree)]

    min_etest = criteria.get("min_etest_p")
    if min_etest is not None and "etest_p" in df.columns:
        df = df[df["etest_p"] >= float(min_etest)]

    min_prob = criteria.get("min_probability")
    if min_prob is not None and prob_col in df.columns:
        df = df[df[prob_col] >= float(min_prob)]

    # Specialisation filter
    spec_filter = criteria.get("specialisation", "All")
    if spec_filter and str(spec_filter) != "All" and "specialisation" in df.columns:
        df = df[df["specialisation"] == spec_filter]

    # Work experience filter
    if criteria.get("workex_required") and "workex" in df.columns:
        df = df[df["workex"].astype(str).str.strip().str.lower() == "yes"]

    # Gender filter
    gender_filter = criteria.get("gender", "All")
    if gender_filter and str(gender_filter) != "All" and "gender" in df.columns:
        df = df[df["gender"] == gender_filter]

    # Sort by probability descending
    if prob_col in df.columns:
        df = df.sort_values(by=prob_col, ascending=False)

    # Ensure risk_level column exists
    if "Risk_Level" not in df.columns and prob_col in df.columns:
        df["Risk_Level"] = df[prob_col].apply(lambda p: categorize_risk(float(p))[0])

    # Select and order output columns
    desired_cols = []
    for c in ["student_id", "sl_no", "name"]:
        if c in df.columns:
            desired_cols.append(c)
    for c in ["gender", "specialisation", "degree_p", "etest_p", "mba_p",
              "workex", "ssc_p", "hsc_p", prob_col, "Risk_Level"]:
        if c in df.columns and c not in desired_cols:
            desired_cols.append(c)
    # Include any remaining columns not yet selected
    remaining = [c for c in df.columns if c not in desired_cols]
    output_cols = desired_cols + remaining

    return df[output_cols].reset_index(drop=True)


# ---------------------------------------------------------
# 1.3 Excel Export with Color Coding
# ---------------------------------------------------------
def generate_shortlist_excel(
    shortlist_df: pd.DataFrame,
    company_name: str,
) -> bytes:
    """Create a colour-coded Excel workbook for the shortlisted students.

    Features:
        - Header row with company name and generation date.
        - Colour-coded rows: green (≥65 % prob), yellow (40-65 %),
          red (<40 %).
        - Auto-adjusted column widths for readability.

    Args:
        shortlist_df: DataFrame of shortlisted students (from
                      ``generate_company_shortlist``).
        company_name: Target company name (shown in the header).

    Returns:
        Raw bytes of the ``.xlsx`` file suitable for ``st.download_button``.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        shortlist_df.to_excel(writer, index=False, sheet_name="Shortlist", startrow=2)
        wb = writer.book
        ws = writer.sheets["Shortlist"]

        # --- Header ---
        ws.merge_cells(start_row=1, start_column=1, end_row=1,
                       end_column=max(len(shortlist_df.columns), 3))
        header_cell = ws.cell(row=1, column=1)
        header_cell.value = (
            f"Campus Placement Shortlist — {company_name}  |  "
            f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
        )
        header_cell.font = Font(size=13, bold=True, color="1F4E79")
        header_cell.alignment = Alignment(horizontal="left")

        # --- Style column headers ---
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=10)
        for col_idx in range(1, len(shortlist_df.columns) + 1):
            cell = ws.cell(row=3, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        # --- Color-code data rows based on probability ---
        green_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")
        yellow_fill = PatternFill(start_color="FEF9E7", end_color="FEF9E7", fill_type="solid")
        red_fill = PatternFill(start_color="FADBD8", end_color="FADBD8", fill_type="solid")

        prob_col_name = None
        for candidate in ("Placement_Probability", "probability"):
            if candidate in shortlist_df.columns:
                prob_col_name = candidate
                break

        prob_col_idx: Optional[int] = None
        if prob_col_name is not None:
            prob_col_idx = list(shortlist_df.columns).index(prob_col_name)

        data_start_row = 4  # row 1 = header, 2 = blank, 3 = col headers
        for row_offset in range(len(shortlist_df)):
            xl_row = data_start_row + row_offset
            fill = None
            if prob_col_idx is not None:
                prob_val = shortlist_df.iloc[row_offset, prob_col_idx]
                try:
                    prob_val = float(prob_val)
                except (TypeError, ValueError):
                    prob_val = 0.0

                if prob_val >= THRESHOLD_MODERATE_RISK:
                    fill = green_fill
                elif prob_val >= THRESHOLD_HIGH_RISK:
                    fill = yellow_fill
                else:
                    fill = red_fill

            for col_idx in range(1, len(shortlist_df.columns) + 1):
                cell = ws.cell(row=xl_row, column=col_idx)
                cell.alignment = Alignment(horizontal="center")
                if fill:
                    cell.fill = fill

        # --- Auto column widths ---
        for col_idx in range(1, len(shortlist_df.columns) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = max(
                len(str(ws.cell(row=3, column=col_idx).value or "")),
                *(
                    len(str(ws.cell(row=data_start_row + r, column=col_idx).value or ""))
                    for r in range(len(shortlist_df))
                ),
            ) if len(shortlist_df) > 0 else len(str(ws.cell(row=3, column=col_idx).value or ""))
            ws.column_dimensions[col_letter].width = min(max_len + 4, 35)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------
# 1.4 Cohort Analytics
# ---------------------------------------------------------
def get_cohort_analytics(
    dataset_df: pd.DataFrame,
    students: List[Dict],
) -> Dict[str, Any]:
    """Compute aggregate cohort analytics for the TPO dashboard.

    Args:
        dataset_df: Original dataset DataFrame (for baseline statistics).
        students: List of student dicts with at minimum ``probability``,
                  ``specialisation``, ``workex``, and identifying fields.

    Returns:
        Dictionary with:
            ``avg_prob_by_specialisation`` — dict mapping spec → avg probability
            ``avg_prob_by_workex``         — dict mapping workex status → avg probability
            ``top_10``                     — list of top-10 students (highest probability)
            ``bottom_10``                  — list of bottom-10 students (highest risk)
            ``overall_avg_probability``    — float
            ``score_distribution``         — dict mapping numeric feature → mean value
            ``week_over_week_trend``       — list of dicts with ``week``, ``avg_prob``
                                             (empty if no checkin data available)
    """
    if not students:
        return {
            "avg_prob_by_specialisation": {},
            "avg_prob_by_workex": {},
            "top_10": [],
            "bottom_10": [],
            "overall_avg_probability": 0.0,
            "score_distribution": {},
            "week_over_week_trend": [],
        }

    df_stu = pd.DataFrame(students)

    # Normalise probability column name
    prob_col = "probability"
    if prob_col not in df_stu.columns and "Placement_Probability" in df_stu.columns:
        df_stu.rename(columns={"Placement_Probability": prob_col}, inplace=True)

    # Ensure numeric
    if prob_col in df_stu.columns:
        df_stu[prob_col] = pd.to_numeric(df_stu[prob_col], errors="coerce").fillna(0.0)
    else:
        df_stu[prob_col] = 0.0

    # Average probability by specialisation
    avg_by_spec: Dict[str, float] = {}
    if "specialisation" in df_stu.columns:
        avg_by_spec = (
            df_stu.groupby("specialisation")[prob_col]
            .mean()
            .round(4)
            .to_dict()
        )

    # Average probability by work experience
    avg_by_workex: Dict[str, float] = {}
    if "workex" in df_stu.columns:
        avg_by_workex = (
            df_stu.groupby("workex")[prob_col]
            .mean()
            .round(4)
            .to_dict()
        )

    # Top 10 and bottom 10
    sorted_asc = df_stu.sort_values(by=prob_col, ascending=True)
    sorted_desc = df_stu.sort_values(by=prob_col, ascending=False)
    top_10 = sorted_desc.head(10).to_dict(orient="records")
    bottom_10 = sorted_asc.head(10).to_dict(orient="records")

    # Overall average probability
    overall_avg = float(df_stu[prob_col].mean())

    # Score distribution from original dataset
    score_dist: Dict[str, float] = {}
    if dataset_df is not None and not dataset_df.empty:
        for feat in NUMERIC_FEATURES:
            if feat in dataset_df.columns:
                score_dist[feat] = round(float(dataset_df[feat].mean()), 2)

    # Week-over-week trend (best-effort from checkin dates)
    wow_trend: List[Dict[str, Any]] = []
    if "date" in df_stu.columns:
        try:
            df_stu["_date_parsed"] = pd.to_datetime(df_stu["date"], errors="coerce")
            df_stu["_week"] = df_stu["_date_parsed"].dt.isocalendar().week.astype(int)
            df_stu["_year"] = df_stu["_date_parsed"].dt.year
            weekly = (
                df_stu.dropna(subset=["_date_parsed"])
                .groupby(["_year", "_week"])[prob_col]
                .mean()
                .reset_index()
            )
            weekly.columns = ["year", "week", "avg_prob"]
            weekly = weekly.sort_values(by=["year", "week"])
            wow_trend = weekly.to_dict(orient="records")
        except Exception:
            pass

    return {
        "avg_prob_by_specialisation": avg_by_spec,
        "avg_prob_by_workex": avg_by_workex,
        "top_10": top_10,
        "bottom_10": bottom_10,
        "overall_avg_probability": round(overall_avg, 4),
        "score_distribution": score_dist,
        "week_over_week_trend": wow_trend,
    }
