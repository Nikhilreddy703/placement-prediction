"""Resume parsing, skill extraction, and Job Description matching engine.

Provides PDF text extraction, keyword-based skill identification across
multiple categories, ATS-style match scoring between resume and JD, and
actionable resume improvement suggestions.
"""
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Skills taxonomy — organized by category for category-level ATS breakdown
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "C\\+\\+", "C#", "JavaScript", "TypeScript",
        "R", "Go", "Rust", "Kotlin", "Swift", "Scala", "Ruby", "PHP",
        "MATLAB", "SAS", "Perl", "Shell Scripting", "Bash", "VBA",
    ],
    "Web & Frontend": [
        "HTML", "CSS", "React", "Angular", "Vue\\.js", "Next\\.js",
        "Node\\.js", "Express\\.js", "Django", "Flask", "Spring Boot",
        "REST API", "GraphQL", "Bootstrap", "Tailwind CSS",
    ],
    "Data & ML": [
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "NLP", "Computer Vision", "Data Analysis", "Data Science",
        "Data Engineering", "Data Visualization", "Statistical Modeling",
        "Tableau", "Power BI", "Excel", "Pandas", "NumPy", "Scikit-learn",
        "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM",
        "Apache Spark", "Hadoop", "ETL", "Data Warehousing",
        "Feature Engineering", "A/B Testing", "Regression Analysis",
        "Time Series", "Predictive Modeling", "Big Data",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",
        "Oracle", "SQLite", "DynamoDB", "Elasticsearch", "Neo4j",
        "NoSQL", "Data Modeling",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "Google Cloud", "GCP", "Docker", "Kubernetes",
        "Terraform", "CI/CD", "Jenkins", "GitHub Actions", "Ansible",
        "Linux", "Nginx", "Apache Kafka", "Microservices",
    ],
    "Tools & Platforms": [
        "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
        "Slack", "Figma", "Postman", "VS Code", "Jupyter", "Colab",
        "Notion", "Trello", "Asana",
    ],
    "Business & Finance": [
        "Financial Modeling", "Financial Analysis", "Accounting",
        "Marketing Analytics", "Digital Marketing", "SEO", "SEM",
        "Supply Chain", "Operations Management", "Business Analysis",
        "Business Intelligence", "CRM", "Salesforce", "SAP",
        "ERP", "Six Sigma", "Lean Management", "Consulting",
        "Market Research", "Competitive Analysis", "Budgeting",
        "Risk Management", "Valuation", "Investment Banking",
        "Portfolio Management", "Mergers and Acquisitions",
    ],
    "Soft Skills": [
        "Leadership", "Communication", "Teamwork", "Problem Solving",
        "Critical Thinking", "Time Management", "Adaptability",
        "Presentation", "Negotiation", "Conflict Resolution",
        "Project Management", "Stakeholder Management",
        "Cross-functional Collaboration", "Public Speaking",
        "Analytical Thinking", "Decision Making", "Mentoring",
        "Strategic Planning", "Agile", "Scrum",
    ],
    "Certifications & Standards": [
        "PMP", "CFA", "CPA", "ACCA", "FRM", "Google Analytics",
        "AWS Certified", "Azure Certified", "Certified Scrum Master",
        "ITIL", "ISO", "TOGAF", "CISSP", "CompTIA",
    ],
}

# Build a flat lookup: canonical_skill_lower -> (display_name, category)
_SKILL_LOOKUP: Dict[str, Tuple[str, str]] = {}
_SKILL_PATTERNS: List[Tuple[re.Pattern, str, str]] = []

for _cat, _skills in SKILLS_TAXONOMY.items():
    for _sk in _skills:
        # The display name is the pattern with regex escapes removed
        _display = _sk.replace("\\", "")
        _key = _display.lower()
        _SKILL_LOOKUP[_key] = (_display, _cat)
        # Build word-boundary regex for accurate matching
        _SKILL_PATTERNS.append(
            (re.compile(rf"\b{_sk}\b", re.IGNORECASE), _display, _cat)
        )


# ---------------------------------------------------------------------------
# 1. PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF file object.

    Attempts to use PyPDF2 first; falls back to raw byte decoding if
    the library is unavailable.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` or any file-like object
            with a ``.read()`` method and an optional ``.seek()`` method.

    Returns:
        Extracted text content as a single string.
    """
    raw_bytes = uploaded_file.read()
    # Reset stream position so the caller can re-read if needed
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # --- Attempt 1: PyPDF2 ---
    try:
        from PyPDF2 import PdfReader
        import io

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        if pages_text:
            return "\n".join(pages_text)
    except ImportError:
        pass
    except Exception:
        pass  # Corrupted PDF or parsing error — try fallback

    # --- Attempt 2: pdfplumber ---
    try:
        import pdfplumber
        import io

        pages_text = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
        if pages_text:
            return "\n".join(pages_text)
    except ImportError:
        pass
    except Exception:
        pass

    # --- Fallback: decode raw bytes best-effort ---
    try:
        text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = raw_bytes.decode("latin-1", errors="ignore")

    # Strip non-printable characters that sneak in from binary PDF streams
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Skill extraction helpers
# ---------------------------------------------------------------------------

def _extract_skills(text: str) -> List[Dict[str, str]]:
    """Internal: extract skills from arbitrary text.

    Returns a list of dicts ``{skill, category}`` with deduplication
    preserving first-seen order.
    """
    found: Dict[str, str] = {}  # display_name_lower -> category
    display_map: Dict[str, str] = {}  # lower -> display_name

    for pattern, display, category in _SKILL_PATTERNS:
        if pattern.search(text):
            key = display.lower()
            if key not in found:
                found[key] = category
                display_map[key] = display

    results = [
        {"skill": display_map[k], "category": found[k]}
        for k in found
    ]
    return results


def extract_skills_from_text(text: str) -> List[str]:
    """Extract and deduplicate skills found in resume text.

    Args:
        text: Plain-text content of a resume.

    Returns:
        Deduplicated list of skill names (display casing).
    """
    return [item["skill"] for item in _extract_skills(text)]


def extract_skills_from_jd(jd_text: str) -> List[str]:
    """Extract and deduplicate skills found in a Job Description.

    Args:
        jd_text: Plain-text content of a job description.

    Returns:
        Deduplicated list of skill names (display casing).
    """
    return [item["skill"] for item in _extract_skills(jd_text)]


# ---------------------------------------------------------------------------
# 3. ATS match scoring
# ---------------------------------------------------------------------------

def _categorize_skill(skill: str) -> str:
    """Return the taxonomy category for a skill, or 'Other'."""
    key = skill.lower()
    if key in _SKILL_LOOKUP:
        return _SKILL_LOOKUP[key][1]
    return "Other"


def compute_ats_match(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Dict[str, Any]:
    """Compute ATS-style match score between resume skills and JD skills.

    Args:
        resume_skills: Skills extracted from the candidate's resume.
        jd_skills: Skills extracted from the target job description.

    Returns:
        Dictionary containing:
        - ``match_score``: float 0–100 (percentage of JD skills matched).
        - ``matched_skills``: skills present in both.
        - ``missing_skills``: skills in JD but absent in resume.
        - ``extra_skills``: skills in resume but not required by JD.
        - ``category_breakdown``: per-category matched/total counts.
    """
    resume_set: Set[str] = {s.lower() for s in resume_skills}
    jd_set: Set[str] = {s.lower() for s in jd_skills}

    matched_lower = resume_set & jd_set
    missing_lower = jd_set - resume_set
    extra_lower = resume_set - jd_set

    # Map back to display casing from JD / resume lists
    _jd_display = {s.lower(): s for s in jd_skills}
    _res_display = {s.lower(): s for s in resume_skills}

    matched_display = sorted([_jd_display.get(s, s) for s in matched_lower])
    missing_display = sorted([_jd_display.get(s, s) for s in missing_lower])
    extra_display = sorted([_res_display.get(s, s) for s in extra_lower])

    # Overall match score
    if len(jd_set) == 0:
        match_score = 100.0 if len(resume_set) == 0 else 0.0
    else:
        match_score = round(len(matched_lower) / len(jd_set) * 100, 1)

    # Category breakdown
    category_breakdown: Dict[str, Dict[str, int]] = {}
    for skill_lower in jd_set:
        display = _jd_display.get(skill_lower, skill_lower)
        cat = _categorize_skill(display)
        if cat not in category_breakdown:
            category_breakdown[cat] = {"matched": 0, "total": 0}
        category_breakdown[cat]["total"] += 1
        if skill_lower in matched_lower:
            category_breakdown[cat]["matched"] += 1

    return {
        "match_score": match_score,
        "matched_skills": matched_display,
        "missing_skills": missing_display,
        "extra_skills": extra_display,
        "category_breakdown": category_breakdown,
    }


# ---------------------------------------------------------------------------
# 4. Resume improvement suggestions
# ---------------------------------------------------------------------------

def generate_resume_suggestions(
    missing_skills: List[str],
    profile: Dict[str, Any],
) -> List[str]:
    """Generate actionable resume improvement suggestions.

    Uses the student's academic profile and the gap between their resume
    and the target JD to produce prioritised, specific recommendations.

    Args:
        missing_skills: Skills present in the JD but absent from the resume.
        profile: Student profile dict (may contain specialisation, workex,
            degree_t, etest_p, mba_p, etc.).

    Returns:
        List of bullet-point suggestion strings.
    """
    suggestions: List[str] = []
    specialisation = str(profile.get("specialisation", "")).lower()
    workex = str(profile.get("workex", "No")).strip().lower()
    degree_t = str(profile.get("degree_t", "")).lower()

    if not missing_skills:
        suggestions.append(
            "✅ Your resume covers all skills listed in the JD — great match! "
            "Focus on quantifying your achievements with metrics and outcomes."
        )
        return suggestions

    # Categorize missing skills
    missing_by_cat: Dict[str, List[str]] = {}
    for sk in missing_skills:
        cat = _categorize_skill(sk)
        missing_by_cat.setdefault(cat, []).append(sk)

    # Technical / programming gaps
    tech_cats = {"Programming Languages", "Web & Frontend", "Databases", "Cloud & DevOps", "Tools & Platforms"}
    tech_missing = []
    for cat in tech_cats:
        tech_missing.extend(missing_by_cat.get(cat, []))

    if tech_missing:
        top = ", ".join(tech_missing[:5])
        suggestions.append(
            f"🛠️ **Add Technical Skills:** Your resume is missing: {top}. "
            "Consider adding relevant coursework projects, certifications, or "
            "self-directed projects that demonstrate these competencies."
        )

    # Data / ML gaps
    data_missing = missing_by_cat.get("Data & ML", [])
    if data_missing:
        top = ", ".join(data_missing[:4])
        suggestions.append(
            f"📊 **Data & Analytics Skills Gap:** The JD requires: {top}. "
            "Highlight any analytics coursework, Kaggle projects, or "
            "data-driven case studies in your resume."
        )

    # Business gaps
    biz_missing = missing_by_cat.get("Business & Finance", [])
    if biz_missing:
        top = ", ".join(biz_missing[:4])
        if "mkt&fin" in specialisation or "comm" in degree_t:
            suggestions.append(
                f"💼 **Leverage Your Finance/Business Background:** The JD lists: {top}. "
                "Ensure your resume explicitly mentions relevant coursework, "
                "case competitions, or projects in these domains."
            )
        else:
            suggestions.append(
                f"💼 **Business Skills Gap:** Consider acquiring exposure to: {top} "
                "through online certifications (e.g., Coursera, edX) or "
                "cross-functional projects."
            )

    # Soft skills gaps
    soft_missing = missing_by_cat.get("Soft Skills", [])
    if soft_missing:
        top = ", ".join(soft_missing[:4])
        suggestions.append(
            f"🤝 **Demonstrate Soft Skills:** The JD values: {top}. "
            "Weave these into your experience descriptions using the STAR "
            "format (Situation → Task → Action → Result) rather than just "
            "listing them."
        )

    # Certifications gaps
    cert_missing = missing_by_cat.get("Certifications & Standards", [])
    if cert_missing:
        top = ", ".join(cert_missing[:3])
        suggestions.append(
            f"🎓 **Certification Gaps:** The JD prefers: {top}. "
            "Plan to obtain at least one of these before the application deadline."
        )

    # Work experience
    if workex == "no":
        suggestions.append(
            "💼 **Internship / Project Experience:** Since you lack formal work "
            "experience, ensure your resume prominently features capstone projects, "
            "hackathons, pro-bono consulting, or freelance engagements."
        )

    # General formatting tip
    suggestions.append(
        "📝 **ATS Formatting Tip:** Use exact skill phrases from the JD "
        "(matching capitalisation and spelling) so automated screening "
        "systems can detect them. Avoid graphics-heavy resume templates."
    )

    return suggestions
