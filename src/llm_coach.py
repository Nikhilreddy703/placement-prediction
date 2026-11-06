"""LLM Integration Module for Genuine AI Resume Critique & Mock Interview Feedback.

Supports Google Gemini API via lightweight HTTP REST calls with zero additional dependencies.
Gracefully falls back to local rule-based evaluations when an API key is not configured.
"""
import json
import os
from typing import Any, Dict, Optional
import requests


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def get_api_key(custom_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Gemini API key from explicit argument, environment variable, or .env."""
    if custom_key and custom_key.strip():
        return custom_key.strip()
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def is_llm_available(api_key: Optional[str] = None) -> bool:
    """Check if a valid Gemini API key is configured."""
    return bool(get_api_key(api_key))


def call_gemini(prompt: str, api_key: Optional[str] = None, timeout: int = 15) -> Optional[str]:
    """Call Google Gemini REST endpoint with prompt and return generated text."""
    key = get_api_key(api_key)
    if not key:
        return None

    url = f"{GEMINI_API_URL}?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return None
    except Exception:
        return None


def analyze_resume_with_llm(
    resume_text: str,
    jd_text: str,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate in-depth semantic ATS critique using Gemini LLM."""
    if not is_llm_available(api_key):
        return None

    prompt = f"""You are a Senior Technical Recruiter and ATS Specialist.
Analyze the candidate's resume text against the provided Job Description (JD).

Resume Text:
\"\"\"{resume_text[:2500]}\"\"\"

Job Description:
\"\"\"{jd_text[:2500]}\"\"\"

Provide your analysis in EXACT JSON format with these exact keys:
{{
    "match_percentage": <integer 0 to 100>,
    "executive_summary": "<2-3 sentence overview of candidate suitability>",
    "critical_missing_skills": ["<skill1>", "<skill2>", ...],
    "strengths": ["<strength1>", "<strength2>", ...],
    "bullet_rewrite_suggestions": [
        {{
            "original_or_area": "<area in resume to improve>",
            "improved_bullet": "<quantified, impact-driven bullet point>"
        }}
    ],
    "next_steps": ["<action item 1>", "<action item 2>"]
}}
Return ONLY raw JSON, with no markdown code fences or backticks.
"""
    raw_response = call_gemini(prompt, api_key=api_key)
    if not raw_response:
        return None

    try:
        clean_text = raw_response.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```", 2)[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:].strip()
        return json.loads(clean_text)
    except Exception:
        return None


def evaluate_interview_with_llm(
    question: str,
    user_answer: str,
    criteria: str,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate candidate interview response using Gemini LLM against the STAR methodology."""
    if not is_llm_available(api_key):
        return None

    prompt = f"""You are an Expert Campus Recruitment Technical & Behavioral Interviewer.
Evaluate the candidate's answer to the following question against the STAR rubric and domain criteria.

Question:
\"\"\"{question}\"\"\"

Domain Evaluation Criteria:
\"\"\"{criteria}\"\"\"

Candidate Answer:
\"\"\"{user_answer}\"\"\"

Provide an assessment in EXACT JSON format with these exact keys:
{{
    "overall_score": <integer 0 to 100>,
    "star_ratings": {{
        "situation": "<Good / Missing / Weak>",
        "task": "<Good / Missing / Weak>",
        "action": "<Good / Missing / Weak>",
        "result": "<Good / Missing / Weak>"
    }},
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvement_areas": ["<area 1>", "<area 2>"],
    "suggested_sample_answer": "<A concise, high-scoring STAR answer to this exact question>",
    "interviewer_verdict": "<Hire / Leaning Hire / Leaning No / No Hire>"
}}
Return ONLY raw JSON, with no markdown code fences or backticks.
"""
    raw_response = call_gemini(prompt, api_key=api_key)
    if not raw_response:
        return None

    try:
        clean_text = raw_response.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```", 2)[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:].strip()
        return json.loads(clean_text)
    except Exception:
        return None

