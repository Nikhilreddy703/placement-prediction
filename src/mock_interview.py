"""Mock Interview Question Generator and Answer Evaluator.

Provides specialisation-aware interview question banks, keyword-based answer
scoring, and personalised interview preparation tips for placement candidates.
"""
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import THRESHOLD_HIGH_RISK, THRESHOLD_MODERATE_RISK


# ═══════════════════════════════════════════════════════════
# Question Bank — organised by specialisation and category
# ═══════════════════════════════════════════════════════════

_TECHNICAL_MKT_FIN: List[Dict[str, str]] = [
    {
        "question_text": (
            "Walk me through a Discounted Cash Flow (DCF) valuation model. "
            "How would you estimate the intrinsic value of a mid-cap FMCG company?"
        ),
        "evaluation_criteria": (
            "DCF free-cash-flow projection WACC terminal-value discount-rate "
            "sensitivity-analysis growth-rate enterprise-value equity-value"
        ),
        "sample_framework": (
            "1. Project free cash flows for 5-10 years using revenue growth assumptions.\n"
            "2. Estimate WACC (cost of equity via CAPM + cost of debt).\n"
            "3. Calculate terminal value (Gordon Growth or exit multiple).\n"
            "4. Discount all cash flows to present value.\n"
            "5. Subtract net debt to arrive at equity value; divide by shares outstanding."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "Explain the difference between enterprise value and equity value. "
            "When would you use EV/EBITDA versus P/E as a valuation multiple?"
        ),
        "evaluation_criteria": (
            "enterprise-value equity-value market-cap debt cash EV/EBITDA P/E "
            "capital-structure comparables leverage"
        ),
        "sample_framework": (
            "1. Define EV = Market Cap + Debt − Cash.\n"
            "2. Explain EV is capital-structure neutral; equity value is not.\n"
            "3. EV/EBITDA preferred for capital-intensive or levered firms.\n"
            "4. P/E simpler but distorted by leverage differences.\n"
            "5. Provide a sector example (e.g., telecom vs. IT services)."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "What is portfolio diversification? Explain the efficient frontier "
            "and how Modern Portfolio Theory guides asset allocation."
        ),
        "evaluation_criteria": (
            "diversification correlation efficient-frontier risk-return "
            "Markowitz standard-deviation Sharpe-ratio asset-allocation systematic-risk"
        ),
        "sample_framework": (
            "1. Define diversification and why uncorrelated assets reduce portfolio risk.\n"
            "2. Introduce the Markowitz efficient frontier as a set of optimal portfolios.\n"
            "3. Explain the role of the Sharpe ratio in selecting the tangency portfolio.\n"
            "4. Distinguish systematic vs. idiosyncratic risk.\n"
            "5. Give a practical example of a two-asset portfolio."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "How would you measure the ROI of a digital marketing campaign? "
            "Walk me through the key metrics and attribution models."
        ),
        "evaluation_criteria": (
            "ROI ROAS conversion-rate CPA CLV attribution first-touch last-touch "
            "multi-touch analytics funnel impressions CTR CAC"
        ),
        "sample_framework": (
            "1. Define ROI and ROAS in the context of digital spend.\n"
            "2. List funnel metrics: impressions → CTR → conversions → revenue.\n"
            "3. Explain first-touch, last-touch, and multi-touch attribution.\n"
            "4. Discuss CAC vs. CLV for long-term profitability.\n"
            "5. Mention tools (Google Analytics, UTM parameters, A/B testing)."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "A company is considering launching a premium credit card product. "
            "What financial metrics would you evaluate before the launch, and "
            "how would you structure the pricing?"
        ),
        "evaluation_criteria": (
            "break-even NPV interest-income interchange-fee annual-fee "
            "default-rate credit-risk pricing customer-segmentation "
            "competitive-analysis cost-of-funds"
        ),
        "sample_framework": (
            "1. Segment target customers (income, spending, credit history).\n"
            "2. Revenue streams: annual fee, interchange, interest income.\n"
            "3. Cost drivers: rewards, defaults, servicing, marketing.\n"
            "4. Build a simple NPV model with assumptions on take-up rate.\n"
            "5. Benchmark pricing against top-3 competitors."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "Explain the concept of brand equity. How can a company measure "
            "and improve it using quantitative marketing analytics?"
        ),
        "evaluation_criteria": (
            "brand-equity brand-awareness perceived-quality loyalty associations "
            "NPS conjoint-analysis brand-tracking survey sentiment premium-pricing"
        ),
        "sample_framework": (
            "1. Define brand equity using Keller's or Aaker's model.\n"
            "2. Measurement: brand-tracking surveys, NPS, conjoint analysis.\n"
            "3. Digital proxies: search volume, social sentiment, share of voice.\n"
            "4. Link brand equity to pricing power and customer lifetime value.\n"
            "5. Suggest improvement levers: consistency, experience, storytelling."
        ),
        "difficulty": "Medium",
    },
]

_TECHNICAL_MKT_HR: List[Dict[str, str]] = [
    {
        "question_text": (
            "How would you design a talent acquisition strategy for a "
            "high-growth start-up that needs to hire 200 engineers in 12 months?"
        ),
        "evaluation_criteria": (
            "talent-acquisition employer-branding sourcing pipeline "
            "ATS time-to-hire cost-per-hire campus-hiring referral "
            "diversity DEI screening assessment"
        ),
        "sample_framework": (
            "1. Define hiring plan: roles, skills, timeline, budget.\n"
            "2. Build employer brand: careers page, LinkedIn, Glassdoor.\n"
            "3. Multi-channel sourcing: referrals, job boards, campus, agencies.\n"
            "4. Structured screening: coding test → technical round → culture fit.\n"
            "5. Track KPIs: time-to-hire, offer-acceptance rate, quality-of-hire."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "What are the key drivers of employee engagement? How would you "
            "measure engagement and design an intervention for a team with "
            "low morale?"
        ),
        "evaluation_criteria": (
            "engagement survey eNPS motivation recognition autonomy growth "
            "Herzberg two-factor Maslow feedback pulse-survey retention"
        ),
        "sample_framework": (
            "1. Define engagement vs. satisfaction (discretionary effort).\n"
            "2. Key drivers: purpose, autonomy, mastery, recognition, growth.\n"
            "3. Measurement: annual survey, pulse surveys, eNPS.\n"
            "4. Diagnosis: identify low-scoring dimensions.\n"
            "5. Interventions: manager coaching, recognition programs, career paths."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "Explain the components of a total compensation and benefits "
            "strategy. How do you ensure internal equity while remaining "
            "competitive externally?"
        ),
        "evaluation_criteria": (
            "compensation base-pay variable-pay benefits equity "
            "benchmarking market-data job-evaluation pay-bands "
            "internal-equity Compa-ratio total-rewards"
        ),
        "sample_framework": (
            "1. Components: base salary, variable pay, benefits, equity/ESOP.\n"
            "2. Job evaluation methods: point-factor, market pricing.\n"
            "3. External benchmarking using salary survey data.\n"
            "4. Internal equity checks: Compa-ratio, pay-band analysis.\n"
            "5. Communication: total rewards statements, transparency policy."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "Describe how you would implement a change management initiative "
            "for a company migrating from on-premise tools to a fully "
            "cloud-based HR platform."
        ),
        "evaluation_criteria": (
            "change-management Kotter ADKAR stakeholder communication "
            "training resistance adoption champion pilot rollout feedback"
        ),
        "sample_framework": (
            "1. Create urgency and a compelling vision (Kotter Step 1-3).\n"
            "2. Identify stakeholders and change champions.\n"
            "3. ADKAR: Awareness → Desire → Knowledge → Ability → Reinforcement.\n"
            "4. Phased rollout: pilot → department → company-wide.\n"
            "5. Measure adoption metrics and address resistance continuously."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "What is HR Analytics? Walk me through a project where you would "
            "use people data to reduce employee attrition."
        ),
        "evaluation_criteria": (
            "HR-analytics people-analytics attrition logistic-regression "
            "survival-analysis feature-importance exit-interview "
            "retention predictive-model dashboard"
        ),
        "sample_framework": (
            "1. Define the business problem: high attrition cost.\n"
            "2. Data collection: HRIS demographics, tenure, performance, survey.\n"
            "3. Exploratory analysis: attrition rates by department, tenure band.\n"
            "4. Predictive model: logistic regression or survival analysis.\n"
            "5. Action plan: targeted retention interventions for high-risk groups."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "How would you design an effective employee onboarding programme "
            "that improves 90-day retention by 20 %?"
        ),
        "evaluation_criteria": (
            "onboarding orientation buddy-system 30-60-90 cultural-integration "
            "training retention engagement pre-boarding checklist feedback"
        ),
        "sample_framework": (
            "1. Pre-boarding: welcome kit, account setup, manager introduction.\n"
            "2. Day 1: culture immersion, values session, team lunch.\n"
            "3. 30-60-90 day plan with clear milestones and check-ins.\n"
            "4. Assign a buddy or mentor for informal guidance.\n"
            "5. Collect feedback at 30 and 90 days; iterate the programme."
        ),
        "difficulty": "Easy",
    },
]

_BEHAVIORAL_QUESTIONS: List[Dict[str, str]] = [
    {
        "question_text": (
            "Tell me about a time you led a team through a challenging project. "
            "What was the outcome?"
        ),
        "evaluation_criteria": (
            "leadership delegation communication challenge outcome "
            "accountability team-building initiative results"
        ),
        "sample_framework": (
            "STAR Method:\n"
            "  Situation — Describe the project and why it was challenging.\n"
            "  Task — Your specific role and responsibility as the leader.\n"
            "  Action — Steps you took: delegation, motivation, conflict resolution.\n"
            "  Result — Quantifiable outcome and what you learned."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "Describe a situation where you had a significant disagreement or "
            "conflict with a team member. How did you handle it?"
        ),
        "evaluation_criteria": (
            "conflict-resolution empathy active-listening compromise "
            "professionalism emotional-intelligence perspective feedback"
        ),
        "sample_framework": (
            "STAR Method:\n"
            "  Situation — Context of the disagreement.\n"
            "  Task — What needed to be resolved and stakes involved.\n"
            "  Action — Steps: active listening, finding common ground, proposing a solution.\n"
            "  Result — Resolution achieved and relationship maintained."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "Tell me about a time you failed at something important. "
            "What did you learn from the experience?"
        ),
        "evaluation_criteria": (
            "failure self-awareness growth-mindset resilience learning "
            "accountability reflection improvement honesty"
        ),
        "sample_framework": (
            "STAR Method:\n"
            "  Situation — What the goal was and why it mattered.\n"
            "  Task — Your responsibility in the outcome.\n"
            "  Action — What went wrong and your immediate response.\n"
            "  Result — Lessons learned and how you applied them subsequently."
        ),
        "difficulty": "Medium",
    },
    {
        "question_text": (
            "Describe a time when you had to deliver results under extreme "
            "time pressure. How did you prioritise and manage stress?"
        ),
        "evaluation_criteria": (
            "time-management prioritisation stress-management deadline "
            "focus efficiency delegation planning results"
        ),
        "sample_framework": (
            "STAR Method:\n"
            "  Situation — Tight deadline and high stakes.\n"
            "  Task — Deliverables and constraints.\n"
            "  Action — Prioritisation, breaking tasks down, seeking help.\n"
            "  Result — Whether you met the deadline and quality of the output."
        ),
        "difficulty": "Medium",
    },
]

_CASE_STUDIES: List[Dict[str, str]] = [
    {
        "question_text": (
            "Case Study: A mid-size Indian FMCG company wants to enter the "
            "premium organic food segment. The CEO has asked you to evaluate "
            "whether this is a viable market entry opportunity. Present your "
            "analysis framework and recommendation."
        ),
        "evaluation_criteria": (
            "market-entry market-sizing TAM SAM SOM competitive-landscape "
            "Porter five-forces distribution pricing SWOT investment "
            "break-even go-to-market revenue-model customer-segment"
        ),
        "sample_framework": (
            "1. Market sizing: TAM → SAM → SOM for premium organic foods in India.\n"
            "2. Industry analysis: Porter's Five Forces, growth trends.\n"
            "3. Competitive landscape: incumbents, private labels, D2C brands.\n"
            "4. Internal capabilities: existing supply chain, brand equity, distribution.\n"
            "5. Financial viability: investment required, break-even timeline.\n"
            "6. Recommendation: enter / don't enter / phased pilot."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "Case Study: A tech company's operating costs have grown 30 % "
            "year-over-year while revenue grew only 12 %. The CFO wants a "
            "cost reduction plan that preserves innovation capacity. "
            "What would you recommend?"
        ),
        "evaluation_criteria": (
            "cost-reduction OpEx CapEx headcount cloud-optimization "
            "vendor-consolidation process-automation KPI ROI "
            "zero-based-budgeting prioritisation trade-offs"
        ),
        "sample_framework": (
            "1. Diagnose cost drivers: categorize OpEx (people, infra, vendors).\n"
            "2. Benchmark against industry peers.\n"
            "3. Quick wins: vendor renegotiation, license rationalization.\n"
            "4. Medium-term: process automation, cloud cost optimization.\n"
            "5. Protect innovation: ring-fence R&D budget, cut non-core spending.\n"
            "6. Implementation roadmap with quarterly savings targets."
        ),
        "difficulty": "Hard",
    },
    {
        "question_text": (
            "Case Study: An e-commerce marketplace is experiencing "
            "plateauing monthly active users despite increasing marketing spend. "
            "Design a growth strategy to achieve 40 % user growth in 12 months."
        ),
        "evaluation_criteria": (
            "growth-strategy retention acquisition activation referral "
            "AARRR funnel cohort-analysis LTV CAC product-market-fit "
            "network-effects virality engagement"
        ),
        "sample_framework": (
            "1. Funnel analysis (AARRR): where are users dropping off?\n"
            "2. Cohort analysis: retention curves by acquisition channel.\n"
            "3. Acquisition levers: referral programme, partnerships, SEO.\n"
            "4. Activation & engagement: onboarding flow, push notifications.\n"
            "5. Retention: loyalty programme, personalisation, community.\n"
            "6. Financial model: target CAC/LTV ratio; reallocate spend."
        ),
        "difficulty": "Hard",
    },
]


# ═══════════════════════════════════════════════════════════
# 2.1 Question Generator
# ═══════════════════════════════════════════════════════════

def generate_interview_questions(
    specialisation: str,
    profile: Dict[str, Any],
    difficulty: str = "Medium",
) -> Dict[str, Any]:
    """Generate a curated set of mock interview questions for a student.

    Based on the student's MBA specialisation and profile, this returns:
        - 3 technical / domain questions
        - 2 behavioral / HR questions
        - 1 case study prompt

    Each question includes evaluation criteria, a sample answer framework,
    and difficulty rating.

    Args:
        specialisation: ``"Mkt&Fin"`` or ``"Mkt&HR"``.
        profile: Student profile dict (used to adjust question selection
                 based on work experience and academic scores).
        difficulty: ``"Easy"``, ``"Medium"``, or ``"Hard"``.

    Returns:
        Dictionary with:
            ``specialisation`` — echo of the input specialisation
            ``difficulty``     — requested difficulty level
            ``questions``      — list of question dicts (6 items)
    """
    diff_lower = difficulty.strip().lower()

    # Select the domain question bank
    if specialisation == "Mkt&HR":
        tech_bank = list(_TECHNICAL_MKT_HR)
    else:
        tech_bank = list(_TECHNICAL_MKT_FIN)

    # Filter by difficulty preference (fallback to all if insufficient)
    filtered_tech = [q for q in tech_bank if q.get("difficulty", "Medium").lower() == diff_lower]
    if len(filtered_tech) < 3:
        filtered_tech = tech_bank  # use entire bank

    # Adaptive selection: students without work experience get more foundational questions
    has_workex = str(profile.get("workex", "No")).strip().lower() == "yes"
    if not has_workex:
        # Prefer easier / medium questions
        filtered_tech.sort(key=lambda q: {"easy": 0, "medium": 1, "hard": 2}.get(
            q.get("difficulty", "Medium").lower(), 1
        ))

    # Pick 3 technical questions
    selected_tech = filtered_tech[:3]
    tech_questions = [
        {
            "question_text": q["question_text"],
            "category": "Technical",
            "difficulty": q.get("difficulty", difficulty),
            "evaluation_criteria": q["evaluation_criteria"],
            "sample_framework": q["sample_framework"],
        }
        for q in selected_tech
    ]

    # Pick 2 behavioral questions
    behavioral_bank = list(_BEHAVIORAL_QUESTIONS)
    filtered_beh = [q for q in behavioral_bank if q.get("difficulty", "Medium").lower() == diff_lower]
    if len(filtered_beh) < 2:
        filtered_beh = behavioral_bank
    selected_beh = filtered_beh[:2]
    beh_questions = [
        {
            "question_text": q["question_text"],
            "category": "Behavioral",
            "difficulty": q.get("difficulty", difficulty),
            "evaluation_criteria": q["evaluation_criteria"],
            "sample_framework": q["sample_framework"],
        }
        for q in selected_beh
    ]

    # Pick 1 case study
    case_bank = list(_CASE_STUDIES)
    filtered_case = [q for q in case_bank if q.get("difficulty", "Hard").lower() == diff_lower]
    if len(filtered_case) < 1:
        filtered_case = case_bank
    selected_case = filtered_case[:1]
    case_questions = [
        {
            "question_text": q["question_text"],
            "category": "Case Study",
            "difficulty": q.get("difficulty", difficulty),
            "evaluation_criteria": q["evaluation_criteria"],
            "sample_framework": q["sample_framework"],
        }
        for q in selected_case
    ]

    all_questions = tech_questions + beh_questions + case_questions

    return {
        "specialisation": specialisation,
        "difficulty": difficulty,
        "questions": all_questions,
    }


# ═══════════════════════════════════════════════════════════
# 2.2 Answer Quality Evaluator
# ═══════════════════════════════════════════════════════════

def _tokenise(text: str) -> List[str]:
    """Lowercase tokenisation with basic punctuation removal."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def _check_star_elements(text: str) -> Dict[str, bool]:
    """Detect STAR method elements in the answer text."""
    text_lower = text.lower()
    elements = {
        "Situation": any(kw in text_lower for kw in [
            "situation", "context", "background", "scenario", "setting",
            "at that time", "when i was", "during my",
        ]),
        "Task": any(kw in text_lower for kw in [
            "task", "goal", "objective", "responsibility", "charged with",
            "my role", "i was responsible", "assigned to", "challenge was",
        ]),
        "Action": any(kw in text_lower for kw in [
            "action", "i did", "i decided", "i implemented", "steps i took",
            "approach", "strategy", "initiative", "i led", "i organised",
            "i organized", "i built", "i developed", "i created",
        ]),
        "Result": any(kw in text_lower for kw in [
            "result", "outcome", "impact", "achieved", "increased",
            "decreased", "improved", "delivered", "led to", "consequently",
            "as a result", "which resulted", "success", "revenue", "savings",
        ]),
    }
    return elements


def evaluate_answer_quality(
    answer_text: str,
    question: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a candidate's answer using keyword matching and structure analysis.

    Scoring dimensions (no LLM — fully rule-based):
        - **Relevance** (0-100): keyword overlap with evaluation criteria.
        - **Structure** (0-100): checks for STAR elements (Situation, Task,
          Action, Result).
        - **Depth** (0-100): word count, specificity (numbers, metrics,
          proper nouns), and detail indicators.
        - **Overall** : weighted average (relevance 40 %, structure 30 %,
          depth 30 %).

    Args:
        answer_text: The candidate's written answer.
        question: The question dict (must contain ``evaluation_criteria``).

    Returns:
        Dictionary with ``relevance_score``, ``structure_score``,
        ``depth_score``, ``overall_score``, and ``feedback`` list.
    """
    answer_text = (answer_text or "").strip()
    if not answer_text:
        return {
            "relevance_score": 0,
            "structure_score": 0,
            "depth_score": 0,
            "overall_score": 0,
            "feedback": ["Your answer is empty. Please provide a substantive response."],
        }

    answer_tokens = set(_tokenise(answer_text))
    criteria_text = question.get("evaluation_criteria", "")
    criteria_tokens = set(_tokenise(criteria_text))

    # ── Relevance Score ──
    if criteria_tokens:
        overlap = answer_tokens & criteria_tokens
        relevance_ratio = len(overlap) / len(criteria_tokens)
    else:
        relevance_ratio = 0.5  # neutral if no criteria

    relevance_score = min(100, int(relevance_ratio * 100 * 1.5))  # scale up slightly

    # ── Structure Score (STAR) ──
    star = _check_star_elements(answer_text)
    star_count = sum(star.values())
    structure_score = int((star_count / 4) * 100)

    # Bonus for clear paragraph structure (multiple sentences / paragraphs)
    sentence_count = len(re.split(r'[.!?]+', answer_text))
    if sentence_count >= 5:
        structure_score = min(100, structure_score + 10)

    # ── Depth Score ──
    word_count = len(answer_text.split())

    # Word count component (0-40 points: <30 words = 0, 30-60 = 15, 60-120 = 25, 120-200 = 35, 200+ = 40)
    if word_count >= 200:
        wc_points = 40
    elif word_count >= 120:
        wc_points = 35
    elif word_count >= 60:
        wc_points = 25
    elif word_count >= 30:
        wc_points = 15
    else:
        wc_points = 5

    # Numbers / metrics mentioned (0-30 points)
    numbers_found = re.findall(r'\d+[\d,.%]*', answer_text)
    metric_points = min(30, len(numbers_found) * 8)

    # Specificity indicators (0-30 points)
    specificity_kws = [
        "for example", "specifically", "in particular", "such as",
        "we achieved", "the result was", "which led to", "measured by",
        "data showed", "according to", "analysis", "framework",
    ]
    specificity_hits = sum(1 for kw in specificity_kws if kw in answer_text.lower())
    specificity_points = min(30, specificity_hits * 8)

    depth_score = min(100, wc_points + metric_points + specificity_points)

    # ── Overall Score (weighted) ──
    overall_score = int(
        relevance_score * 0.40
        + structure_score * 0.30
        + depth_score * 0.30
    )

    # ── Feedback Generation ──
    feedback: List[str] = []

    if relevance_score < 40:
        feedback.append(
            "🔑 **Relevance:** Your answer doesn't address enough key concepts. "
            "Review the evaluation criteria and incorporate more domain-specific "
            "terminology and frameworks."
        )
    elif relevance_score < 70:
        feedback.append(
            "🔑 **Relevance:** Good start, but try to cover more of the core "
            "concepts expected for this question."
        )

    missing_star = [elem for elem, found in star.items() if not found]
    if missing_star:
        feedback.append(
            f"📋 **Structure (STAR):** Your answer is missing the following "
            f"STAR elements: {', '.join(missing_star)}. Use the STAR method "
            f"(Situation → Task → Action → Result) for behavioral answers."
        )

    if word_count < 50:
        feedback.append(
            "📝 **Depth:** Your answer is too brief. Aim for at least 80-120 "
            "words with concrete details, examples, and measurable outcomes."
        )
    elif word_count < 100:
        feedback.append(
            "📝 **Depth:** Consider expanding with specific examples, metrics, "
            "or quantifiable results to strengthen your answer."
        )

    if not numbers_found:
        feedback.append(
            "📊 **Quantify Impact:** Include specific numbers, percentages, or "
            "metrics (e.g., 'reduced costs by 15%', 'managed a team of 8') to "
            "make your answer more compelling."
        )

    if overall_score >= 75 and not feedback:
        feedback.append(
            "✅ **Strong answer!** You've demonstrated good relevance, structure, "
            "and depth. Consider practising delivery and timing for the actual interview."
        )

    return {
        "relevance_score": relevance_score,
        "structure_score": structure_score,
        "depth_score": depth_score,
        "overall_score": overall_score,
        "feedback": feedback,
    }


# ═══════════════════════════════════════════════════════════
# 2.3 Personalised Interview Tips
# ═══════════════════════════════════════════════════════════

def get_interview_tips(
    profile: Dict[str, Any],
    prob: float,
) -> List[str]:
    """Generate personalised interview preparation tips based on the student's profile.

    The tips are tailored to the student's strengths, weaknesses, specialisation,
    and overall placement probability.

    Args:
        profile: Student profile dict with academic scores, specialisation,
                 and work experience fields.
        prob: Placement probability (0.0 to 1.0).

    Returns:
        List of actionable tip strings.
    """
    tips: List[str] = []

    spec = str(profile.get("specialisation", "Mkt&Fin"))
    workex = str(profile.get("workex", "No")).strip().lower()
    etest_p = float(profile.get("etest_p", 70.0))
    degree_p = float(profile.get("degree_p", 68.0))
    mba_p = float(profile.get("mba_p", 65.0))
    ssc_p = float(profile.get("ssc_p", 70.0))
    hsc_p = float(profile.get("hsc_p", 70.0))

    # ── General tips based on probability ──
    if prob < THRESHOLD_HIGH_RISK:
        tips.append(
            "🎯 **Intensive Preparation:** With your current profile, focus on "
            "building confidence through daily mock interview practice (minimum "
            "2 questions per day for the next 4 weeks)."
        )
        tips.append(
            "📚 **Knowledge Gaps:** Dedicate 1 hour daily to core domain concepts. "
            "Use structured resources like CFA Institute materials (Finance) or "
            "SHRM guides (HR)."
        )
    elif prob < THRESHOLD_MODERATE_RISK:
        tips.append(
            "⚡ **Sharpen Your Edge:** You're in the borderline zone. A strong "
            "interview performance can tip the scales. Practice at least 3 mock "
            "interviews per week."
        )
    else:
        tips.append(
            "💪 **Leverage Your Strengths:** Your academic profile is strong. "
            "Focus on demonstrating thought leadership and strategic thinking "
            "in interviews."
        )

    # ── Specialisation-specific tips ──
    if spec == "Mkt&Fin":
        tips.append(
            "📈 **Finance Prep:** Be ready to discuss current market trends, "
            "recent IPOs, RBI monetary policy, and global macro events. Read "
            "the Economic Times / Mint daily."
        )
        tips.append(
            "🧮 **Quant Skills:** Practice mental math, financial ratios, and "
            "basic valuation problems (DCF, comparable analysis). Interviewers "
            "often test numerical agility."
        )
    else:
        tips.append(
            "👥 **HR Domain:** Stay updated on employment law changes, hybrid "
            "work trends, and major HR technology platforms (Workday, SuccessFactors, "
            "Darwinbox)."
        )
        tips.append(
            "🧠 **People Analytics:** Be prepared to discuss how data can drive "
            "HR decisions — attrition prediction, engagement surveys, workforce "
            "planning metrics."
        )

    # ── Work experience tips ──
    if workex == "yes":
        tips.append(
            "💼 **Leverage Your Experience:** Prepare 3-4 detailed stories from "
            "your work experience using the STAR method. Quantify your impact "
            "(revenue generated, costs saved, team size managed)."
        )
    else:
        tips.append(
            "🎓 **Academic Projects as Experience:** Since you don't have formal "
            "work experience, prepare to discuss academic projects, case "
            "competitions, club activities, and internships as if they were "
            "professional assignments. Use STAR format."
        )

    # ── Score-based tips ──
    if etest_p < 65:
        tips.append(
            "🧪 **Aptitude Readiness:** Your employability test score suggests "
            "room for improvement. Many companies use aptitude screening before "
            "interviews — practice quant, verbal, and logical reasoning daily."
        )

    if degree_p < 60:
        tips.append(
            "📋 **Address Academic Gaps:** If asked about your undergraduate "
            "performance, be honest and pivot to growth: highlight how your "
            "MBA scores, certifications, or projects demonstrate improvement."
        )

    if mba_p >= 75:
        tips.append(
            "🏆 **Highlight Academic Excellence:** Your strong MBA scores are "
            "a differentiator. Mention specific courses or projects that relate "
            "to the role you're interviewing for."
        )

    # ── General best practices ──
    tips.append(
        "🔄 **Practice Out Loud:** Rehearse your answers aloud, not just in "
        "your head. Record yourself and review for filler words, pacing, and "
        "confidence."
    )
    tips.append(
        "🏢 **Research the Company:** Before every interview, spend 30 minutes "
        "reading the company's annual report, recent news, and Glassdoor "
        "reviews. Tailor your answers to their business challenges."
    )

    return tips
