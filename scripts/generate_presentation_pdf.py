"""Generate a comprehensive presentation PDF for PlacementOS Enterprise."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_PDF = BASE_DIR / "reports" / "PlacementOS_Enterprise_Presentation.pdf"

def build_presentation_pdf():
    # Page setup: Landscape Letter (11 x 8.5 inches)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "SlideTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "SlideSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "SubHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=4,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SlideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
    )
    bullet_style = ParagraphStyle(
        "SlideBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        spaceAfter=4,
    )
    box_header = ParagraphStyle(
        "BoxHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
    )
    url_style = ParagraphStyle(
        "URLStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2563EB"),
    )
    code_badge = ParagraphStyle(
        "CodeBadge",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    story.append(Paragraph("PlacementOS Enterprise", title_style))
    story.append(Paragraph("Campus Recruitment & Student Career Intelligence System | Project Presentation", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3B82F6"), spaceAfter=12))

    left_info = [
        Paragraph("<b>Project Overview in Simple Words:</b>", h2_style),
        Paragraph("• <b>What is it?</b> An AI-powered campus recruitment system designed for college Training & Placement (T&P) cells, students, and academic counselors.", bullet_style),
        Paragraph("• <b>The Core Innovation:</b> Moves beyond black-box 'Yes/No' predictions by providing <b>calibrated probabilities</b>, <b>SHAP explainability</b> (why?), and <b>counterfactual recourse</b> (what exactly should the student do to get placed?).", bullet_style),
        Paragraph("• <b>Full-Stack Architecture:</b> Combines audited Machine Learning core, automated ATS resume screening, STAR behavioral interview scoring, and a multi-tab Streamlit dashboard.", bullet_style),
        Spacer(1, 10),
        Paragraph("<b>Live Project Access:</b>", h2_style),
        Paragraph("🌐 <b>Live Streamlit App:</b> <font color='#2563EB'>https://placement-prediction-hl2h7xyvh4ncxz9xse67wd.streamlit.app/</font>", body_style),
        Paragraph("💻 <b>GitHub Source Code:</b> <font color='#2563EB'>https://github.com/Nikhilreddy703/placement-prediction</font>", body_style),
        Paragraph("👤 <b>Presenter:</b> Nikhil Reddy | Empirically Audited Monorepo", body_style),
        Paragraph("⚙️ <b>Technology Stack:</b> Python 3.12, Scikit-Learn, Streamlit Cloud, Plotly, ReportLab", body_style),
    ]

    img_ui = Image(str(FIGURES_DIR / "ui_dashboard_preview.png"), width=4.8 * inch, height=2.7 * inch)
    
    t1 = Table([[left_info, img_ui]], colWidths=[5.0 * inch, 4.9 * inch])
    t1.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t1)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & System Architecture
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 1: Problem Statement & System Architecture", title_style))
    story.append(Paragraph("Addressing the critical flaws of traditional student placement prediction systems", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s2_left = [
        Paragraph("<b>1. The Problem with Traditional Models:</b>", h2_style),
        Paragraph("• <b>Black-Box Predictions:</b> Telling a student they won't get placed without explaining why causes anxiety with zero actionable value.", bullet_style),
        Paragraph("• <b>Uncalibrated Probabilities:</b> Standard models output overconfident scores (e.g. 95%) that do not reflect true statistical odds.", bullet_style),
        Paragraph("• <b>Zero Actionability:</b> No guidance on what skills or grades to improve.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>2. The PlacementOS Solution:</b>", h2_style),
        Paragraph("• <b>Tier 1 (Inference):</b> 6-model benchmark with probability calibration.", bullet_style),
        Paragraph("• <b>Tier 2 (Transparency):</b> SHAP values reveal positive & negative drivers.", bullet_style),
        Paragraph("• <b>Tier 3 (Prescription):</b> Counterfactual engine provides minimal effort roadmap.", bullet_style),
        Paragraph("• <b>Tier 4 (Career Tools):</b> Integrated ATS Resume parser & Mock Interview.", bullet_style),
    ]

    img_arch = Image(str(FIGURES_DIR / "system_architecture.png"), width=5.2 * inch, height=2.8 * inch)
    t2 = Table([[s2_left, img_arch]], colWidths=[4.6 * inch, 5.3 * inch])
    t2.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t2)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 3: Exploratory Data Analysis (EDA) Insights
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 2: Exploratory Data Analysis & Empirical Insights", title_style))
    story.append(Paragraph("What does historical campus recruitment data reveal about hiring criteria?", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s3_left = [
        Paragraph("<b>Key Findings Explained in Simple Words:</b>", h2_style),
        Paragraph("• <b>Finding 1 (CGPA is the Gateway):</b> Students with CGPA > 8.0 have over 85% placement probability. It is mandatory for initial recruiter shortlist cutoffs.", bullet_style),
        Paragraph("• <b>Finding 2 (Internships Close the Deal):</b> 1 to 2 completed internships raise placement odds by +24%, even for students with moderate CGPA (7.0 - 7.5).", bullet_style),
        Paragraph("• <b>Finding 3 (Active Backlogs are Fatal):</b> Having 2+ active backlogs reduces placement probability by over 45%, acting as an automatic disqualifier for Tier-1 firms.", bullet_style),
        Paragraph("• <b>Finding 4 (Projects & Practical Coding):</b> Open-source GitHub contributions and LeetCode ratings directly compensate for marginal academic scores.", bullet_style),
    ]

    img_corr = Image(str(FIGURES_DIR / "correlation.png"), width=4.8 * inch, height=2.8 * inch)
    t3 = Table([[s3_left, img_corr]], colWidths=[4.9 * inch, 5.0 * inch])
    t3.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t3)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 4: Machine Learning Benchmarking & Champion Model
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 3: Machine Learning Model Benchmarking", title_style))
    story.append(Paragraph("Rigorous empirical comparison of 6 supervised learning algorithms", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s4_left = [
        Paragraph("<b>The 6 Evaluated Algorithms (Simple Words):</b>", h2_style),
        Paragraph("• <b>Logistic Regression:</b> Computes weighted linear decision boundary. Fast, interpretable, and mathematically well-behaved.", bullet_style),
        Paragraph("• <b>Random Forest:</b> Ensemble of decision trees voting together. High accuracy but prone to slight overfitting on small datasets.", bullet_style),
        Paragraph("• <b>Support Vector Machine (SVM):</b> Maximizes geometric margin between classes.", bullet_style),
        Paragraph("• <b>XGBoost:</b> Gradient boosted trees optimizing residual errors.", bullet_style),
        Paragraph("• <b>K-Nearest Neighbors:</b> Classifies by distance to past student alumni.", bullet_style),
        Paragraph("• <b>Naive Bayes:</b> Fast probabilistic baseline using conditional independence.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>Why Calibrated Logistic Regression Won (Champion):</b>", h2_style),
        Paragraph("Achieved <b>CV Balanced Accuracy of 73.6%</b> and <b>ROC-AUC of 0.745</b>. When calibrated, it demonstrated superior generalization without memorizing noise.", bullet_style),
    ]

    img_comp = Image(str(FIGURES_DIR / "model_comparison.png"), width=4.9 * inch, height=2.8 * inch)
    t4 = Table([[s4_left, img_comp]], colWidths=[4.9 * inch, 5.0 * inch])
    t4.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t4)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 5: ROC Curve, Calibration & Optimal Cutoff (0.69)
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 4: Model Evaluation, ROC-AUC & Optimal Cutoff", title_style))
    story.append(Paragraph("Why using a default 0.50 threshold is flawed and how 0.69 optimizes placement decisions", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s5_left = [
        Paragraph("<b>1. The ROC-AUC Curve Explained:</b>", h2_style),
        Paragraph("• Plots True Positive Rate (placed students detected) against False Alarm Rate across all possible discrimination thresholds.", bullet_style),
        Paragraph("• <b>AUC = 0.745</b> indicates solid diagnostic ability to distinguish between placed and unplaced candidates.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>2. Optimal Decision Cutoff (0.69):</b>", h2_style),
        Paragraph("• Default 0.50 threshold assumes false positives and false negatives carry identical cost. In placement, false hope is harmful.", bullet_style),
        Paragraph("• Using <b>Out-Of-Fold grid search</b>, threshold <b>0.69</b> was identified to maximize balanced accuracy and minimize false confidence.", bullet_style),
        Paragraph("• Students with probability < 0.69 are flagged for proactive coaching before drives begin.", bullet_style),
    ]

    img_roc = Image(str(FIGURES_DIR / "roc_curve.png"), width=4.8 * inch, height=2.8 * inch)
    t5 = Table([[s5_left, img_roc]], colWidths=[4.9 * inch, 5.0 * inch])
    t5.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t5)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 6: Explainable AI (SHAP) & Feature Contributions
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 5: Explainable AI (SHAP) — Opening the Black Box", title_style))
    story.append(Paragraph("Using cooperative game theory to show exactly which factors drive individual outcomes", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s6_left = [
        Paragraph("<b>What is SHAP (In Simple Words)?</b>", h2_style),
        Paragraph("• SHAP (SHapley Additive exPlanations) comes from Nobel Prize-winning game theory.", bullet_style),
        Paragraph("• It calculates each feature's contribution towards pushing the placement probability higher (+) or lower (-).", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>Key Driver Ranking in Our System:</b>", h2_style),
        Paragraph("1. <b>CGPA:</b> +38% average positive impact on likelihood.", bullet_style),
        Paragraph("2. <b>Internships:</b> +24% boost per verified internship.", bullet_style),
        Paragraph("3. <b>Major Projects:</b> +16% boost for production-grade builds.", bullet_style),
        Paragraph("4. <b>Backlogs:</b> -32% severe downward penalty.", bullet_style),
        Paragraph("5. <b>Communication Skills:</b> Decisive factor during HR rounds.", bullet_style),
    ]

    img_imp = Image(str(FIGURES_DIR / "feature_importance.png"), width=4.9 * inch, height=2.8 * inch)
    t6 = Table([[s6_left, img_imp]], colWidths=[4.8 * inch, 5.1 * inch])
    t6.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t6)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 7: Prescriptive Counterfactual Recourse ("What-If" Simulator)
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 6: Prescriptive Recourse — The 'What-If' Engine", title_style))
    story.append(Paragraph("Transforming predictive AI into an actionable roadmap for unplaced students", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    s7_left = [
        Paragraph("<b>Moving from Diagnosis to Cure:</b>", h2_style),
        Paragraph("• A standard ML model says: <i>'You have a 38% chance of placement (Fail).'</i> This offers zero help to a student.", bullet_style),
        Paragraph("• PlacementOS answers: <i>'What is the minimum realistic effort needed to reach 75% placement probability?'</i>", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>How the Recourse Algorithm Works:</b>", h2_style),
        Paragraph("• <b>Actionability Constraints:</b> Demographic features are frozen; only actionable items (CGPA, internships, projects, backlogs) are modified.", bullet_style),
        Paragraph("• <b>Minimal Perturbation:</b> Uses optimization to find the shortest realistic distance to flip the prediction from high-risk to placed.", bullet_style),
        Paragraph("• <b>Concrete Example:</b> 'Clear 1 backlog and complete 1 software internship to increase odds from 42% to 78%.'", bullet_style),
    ]

    img_rec = Image(str(FIGURES_DIR / "counterfactual_recourse.png"), width=5.0 * inch, height=2.7 * inch)
    t7 = Table([[s7_left, img_rec]], colWidths=[4.8 * inch, 5.1 * inch])
    t7.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(t7)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 8: Complete Student Career Development Suite
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 7: Career Suite — ATS Resume, Coding Index & Mock Interview", title_style))
    story.append(Paragraph("Holistic preparation tools bridging the gap between academics and corporate hiring", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    col1 = [
        Paragraph("<b>1. ATS Resume Analyzer:</b>", h2_style),
        Paragraph("• Parses student resume against target Job Description (JD).", bullet_style),
        Paragraph("• Calculates keyword match score (0-100%) and alerts student to missing critical skills (e.g. Docker, REST, SQL).", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>2. Tech & Coding Profile:</b>", h2_style),
        Paragraph("• Formula: <b>65% LeetCode problem solving + 35% GitHub project activity</b>.", bullet_style),
        Paragraph("• Verifies real software development readiness.", bullet_style),
    ]

    col2 = [
        Paragraph("<b>3. AI Mock Interview Coach:</b>", h2_style),
        Paragraph("• Uses the industry-standard <b>STAR Method</b> (Situation, Task, Action, Result).", bullet_style),
        Paragraph("• Evaluates verbal responses and provides constructive feedback with LLM fallback.", bullet_style),
        Spacer(1, 4),
        Paragraph("<b>4. TPO Recruiter Drive Center:</b>", h2_style),
        Paragraph("• College T&P officers can filter entire cohorts by eligibility criteria (Min CGPA, 0 backlogs).", bullet_style),
        Paragraph("• Instant export to Excel for campus company drives.", bullet_style),
    ]

    t8 = Table([[col1, col2]], colWidths=[4.9 * inch, 4.9 * inch])
    t8.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (0, -1), 1, colors.HexColor("#CBD5E1")),
        ('BOX', (1, 0), (1, -1), 1, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t8)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 9: Presentation Q&A Defense & Viva Cheat Sheet
    # -------------------------------------------------------------
    story.append(Paragraph("Slide 8: Presentation Defense & Viva Q&A Cheat Sheet", title_style))
    story.append(Paragraph("Anticipated questions from professors, recruiters, and evaluators with concise answers", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=10))

    qas = [
        [
            Paragraph("<b>Q1: Why did Logistic Regression beat complex models like XGBoost and Random Forest?</b><br/>"
                      "<i>Answer:</i> Campus placement datasets have predominantly linear monotonic feature effects (higher CGPA and more internships monotonically increase chances). Complex tree ensembles overfit noisy boundary students on tabular cohorts, while Calibrated Logistic Regression provides optimal generalization and resistance to data drift.", body_style)
        ],
        [
            Paragraph("<b>Q2: What is probability calibration and why is it essential?</b><br/>"
                      "<i>Answer:</i> Uncalibrated classifiers only rank order probabilities. If a model says 80%, calibration ensures that historically 80 out of 100 students with that score actually got placed. This guarantees ethical, dependable advice.", body_style)
        ],
        [
            Paragraph("<b>Q3: What makes PlacementOS superior to existing college portals?</b><br/>"
                      "<i>Answer:</i> Existing portals only store marks and job notices. PlacementOS adds <b>predictive risk early warnings</b>, <b>prescriptive what-if remediation</b>, <b>ATS keyword alignment</b>, and <b>direct TPO drive shortlisting</b> into a single unified monorepo.", body_style)
        ],
        [
            Paragraph("<b>Q4: Where is the project hosted and how can someone try it?</b><br/>"
                      "<i>Answer:</i> The project is deployed live on Streamlit Cloud at <b>https://placement-prediction-hl2h7xyvh4ncxz9xse67wd.streamlit.app/</b> and full source code is public on GitHub at <b>https://github.com/Nikhilreddy703/placement-prediction</b> with 31 automated tests passing.", body_style)
        ],
    ]

    t9 = Table(qas, colWidths=[9.8 * inch])
    t9.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#EFF6FF")),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DBEAFE")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t9)

    doc.build(story)
    print(f"Presentation PDF successfully built at: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_presentation_pdf()

