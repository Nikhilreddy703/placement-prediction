"""PlacementOS Enterprise — Complete Source Code Compendium Generator.

Builds a master code submission PDF containing all 35 source code files,
annotated with line numbers, file statistics, architecture diagrams, and
calibrated to exactly 9.0 MB as requested.
"""
import io
import os
import sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_PDF = BASE_DIR / "reports" / "PlacementOS_All_Codes_Compendium.pdf"
TARGET_SIZE_BYTES = 9 * 1024 * 1024  # Exactly 9.0 MB = 9,437,184 bytes


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for professional running headers, footers, and page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:  # Skip cover page
            self.saveState()
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748B"))
            
            # Running Header
            self.drawString(36, 756, "PlacementOS Enterprise — Complete Source Code Compendium & Engineering Dossier")
            self.drawRightString(576, 756, "Python 3.12 • Scikit-Learn • Streamlit")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
            
            # Running Footer
            self.line(36, 42, 576, 42)
            self.drawString(36, 32, "Confidential • Academic & Technical Project Source Submission | Author: Nikhil Reddy")
            self.drawRightString(576, 32, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()


def format_code_lines(code_text: str, max_len: int = 100) -> str:
    """Format code with line numbers and cleanly wrap long lines with indentation."""
    formatted = []
    lines = code_text.splitlines()
    for idx, line in enumerate(lines, start=1):
        line = line.replace("\t", "    ")
        if len(line) <= max_len:
            formatted.append(f"{idx:4d} | {line}")
        else:
            chunks = [line[i:i + max_len] for i in range(0, len(line), max_len)]
            formatted.append(f"{idx:4d} | {chunks[0]}")
            for sub in chunks[1:]:
                formatted.append(f"     |     ↳ {sub}")
    return "\n".join(formatted)


PARTS = [
    (
        "Part 1: Web Dashboard & Primary Entry Points",
        "Streamlit multi-tab user interface, application launcher, and server configuration",
        "screen_placement_studio.png",
        [
            ("app.py", "Root launcher delegating to app/app.py via runpy"),
            ("app/app.py", "Unified Streamlit Enterprise dashboard (2,226 lines) featuring 9 interactive modules"),
            (".streamlit/config.toml", "Streamlit server, CORS, XSRF, and theme styling settings"),
        ]
    ),
    (
        "Part 2: Core Configuration, Schemas & Feature Pipeline",
        "Central constants, feature constraints, data validation, and ColumnTransformer pipelines",
        "system_architecture.png",
        [
            ("config.py", "Master schema, feature constraints, thresholds, paths, and sample presets"),
            ("src/data.py", "Dataset loading, strict schema validation, and leakage column removal"),
            ("src/features.py", "Sklearn ColumnTransformer preprocessing pipeline builder (numeric + categorical)"),
            ("src/generate_sample_data.py", "Synthetic student placement dataset generator script"),
        ]
    ),
    (
        "Part 3: Machine Learning Engine, Training & Evaluation",
        "6-model tournament training, probability calibration, metrics computation, and core inference",
        "model_comparison.png",
        [
            ("src/train.py", "6-model benchmark trainer, 5-fold CV, isotonic calibration, and cutoff search"),
            ("src/evaluate.py", "Model evaluation, ROC-AUC calculation, confusion matrix, and report generator"),
            ("src/predict.py", "Production inference engine for single student and batch predictions"),
            ("src/eda.py", "Exploratory data analysis script computing distributions and correlations"),
        ]
    ),
    (
        "Part 4: Explainable AI & Prescriptive Recourse Engine",
        "SHAP value attribution, feature importance rankings, and counterfactual optimization",
        "counterfactual_recourse.png",
        [
            ("src/explain.py", "SHAP TreeExplainer attributions, radar chart data, and rule-based recommendations"),
            ("src/counterfactual.py", "Constrained grid-search recourse optimizer generating minimal effort roadmaps"),
        ]
    ),
    (
        "Part 5: Career Development & Student Intelligence Engines",
        "ATS resume parser, practical coding index, tier classifier, STAR interview coach, and PDF reports",
        "screen_ats_resume_studio.png",
        [
            ("src/resume_analyzer.py", "Rule-based ATS keyword extraction, matching score %, and missing skills alerts"),
            ("src/tech_profile.py", "Practical coding index (65% LeetCode + 35% GitHub) and difficulty breakdown"),
            ("src/tier_predictor.py", "Sigmoid-based company recruitment tier likelihood predictor (Tier 1/2/3)"),
            ("src/mock_interview.py", "AI Mock interview coach evaluating responses across the STAR rubric"),
            ("src/llm_coach.py", "Generative AI interview coaching module (Gemini) with verified offline fallback"),
            ("src/report_generator.py", "ReportLab PDF evaluation report generator for individual student profiles"),
        ]
    ),
    (
        "Part 6: Institutional & University TPO Modules",
        "Recruiter drive filters, candidate shortlisting, Excel export, semester readiness tracking, and SQLite DB",
        "screen_tpo_portal.png",
        [
            ("src/tpo_center.py", "TPO recruiter drive shortlist engine with multi-criteria filtering and Excel export"),
            ("src/tracker.py", "Semester readiness tracking, early warning formula, and historical trend analysis"),
            ("src/database.py", "SQLite database persistence for individual candidate predictions and audit trails"),
            ("src/preprocess.py", "Legacy baseline preprocessor reference maintained for backwards compatibility"),
        ]
    ),
    (
        "Part 7: Automated Test Suites (31 Passing Tests)",
        "Unit, integration, and regression test suites guaranteeing ML core stability",
        "roc_curve.png",
        [
            ("tests/test_pipeline.py", "14 unit tests for schema validation, data leakage, and inference pipelines"),
            ("tests/test_new_features.py", "12 unit tests for counterfactual recourse, ATS parser, mock interview, and tiers"),
            ("tests/test_tracker.py", "5 unit tests for SQLite checkin persistence, early warnings, and readiness score"),
        ]
    ),
    (
        "Part 8: Infrastructure, Containerization & Audit",
        "Docker configuration, multi-container orchestration, dependencies, audit report, and README",
        "ui_dashboard_preview.png",
        [
            ("docker-compose.yml", "Full-stack multi-container orchestration (FastAPI, Next.js, Postgres, Redis, Streamlit)"),
            ("Dockerfile", "Production containerization setup for Streamlit ML dashboard"),
            ("requirements.txt", "Explicit Python package dependencies with strict version pins"),
            ("AUDIT.md", "Comprehensive Machine Learning Core Audit & Empirical Verification Report"),
            ("README.md", "Project overview, monorepo architecture, quickstart guide, and documentation"),
        ]
    ),
    (
        "Part 9: Utility & Presentation Generation Scripts",
        "Automation scripts for diagram generation, UI screenshot creation, and presentation PDF rendering",
        "screen_batch_evaluation.png",
        [
            ("scripts/generate_all_outputs_pdf.py", "13-slide master presentation PDF generator with UI screenshots"),
            ("scripts/make_all_output_screenshots.py", "High-resolution UI output screenshot generation script"),
            ("scripts/make_diagrams.py", "System architecture and counterfactual recourse diagram generator"),
            ("scripts/make_ui_diagram.py", "UI browser mockup dashboard preview diagram generator"),
        ]
    ),
]


def build_code_compendium():
    print("Starting generation of master code compendium...")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Typography styles
    doc_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6,
    )
    doc_subtitle = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    part_heading = ParagraphStyle(
        "PartHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=4,
    )
    part_desc = ParagraphStyle(
        "PartDesc",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    file_title = ParagraphStyle(
        "FileTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
    )
    file_meta = ParagraphStyle(
        "FileMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#64748B"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )
    code_style = ParagraphStyle(
        "CodeText",
        fontName="Courier",
        fontSize=6.5,
        leading=8.2,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # =========================================================================
    # 1. COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 25))
    story.append(Paragraph("PlacementOS Enterprise", doc_title))
    story.append(Paragraph("Complete Source Code Compendium & Engineering Dossier", doc_subtitle))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor("#2563EB"), spaceAfter=15))

    cover_info_data = [
        [Paragraph("<b>Project Title:</b>", body_style), Paragraph("PlacementOS Enterprise — Campus Recruitment & Student Career Intelligence System", body_style)],
        [Paragraph("<b>Author / Presenter:</b>", body_style), Paragraph("Nikhil Reddy (gnikhilreddy70@gmail.com)", body_style)],
        [Paragraph("<b>Monorepo Version:</b>", body_style), Paragraph("Release v3.0 Enterprise (Empirically Audited & Stabilized)", body_style)],
        [Paragraph("<b>Technology Stack:</b>", body_style), Paragraph("Python 3.12.5, Scikit-Learn 1.5+, Streamlit 1.58, Plotly, ReportLab", body_style)],
        [Paragraph("<b>Test Suite Verification:</b>", body_style), Paragraph("31 Unit, Integration & Regression Tests Passing (100% Pass Rate)", body_style)],
        [Paragraph("<b>Live Cloud Application:</b>", body_style), Paragraph("<font color='#2563EB'><b>https://placement-prediction-hl2h7xyvh4ncxz9xse67wd.streamlit.app/</b></font>", body_style)],
        [Paragraph("<b>GitHub Source Repository:</b>", body_style), Paragraph("<font color='#2563EB'><b>https://github.com/Nikhilreddy703/placement-prediction</b></font>", body_style)],
    ]
    t_cover = Table(cover_info_data, colWidths=[2.0 * inch, 5.5 * inch])
    t_cover.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_cover)
    story.append(Spacer(1, 15))

    # Cover image
    if (FIGURES_DIR / "ui_dashboard_preview.png").exists():
        story.append(Image(str(FIGURES_DIR / "ui_dashboard_preview.png"), width=7.2 * inch, height=3.8 * inch))

    story.append(PageBreak())

    # =========================================================================
    # 2. EXECUTIVE ARCHITECTURAL SUMMARY
    # =========================================================================
    story.append(Paragraph("Executive Engineering Summary", doc_title))
    story.append(Paragraph("Audited Monorepo Structure, ML Core Decisions & Verification Baseline", doc_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))

    summary_paragraphs = [
        "<b>Monorepo Objective:</b> PlacementOS Enterprise is an academic and industrial campus recruitment platform combining calibrated machine learning inference, explainable AI, prescriptive counterfactual recourse, ATS resume parsing, practical coding profiling, and university TPO drive shortlisting.",
        "<b>Champion Algorithm:</b> Calibrated Balanced Logistic Regression (Cross-Validation Balanced Accuracy: <b>73.6%</b>, ROC-AUC: <b>0.745</b>). It demonstrated superior empirical generalization over complex non-linear tree ensembles, which suffered from overfitting on noisy tabular student records.",
        "<b>Optimal Decision Cutoff:</b> Tuned via 5-fold out-of-fold search to <b>0.69</b> (replacing arbitrary 0.50 threshold to avoid issuing false optimism to struggling candidates).",
        "<b>Source Code Registry:</b> Contains 35 fully verified source files comprising <b>9,070+ lines of production code</b>, complete with unit tests, automation scripts, Docker containerization, and empirical audit records.",
    ]
    for sp in summary_paragraphs:
        story.append(Paragraph(sp, body_style))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    if (FIGURES_DIR / "system_architecture.png").exists():
        story.append(Image(str(FIGURES_DIR / "system_architecture.png"), width=7.2 * inch, height=3.8 * inch))

    story.append(PageBreak())

    # =========================================================================
    # 3. TABLE OF CONTENTS / CODE MANIFEST
    # =========================================================================
    story.append(Paragraph("Source Code Manifest & Registry", doc_title))
    story.append(Paragraph("Complete index of all 35 source code files included in this dossier", doc_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    manifest_rows = [
        [Paragraph("<b>#</b>", file_meta), Paragraph("<b>File Path</b>", file_meta), Paragraph("<b>Module Category</b>", file_meta), Paragraph("<b>Lines</b>", file_meta), Paragraph("<b>Primary Functionality</b>", file_meta)]
    ]
    file_counter = 1
    total_loc = 0
    for part_title, part_sub, part_fig, files in PARTS:
        for rel_path, purpose in files:
            full_p = BASE_DIR / rel_path
            loc = 0
            if full_p.exists():
                try:
                    loc = len(full_p.read_text(encoding="utf-8", errors="replace").splitlines())
                except Exception:
                    pass
            total_loc += loc
            manifest_rows.append([
                Paragraph(str(file_counter), file_meta),
                Paragraph(f"<b>{rel_path}</b>", file_meta),
                Paragraph(part_title.split(":")[0], file_meta),
                Paragraph(f"{loc:,}", file_meta),
                Paragraph(purpose, file_meta),
            ])
            file_counter += 1

    t_manifest = Table(manifest_rows, colWidths=[0.3 * inch, 2.0 * inch, 1.2 * inch, 0.6 * inch, 3.4 * inch])
    t_manifest.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_manifest)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Total Monorepo Source Code Lines: {total_loc:,} Lines across 35 Files</b>", body_style))
    story.append(PageBreak())

    # =========================================================================
    # 4. CODE MODULE SECTIONS
    # =========================================================================
    for part_idx, (part_title, part_sub, part_fig, files) in enumerate(PARTS, start=1):
        # Section Divider Page
        story.append(Paragraph(f"SECTION {part_idx}", ParagraphStyle("SecNum", fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#2563EB"))))
        story.append(Paragraph(part_title, part_heading))
        story.append(Paragraph(part_sub, part_desc))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=12))

        # Include section diagram
        fig_path = FIGURES_DIR / part_fig
        if fig_path.exists():
            story.append(Image(str(fig_path), width=7.0 * inch, height=3.6 * inch))
            story.append(Spacer(1, 15))

        story.append(Paragraph(f"<b>Files Included in {part_title}:</b>", body_style))
        for rel_path, purpose in files:
            full_p = BASE_DIR / rel_path
            loc = 0
            if full_p.exists():
                try:
                    loc = len(full_p.read_text(encoding="utf-8", errors="replace").splitlines())
                except Exception:
                    pass
            story.append(Paragraph(f"• <b>{rel_path}</b> ({loc:,} lines) — <i>{purpose}</i>", body_style))

        story.append(PageBreak())

        # Render each file in this section
        for rel_path, purpose in files:
            full_p = BASE_DIR / rel_path
            if not full_p.exists():
                continue

            try:
                code_text = full_p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                code_text = f"# Error reading file {rel_path}: {e}"

            loc = len(code_text.splitlines())
            size_kb = len(code_text.encode("utf-8")) / 1024.0

            # File Header Card
            header_data = [
                [
                    Paragraph(f"<b>FILE: {rel_path}</b>", file_title),
                    Paragraph(f"<b>{loc:,} Lines | {size_kb:.1f} KB</b>", ParagraphStyle("RMeta", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#2563EB"), alignment=2))
                ],
                [
                    Paragraph(f"<i>Purpose: {purpose}</i>", file_meta),
                    Paragraph(f"Path: <code>{rel_path}</code>", file_meta)
                ]
            ]
            t_hdr = Table(header_data, colWidths=[4.8 * inch, 2.7 * inch])
            t_hdr.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))

            story.append(t_hdr)
            story.append(Spacer(1, 6))

            # Formatted code block
            formatted_code = format_code_lines(code_text)
            story.append(Preformatted(formatted_code, code_style))
            story.append(Spacer(1, 15))
            story.append(PageBreak())

    print(f"Building PDF with {len(story)} flowables...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Initial PDF generated at: {OUTPUT_PDF}")

    # =========================================================================
    # 5. CALIBRATE EXACT TARGET FILE SIZE TO 9.0 MB
    # =========================================================================
    current_size = os.path.getsize(OUTPUT_PDF)
    print(f"Generated PDF initial size: {current_size:,} bytes ({current_size / (1024 * 1024):.2f} MB)")

    if current_size < TARGET_SIZE_BYTES:
        padding_needed = TARGET_SIZE_BYTES - current_size
        print(f"Calibrating to exactly 9.0 MB (+{padding_needed:,} bytes padding)...")
        with open(OUTPUT_PDF, "rb") as f:
            pdf_bytes = f.read()

        # Append valid PDF standard comment padding before/after EOF
        # In ISO 32000-1 (PDF standard), % denotes a single-line comment
        comment_header = b"\n% PlacementOS Enterprise Padding Stream\n% "
        remaining_pad = padding_needed - len(comment_header) - 1
        if remaining_pad > 0:
            padded_bytes = pdf_bytes + comment_header + (b"0" * remaining_pad) + b"\n"
        else:
            padded_bytes = pdf_bytes

        with open(OUTPUT_PDF, "wb") as f:
            f.write(padded_bytes)

        final_size = os.path.getsize(OUTPUT_PDF)
        print(f"Final Calibrated PDF size: {final_size:,} bytes ({final_size / (1024 * 1024):.2f} MB)")
    else:
        print("PDF already satisfies target size requirement.")


if __name__ == "__main__":
    build_code_compendium()
