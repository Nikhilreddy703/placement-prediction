"""Generate realistic, high-resolution visual UI screenshots for each PlacementOS module output."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Helper for drawing rounded cards
def draw_card(ax, x, y, w, h, bg="#1E293B", border="#334155", radius=0.02):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}",
                                  facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    return rect

# ==============================================================================
# SCREEN 1: Student Placement Studio Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

# Header
ax.text(0.04, 0.94, 'PlacementOS Enterprise • Student Placement Studio', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: Calibrated Placement Probability, Risk Status & SHAP Dimension Radar', fontsize=9.5, color='#94A3B8')

# Left Card: Candidate Metrics
draw_card(ax, 0.04, 0.06, 0.44, 0.80)
ax.text(0.07, 0.80, 'Candidate Profile & Inputs', fontsize=11, fontweight='bold', color='#E2E8F0')

inputs = [
    ("Target Candidate:", "Rahul Sharma (Roll: CS2024-042)"),
    ("Active Model:", "Calibrated Logistic Regression (Champion)"),
    ("Cumulative GPA (CGPA):", "8.25 / 10.00"),
    ("Internships Completed:", "2 Verified Internships"),
    ("Major Technical Projects:", "3 Production Builds"),
    ("Aptitude / Coding Score:", "84.0%"),
    ("Active Academic Backlogs:", "0 Backlogs"),
    ("Communication Rating:", "4.5 / 5.0 (Advanced)"),
]
for idx, (label, val) in enumerate(inputs):
    yy = 0.73 - idx * 0.08
    ax.text(0.07, yy, label, fontsize=8.8, color='#94A3B8')
    ax.text(0.24, yy, val, fontsize=8.8, fontweight='bold', color='#38BDF8')

# Right Top Card: Big Likelihood Gauge Output
draw_card(ax, 0.51, 0.48, 0.45, 0.38, bg="#064E3B", border="#10B981")
ax.text(0.735, 0.78, '87.4%', ha='center', va='center', fontsize=32, fontweight='bold', color='#34D399')
ax.text(0.735, 0.68, 'Calibrated Placement Probability', ha='center', va='center', fontsize=11, fontweight='bold', color='#FFFFFF')
ax.text(0.735, 0.60, 'STATUS: LOW RISK • RECRUITER DRIVE READY', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#A7F3D0')
ax.text(0.735, 0.53, 'Calibrated: 87.4%  |  Uncalibrated Raw: 92.1%  |  Optimal Cutoff: 0.69', ha='center', va='center', fontsize=8, color='#6EE7B7')

# Right Bottom Card: SHAP Radar Profile
draw_card(ax, 0.51, 0.06, 0.45, 0.39)
ax.text(0.54, 0.40, 'Student Competency Dimensions (SHAP Multi-Axis)', fontsize=10, fontweight='bold', color='#E2E8F0')

# Mini Radar / Bar representation
dims = [("Academics (CGPA)", 0.85, '#38BDF8'), ("Internships", 0.90, '#34D399'), 
        ("Projects", 0.75, '#818CF8'), ("Coding / Aptitude", 0.82, '#F472B6'), ("Backlog Safety", 1.0, '#4ADE80')]
for idx, (name, val, col) in enumerate(dims):
    yy = 0.33 - idx * 0.065
    ax.text(0.54, yy, name, fontsize=8.5, color='#94A3B8')
    # background bar
    rect_b = patches.FancyBboxPatch((0.70, yy-0.01), 0.22, 0.03, boxstyle="round,pad=0.005", facecolor='#334155', edgecolor='none')
    ax.add_patch(rect_b)
    # fill bar
    rect_f = patches.FancyBboxPatch((0.70, yy-0.01), 0.22 * val, 0.03, boxstyle="round,pad=0.005", facecolor=col, edgecolor='none')
    ax.add_patch(rect_f)
    ax.text(0.93, yy, f"{int(val*100)}%", fontsize=8, fontweight='bold', color=col)

plt.tight_layout()
plt.savefig(out_dir / 'screen_placement_studio.png', dpi=180, bbox_inches='tight')
plt.close()

# ==============================================================================
# SCREEN 2: ATS Resume Studio Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

ax.text(0.04, 0.94, 'PlacementOS Enterprise • ATS Resume & JD Match Studio', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: Rule-Based Keyword Match %, Missing Skills & Recruiter JD Alignment', fontsize=9.5, color='#94A3B8')

# Left Card: Overall Match Gauge & Summary
draw_card(ax, 0.04, 0.06, 0.42, 0.80)
ax.text(0.07, 0.80, 'ATS Match Diagnostic', fontsize=11, fontweight='bold', color='#E2E8F0')

# Big Match Score Circle
ax.add_patch(plt.Circle((0.25, 0.62), 0.12, color='#1E293B', ec='#6366F1', lw=3))
ax.text(0.25, 0.63, '82%', ha='center', va='center', fontsize=26, fontweight='bold', color='#818CF8')
ax.text(0.25, 0.55, 'JD Match Score', ha='center', va='center', fontsize=9, color='#C7D2FE')

summary_bullets = [
    "Target Role: Full-Stack / ML Software Engineer",
    "Company: Tier-1 Product Organization",
    "Total Required Keywords: 28 Extracted",
    "Direct Keyword Hits: 23 Matched (82.1%)",
    "Missing Critical Keywords: 5 Detected",
    "ATS Verdict: STRONG CANDIDATE (Passed Screening)",
]
for idx, txt in enumerate(summary_bullets):
    yy = 0.44 - idx * 0.065
    ax.text(0.07, yy, f"• {txt}", fontsize=8.5, color='#CBD5E1')

# Right Card: Matched vs Missing Skills Pills
draw_card(ax, 0.49, 0.06, 0.47, 0.80)
ax.text(0.52, 0.80, 'Matched Technical Skills (Verified in Resume)', fontsize=10.5, fontweight='bold', color='#34D399')

matched_skills = ["Python 3.12", "FastAPI", "React / Next.js", "Scikit-Learn", "PostgreSQL", 
                  "Git / GitHub", "REST APIs", "Pandas & NumPy", "Tailwind CSS", "Data Structures"]
for idx, skill in enumerate(matched_skills):
    col_idx = idx % 2
    row_idx = idx // 2
    bx = 0.52 + col_idx * 0.22
    by = 0.72 - row_idx * 0.075
    rect_s = patches.FancyBboxPatch((bx, by), 0.20, 0.05, boxstyle="round,pad=0.005", facecolor='#064E3B', edgecolor='#10B981', lw=1)
    ax.add_patch(rect_s)
    ax.text(bx + 0.10, by + 0.025, f"✓ {skill}", ha='center', va='center', fontsize=8, fontweight='bold', color='#6EE7B7')

ax.text(0.52, 0.32, 'Missing High-Value Keywords (Prescribed Action)', fontsize=10.5, fontweight='bold', color='#F87171')
missing_skills = [("Docker & Containerization", "High Priority"), ("Redis In-Memory Cache", "Medium Priority"), 
                  ("CI/CD GitHub Actions", "High Priority"), ("Kubernetes Orchestration", "Nice to Have")]
for idx, (ms, pri) in enumerate(missing_skills):
    by = 0.24 - idx * 0.058
    rect_m = patches.FancyBboxPatch((0.52, by), 0.42, 0.045, boxstyle="round,pad=0.005", facecolor='#450A0A', edgecolor='#EF4444', lw=1)
    ax.add_patch(rect_m)
    ax.text(0.54, by + 0.022, f"✗ {ms}", va='center', fontsize=8, fontweight='bold', color='#FCA5A5')
    ax.text(0.92, by + 0.022, pri, ha='right', va='center', fontsize=7.5, color='#F87171')

plt.tight_layout()
plt.savefig(out_dir / 'screen_ats_resume_studio.png', dpi=180, bbox_inches='tight')
plt.close()

# ==============================================================================
# SCREEN 3: Tech Profile & Tier Predictor Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

ax.text(0.04, 0.94, 'PlacementOS Enterprise • Tech Profile & Company Tier Predictor', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: Practical Coding Matrix (65/35 Index) & Tier-1/2/3 Recruitment Probabilities', fontsize=9.5, color='#94A3B8')

# Left Card: Coding Matrix
draw_card(ax, 0.04, 0.06, 0.44, 0.80)
ax.text(0.07, 0.80, 'Practical Coding & Developer Index', fontsize=11, fontweight='bold', color='#E2E8F0')

draw_card(ax, 0.07, 0.52, 0.38, 0.24, bg="#1E1B4B", border="#6366F1")
ax.text(0.26, 0.67, '86.5 / 100', ha='center', va='center', fontsize=28, fontweight='bold', color='#A5B4FC')
ax.text(0.26, 0.58, 'Practical Tech Profile Score (Formula: 65% LeetCode + 35% GitHub)', ha='center', va='center', fontsize=7.8, color='#C7D2FE')

code_metrics = [
    ("LeetCode Problems Solved:", "342 (120 Easy, 185 Medium, 37 Hard)"),
    ("Contest Rating:", "1785 (Knight / Top 8%)"),
    ("GitHub Public Repositories:", "14 Active Repositories"),
    ("Total Commits in Past Year:", "486 Verified Commits"),
    ("Production Frameworks:", "FastAPI, React, PyTorch, Docker"),
]
for idx, (lbl, val) in enumerate(code_metrics):
    yy = 0.44 - idx * 0.07
    ax.text(0.07, yy, lbl, fontsize=8.5, color='#94A3B8')
    ax.text(0.07, yy - 0.03, val, fontsize=8.5, fontweight='bold', color='#38BDF8')

# Right Card: Tier Likelihoods
draw_card(ax, 0.51, 0.06, 0.45, 0.80)
ax.text(0.54, 0.80, 'Recruiter Company Tier Likelihoods', fontsize=11, fontweight='bold', color='#E2E8F0')

tiers = [
    ("Tier 1: Global Tech / FAANG / Core Product (18+ LPA)", 0.84, "#10B981", "#064E3B", "84.2% (Eligible)"),
    ("Tier 2: Mid-Size Product / FinTech / Unicorns (8-18 LPA)", 0.95, "#3B82F6", "#1E3A8A", "95.1% (High Match)"),
    ("Tier 3: IT Services / Systems Integrators (4-8 LPA)", 0.99, "#8B5CF6", "#4C1D95", "99.0% (Guaranteed)"),
]
for idx, (tname, val, col, bg_c, tag) in enumerate(tiers):
    by = 0.52 - idx * 0.22
    draw_card(ax, 0.54, by, 0.39, 0.18, bg=bg_c, border=col)
    ax.text(0.56, by + 0.14, tname, fontsize=8.5, fontweight='bold', color='#FFFFFF')
    ax.text(0.56, by + 0.09, f"Likelihood: {tag}", fontsize=9, fontweight='bold', color=col)
    # progress bar
    rect_tb = patches.FancyBboxPatch((0.56, by + 0.03), 0.35, 0.025, boxstyle="round,pad=0.002", facecolor='#0F172A', edgecolor='none')
    ax.add_patch(rect_tb)
    rect_tf = patches.FancyBboxPatch((0.56, by + 0.03), 0.35 * val, 0.025, boxstyle="round,pad=0.002", facecolor=col, edgecolor='none')
    ax.add_patch(rect_tf)

plt.tight_layout()
plt.savefig(out_dir / 'screen_tech_coding_matrix.png', dpi=180, bbox_inches='tight')
plt.close()

# ==============================================================================
# SCREEN 4: Mock Interview & STAR Rubric Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

ax.text(0.04, 0.94, 'PlacementOS Enterprise • AI Mock Interview & STAR Rubric', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: Behavioral Response Evaluation (Situation, Task, Action, Result Scoring)', fontsize=9.5, color='#94A3B8')

# Prompt Box
draw_card(ax, 0.04, 0.68, 0.92, 0.18, bg="#1E293B", border="#475569")
ax.text(0.06, 0.81, 'Behavioral Interview Question Prompt:', fontsize=9.5, fontweight='bold', color='#F59E0B')
ax.text(0.06, 0.74, '"Describe a situation where a technical project failed or encountered a critical bug before production. How did you handle it?"',
        fontsize=9, fontstyle='italic', color='#E2E8F0')

# S - T - A - R Score Cards
star_items = [
    ("Situation", "88%", "#10B981", "#064E3B", "Clear business context and deployment timeline established."),
    ("Task", "82%", "#3B82F6", "#1E3A8A", "Defined exact personal ownership vs team responsibilities."),
    ("Action", "92%", "#8B5CF6", "#4C1D95", "Superb depth: Debugged latency bottleneck with profiler."),
    ("Result", "78%", "#F59E0B", "#78350F", "Good resolution, but could quantify business metrics more.")
]
for idx, (title, score, col, bg_c, feedback) in enumerate(star_items):
    bx = 0.04 + idx * 0.235
    draw_card(ax, bx, 0.22, 0.22, 0.42, bg=bg_c, border=col)
    ax.text(bx + 0.11, 0.58, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax.text(bx + 0.11, 0.48, score, ha='center', va='center', fontsize=24, fontweight='bold', color=col)
    ax.text(bx + 0.11, 0.38, 'STAR Score', ha='center', va='center', fontsize=8, color='#CBD5E1')
    ax.text(bx + 0.02, 0.28, feedback, fontsize=7.5, color='#E2E8F0', linespacing=1.3)

# Bottom Coach Summary Card
draw_card(ax, 0.04, 0.04, 0.92, 0.14, bg="#064E3B", border="#10B981")
ax.text(0.06, 0.13, '★ AI Interview Coach Assessment: EXCELLENT BEHAVIORAL MATURITY (Overall Score: 85/100)',
        fontsize=10, fontweight='bold', color='#34D399')
ax.text(0.06, 0.07, 'Coach Tip: Quantify the final impact with exact numbers (e.g. "Reduced API response latency by 45% and prevented 2h downtime").',
        fontsize=8.5, color='#A7F3D0')

plt.tight_layout()
plt.savefig(out_dir / 'screen_mock_interview.png', dpi=180, bbox_inches='tight')
plt.close()

# ==============================================================================
# SCREEN 5: TPO & Recruiter Portal Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

ax.text(0.04, 0.94, 'PlacementOS Enterprise • TPO & Recruiter Drive Portal', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: Recruitment Drive Candidate Shortlisting, Eligibility Filtering & Instant Excel Export', fontsize=9.5, color='#94A3B8')

# Top Drive Filter Criteria Bar
draw_card(ax, 0.04, 0.72, 0.92, 0.14, bg="#1E293B", border="#3B82F6")
ax.text(0.06, 0.81, 'Active Recruiter Drive Filter: "Microsoft Full-Stack Campus Recruitment Drive"', fontsize=10, fontweight='bold', color='#38BDF8')
ax.text(0.06, 0.75, 'Criteria: Min CGPA >= 7.50  |  Active Backlogs = 0  |  Min Internships >= 1  |  Tech Index >= 70.0  |  Sort: Placement Probability DESC',
        fontsize=8.5, color='#94A3B8')

# Shortlist Candidate Table Mockup
draw_card(ax, 0.04, 0.06, 0.92, 0.62)
ax.text(0.06, 0.63, 'Shortlisted Eligible Candidates: 42 of 100 Students Qualified (42.0% Conversion)', fontsize=11, fontweight='bold', color='#E2E8F0')

# Table header
headers = ["Roll Number", "Student Name", "CGPA", "Internships", "Coding Index", "Placement Prob", "Risk Level", "Drive Status"]
hx_positions = [0.06, 0.18, 0.32, 0.40, 0.50, 0.62, 0.74, 0.86]
for hx, htext in zip(hx_positions, headers):
    ax.text(hx, 0.57, htext, fontsize=8.5, fontweight='bold', color='#94A3B8')

# Table rows
rows = [
    ("CS2024-042", "Rahul Sharma", "8.25", "2", "86.5", "87.4%", "Low Risk", "Shortlisted ✓"),
    ("CS2024-015", "Ananya Verma", "9.10", "3", "92.0", "94.6%", "Low Risk", "Shortlisted ✓"),
    ("CS2024-088", "Karthik Reddy", "7.80", "1", "78.4", "79.2%", "Low Risk", "Shortlisted ✓"),
    ("CS2024-031", "Sneha Patel", "8.45", "2", "81.0", "84.5%", "Low Risk", "Shortlisted ✓"),
    ("CS2024-102", "Vikram Rao", "7.60", "1", "74.0", "72.8%", "Low Risk", "Shortlisted ✓"),
]
for r_idx, r_data in enumerate(rows):
    ry = 0.50 - r_idx * 0.065
    row_bg = '#1E293B' if r_idx % 2 == 0 else '#0F172A'
    rect_r = patches.FancyBboxPatch((0.05, ry-0.015), 0.90, 0.05, boxstyle="round,pad=0.002", facecolor=row_bg, edgecolor='none')
    ax.add_patch(rect_r)
    for hx, val in zip(hx_positions, r_data):
        v_col = '#34D399' if "Shortlisted" in val or "Low" in val or "9" in val[:1] else '#E2E8F0'
        ax.text(hx, ry, val, fontsize=8, color=v_col)

# Bottom Action Button
draw_card(ax, 0.66, 0.10, 0.28, 0.08, bg="#15803D", border="#22C55E")
ax.text(0.80, 0.14, 'Download Drive Shortlist (Excel .xlsx)', ha='center', va='center', fontsize=9, fontweight='bold', color='#FFFFFF')

plt.tight_layout()
plt.savefig(out_dir / 'screen_tpo_portal.png', dpi=180, bbox_inches='tight')
plt.close()

# ==============================================================================
# SCREEN 6: Batch Student Evaluation Output
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

ax.text(0.04, 0.94, 'PlacementOS Enterprise • Batch Student Evaluation Studio', fontsize=15, fontweight='bold', color='#FFFFFF')
ax.text(0.04, 0.89, 'OUTPUT: High-Throughput Cohort CSV Upload, Inference & Institutional Risk Breakdown', fontsize=9.5, color='#94A3B8')

# Left Card: Summary Stats
draw_card(ax, 0.04, 0.06, 0.44, 0.80)
ax.text(0.07, 0.80, 'Batch Processing Summary', fontsize=11, fontweight='bold', color='#E2E8F0')

batch_stats = [
    ("Uploaded File:", "Final_Year_BTech_Batch_2025.csv"),
    ("Total Records Processed:", "400 Students Ingested"),
    ("Inference Processing Time:", "0.82 seconds (Vectorized ML Core)"),
    ("Low Risk (Likely Placed):", "268 Students (67.0%)"),
    ("Moderate Risk (Needs Coaching):", "84 Students (21.0%)"),
    ("High Risk (Critical Intervention):", "48 Students (12.0%)"),
]
for idx, (lbl, val) in enumerate(batch_stats):
    yy = 0.72 - idx * 0.08
    ax.text(0.07, yy, lbl, fontsize=8.8, color='#94A3B8')
    ax.text(0.07, yy - 0.03, val, fontsize=8.8, fontweight='bold', color='#38BDF8')

# Right Card: Cohort Risk Distribution Chart Mockup
draw_card(ax, 0.51, 0.06, 0.45, 0.80)
ax.text(0.54, 0.80, 'Cohort Risk Stratification (Actionable Categories)', fontsize=10.5, fontweight='bold', color='#E2E8F0')

r_cats = [
    ("Low Risk • Prob >= 0.70", "268 (67%)", 0.67, "#10B981", "Cleared for Top-Tier Placement Drives"),
    ("Moderate Risk • 0.50 <= P < 0.70", "84 (21%)", 0.21, "#F59E0B", "Assigned to Mock Interview & Resume Clinic"),
    ("High Risk • Prob < 0.50", "48 (12%)", 0.12, "#EF4444", "Urgent Intervention: Clear Backlogs + Projects"),
]
for idx, (cat_name, count_str, pct, col, note) in enumerate(r_cats):
    by = 0.54 - idx * 0.22
    draw_card(ax, 0.54, by, 0.39, 0.18, bg="#1E293B", border=col)
    ax.text(0.56, by + 0.13, cat_name, fontsize=8.5, fontweight='bold', color='#FFFFFF')
    ax.text(0.90, by + 0.13, count_str, ha='right', fontsize=9, fontweight='bold', color=col)
    # progress bar
    rect_pb = patches.FancyBboxPatch((0.56, by + 0.06), 0.35, 0.025, boxstyle="round,pad=0.002", facecolor='#0F172A', edgecolor='none')
    ax.add_patch(rect_pb)
    rect_pf = patches.FancyBboxPatch((0.56, by + 0.06), 0.35 * pct, 0.025, boxstyle="round,pad=0.002", facecolor=col, edgecolor='none')
    ax.add_patch(rect_pf)
    ax.text(0.56, by + 0.02, note, fontsize=7.2, color='#94A3B8')

plt.tight_layout()
plt.savefig(out_dir / 'screen_batch_evaluation.png', dpi=180, bbox_inches='tight')
plt.close()

print('All UI output screenshots generated successfully!')
