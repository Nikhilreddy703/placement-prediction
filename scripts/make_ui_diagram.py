import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

out_dir = Path("reports/figures")

fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')
ax.axis('off')

# Window frame (browser mockup)
browser_rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle='round,pad=0.015,rounding_size=0.025',
                                     facecolor='#1E293B', edgecolor='#334155', linewidth=2)
ax.add_patch(browser_rect)

# Browser header dots
ax.add_patch(plt.Circle((0.06, 0.91), 0.012, color='#EF4444'))
ax.add_patch(plt.Circle((0.09, 0.91), 0.012, color='#F59E0B'))
ax.add_patch(plt.Circle((0.12, 0.91), 0.012, color='#10B981'))

# URL bar
url_rect = patches.FancyBboxPatch((0.17, 0.88), 0.78, 0.055, boxstyle='round,pad=0.01,rounding_size=0.02',
                                 facecolor='#0F172A', edgecolor='#475569', linewidth=1)
ax.add_patch(url_rect)
ax.text(0.19, 0.907, 'https://placement-prediction-hl2h7xyvh4ncxz9xse67wd.streamlit.app',
        fontsize=10.5, color='#94A3B8', va='center', fontfamily='monospace')

# Main app title inside browser
ax.text(0.06, 0.81, 'PlacementOS Enterprise', fontsize=18, fontweight='bold', color='#FFFFFF')
ax.text(0.40, 0.81, 'Campus Placement Decision & Career Intelligence Platform', fontsize=12, color='#94A3B8')

# Active Tab Navigation Bar
tabs = ['Prediction Studio', 'Explainability (SHAP)', 'What-If Recourse', 'ATS Resume', 'Mock Interview', 'TPO Center']
for i, tab in enumerate(tabs):
    tx = 0.06 + i * 0.148
    bg_c = '#4F46E5' if i == 0 else '#334155'
    txt_c = '#FFFFFF' if i == 0 else '#94A3B8'
    t_rect = patches.FancyBboxPatch((tx, 0.70), 0.138, 0.05, boxstyle='round,pad=0.008,rounding_size=0.015',
                                    facecolor=bg_c, edgecolor='none')
    ax.add_patch(t_rect)
    ax.text(tx + 0.069, 0.725, tab, ha='center', va='center', fontsize=8.5, fontweight='bold', color=txt_c)

# Left Column: Student Input Card Mockup
card1 = patches.FancyBboxPatch((0.06, 0.08), 0.38, 0.58, boxstyle='round,pad=0.01,rounding_size=0.02',
                              facecolor='#0F172A', edgecolor='#334155', linewidth=1.5)
ax.add_patch(card1)
ax.text(0.09, 0.61, 'Candidate Input & Profile', fontsize=12, fontweight='bold', color='#E2E8F0')

inputs = [
    ('CGPA:', '8.4 / 10.0'),
    ('Internships Completed:', '2 Summer Internships'),
    ('Major Projects:', '3 Full-Stack & ML Projects'),
    ('Aptitude / Coding Score:', '85%'),
    ('Active Backlogs:', '0 Backlogs'),
    ('Tech Profile Index:', '88 / 100 (LeetCode & GitHub)'),
]
for j, (lbl, val) in enumerate(inputs):
    yy = 0.53 - j * 0.075
    ax.text(0.09, yy, lbl, fontsize=9.5, color='#94A3B8')
    ax.text(0.26, yy, val, fontsize=9.5, fontweight='bold', color='#38BDF8')

# Right Column: Prediction Gauge & Recourse Card Mockup
card2 = patches.FancyBboxPatch((0.48, 0.08), 0.47, 0.58, boxstyle='round,pad=0.01,rounding_size=0.02',
                              facecolor='#0F172A', edgecolor='#334155', linewidth=1.5)
ax.add_patch(card2)
ax.text(0.51, 0.61, 'Calibrated Inference & Actionable Recourse', fontsize=12, fontweight='bold', color='#E2E8F0')

# Big Likelihood Metric
metric_bg = patches.FancyBboxPatch((0.51, 0.36), 0.41, 0.20, boxstyle='round,pad=0.01,rounding_size=0.02',
                                   facecolor='#064E3B', edgecolor='#10B981', linewidth=1.5)
ax.add_patch(metric_bg)
ax.text(0.715, 0.50, '87.4%', ha='center', va='center', fontsize=26, fontweight='bold', color='#34D399')
ax.text(0.715, 0.41, 'Placement Likelihood (Low Risk • Tier-1 Ready)', ha='center', va='center', fontsize=9.5, color='#A7F3D0')

# Recourse prescription box
rec_bg = patches.FancyBboxPatch((0.51, 0.12), 0.41, 0.20, boxstyle='round,pad=0.01,rounding_size=0.02',
                                facecolor='#1E1B4B', edgecolor='#6366F1', linewidth=1.5)
ax.add_patch(rec_bg)
ax.text(0.53, 0.27, 'Prescriptive Career Coaching', fontsize=10, fontweight='bold', color='#A5B4FC')
ax.text(0.53, 0.21, '✓ Strong academic foundation (CGPA 8.4 boosts +18% prob)', fontsize=8.5, color='#C7D2FE')
ax.text(0.53, 0.16, '★ Next step: Target Tier-1 Core drives; refine STAR behavioral answers', fontsize=8.5, color='#FDE047')

plt.tight_layout()
plt.savefig(out_dir / 'ui_dashboard_preview.png', dpi=200, bbox_inches='tight')
plt.close()
print('UI Dashboard preview diagram saved!')
