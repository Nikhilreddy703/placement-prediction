import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Diagram 2: Counterfactual Recourse
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Prescriptive Counterfactual Recourse: The "What-If" Engine', 
        ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
ax.text(0.5, 0.89, 'Automated search for minimal realistic changes required to convert high-risk students to placed status', 
        ha='center', va='center', fontsize=10, color='#475569')

steps = [
    (0.03, 0.25, 0.20, 0.52, 'Step 1: Current State', 'Student Profile:\n• CGPA: 6.2\n• Internships: 0\n• Projects: 1\n• Backlogs: 2\n\nProbability: 38%\nStatus: High Risk', '#FEF2F2', '#EF4444'),
    (0.27, 0.25, 0.22, 0.52, 'Step 2: Constrained Search', 'Recourse Rules:\n• Freeze immutable traits\n• Bound realistic changes:\n  +CGPA <= 0.8\n  +Internships <= 2\n  +Projects <= 3\n• Clear active backlogs', '#EFF6FF', '#3B82F6'),
    (0.53, 0.25, 0.21, 0.52, 'Step 3: Optimization', 'Min Effort Distance:\nargmin  Cost(delta)\ns.t. P(placed) >= 0.70\n\nIdentifies smallest\nactionable step that\nflips the prediction', '#FAF5FF', '#9333EA'),
    (0.77, 0.25, 0.20, 0.52, 'Step 4: Prescriptive Plan', 'Target Goal:\n• Clear 2 Backlogs\n• Complete 1 Internship\n• Build 1 Major Project\n\nNew Prob: 76%\nStatus: Placed!', '#F0FDF4', '#16A34A')
]

for x, y, w, h, title, text, bg, border in steps:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.03',
                                  facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.05, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')
    ax.text(x + 0.02, y + h/2 - 0.04, text, ha='left', va='center', fontsize=9.5, color='#334155', linespacing=1.35)

# Connect with arrows
ax.annotate('', xy=(0.26, 0.51), xytext=(0.23, 0.51), arrowprops=dict(arrowstyle='->', lw=2.5, color='#94A3B8'))
ax.annotate('', xy=(0.52, 0.51), xytext=(0.49, 0.51), arrowprops=dict(arrowstyle='->', lw=2.5, color='#94A3B8'))
ax.annotate('', xy=(0.76, 0.51), xytext=(0.74, 0.51), arrowprops=dict(arrowstyle='->', lw=2.5, color='#94A3B8'))

# Bottom callout
rect_bot = patches.FancyBboxPatch((0.15, 0.04), 0.70, 0.14, boxstyle='round,pad=0.015,rounding_size=0.02',
                                  facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1)
ax.add_patch(rect_bot)
ax.text(0.5, 0.11, 'Key Educational Takeaway: Moves beyond diagnosing failure to providing actionable, personalized guidance.',
        ha='center', va='center', fontsize=9.5, fontstyle='italic', color='#1E293B')

plt.tight_layout()
plt.savefig(out_dir / 'counterfactual_recourse.png', dpi=200, bbox_inches='tight')
plt.close()
print('Recourse diagram saved successfully!')
