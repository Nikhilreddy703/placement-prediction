"""Generate a SYNTHETIC placement dataset so the project runs immediately.

The columns match the popular Kaggle "Campus Recruitment" dataset.
For your final submission, replace data/raw/placement.csv with a real dataset
(Kaggle download or data collected from your own college) and say so in your report.

Usage:  python src/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)
N = 400

gender = rng.choice(["M", "F"], N, p=[0.6, 0.4])
ssc_p = np.clip(rng.normal(67, 10, N), 40, 95).round(2)
ssc_b = rng.choice(["Central", "Others"], N)
hsc_p = np.clip(rng.normal(66, 10, N), 40, 95).round(2)
hsc_b = rng.choice(["Central", "Others"], N)
hsc_s = rng.choice(["Commerce", "Science", "Arts"], N, p=[0.5, 0.4, 0.1])
degree_p = np.clip(rng.normal(66, 7, N), 50, 91).round(2)
degree_t = rng.choice(["Comm&Mgmt", "Sci&Tech", "Others"], N, p=[0.6, 0.3, 0.1])
workex = rng.choice(["Yes", "No"], N, p=[0.35, 0.65])
etest_p = np.clip(rng.normal(72, 13, N), 50, 98).round(2)
specialisation = rng.choice(["Mkt&Fin", "Mkt&HR"], N, p=[0.55, 0.45])
mba_p = np.clip(rng.normal(62, 6, N), 51, 78).round(2)

# Hidden rule used to create the labels (higher scores and work experience -> more likely placed)
z = (
    0.07 * (ssc_p - 67)
    + 0.05 * (hsc_p - 66)
    + 0.12 * (degree_p - 66)
    + 0.03 * (etest_p - 72)
    + 0.05 * (mba_p - 62)
    + 1.2 * (workex == "Yes")
    + 0.4
    + rng.normal(0, 0.8, N)
)
placed = rng.random(N) < 1 / (1 + np.exp(-z))
status = np.where(placed, "Placed", "Not Placed")
salary = np.where(placed, rng.integers(200000, 400000, N), np.nan)  # exists only for placed students

df = pd.DataFrame({
    "sl_no": np.arange(1, N + 1), "gender": gender, "ssc_p": ssc_p, "ssc_b": ssc_b,
    "hsc_p": hsc_p, "hsc_b": hsc_b, "hsc_s": hsc_s, "degree_p": degree_p, "degree_t": degree_t,
    "workex": workex, "etest_p": etest_p, "specialisation": specialisation, "mba_p": mba_p,
    "status": status, "salary": salary,
})
out = ROOT / "data" / "raw" / "placement.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}  (SYNTHETIC data - replace with real data for submission)")
