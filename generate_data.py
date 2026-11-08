"""Generate a SYNTHETIC student-performance dataset (data/students.csv).

The data is simulated, not collected from real students. The relationships (more study time and
attendance -> higher scores, etc.) are assumptions built into this script, so the analysis demonstrates
the *workflow* (cleaning, EDA, modelling), not real-world findings.

Usage:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 500


def main() -> None:
    rng = np.random.default_rng(SEED)

    study = np.clip(rng.normal(12, 5, N), 0, 40)                      # hours per week
    attendance = np.clip(100 - rng.gamma(2.0, 6.0, N), 40, 100)       # percent
    sleep = np.clip(rng.normal(6.8, 1.1, N), 4, 10)                   # hours per night
    prior_gpa = np.clip(rng.normal(3.0, 0.5, N), 1.5, 4.0)
    tutoring = rng.binomial(1, 0.3, N)                                # 1 = attends tutoring
    internet = rng.binomial(1, 0.85, N)                               # 1 = reliable internet at home

    noise = rng.normal(0, 6, N)
    score = (
        8
        + 1.4 * study
        + 0.28 * attendance
        + 2.0 * (sleep - 6.8)
        + 9.0 * (prior_gpa - 3.0)
        + 4.5 * tutoring
        + 2.0 * internet
        + noise
    )
    score = np.clip(score, 0, 100).round(1)

    df = pd.DataFrame(
        {
            "student_id": np.arange(1, N + 1),
            "study_hours_per_week": study.round(1),
            "attendance_pct": attendance.round(1),
            "sleep_hours": sleep.round(1),
            "prior_gpa": prior_gpa.round(2),
            "tutoring": tutoring,
            "internet_access": internet,
            "exam_score": score,
        }
    )

    # Inject realistic data-quality problems so the cleaning step has something to do.
    for col, frac in [("sleep_hours", 0.04), ("attendance_pct", 0.03)]:
        idx = rng.choice(N, size=int(N * frac), replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(5, random_state=SEED)], ignore_index=True)  # a few duplicate rows

    out = Path(__file__).parent / "data" / "students.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
