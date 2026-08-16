"""
generate_dataset.py
--------------------
Generates a synthetic dataset of student academic records and saves it
as data/student_data.csv. This dataset is used to train the
performance-prediction model in train_model.py.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
NUM_STUDENTS = 1500

def generate_dataset(num_students: int = NUM_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(4, 2, num_students), 0, 12)
    attendance = np.clip(rng.normal(80, 12, num_students), 30, 100)
    previous_scores = np.clip(rng.normal(65, 15, num_students), 0, 100)
    assignments_completed = np.clip(rng.normal(80, 15, num_students), 0, 100)
    sleep_hours = np.clip(rng.normal(6.5, 1.5, num_students), 3, 10)
    extracurricular = rng.choice([0, 1], size=num_students, p=[0.55, 0.45])
    parental_support = rng.choice([0, 1, 2], size=num_students, p=[0.2, 0.5, 0.3])  # 0=low,1=medium,2=high
    internet_access = rng.choice([0, 1], size=num_students, p=[0.15, 0.85])

    # Build a realistic score using a weighted formula + noise
    noise = rng.normal(0, 5, num_students)
    final_score = (
        0.28 * study_hours * 8
        + 0.22 * attendance
        + 0.20 * previous_scores
        + 0.15 * assignments_completed
        + 0.05 * sleep_hours * 5
        + 0.04 * extracurricular * 10
        + 0.06 * parental_support * 10
        + 0.03 * internet_access * 10
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "study_hours": np.round(study_hours, 1),
        "attendance": np.round(attendance, 1),
        "previous_scores": np.round(previous_scores, 1),
        "assignments_completed": np.round(assignments_completed, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "extracurricular": extracurricular,
        "parental_support": parental_support,
        "internet_access": internet_access,
        "final_score": np.round(final_score, 1),
    })

    df["pass_fail"] = np.where(df["final_score"] >= 40, "Pass", "Fail")

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset()
    output_path = os.path.join("data", "student_data.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Dataset generated successfully: {output_path}")
    print(dataset.head())
    print(f"\nShape: {dataset.shape}")
