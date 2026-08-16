"""
app.py
-------
Flask web application for the AI-Driven Student Performance Prediction
System. Loads the trained RandomForest model and scaler, serves an
input form, and returns a predicted final score, letter grade, and
pass/fail status.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

from train_model import FEATURE_COLUMNS, MODEL_PATH, SCALER_PATH, train

app = Flask(__name__)
app.secret_key = "student-performance-secret-key"

model = None
scaler = None


def load_artifacts():
    """Load the trained model and scaler, training them first if missing."""
    global model, scaler
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("No trained model found - training a new one now...")
        train()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    if score >= 40:
        return "E"
    return "F"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        input_values = {
            "study_hours": float(form.get("study_hours", 0)),
            "attendance": float(form.get("attendance", 0)),
            "previous_scores": float(form.get("previous_scores", 0)),
            "assignments_completed": float(form.get("assignments_completed", 0)),
            "sleep_hours": float(form.get("sleep_hours", 0)),
            "extracurricular": 1 if form.get("extracurricular") == "yes" else 0,
            "parental_support": int(form.get("parental_support", 1)),
            "internet_access": 1 if form.get("internet_access") == "yes" else 0,
        }

        features = np.array([[input_values[col] for col in FEATURE_COLUMNS]])
        features_scaled = scaler.transform(features)
        predicted_score = float(model.predict(features_scaled)[0])
        predicted_score = round(max(0, min(100, predicted_score)), 1)

        grade = score_to_grade(predicted_score)
        status = "Pass" if predicted_score >= 40 else "Fail"

        return render_template(
            "result.html",
            student_name=form.get("student_name", "Student"),
            score=predicted_score,
            grade=grade,
            status=status,
            inputs=input_values,
        )
    except (ValueError, TypeError) as exc:
        flash(f"Invalid input: {exc}")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
else:
    load_artifacts()
