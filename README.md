# Student-Performance-Prediction
# 🎓 AI-Driven Student Performance Prediction System

A web app that predicts a student's final academic score using a machine
learning model, built with **Python 3.14**, **Flask**, **scikit-learn**,
**HTML**, and **CSS**.

## What It Does

Takes inputs like study hours, attendance, previous scores, assignments
completed, sleep hours, extracurricular activity, parental support, and
internet access — and predicts:
- Final score (0–100)
- Letter grade (A+ to F)
- Pass/Fail status

## Tech Stack

- **Backend:** Python 3.14, Flask
- **ML:** scikit-learn (Random Forest Regressor), pandas, NumPy, joblib
- **Frontend:** HTML5, CSS3 (Jinja2 templates)

## Project Structure
├── app.py # Flask app (routes + prediction logic)
├── train_model.py # Trains and saves the ML model
├── generate_dataset.py # Generates synthetic training data
├── requirements.txt
├── data/ # Dataset (CSV)
├── model/ # Trained model + scaler (.pkl)
├── static/ # CSS + charts
└── templates/ # HTML pages
## How to Run

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Model Performance

Trained on synthetic data (1,500 records) using a Random Forest Regressor:
- MAE: ~4.3
- RMSE: ~5.3
- R²: ~0.55

## Future Scope

- Train on real student data
- Add risk-level classification (High/Medium/Low)
- Add database + login for teachers/admins
- Deploy to cloud
