# AI-Driven Student Performance Prediction System

A web application that predicts a student's final academic score using a
machine-learning model (Random Forest Regression), built with **Python 3.14**,
**Flask**, **scikit-learn**, **HTML**, and **CSS**.

## Features

- Synthetic dataset generator for student academic/lifestyle data
- Random Forest Regression model (scikit-learn) with feature scaling
- Flask web app with a form to enter student details
- Instant prediction of final score, letter grade (A+ to F), and Pass/Fail status
- Feature-importance chart generated during training
- Clean, responsive HTML/CSS interface (no external CSS frameworks needed)

## Project Structure

```
student_performance_predictor/
│
├── app.py                     # Flask application (routes, prediction logic)
├── train_model.py             # Trains and saves the ML model
├── generate_dataset.py        # Generates the synthetic training dataset
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   └── student_data.csv       # Generated dataset (created on first run)
│
├── model/
│   ├── student_model.pkl      # Trained model (created by train_model.py)
│   └── scaler.pkl             # Fitted StandardScaler
│
├── static/
│   ├── style.css               # App styling
│   └── feature_importance.png  # Chart generated during training
│
└── templates/
    ├── base.html               # Shared layout
    ├── index.html              # Input form (home page)
    ├── result.html             # Prediction result page
    └── about.html              # About / how-it-works page
```

## Setup & Run (Python 3.14)

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python3.14 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Generate the dataset and train the model manually.**
   If you skip this step, `app.py` will do it automatically on first run.

   ```bash
   python generate_dataset.py
   python train_model.py
   ```

4. **Run the web app:**

   ```bash
   python app.py
   ```

5. Open your browser at **http://127.0.0.1:5000**

## How It Works

1. `generate_dataset.py` creates 1,500 synthetic student records with features:
   study hours, attendance, previous scores, assignments completed, sleep
   hours, extracurricular participation, parental support, and internet access.
2. `train_model.py` scales the features with `StandardScaler`, trains a
   `RandomForestRegressor` to predict `final_score` (0–100), evaluates it
   (MAE, RMSE, R²), and saves the model + scaler with `joblib`.
3. `app.py` loads the saved model/scaler and serves:
   - `GET /` — the input form
   - `POST /predict` — runs the prediction and renders the result page
   - `GET /about` — project explanation page

## Notes

- The dataset is synthetic (randomly generated with realistic relationships)
  for demonstration purposes. To use real data, replace `data/student_data.csv`
  with your own dataset that has the same column names, then re-run
  `train_model.py`.
- Model performance (R² ≈ 0.55 on synthetic data) will vary depending on the
  random seed and the strength of relationships in real-world data.
