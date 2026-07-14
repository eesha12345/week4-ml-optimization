import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_grade(study_hours, attendance, participation, total_score):

    data = pd.DataFrame({
        "weekly_self_study_hours": [study_hours],
        "attendance_percentage": [attendance],
        "class_participation": [participation],
        "total_score": [total_score]
    })

    data = scaler.transform(data)

    prediction = model.predict(data)

    return prediction[0]