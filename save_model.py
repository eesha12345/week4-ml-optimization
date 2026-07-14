import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../dataset.csv").head(100000)

# Features
X = df[[
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score"
]]

# Target
y = df["grade"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the best model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/best_model.pkl")

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")