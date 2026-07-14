import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest model
rf = RandomForestClassifier(random_state=42)

# Small parameter grid (faster)
params = {
    "n_estimators": [50],
    "max_depth": [10]
}

# Grid Search
grid = GridSearchCV(
    estimator=rf,
    param_grid=params,
    cv=3,
    scoring="accuracy"
)

# Train
grid.fit(X_train, y_train)

# Results
print("Best Parameters:")
print(grid.best_params_)

print("Best Accuracy:")
print(grid.best_score_)