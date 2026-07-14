# Student Performance Prediction Dashboard

## Objective

The objective of this project is to predict student performance using Machine Learning. The project includes data preprocessing, feature scaling, model comparison, hyperparameter tuning, and an interactive Streamlit dashboard.

---

## Dataset

Student Performance Dataset

---

## Feature Engineering

- Data Cleaning
- Missing Value Handling
- Feature Scaling using StandardScaler

---

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

---

## Hyperparameter Tuning

GridSearchCV was used to find the best parameters for the Random Forest model.

---

## Best Model

Random Forest Classifier

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## Installation

Install all required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
streamlit run app.py
```

---

## Project Structure

```text
week4-ml-optimization/
│
├── app.py
├── dataset.csv
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── optimize_model.py
│   ├── save_model.py
│   └── predict.py
│
└── assets/
    └── screenshots/
```

---

## Dashboard

Run the application using:

```bash
streamlit run app.py
```

Take screenshots of the dashboard and save them inside the `assets/screenshots` folder.