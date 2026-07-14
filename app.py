import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page Settings
st.set_page_config(
    page_title="Student Performance Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
menu = st.sidebar.selectbox(
    "Select Page",
    [
        "Project Overview",
        "Dataset Information",
        "Data Preprocessing",
        "Model Comparison",
        "Hyperparameter Tuning",
        "Prediction"
    ]
)

# ---------------- Project Overview ----------------
if menu == "Project Overview":

    st.title("🎓 Student Performance Prediction Dashboard")

    st.write("""
This project predicts student grades using Machine Learning.

### Models Used
- Logistic Regression
- Decision Tree
- Random Forest

The best model was selected using GridSearchCV.
""")

# ---------------- Dataset ----------------
elif menu == "Dataset Information":

    st.title("Dataset Information")

    df = pd.read_csv("dataset.csv")

    st.write("First 5 Rows")

    st.dataframe(df.head())

    st.write("Dataset Shape")

    st.write(df.shape)

# ---------------- Data Preprocessing ----------------
elif menu == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing & Feature Engineering")
    st.write("Summary of steps taken to prepare the 1,000,000 rows for model training:")

    with st.expander("✔️ Missing values handled"):
        st.write("Checked for null values across all columns using `df.isnull().sum()`.")
        st.success("Result: 0 missing values found. Dataset was already clean!")

    with st.expander("✔️ Feature scaling completed"):
        st.write("To ensure uniform numerical distribution and handle text categories:")
        st.markdown("- **StandardScaler**: Applied to numerical features like study hours and attendance.")
        st.markdown("- **Label/One-Hot Encoding**: Applied to categorical columns.")

    with st.expander("✔️ Dataset cleaned"):
        st.write("Irrelevant tracking features like `student_id` were removed to prevent model bias.")
        st.code("df.drop(columns=['student_id'], inplace=True)")

    

# ---------------- Model Comparison ----------------

elif menu == "Model Comparison":
    st.title("📈 Model Comparison")
    st.write("Evaluation metrics for the three trained baseline models:")

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.9956, 0.9980, 0.9981],
        "Precision": [0.9940, 0.9975, 0.9980],
        "Recall": [0.9935, 0.9970, 0.9978],
        "F1 Score": [0.9937, 0.9972, 0.9979]
    })
    
    st.table(comparison)
    
    st.subheader("Visual Performance Metrics Comparison")
    st.bar_chart(comparison.set_index("Model"))

# ---------------- Hyperparameter Tuning ----------------
elif menu == "Hyperparameter Tuning":

    st.title("Hyperparameter Tuning")

    st.write("Best Parameters")

    st.code("""
max_depth = 10
n_estimators = 50
""")

    st.success("Best Accuracy : 0.99796")

# ---------------- Prediction ----------------
elif menu == "Prediction":
        st.title("🔮 Student Performance Live Prediction")
        st.write("Provide the student metrics below to evaluate the prediction outcome:")

        # 1. Input fields that prevent negative values using min_value=0.0
        study_hours = st.number_input("Weekly Self-Study Hours", min_value=0.0, max_value=168.0, value=15.0, step=1.0)
        attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        participation = st.number_input("Class Participation", min_value=0, max_value=10, value=5, step=1)
        total_score = st.number_input("Total Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

        if st.button("Predict"):
            # 2. Check if the user left critical input values completely at 0
            if total_score == 0.0 and attendance == 0.0:
                st.warning("⚠️ Please input valid student metrics above to generate a realistic prediction.")
            
            # 3. Automated rule: If total score is between 0 and 25, declare an instant Fail
            elif 0.0 <= total_score <= 25.0:
                st.error("📉 Predicted Grade: Fail (Score below baseline passing threshold)")
            
            # 4. If the score is above 25, run it through your trained ML model pipeline
            else:
                with st.spinner("Processing scaled feature vector through optimized pipeline..."):
                    # Explicitly round off the variables to clean up floating decimals
                    final_study = round(study_hours)
                    final_attendance = round(attendance)
                    final_total = round(total_score)

                    # Build the DataFrame structure for your pipeline
                    input_data = pd.DataFrame({
                        "weekly_self_study_hours": [final_study],
                        "attendance_percentage": [final_attendance],
                        "class_participation": [participation],
                        "total_score": [final_total]
                    })
                    
                    try:
                        # Transform data using your saved scaler and predict via your model
                        input_scaled = scaler.transform(input_data)
                        prediction = model.predict(input_scaled)
                        
                        # Extract the raw string prediction value safely out of the array format
                        final_grade = prediction[0]
                        st.success(f"🎉 Predicted Grade: {final_grade}")
                        
                    except Exception as e:
                        st.error(f"Execution Error: Ensure feature columns match your trained scaler exactly. Detail: {e}")




