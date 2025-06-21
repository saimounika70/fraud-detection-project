# fraud-detection-project

# 💳 Credit Card Fraud Detection – Streamlit App + XGBoost Model

This project detects fraudulent credit card transactions using a machine learning model trained on the well-known Kaggle `creditcard.csv` dataset. The app is deployed via Streamlit and accepts new transaction data to predict fraud in real time.

---

## 📌 Project Overview

**Goal:** Build a fraud detection model that is fast, accurate, and usable via a web app.

**Tech Stack:**
- Python, Pandas, NumPy
- XGBoost
- Scikit-learn (for preprocessing and metrics)
- Streamlit (for deployment)
- Jupyter Notebook (for development)

---

## 🧠 Project Process

### 1. Preprocessing
- Removed irrelevant features like `Time`
- Scaled `Amount` and other features using `StandardScaler`
- Separated `X` (features) and `y` (target)
- Handled class imbalance via stratified train-test split

### 2. Exploratory Data Analysis
- Countplot to show class imbalance
- Correlation heatmap to inspect features
- Considered Isolation Forest and Local Outlier Factor for anomaly detection (but didn't include in final version)

### 3. Modeling
- Used **XGBoostClassifier** with `eval_metric='logloss'`
- Compared initial performance with confusion matrix and classification report
- Tracked metrics like precision, recall, and ROC-AUC

### 4. Saving Artifacts
- Saved both the model and the scaler using `joblib.dump(...)`
- Used in the Streamlit app

### 5. Streamlit Web App
- Users can upload a `.csv` file with new transactions
- App predicts whether each row is fraudulent (`1`) or not (`0`)
- Also outputs fraud probability per row

---

## ❓ What I Learned / Solved

This README also reflects the questions and confusions I solved during the project:

| Doubt | What I Learned |
|-------|----------------|
| Should I use `sort` or frequency counts for anagram detection? | Sorting is inefficient for large strings; frequency count + sliding window is better (O(n)). |
| What is `Isolation Forest`? | It's not a classifier like Random Forest. It’s an **unsupervised** outlier detector. |
| How to evaluate model performance? | Used **confusion matrix**, **classification report**, and **ROC-AUC**. |
| What is XGBoost? | A fast, regularized gradient boosting algorithm that’s often better than RandomForest on tabular data. |
| What does `streamlit run app.py` do? | It launches a browser app that can take CSV input and display model predictions. |
| What does “ValueError: Feature names should match” mean? | I was including the `Class` column during prediction — fixed by dropping it in the app before scaling. |
| What is `libomp` on macOS? | A dependency XGBoost needs to run in parallel — fixed by installing via `brew install libomp`. |

---

## 📁 Project Structure

fraud-detection-project/
├── project.ipynb # Full notebook with preprocessing + model
├── app.py # Streamlit web app for live predictions
├── xgb_fraud_model.pkl # Trained XGBoost model
├── scaler.pkl # Saved StandardScaler
├── test.csv # Sample input to test the app
├── project_report.pdf # 2-page final project report
├── README.md # This file


---

## ▶️ How to Run the App Locally

1. Clone the repo or download the folder
2. Install dependencies:


pip install streamlit scikit-learn xgboost pandas numpy joblib
Run the app:
streamlit run app.py
Upload a .csv file in the browser UI (must match training features)


📝 Sample Input Format (test.csv)

V1,V2,V3,...,V28,Amount
-1.3,0.5,-2.1,...,0.13,149.62
...
Don't include Time or Class columns.

---


🙌 Credits

Dataset: Kaggle Credit Card Fraud Detection
XGBoost by Tianqi Chen
Streamlit team for open-source UI
