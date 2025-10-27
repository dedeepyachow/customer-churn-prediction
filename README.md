# 🧠 Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn for a telecom company.  
This project covers data cleaning, exploratory data analysis (EDA), model building, evaluation, and feature importance — all in one professional setup.

---

## 🚀 Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Matplotlib**, **Seaborn** (for EDA)
- **Joblib** (for model persistence)
- **Jupyter Notebook**

---

## 📊 Project Workflow

### 1️⃣ Data Understanding & Cleaning
- Removed unnecessary columns (`customerID`)
- Handled missing values in `TotalCharges`
- Converted categorical variables to numeric using `LabelEncoder`

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized churn distribution and correlations
- Examined categorical variables’ influence on churn
- Identified key churn drivers like contract type and tenure  
📓 View notebook → [`notebooks/churn_analysis.ipynb`](notebooks/churn_analysis.ipynb)

### 3️⃣ Feature Engineering & Scaling
- Encoded categorical features
- Scaled numerical features using `StandardScaler`

### 4️⃣ Model Building & Evaluation
- Trained a **Random Forest Classifier**
- Evaluated using accuracy, precision, recall, and F1-score
- Feature importance visualization to explain predictions

---
