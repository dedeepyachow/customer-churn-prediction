import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Replace ' ' with NaN in TotalCharges and convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "Churn":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Target encode
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature scaling
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
