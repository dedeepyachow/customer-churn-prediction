from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from src.data_preprocessing import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Telco-Customer-Churn.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    dump(model, "models/churn_model.joblib")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
