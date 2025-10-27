from joblib import load
from src.data_preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Telco-Customer-Churn.csv")

    model = load("models/churn_model.joblib")
    y_pred = model.predict(X_test)

    print("Model Evaluation Results:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    evaluate_model()
