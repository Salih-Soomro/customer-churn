import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    # Step 1 — Load data
    customer_data = pd.read_csv(csv_path)

    # Step 2 — Drop customerID column
    customer_data = customer_data.drop(columns=["customerID"])

    # Step 3 — Convert TotalCharges to numeric
    customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"], errors="coerce")

    # Step 4 — Fill missing TotalCharges with median
    median_value = customer_data["TotalCharges"].median()
    customer_data["TotalCharges"] = customer_data["TotalCharges"].fillna(median_value)

    # Step 5 — Encode target column Churn
    customer_data["Churn"] = customer_data["Churn"].map({"Yes": 1, "No": 0})

    # Step 6 — Separate features and target
    target = customer_data["Churn"]
    features = customer_data.drop(columns=["Churn"])

    # Step 7 — One-hot encode all categorical (object) columns
    features = pd.get_dummies(features)
    feature_names = features.columns.tolist()

    # Step 8 — Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Step 9 — Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 10 — Return
    return X_train, X_test, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess("data/raw/telco_churn.csv")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Number of features:", len(feature_names))
    print("Preprocessing complete.")
