import pandas as pd


def load_and_preprocess(csv_path):
	customer_data = pd.read_csv(csv_path)
	customer_data = customer_data.drop(columns=["customerID"])
	customer_data["TotalCharges"] = pd.to_numeric(
		customer_data["TotalCharges"], errors="coerce"
	)
	median_value = customer_data["TotalCharges"].median()
	customer_data["TotalCharges"] = customer_data["TotalCharges"].fillna(
		median_value
	)
	customer_data["Churn"] = customer_data["Churn"].map({"Yes": 1, "No": 0})
	target = customer_data["Churn"]
	features = customer_data.drop(columns=["Churn"])
	features = pd.get_dummies(features)
	feature_names = features.columns.tolist()
	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(
		features, target, test_size=0.2, random_state=42
	)
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
	X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
		"data/raw/telco_churn.csv"
	)
	print("X_train shape:", X_train.shape)
	print("X_test shape:", X_test.shape)
	print("y_train shape:", y_train.shape)
	print("y_test shape:", y_test.shape)
	print("Number of features:", len(feature_names))
	print("Preprocessing complete.")

