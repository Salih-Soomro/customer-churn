import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash

# Fix 4 — Absolute paths instead of hardcoded relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the app can import src/recommend.py
sys.path.append(os.path.join(BASE_DIR, "src"))
from recommend import get_retention_recommendation

app = Flask(__name__)
app.secret_key = "churn-secret-key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Step 1 — Check if file was uploaded
    if "file" not in request.files:
        flash("No file uploaded. Please upload a CSV file.")
        return redirect(url_for("index"))

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        flash("No file selected. Please choose a CSV file.")
        return redirect(url_for("index"))

    if not uploaded_file.filename.endswith(".csv"):
        flash("Wrong file type. Please upload a .csv file only.")
        return redirect(url_for("index"))

    try:
        # Step 2 — Load the uploaded CSV
        customer_data = pd.read_csv(uploaded_file)

        # Step 3 — Keep a copy of original data for recommendations
        original_data = customer_data.copy()

        # Step 4 — Load saved model files (using absolute paths)
        best_model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
        
        # Fix B — Load optimal threshold
        threshold_path = os.path.join(BASE_DIR, "models", "threshold.pkl")
        if os.path.exists(threshold_path):
            optimal_threshold = joblib.load(threshold_path)
        else:
            optimal_threshold = 0.5

        # Step 5 — Preprocess the uploaded data
        # ... (rest of preprocessing remains the same)
        if "customerID" in customer_data.columns:
            customer_data = customer_data.drop(columns=["customerID"])

        if "Churn" in customer_data.columns:
            customer_data = customer_data.drop(columns=["Churn"])

        customer_data["TotalCharges"] = pd.to_numeric(
            customer_data["TotalCharges"], errors="coerce"
        )
        median_value = customer_data["TotalCharges"].median()
        customer_data["TotalCharges"] = customer_data["TotalCharges"].fillna(median_value)

        # Feature engineering (must match preprocess.py)
        customer_data["AvgMonthlyCharge"] = customer_data["TotalCharges"] / (customer_data["tenure"] + 1)
        customer_data["IsNewCustomer"] = (customer_data["tenure"] < 12).astype(int)
        service_columns = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        existing_service_cols = [col for col in service_columns if col in customer_data.columns]
        customer_data["HasMultipleServices"] = (
            customer_data[existing_service_cols]
            .apply(lambda row: (row == "Yes").sum(), axis=1)
        )

        customer_data = pd.get_dummies(customer_data)

        # Align columns to match exactly what the model was trained on
        customer_data = customer_data.reindex(columns=feature_names, fill_value=0)

        scaled_data = scaler.transform(customer_data)

        # Step 6 — Run predictions and get churn probabilities
        churn_probabilities = best_model.predict_proba(scaled_data)[:, 1]
        # Use optimal threshold
        predictions = (churn_probabilities >= optimal_threshold).astype(int)

        # Step 7 — Build results list
        results = []

        for i in range(len(predictions)):
            churn_predicted = int(predictions[i])

            if churn_predicted == 1:
                customer_row = original_data.iloc[i].to_dict()
                recommendations = get_retention_recommendation(customer_row)
            else:
                recommendations = []

            results.append({
                "customer_index": i + 1,
                "churn_predicted": churn_predicted,
                "churn_probability": round(float(churn_probabilities[i]) * 100, 1),
                "recommendations": recommendations
            })

        # Step 8 — Count churners and pass to template
        total_customers = len(results)
        total_churners = sum(1 for r in results if r["churn_predicted"] == 1)

        return render_template(
            "results.html",
            results=results,
            total_customers=total_customers,
            total_churners=total_churners
        )

    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
