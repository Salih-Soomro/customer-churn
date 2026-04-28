import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from recommend import get_retention_recommendation

app = Flask(__name__)
app.secret_key = "churn-secret-key"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
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
        customer_data = pd.read_csv(uploaded_file)
        original_data = customer_data.copy()

        best_model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")

        if "customerID" in customer_data.columns:
            customer_data = customer_data.drop(columns=["customerID"])

        if "Churn" in customer_data.columns:
            customer_data = customer_data.drop(columns=["Churn"])

        customer_data["TotalCharges"] = pd.to_numeric(
            customer_data["TotalCharges"], errors="coerce"
        )
        median_value = customer_data["TotalCharges"].median()
        customer_data["TotalCharges"] = customer_data["TotalCharges"].fillna(
            median_value
        )

        customer_data = pd.get_dummies(customer_data)

        # Align columns to match exactly what the model was trained on
        customer_data = customer_data.reindex(columns=feature_names, fill_value=0)

        scaled_data = scaler.transform(customer_data)

        predictions = best_model.predict(scaled_data)

        results = []

        for i in range(len(predictions)):
            churn_predicted = int(predictions[i])

            if churn_predicted == 1:
                customer_row = original_data.iloc[i].to_dict()
                recommendations = get_retention_recommendation(customer_row)
            else:
                recommendations = []

            results.append(
                {
                    "customer_index": i + 1,
                    "churn_predicted": churn_predicted,
                    "recommendations": recommendations,
                }
            )

        total_customers = len(results)
        total_churners = sum(1 for r in results if r["churn_predicted"] == 1)

        return render_template(
            "results.html",
            results=results,
            total_customers=total_customers,
            total_churners=total_churners,
        )

    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
 
