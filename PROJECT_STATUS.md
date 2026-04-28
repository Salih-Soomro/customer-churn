# Project Status & Blueprint

## Current Status
Step: 1 — EDA complete
Last updated: 2026-04-28

## Steps Checklist
- [x] Step 0: Project structure, README, this file
- [x] Step 1: Data exploration and EDA
- [ ] Step 2: Data preprocessing
- [ ] Step 3: Model training
- [ ] Step 4: Model evaluation
- [ ] Step 5: Retention recommendation agent
- [ ] Step 6: Flask web app backend
- [ ] Step 7: HTML/CSS frontend
- [ ] Step 8: End-to-end testing and final cleanup

## Dataset Notes
- File: data/raw/telco_churn.csv
- Shape: 7043 rows × 21 columns
- Target column: Churn
- Churn distribution: No = 5174, Yes = 1869 (imbalanced)
- Columns: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
- Missing values: 0 reported (but TotalCharges has blank strings that become NaN after numeric conversion)
- Known issues: TotalCharges is stored as string (object) dtype — requires pd.to_numeric(errors="coerce") conversion
- Features used: [fill after preprocessing]

## Model Performance (fill after Step 4)
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression |          |           |        |          |
| Decision Tree       |          |           |        |          |
| Random Forest       |          |           |        |          |

## Best Model
Name: [fill after Step 4]
Saved as: models/best_model.pkl

## Retention Rules (fill after Step 5)
[document the if/else rules here]

## Known Issues / TODOs
[list anything incomplete]
