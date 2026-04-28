# Project Status & Blueprint

## Current Status
Step: 4 — Evaluation complete
Last updated: 2026-04-28

## Steps Checklist
- [x] Step 0: Project structure, README, this file
- [x] Step 1: Data exploration and EDA
- [x] Step 2: Data preprocessing
- [x] Step 3: Model training
- [x] Step 4: Model evaluation
- [x] Step 5: Retention recommendation agent
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
- Features used: 45

## Model Performance (fill after Step 4)
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.8197   | 0.6831    | 0.5952 | 0.6361   |
| Decision Tree       | 0.7154   |           |        | 0.4744   |
| Random Forest       | 0.7970   |           |        | 0.5531   |

## Best Model
Name: Logistic Regression
Saved as: models/best_model.pkl

## Retention Rules (fill after Step 5)
1. MonthlyCharges > 70       → Offer a discounted monthly plan
2. tenure < 12               → Assign a dedicated customer success manager
3. Contract == Month-to-month → Offer a 1-year or 2-year contract discount
4. TechSupport == No         → Offer free tech support upgrade for 3 months
5. Fiber optic + charges > 80 → Offer fiber loyalty discount
6. Always                    → Send a personalized retention email

## Known Issues / TODOs
[list anything incomplete]
