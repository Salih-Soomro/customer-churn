# Project Status & Blueprint

## Current Status
Step: 8 — PROJECT COMPLETE (Accuracy Fix Applied)
Last updated: 2026-04-28

## Steps Checklist
- [x] Step 0: Project structure, README, this file
- [x] Step 1: Data exploration and EDA
- [x] Step 2: Data preprocessing
- [x] Step 3: Model training
- [x] Step 4: Model evaluation
- [x] Step 5: Retention recommendation agent
- [x] Step 6: Flask web app backend
- [x] Step 7: HTML/CSS frontend
- [x] Step 8: End-to-end testing and final cleanup

## Dataset Notes
- File: data/raw/telco_churn.csv
- Target column: Churn
- Known issues: TotalCharges stored as string, converted to numeric
- Engineered features: AvgMonthlyCharge, IsNewCustomer, HasMultipleServices
- Features used: 48+

## Model Performance
| Model               | Accuracy | Precision | Recall | F1 Score | CV F1  |
|---------------------|----------|-----------|--------|----------|--------|
| Logistic Regression | 0.7622   | 0.5330    | 0.8231 | 0.6470   | 0.6226 |
| Decision Tree       | 0.7466   | 0.5221    | 0.5067 | 0.5143   | 0.4859 |
| Random Forest       | 0.7963   | 0.6680    | 0.4584 | 0.5437   | 0.5410 |
| Gradient Boosting   | 0.7622   | 0.5321    | 0.8445 | 0.6528   | 0.7730 |

## Best Model
Name: Gradient Boosting (tuned via GridSearchCV with scoring='f1')
Best params: learning_rate=0.05, max_depth=3, n_estimators=100
Saved as: models/best_model.pkl

## Improvements Applied
- class_weight='balanced' added to LR, DT, and RF
- compute_sample_weight('balanced') used for GradientBoosting
- GridSearchCV scoring corrected to 'f1' (was defaulting to accuracy)
- Feature engineering: AvgMonthlyCharge, IsNewCustomer, HasMultipleServices
- 5-fold cross-validation for all models
- Expanded hyperparameter grids for GridSearchCV
- Churn probability display (predict_proba) in results
- Table filtering (Show All / High Risk Only)
- Pagination (50 rows per page)
- Feature importance chart saved to outputs/evaluation/feature_importance.png
- Hardcoded paths replaced with absolute paths (BASE_DIR)

## Retention Rules
1. MonthlyCharges > 70        → Offer a discounted monthly plan
2. tenure < 12                → Assign a dedicated customer success manager
3. Contract == Month-to-month → Offer a 1-year or 2-year contract discount
4. TechSupport == No          → Offer free tech support upgrade for 3 months
5. Fiber optic + charges > 80 → Offer fiber loyalty discount
6. Always                     → Send a personalized retention email

## Final Summary
This project is an AI-powered web application that predicts customer churn
using four machine learning models trained on the Kaggle Telco dataset.
The best performing model (Gradient Boosting) is saved and served via a Flask
web interface. Users can upload a CSV file, view churn predictions with
probability scores per customer, and see personalized retention recommendations
for high-risk customers.

## Known Issues / TODOs
None — project is complete.
