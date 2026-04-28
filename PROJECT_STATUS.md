# Project Status & Blueprint

## Current Status
Step: 8 — PROJECT COMPLETE (Improvements Applied)
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
- Features used: 45

## Model Performance
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7509   | 0.5185    | 0.8284 | 0.6378   |
| Decision Tree       | 0.7154   | 0.4641    | 0.4853 | 0.4744   |
| Random Forest       | 0.7871   | 0.6377    | 0.4531 | 0.5298   |
| Gradient Boosting   | 0.8020   | 0.6526    | 0.5389 | 0.5903   |

## Best Model
Name: Logistic Regression (tuned via GridSearchCV, C=0.1, class_weight='balanced')
Saved as: models/best_model.pkl

## Improvements Applied
- class_weight='balanced' added to Logistic Regression and Random Forest
- GradientBoostingClassifier added as 4th model
- 5-fold cross-validation for all models
- GridSearchCV hyperparameter tuning for best model
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
The best performing model is saved and served via a Flask web interface.
Users can upload a CSV file, view churn predictions with probability scores
per customer, and see personalized retention recommendations for high-risk customers.

## Known Issues / TODOs
None — project is complete.
