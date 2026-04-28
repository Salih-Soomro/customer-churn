# Project Status & Blueprint

## Current Status
Step: 8 — PROJECT COMPLETE (Threshold Tuning & Honest CV Applied)
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
| Model                               | Accuracy | Precision | Recall | F1 Score | CV F1 (honest) |
|-------------------------------------|----------|-----------|--------|----------|----------------|
| Logistic Regression                 | 0.7622   | 0.5330    | 0.8231 | 0.6470   | 0.6226         |
| Decision Tree                       | 0.7466   | 0.5221    | 0.5067 | 0.5143   | 0.4859         |
| Random Forest                       | 0.7963   | 0.6680    | 0.4584 | 0.5437   | 0.5410         |
| Gradient Boosting                   | 0.7764   | 0.5551    | 0.7828 | 0.6496   | 0.5551         |
| **Tuned GB + Optimal Threshold**   | **0.7750**| **0.5498**| **0.8284**| **0.6610**| **0.5756**    |

## Best Model
Name: Gradient Boosting (tuned via GridSearchCV, Optimal Threshold applied)
Best params: learning_rate=0.05, max_depth=3, n_estimators=200
Optimal Threshold: 0.52
Saved as: models/best_model.pkl, models/threshold.pkl

## Improvements Applied
- **Threshold Tuning**: Optimal decision threshold (0.52) found to balance Precision/Recall and maximize F1.
- **Honest CV**: GridSearchCV scoring fixed to be unweighted to reflect real-world performance accurately.
- **Class Balancing**: `compute_sample_weight('balanced')` used for GradientBoosting retraining.
- **Feature Engineering**: AvgMonthlyCharge, IsNewCustomer, HasMultipleServices.
- **UI/UX**: Added Churn Probability, Filter Bar, and Pagination to results page.

## Retention Rules
1. MonthlyCharges > 70        → Offer a discounted monthly plan
2. tenure < 12                → Assign a dedicated customer success manager
3. Contract == Month-to-month → Offer a 1-year or 2-year contract discount
4. TechSupport == No          → Offer free tech support upgrade for 3 months
5. Fiber optic + charges > 80 → Offer fiber loyalty discount
6. Always                     → Send a personalized retention email

## Final Summary
This project is an AI-powered web application that predicts customer churn using a tuned Gradient Boosting model.
Users can upload a CSV file, view churn predictions with probability scores, and see personalized retention recommendations.
The model uses an optimal decision threshold of 0.52 to provide the best balance of Precision and Recall.

## Known Issues / TODOs
None — project is complete.
