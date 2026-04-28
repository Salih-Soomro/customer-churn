# Customer Churn Prediction & Retention Recommendation System

## Overview
An AI-powered web application that predicts whether a telecom customer will churn
and recommends retention actions for high-risk customers.

## Team
- Uswa Safdar (23K-3017)
- Amna Kazi (23K-3022)
- Muhammad Salih Soomro (23I-2042)

## Folder Structure
customer-churn/
├── data/raw/          → raw CSV dataset
├── src/               → preprocessing, training, evaluation, recommendations
├── models/            → saved trained model files
├── outputs/           → EDA and evaluation plots
├── templates/         → HTML pages for the Flask web app
├── static/            → CSS styling
├── app.py             → Flask web server
├── requirements.txt   → Python dependencies
├── README.md          → project documentation
└── PROJECT_STATUS.md  → progress tracker

## How to Run
1. pip install -r requirements.txt
2. python src/train.py
3. python app.py
4. Open http://localhost:5000

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, flask, joblib

## Dataset
Kaggle Telco Customer Churn dataset (data/raw/telco_churn.csv)