# ChurnGuard AI — Customer Churn Predictor

## 📋 Overview
ChurnGuard AI is an end-to-end Machine Learning project designed to predict customer churn and provide actionable, rule-based retention recommendations. It uses Logistic Regression, Decision Trees, and Random Forest models trained on the Kaggle Telco Customer Churn dataset. The best performing model is served through a sleek, dark-themed Flask web application.

## 🚀 Features
- **Data Preprocessing Pipeline**: Automated cleaning, missing value imputation, and one-hot encoding.
- **Machine Learning Models**: Automated training script that evaluates multiple algorithms and selects the best performer (Logistic Regression).
- **Evaluation Outputs**: Automatically generates Confusion Matrix and ROC Curve plots based on test data.
- **Rule-Based Recommendation Engine**: Cross-references customer attributes to generate personalized retention strategies.
- **Flask Web Interface**: A professional, responsive dark UI built with pure HTML/CSS to upload CSV datasets and view churn predictions instantly.

## 🛠️ Project Structure
\\	ext
customer-churn/
├── app.py                  # Flask web application entry point
├── requirements.txt        # Python dependencies
├── data/
│   └── raw/
│       └── telco_churn.csv # Raw telco dataset
├── models/                 # Saved pickle files (models, scalers, encoders)
├── notebooks/              # Jupyter notebooks for experimentation
├── outputs/
│   ├── eda/                # Exploratory Data Analysis plots
│   └── evaluation/         # Model evaluation plots (ROC, Confusion Matrix)
├── src/                    # Source code scripts
│   ├── eda.py              # Generates EDA plots
│   ├── preprocess.py       # Data cleaning & preprocessing logic
│   ├── train.py            # Model training & model saving logic
│   ├── evaluate.py         # Model evaluation & plot generation
│   └── recommend.py        # Rule-based retention logic
├── static/
│   └── style.css           # Frontend CSS styles (Dark UI)
└── templates/
    ├── index.html          # Webapp homepage / upload form
    └── results.html        # Prediction insights & recommendations page
\
## 💻 How to Run This Project

### 1. Prerequisites
- Python 3.8+ installed on your system.
- Git for cloning the repository.

### 2. Setup environment
Open your terminal and clone the repository:
\\ash
git clone https://github.com/Salih-Soomro/customer-churn.git
cd customer-churn
\
Create and activate a virtual environment:
\\ash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
\
Install dependencies:
\\ash
pip install -r requirements.txt
\
### 3. Run the Web Application
Start the Flask server:
\\ash
python app.py
\- Open your browser and navigate to **http://localhost:5000**.
- Upload the provided \data/raw/telco_churn.csv\ file (or any new CSV following the same format).
- Click **Run Churn Analysis** to view high-risk customers alongside personalized retention recommendations.

## 📊 How the Scripts Work (For Developers)
If you want to re-run the ML pipeline from scratch, use the following commands sequentially:

1. **Preprocess Data**:
   \\ash
   python src/preprocess.py
   \   *(Validates data loading, splitting, and scaling).*

2. **Train Models**:
   \\ash
   python src/train.py
   \   *(Evaluates Logistic Regression, Decision Tree, Random Forest. Saves the best model to \models/\).*

3. **Evaluate Model Performance**:
   \\ash
   python src/evaluate.py
   \   *(Calculates Accuracy, Precision, Recall, F1 Score, and saves plots to \outputs/evaluation/\).*

4. **Test the Recommendation Engine**:
   \\ash
   python src/recommend.py
   \   *(Checks if the baseline rules trigger correctly for dummy data).*
