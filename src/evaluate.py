import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Ensure the script can import preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess

if __name__ == "__main__":
    # Create outputs folder for plots
    os.makedirs("outputs/evaluation", exist_ok=True)

    # 4b. Load data and model
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess("data/raw/telco_churn.csv")
    best_model = joblib.load("models/best_model.pkl")

    # Fix C — Load optimal threshold
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    threshold_path = os.path.join(BASE_DIR, "models", "threshold.pkl")
    if os.path.exists(threshold_path):
        optimal_threshold = joblib.load(threshold_path)
        print(f"Using optimal threshold: {optimal_threshold:.2f}")
    else:
        optimal_threshold = 0.5

    # 4c. Generate predictions
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= optimal_threshold).astype(int)
    except AttributeError:
        y_pred = best_model.predict(X_test)
        y_prob = y_pred

    # 4d. Print metrics
    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4e. Plot 1 — Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/evaluation/confusion_matrix.png")
    plt.close()
    print("Saved: outputs/evaluation/confusion_matrix.png")

    # 4f. Plot 2 — ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/evaluation/roc_curve.png")
    plt.close()
    print("Saved: outputs/evaluation/roc_curve.png")

    # 4g. Plot 3 — Feature Importance
    feature_names_list = list(feature_names)

    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models (RandomForest, GradientBoosting, DecisionTree)
        importances = best_model.feature_importances_
        title = "Top 15 Feature Importances"
    elif hasattr(best_model, 'coef_'):
        # Linear models (LogisticRegression)
        importances = np.abs(best_model.coef_[0])
        title = "Top 15 Feature Importances (Logistic Regression Coefficients)"
    else:
        importances = None
        title = None

    if importances is not None:
        # Get top 15 features
        indices = np.argsort(importances)[-15:]
        top_features = [feature_names_list[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 7))
        sns.barplot(x=top_importances, y=top_features, palette="Blues_d")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("outputs/evaluation/feature_importance.png")
        plt.close()
        print("Saved: outputs/evaluation/feature_importance.png")
    else:
        print("Warning: Best model has no feature_importances_ or coef_ attribute. Skipping feature importance plot.")
