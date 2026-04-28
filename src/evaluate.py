import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_preprocess

    os.makedirs('outputs/evaluation', exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess('data/raw/telco_churn.csv')
    best_model = joblib.load('models/best_model.pkl')

    y_pred = best_model.predict(X_test)
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred

    print('Model Evaluation Results')
    print('------------------------')
    print(f'Accuracy  : {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision : {precision_score(y_test, y_pred):.4f}')
    print(f'Recall    : {recall_score(y_test, y_pred):.4f}')
    print(f'F1 Score  : {f1_score(y_test, y_pred):.4f}')
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('outputs/evaluation/confusion_matrix.png')
    plt.close()
    print('Saved: outputs/evaluation/confusion_matrix.png')

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/evaluation/roc_curve.png')
    plt.close()
    print('Saved: outputs/evaluation/roc_curve.png')
