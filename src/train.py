import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess('data/raw/telco_churn.csv')

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'f1': f1}
        print(f'Model: {name}')
        print(f'  Accuracy : {acc:.4f}')
        print(f'  F1 Score : {f1:.4f}')

    best_model_name = max(results, key=lambda name: results[name]['f1'])
    best_model = models[best_model_name]

    print(f'\nBest Model: {best_model_name}')
    print(f'Best F1 Score: {results[best_model_name]['f1']:.4f}\n')

    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print('Saved: models/best_model.pkl')
    print('Saved: models/scaler.pkl')
    print('Saved: models/feature_names.pkl')
