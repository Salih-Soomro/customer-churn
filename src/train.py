import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# Ensure the script can import preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess("data/raw/telco_churn.csv")

    # Define models (class_weight='balanced' for LR and RF to handle class imbalance)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    }

    # Train and evaluate all models
    results = {}
    print("=" * 70)
    print(f"{'MODEL TRAINING & EVALUATION':^70}")
    print("=" * 70)

    for name, model in models.items():
        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"\n> {name}")
        print(f"  Cross-Val F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit on full training set
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall   : {rec:.4f}")
        print(f"  Test F1 Score : {f1:.4f}")

    # Find best model by test F1 score
    best_model_name = max(results, key=lambda name: results[name]["f1"])
    best_model = models[best_model_name]

    print("\n" + "=" * 70)
    print(f"  BEST MODEL (by F1): {best_model_name}  —  F1 = {results[best_model_name]['f1']:.4f}")
    print("=" * 70)

    # GridSearchCV tuning for the best model
    print(f"\n> Tuning {best_model_name} with GridSearchCV (5-fold CV)...")

    param_grids = {
        "Logistic Regression": {'C': [0.01, 0.1, 1, 10]},
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10]},
        "Gradient Boosting": {'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]},
        "Decision Tree": {'max_depth': [None, 5, 10, 15]}
    }

    grid_search = GridSearchCV(
        best_model,
        param_grids[best_model_name],
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1 : {grid_search.best_score_:.4f}")

    # Use the tuned estimator
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)

    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_prec = precision_score(y_test, y_pred_tuned)
    tuned_rec = recall_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned)

    print(f"  Tuned Test Accuracy : {tuned_acc:.4f}")
    print(f"  Tuned Test Precision: {tuned_prec:.4f}")
    print(f"  Tuned Test Recall   : {tuned_rec:.4f}")
    print(f"  Tuned Test F1 Score : {tuned_f1:.4f}")

    # Update the results dict with tuned scores
    results[best_model_name] = {"accuracy": tuned_acc, "precision": tuned_prec, "recall": tuned_rec, "f1": tuned_f1}

    # Final summary table
    print("\n" + "=" * 70)
    print(f"{'FINAL SUMMARY TABLE':^70}")
    print("=" * 70)
    print(f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for name, metrics in results.items():
        tag = " *" if name == best_model_name else ""
        print(f"{name:<24} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}{tag}")
    print("-" * 70)
    print(f"* = Best model (tuned via GridSearchCV)")

    # Save the tuned best model, scaler, and feature names
    joblib.dump(tuned_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    print(f"\nSaved: models/best_model.pkl  ({best_model_name}, tuned)")
    print("Saved: models/scaler.pkl")
    print("Saved: models/feature_names.pkl")
