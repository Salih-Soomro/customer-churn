import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

# Ensure the script can import preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # A2. Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess("data/raw/telco_churn.csv")

    # A3. Compute sample weights for GradientBoosting (no class_weight param)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # A4. Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    }

    # A5. Train and evaluate all models
    results = {}
    print("=" * 75)
    print(f"{'MODEL TRAINING & EVALUATION':^75}")
    print("=" * 75)

    for name, model in models.items():
        # Fit
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 5-fold Cross-Validation F1
        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cv_f1": cv_f1}

        print(f"\n> {name}")
        print(f"  CV F1 (5-fold): {cv_f1:.4f}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall   : {rec:.4f}")
        print(f"  Test F1 Score : {f1:.4f}")

    # A6. Pick best model by TEST F1
    best_model_name = max(results, key=lambda n: results[n]["f1"])
    print("\n" + "=" * 75)
    print(f"  BEST MODEL (by Test F1): {best_model_name}  --  F1 = {results[best_model_name]['f1']:.4f}")
    print("=" * 75)

    # A7. Tune the best model with GridSearchCV (scoring='f1')
    print(f"\n> Tuning {best_model_name} with GridSearchCV (scoring='f1', 5-fold CV)...")

    param_grids = {
        "Logistic Regression": {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        "Decision Tree": {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]},
        "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None]},
        "Gradient Boosting": {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5]}
    }

    grid_search = GridSearchCV(
        models[best_model_name],
        param_grids[best_model_name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    if best_model_name == "Gradient Boosting":
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1 : {grid_search.best_score_:.4f}")

    # A8. Evaluate tuned model on test set
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)

    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_prec = precision_score(y_test, y_pred_tuned)
    tuned_rec = recall_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned)

    print(f"\n  Tuned Test Accuracy : {tuned_acc:.4f}")
    print(f"  Tuned Test Precision: {tuned_prec:.4f}")
    print(f"  Tuned Test Recall   : {tuned_rec:.4f}")
    print(f"  Tuned Test F1 Score : {tuned_f1:.4f}")

    # Update results for best model with tuned scores
    results[best_model_name] = {
        "accuracy": tuned_acc, "precision": tuned_prec,
        "recall": tuned_rec, "f1": tuned_f1,
        "cv_f1": grid_search.best_score_
    }

    # A10. Final summary table
    print("\n" + "=" * 75)
    print(f"{'FINAL SUMMARY TABLE':^75}")
    print("=" * 75)
    print(f"{'Model':<24} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV F1':>8}")
    print("-" * 75)
    for name, m in results.items():
        tag = " *" if name == best_model_name else ""
        print(f"{name:<24} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['cv_f1']:>8.4f}{tag}")
    print("-" * 75)
    print("* = Best model (tuned via GridSearchCV with scoring='f1')")

    # A9. Save
    joblib.dump(tuned_model, os.path.join(BASE_DIR, "models", "best_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(BASE_DIR, "models", "feature_names.pkl"))

    print(f"\nSaved: models/best_model.pkl  ({best_model_name}, tuned)")
    print("Saved: models/scaler.pkl")
    print("Saved: models/feature_names.pkl")
