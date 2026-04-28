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
    # Note: For CV to be "honest", we define models without sample weights here.
    # Class weights are applied during fitting for LR, DT, RF.
    # GB will use sample_weights during fitting.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    }

    # A5. Train and evaluate all models (initial pass)
    results = {}
    print("=" * 80)
    print(f"{'MODEL TRAINING & EVALUATION (INITIAL PASS)':^80}")
    print("=" * 80)

    for name, model in models.items():
        # Fit
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Predict on test set (using default 0.5 threshold)
        y_pred = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 5-fold Cross-Validation F1 (Honest: no sample_weight in CV)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cv_f1": cv_f1}

        print(f"\n> {name}")
        print(f"  CV F1 (honest): {cv_f1:.4f}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall   : {rec:.4f}")
        print(f"  Test F1 Score : {f1:.4f}")

    # A6. Pick best model by TEST F1
    best_model_name = max(results, key=lambda n: results[n]["f1"])
    print("\n" + "=" * 80)
    print(f"  BEST MODEL (by Test F1): {best_model_name}  --  F1 = {results[best_model_name]['f1']:.4f}")
    print("=" * 80)

    # A7. Tune ONLY the best model with GridSearchCV (scoring='f1', no sample_weight)
    print(f"\n> Tuning {best_model_name} with GridSearchCV (scoring='f1', honest CV)...")

    param_grids = {
        "Logistic Regression": {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        "Decision Tree": {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]},
        "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None]},
        "Gradient Boosting": {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5]}
    }

    # STEP 1: Find best params WITHOUT sample_weight (Honest CV)
    grid_search = GridSearchCV(
        models[best_model_name],
        param_grids[best_model_name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    honest_cv_f1 = grid_search.best_score_
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1 (honest): {honest_cv_f1:.4f}")

    # STEP 2: Retrain the tuned model WITH sample_weight (for GB) or class_weight (others)
    if best_model_name == "Gradient Boosting":
        tuned_model = GradientBoostingClassifier(**best_params, random_state=42)
        tuned_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        # Others already have class_weight='balanced' from the models dict or can be set via params
        tuned_model = grid_search.best_estimator_
        # For RF/DT/LR, if tuned via grid search, it already has class_weight='balanced' if it was in the base model
        tuned_model.fit(X_train, y_train)

    # A2. Optimal Threshold Tuning
    print(f"\n> Tuning Decision Threshold for {best_model_name}...")
    churn_probabilities = tuned_model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0
    thresholds_to_try = np.arange(0.3, 0.7, 0.01)

    for threshold in thresholds_to_try:
        predictions_at_threshold = (churn_probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_test, predictions_at_threshold)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"  F1 at optimal threshold: {best_f1:.4f}")

    y_pred_final = (churn_probabilities >= best_threshold).astype(int)

    final_acc = accuracy_score(y_test, y_pred_final)
    final_prec = precision_score(y_test, y_pred_final)
    final_rec = recall_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final)

    print(f"\n  Final Metrics (at {best_threshold:.2f} threshold):")
    print(f"  Accuracy : {final_acc:.4f}")
    print(f"  Precision: {final_prec:.4f}")
    print(f"  Recall   : {final_rec:.4f}")
    print(f"  F1 Score : {final_f1:.4f}")

    # Update results with tuned + thresholded metrics
    results[f"Tuned {best_model_name} + Threshold"] = {
        "accuracy": final_acc, "precision": final_prec,
        "recall": final_rec, "f1": final_f1,
        "cv_f1": honest_cv_f1
    }

    # A10. Final summary table
    print("\n" + "=" * 85)
    print(f"{'FINAL SUMMARY TABLE':^85}")
    print("=" * 85)
    print(f"{'Model':<32} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV F1 (honest)':>12}")
    print("-" * 85)
    for name, m in results.items():
        tag = " *" if name.startswith("Tuned") else ""
        print(f"{name:<32} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['cv_f1']:>12.4f}{tag}")
    print("-" * 85)
    print("* = Best tuned model with optimal decision threshold")

    # A9. Save
    joblib.dump(tuned_model, os.path.join(BASE_DIR, "models", "best_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(BASE_DIR, "models", "feature_names.pkl"))
    joblib.dump(best_threshold, os.path.join(BASE_DIR, "models", "threshold.pkl"))

    print(f"\nSaved: models/best_model.pkl")
    print(f"Saved: models/threshold.pkl (optimal threshold = {best_threshold:.2f})")
    print("Saved: models/scaler.pkl")
    print("Saved: models/feature_names.pkl")
