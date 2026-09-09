"""
One-time script to retrain and export the best model from the notebook.

Replicates the EXACT preprocessing and training pipeline from
Telco_Customer_Churn_Prediction.ipynb so that the saved model
produces identical results.

Usage:
    python train_and_export.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    csv_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Data cleaning (mirrors notebook cells 9-10)
    # ------------------------------------------------------------------
    df = df.drop("customerID", axis=1)

    # TotalCharges: blank strings → NaN → fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ------------------------------------------------------------------
    # 3. Encoding (mirrors notebook cell 12)
    # ------------------------------------------------------------------
    # Binary Yes/No → 1/0
    binary_map = {"Yes": 1, "No": 0}
    for col in ["Partner", "Dependents", "PaperlessBilling", "PhoneService", "Churn"]:
        df[col] = df[col].map(binary_map)

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Ordinal service columns
    service_map = {"Yes": 1, "No": 0, "No internet service": 2, "No phone service": 2}
    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].map(service_map)

    # Contract ordinal
    df["Contract"] = df["Contract"].map(
        {"Month-to-month": 1, "One year": 2, "Two year": 3}
    )

    # ------------------------------------------------------------------
    # 4. One-hot encode PaymentMethod & InternetService (notebook cell 13)
    # ------------------------------------------------------------------
    nominal_cols = ["PaymentMethod", "InternetService"]
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(df[nominal_cols])
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoder.get_feature_names_out(nominal_cols)
    )

    df = pd.concat(
        [df.drop(columns=nominal_cols).reset_index(drop=True), encoded_df], axis=1
    )

    # ------------------------------------------------------------------
    # 5. Feature / target split (notebook cell 21)
    # ------------------------------------------------------------------
    y = df["Churn"]
    X = df.drop(columns=["Churn", "gender", "PhoneService", "TotalCharges"])
    feature_columns = X.columns.tolist()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_columns}")

    # ------------------------------------------------------------------
    # 6. Train/test split + SMOTE (notebook cell 23)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE — train size: {X_train_sm.shape[0]}")

    # ------------------------------------------------------------------
    # 7. Train AdaBoost with RandomizedSearchCV (notebook cells 25-26)
    # ------------------------------------------------------------------
    param_distributions = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=AdaBoostClassifier(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train_sm, y_train_sm)

    best_model = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    # Refit on SMOTE-balanced training data (same as notebook cell 28)
    best_model.fit(X_train_sm, y_train_sm)

    # ------------------------------------------------------------------
    # 8. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    print(f"\nTest F1 (Churn): {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ------------------------------------------------------------------
    # 9. Save artifacts
    # ------------------------------------------------------------------
    os.makedirs("model", exist_ok=True)

    joblib.dump(best_model, os.path.join("model", "adaboost_model.joblib"))
    joblib.dump(encoder, os.path.join("model", "onehot_encoder.joblib"))

    with open(os.path.join("model", "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("\nDone! Saved:")
    print("   model/adaboost_model.joblib")
    print("   model/onehot_encoder.joblib")
    print("   model/feature_columns.json")


if __name__ == "__main__":
    main()
