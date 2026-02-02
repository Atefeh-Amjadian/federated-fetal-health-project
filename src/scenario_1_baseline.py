"""
Scenario 1 - Baseline Centralized Model
Fetal Health Classification

This script implements a clean and interpretable baseline
using a public Kaggle dataset.

Models:
1. Multinomial Logistic Regression (interpretability)
2. LightGBM Classifier (performance)

Disclaimer:
This code is for educational and research purposes only.
It must not be used for medical or clinical decision-making.

Author: <Atefeh Amjadian>
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


# =========================
# Load Dataset
# =========================

# Dataset source:
# https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

df = pd.read_csv("../data/fetal_health.csv")

print("Dataset shape:", df.shape)


# =========================
# Feature / Target Split
# =========================

X = df.drop(columns=["fetal_health"])
y = df["fetal_health"] - 1  # convert labels from {1,2,3} to {0,1,2}

print("\nClass distribution:")
print(y.value_counts())


# =========================
# Train / Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Feature Scaling
# =========================

# Scaling is required for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# Model 1: Multinomial Logistic Regression
# ==================================================

print("\n===== Logistic Regression (Multinomial) =====")

lr_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")

print("Accuracy:", acc_lr)
print("F1 macro:", f1_lr)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# ==================================================
# Model 2: LightGBM Classifier
# ==================================================

print("\n===== LightGBM Classifier =====")

lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# LightGBM does not require feature scaling
lgbm_model.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_test)

acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm, average="macro")

print("Accuracy:", acc_lgbm)
print("F1 macro:", f1_lgbm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgbm))


# =========================
# Final Summary
# =========================

print("\n===== Summary =====")
print(f"Logistic Regression -> Accuracy: {acc_lr:.4f}, F1 macro: {f1_lr:.4f}")
print(f"LightGBM            -> Accuracy: {acc_lgbm:.4f}, F1 macro: {f1_lgbm:.4f}")

