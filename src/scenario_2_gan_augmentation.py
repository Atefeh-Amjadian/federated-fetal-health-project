"""
Scenario 2 - GAN-based Data Augmentation (CTGAN)

Goal:
Balance class distribution by generating synthetic samples using CTGAN
on the TRAIN set only. The TEST set remains untouched.

Models are kept unchanged (LogReg + LightGBM) to isolate the effect of augmentation.

Disclaimer:
Educational/research use only. Not for clinical decision-making.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def main() -> None:
    # -------------------------
    # Load dataset
    # -------------------------
    df = pd.read_csv("../data/fetal_health.csv")

    X = df.drop(columns=["fetal_health"])
    y = df["fetal_health"] - 1  # {1,2,3} -> {0,1,2}

    data = X.copy()
    data["label"] = y

    # -------------------------
    # Train/Test split
    # -------------------------
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    print("Class distribution BEFORE GAN (train):")
    print(train_df["label"].value_counts())

    # -------------------------
    # CTGAN training
    # -------------------------
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=100,
        batch_size=500,
    )

    ctgan.fit(train_df)

    # -------------------------
    # Generate synthetic rows to balance classes
    # -------------------------
    counts = train_df["label"].value_counts()
    max_count = counts.max()

    synthetic_list = []
    for lbl in counts.index:
        n_new = max_count - counts[lbl]
        if n_new > 0:
            samples = ctgan.sample(n_new)
            samples["label"] = lbl  # enforce correct label
            synthetic_list.append(samples)

    synthetic_df = pd.concat(synthetic_list, ignore_index=True)
    aug_train = pd.concat([train_df, synthetic_df], ignore_index=True)

    print("\nClass distribution AFTER GAN (train):")
    print(aug_train["label"].value_counts())

    # -------------------------
    # Prepare ML data
    # -------------------------
    X_train = aug_train.drop(columns=["label"])
    y_train = aug_train["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # -------------------------
    # Model 1: Logistic Regression (needs scaling)
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)

    print("\n===== Logistic Regression + GAN =====")
    print("Accuracy:", accuracy_score(y_test, pred_lr))
    print("F1 macro:", f1_score(y_test, pred_lr, average="macro"))
    print(classification_report(y_test, pred_lr))

    # -------------------------
    # Model 2: LightGBM (no scaling needed)
    # -------------------------
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    lgbm.fit(X_train, y_train)
    pred_lgbm = lgbm.predict(X_test)

    print("\n===== LightGBM + GAN =====")
    print("Accuracy:", accuracy_score(y_test, pred_lgbm))
    print("F1 macro:", f1_score(y_test, pred_lgbm, average="macro"))
    print(classification_report(y_test, pred_lgbm))


if __name__ == "__main__":
    main()
