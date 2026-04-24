# Federated Fetal Health Classification

An independent machine learning project based on a public healthcare dataset from Kaggle.

This project explores different approaches for fetal health classification, including centralized learning, data augmentation, and federated learning.

---

## Dataset
Source (Kaggle):  
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

---

## Scenarios

### Scenario 1 — Centralized Baseline
- Multinomial Logistic Regression (interpretable baseline)
- LightGBM (high-performance model)

---

### Scenario 2 — GAN-Based Data Augmentation
- Synthetic data generation using CTGAN
- Improves class balance
- Leads to better classification performance

---

### Scenario 3 — Federated Learning
- 3 simulated clients (hospitals)
- Non-IID data split without overlap
- Manual FedAvg implementation
- No raw data sharing between clients

Results:
- Accuracy: 0.8732
- F1 macro: 0.7866

---

## How to Run

```bash
cd src
python scenario_1_baseline.py
```

You can also run other scenarios:

```bash
python scenario_2_gan.py
python scenario_3_federated.py
```

---

## Disclaimer
This project is intended for educational and research purposes only.  
It is not designed or validated for clinical or medical use.