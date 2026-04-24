"""
Scenario 3 - Federated Learning with Manual FedAvg

This script simulates federated learning for fetal health classification.
Three clients represent hospitals. Each client trains locally, and the
server aggregates model weights using FedAvg.

Educational/research use only. Not for clinical use.
"""

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 42
NUM_CLIENTS = 3
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# Load data
# =========================

df = pd.read_csv("../data/fetal_health.csv")

X = df.drop(columns=["fetal_health"]).values
y = (df["fetal_health"].values - 1).astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)


# =========================
# Non-IID split without overlap
# =========================

def non_iid_split_no_overlap(X, y, num_clients):
    client_data = []
    classes = np.unique(y)

    remaining_indices = list(np.arange(len(y)))
    random.shuffle(remaining_indices)

    for i in range(num_clients):
        dominant_class = classes[i % len(classes)]

        dominant_idx = [
            idx for idx in remaining_indices
            if y[idx] == dominant_class
        ]

        n_dom = int(0.6 * len(dominant_idx))
        selected_dom = np.random.choice(
            dominant_idx,
            size=n_dom,
            replace=False,
        )

        remaining_indices = [
            idx for idx in remaining_indices
            if idx not in selected_dom
        ]

        n_other = int(0.2 * len(remaining_indices))
        selected_other = np.random.choice(
            remaining_indices,
            size=n_other,
            replace=False,
        )

        remaining_indices = [
            idx for idx in remaining_indices
            if idx not in selected_other
        ]

        client_idx = np.concatenate([selected_dom, selected_other])
        client_data.append((X[client_idx], y[client_idx]))

    return client_data


client_datasets = non_iid_split_no_overlap(
    X_train_full,
    y_train_full,
    NUM_CLIENTS,
)

print("Client distributions:")
for i, (_, y_c) in enumerate(client_datasets):
    print(f"Client {i}:", np.bincount(y_c, minlength=3))


# =========================
# Scaling
# =========================

scaler = StandardScaler()
scaler.fit(X_train_full)

X_test_scaled = scaler.transform(X_test)

client_datasets = [
    (scaler.transform(X_c), y_c)
    for X_c, y_c in client_datasets
]


# =========================
# Model
# =========================

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.model(x)


def make_class_weights(y):
    counts = np.bincount(y, minlength=3)
    weights = len(y) / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_local(model, X, y, epochs=3, lr=0.001):
    model.train()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(y))

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()


def evaluate_model(model, X, y):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)

    preds = torch.argmax(outputs, dim=1).numpy()

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    return acc, f1, preds


# =========================
# FedAvg helpers
# =========================

def get_parameters(model):
    return [
        val.detach().cpu().numpy()
        for val in model.state_dict().values()
    ]


def set_parameters(model, parameters):
    state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in zip(state_dict.keys(), parameters):
        new_state_dict[key] = torch.tensor(value)

    model.load_state_dict(new_state_dict)


def fedavg(client_weights, client_sizes):
    total_size = sum(client_sizes)
    new_weights = []

    for weights_per_layer in zip(*client_weights):
        layer_sum = sum(
            w * size
            for w, size in zip(weights_per_layer, client_sizes)
        )
        new_weights.append(layer_sum / total_size)

    return new_weights


# =========================
# Federated training
# =========================

input_dim = X.shape[1]
global_model = Net(input_dim=input_dim)

for round_id in range(NUM_ROUNDS):
    print(f"\n--- Round {round_id + 1} ---")

    global_weights = get_parameters(global_model)

    client_weights = []
    client_sizes = []

    for X_c, y_c in client_datasets:
        local_model = Net(input_dim=input_dim)
        set_parameters(local_model, global_weights)

        train_local(
            local_model,
            X_c,
            y_c,
            epochs=LOCAL_EPOCHS,
        )

        client_weights.append(get_parameters(local_model))
        client_sizes.append(len(X_c))

    new_global_weights = fedavg(client_weights, client_sizes)
    set_parameters(global_model, new_global_weights)

    acc, f1, _ = evaluate_model(global_model, X_test_scaled, y_test)
    print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")


# =========================
# Final evaluation
# =========================

acc, f1, preds = evaluate_model(global_model, X_test_scaled, y_test)

print("\n=== Final Federated Learning Results ===")
print("Accuracy:", acc)
print("F1 macro:", f1)

print("\nClassification report:")
print(classification_report(y_test, preds))


# =========================
# Save results
# =========================

result_text = f"""Scenario 3 - Federated Learning

Setup:
- 3 clients
- non-IID split without overlap
- manual FedAvg
- local epochs: {LOCAL_EPOCHS}
- federated rounds: {NUM_ROUNDS}
- class-weighted loss

Results:
Accuracy: {acc:.4f}
F1 macro: {f1:.4f}

Notes:
- Test set remained untouched.
- Each client used only its own local data.
- Raw data was not shared between clients.
"""

with open("../results/scenario_3_federated.txt", "w") as f:
    f.write(result_text)

print("\nSaved: ../results/scenario_3_federated.txt")