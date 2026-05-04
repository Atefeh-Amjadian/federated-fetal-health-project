"""
Scenario 4 - GAN-Augmented Federated Learning

This script combines GAN-based data augmentation with a manual Federated Learning
setup using FedAvg. Synthetic samples are generated only from the training set.
The test set remains untouched.

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

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


SEED = 42
NUM_CLIENTS = 3
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.001
CTGAN_EPOCHS = 50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


df = pd.read_csv("../data/fetal_health.csv")

X = df.drop(columns=["fetal_health"])
y = (df["fetal_health"] - 1).astype(int)

data = X.copy()
data["label"] = y

print("Dataset shape:", data.shape)
print("Class distribution:")
print(data["label"].value_counts().sort_index())


train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=SEED,
    stratify=data["label"]
)

print("Train distribution before GAN:")
print(train_df["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

ctgan = CTGANSynthesizer(
    metadata=metadata,
    epochs=CTGAN_EPOCHS,
    batch_size=500
)

ctgan.fit(train_df)


counts = train_df["label"].value_counts()
max_count = counts.max()

synthetic_list = []

for label in counts.index:
    n_new = max_count - counts[label]

    if n_new > 0:
        samples = ctgan.sample(n_new)
        samples["label"] = label
        synthetic_list.append(samples)

synthetic_df = pd.concat(synthetic_list, ignore_index=True)

aug_train_df = pd.concat([train_df, synthetic_df], ignore_index=True)

print("Train distribution after GAN:")
print(aug_train_df["label"].value_counts().sort_index())


X_train_aug = aug_train_df.drop(columns=["label"]).values
y_train_aug = aug_train_df["label"].values.astype(int)

X_test = test_df.drop(columns=["label"]).values
y_test = test_df["label"].values.astype(int)

print("Augmented train shape:", X_train_aug.shape)
print("Test shape:", X_test.shape)


def non_iid_split_no_overlap(X, y, num_clients):
    client_indices = [[] for _ in range(num_clients)]

    proportions = np.array([
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.25, 0.25, 0.50],
    ])

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        start = 0
        for client_id in range(num_clients):
            if client_id == num_clients - 1:
                selected = cls_indices[start:]
            else:
                n = int(proportions[client_id, cls] * len(cls_indices))
                selected = cls_indices[start:start + n]
                start += n

            client_indices[client_id].extend(selected)

    client_data = []
    for idx in client_indices:
        idx = np.array(idx)
        np.random.shuffle(idx)
        client_data.append((X[idx], y[idx]))

    return client_data


client_datasets = non_iid_split_no_overlap(
    X_train_aug,
    y_train_aug,
    NUM_CLIENTS
)

for i, (_, y_c) in enumerate(client_datasets):
    print(f"Client {i} class distribution:", np.bincount(y_c, minlength=3))


scaler = StandardScaler()

X_train_aug_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test)

client_datasets = [
    (scaler.transform(X_c), y_c)
    for X_c, y_c in client_datasets
]


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)


def train_local(model, X, y, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    model.train()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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


def get_parameters(model):
    return [val.detach().cpu().numpy() for val in model.state_dict().values()]


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
            w * size for w, size in zip(weights_per_layer, client_sizes)
        )
        new_weights.append(layer_sum / total_size)

    return new_weights


input_dim = X_train_aug.shape[1]
global_model = Net(input_dim=input_dim)

for round_id in range(NUM_ROUNDS):
    print(f"\n--- Round {round_id + 1} ---")

    global_weights = get_parameters(global_model)

    client_weights = []
    client_sizes = []

    for X_c, y_c in client_datasets:
        local_model = Net(input_dim=input_dim)
        set_parameters(local_model, global_weights)

        train_local(local_model, X_c, y_c)

        client_weights.append(get_parameters(local_model))
        client_sizes.append(len(X_c))

    new_global_weights = fedavg(client_weights, client_sizes)
    set_parameters(global_model, new_global_weights)

    acc, f1, _ = evaluate_model(global_model, X_test_scaled, y_test)
    print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")


acc, f1, preds = evaluate_model(global_model, X_test_scaled, y_test)

print("\n=== Final GAN-Augmented Federated Learning Results ===")
print("Accuracy:", acc)
print("F1 macro:", f1)

print("\nClassification report:")
print(classification_report(y_test, preds))


result_text = f"""Scenario 4 - GAN-Augmented Federated Learning

Setup:
- CTGAN augmentation on training data only
- 3 clients
- non-IID split without overlap
- manual FedAvg
- local epochs: {LOCAL_EPOCHS}
- federated rounds: {NUM_ROUNDS}

Results:
Accuracy: {acc:.4f}
F1 macro: {f1:.4f}

Notes:
- Test set remained untouched.
- Synthetic samples were generated only from the training set.
- Raw client data was not shared during federated training.
"""

with open("../results/scenario_4_gan_federated.txt", "w") as f:
    f.write(result_text)

print(result_text)
print("Saved: ../results/scenario_4_gan_federated.txt")
