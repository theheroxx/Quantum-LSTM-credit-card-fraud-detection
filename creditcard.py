import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# Load and shuffle
df = pd.read_csv("/content/drive/MyDrive/creditcard.csv") #your path
df = df.sample(frac=1, random_state=42)

# Balance dataset
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=len(fraud) * 4, random_state=42)
df_balanced = pd.concat([fraud, legit]).sample(frac=1, random_state=42)


#   6 quantum + 6 classical features
quantum_features = ['V14', 'V10', 'V12', 'V17', 'V3', 'V7']
classical_features = ['Time', 'V4', 'V11', 'V2', 'V9', 'V6']
all_features = quantum_features + classical_features

X_q = df_balanced[quantum_features].values
X_c = df_balanced[classical_features].values
y = df_balanced['Class'].values

# normalizing
scaler_q = StandardScaler()
scaler_c = StandardScaler()
X_q_scaled = scaler_q.fit_transform(X_q)
X_c_scaled = scaler_c.fit_transform(X_c)

#Quantum Feature Map for 6 qubits (2^6 = 64)
feature_map = ZZFeatureMap(feature_dimension=6, reps=2)

def quantum_embed(row):
    circuit = feature_map.assign_parameters(row, inplace=False)
    state = Statevector.from_instruction(circuit)
    probs = np.abs(state.data) ** 2
    return probs[:64]  # 2^6 = 64 amplitudes

print(" Embedding quantum features...")
X_q_encoded = np.array([quantum_embed(row) for row in X_q_scaled])

#concatenate with classical
X_combined = np.hstack((X_q_encoded, X_c_scaled))  # [64 + 6] = 70

# Simulate sequences of 10 transactions
seq_len = 10
num_samples = len(X_combined) // seq_len
X_seq = X_combined[:num_samples * seq_len].reshape(num_samples, seq_len, 70)
y_seq = y[:num_samples * seq_len].reshape(num_samples, seq_len)
y_seq = (y_seq.sum(axis=1) > 0).astype(int)

# Convert to tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

# tt split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
)

# LSTM 
class LSTMFraudQuantum(nn.Module):
    def __init__(self, input_size=70, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        return torch.sigmoid(self.fc2(out))

model = LSTMFraudQuantum()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("🧠 Training Quantum-LSTM...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test)
    y_pred = (preds > 0.5).int()
    y_true = y_test.int()

#Report
print(classification_report(y_true, y_pred))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Legit", "Fraud"])
plt.title("Confusion Matrix - Enhanced Quantum-LSTM FraudNet")
plt.grid(False)
plt.show()
