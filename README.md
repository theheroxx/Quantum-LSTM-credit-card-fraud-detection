# Quantum-LSTM-credit-card-fraud-detection
# 🧠 Quantum-LSTM Fraud Detection

This project explores a hybrid quantum-classical approach to credit card fraud detection using a Long Short-Term Memory (LSTM) neural network combined with quantum feature encoding. The model could be improved, so also I'd be glad if you have any suggestions for me!

## 🚀 Overview

- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (from Kaggle)
- **Quantum Encoding**: 6 features mapped to quantum circuits using Qiskit’s `ZZFeatureMap`
- **Classical Features**: Additional transaction features processed with `StandardScaler`
- **Model**: PyTorch-based LSTM classifier operating on sequences of encoded transactions
- **Output**: Binary classification (Fraud / Legitimate)

---

## 📁 Features

- Balances the dataset using downsampling (equal number of fraud and legit samples)
- Extracts sequences of 10 transactions per sample
- Quantum feature embedding with amplitude probability extraction
- Hybrid input (quantum + classical): 70-dimensional vector per timestep
- Binary classification with BCE loss
- Classification metrics and confusion matrix visualization

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- Qiskit
- NumPy, Pandas, Scikit-learn, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
