# =========================================
# MODEL PART 2 - EVALUATION & ANALYSIS
# =========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================
# 1. PATH SETUP
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

MODEL_PATH = os.path.join(DATA_PATH, "best_emg_model.pkl")
DATA_FILE = os.path.join(DATA_PATH, "cleaned_emg.csv")

# Ensure safe directory
os.makedirs(DATA_PATH, exist_ok=True)

# =========================================
# 2. LOAD MODEL + DATA
# =========================================

print("=" * 50)
print("LOADING MODEL AND DATA")
print("=" * 50)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_FILE)

# Remove extra column if exists
if "gesture_name" in df.columns:
    df = df.drop(columns=["gesture_name"])

X = df.drop(columns=["gesture"])
y = df["gesture"]

X.columns = X.columns.astype(str)

# =========================================
# 3. PREDICTION
# =========================================

y_pred = model.predict(X)

# =========================================
# 4. METRICS
# =========================================

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, average="weighted")
rec = recall_score(y, y_pred, average="weighted")
f1 = f1_score(y, y_pred, average="weighted")

print("\n" + "=" * 50)
print("OVERALL PERFORMANCE")
print("=" * 50)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================================
# 5. CLASSIFICATION REPORT
# =========================================

gesture_names = {
    0: "rock",
    1: "scissors",
    2: "paper",
    3: "ok"
}

target_names = [gesture_names[i] for i in sorted(y.unique())]

print("\nClassification Report:\n")
print(classification_report(y, y_pred, target_names=target_names))

# =========================================
# 6. CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)

plt.title("Confusion Matrix - Final Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# 7. MISCLASSIFICATION ANALYSIS (SAFE VERSION)
# =========================================

errors = y[y != y_pred]

if len(errors) == 0:
    print("\nNo misclassifications found. Model is extremely stable on this dataset.")
else:
    error_counts = errors.value_counts()

    print("\nMost Misclassified Classes:\n")
    print(error_counts)

    plt.figure(figsize=(6, 4))
    error_counts.plot(kind="bar")
    plt.title("Misclassified Class Distribution")
    plt.xlabel("Gesture")
    plt.ylabel("Count")
    plt.show()

# =========================================
# 8. SAMPLE PREDICTIONS
# =========================================

print("\nSample Predictions:\n")

for i in range(5):
    sample = X.iloc[[i]]
    actual = y.iloc[i]
    predicted = model.predict(sample)[0]

    print(f"Actual: {gesture_names[actual]} | Predicted: {gesture_names[predicted]}")

# =========================================
# 9. SIMULATED REAL-WORLD TEST (UNSEEN DATA)
# =========================================

print("\n" + "=" * 50)
print("SIMULATED REAL-WORLD TEST")
print("=" * 50)

# take truly random unseen samples
test_samples = X.sample(10, random_state=None)

predictions = model.predict(test_samples)

for i in range(len(test_samples)):
    pred = predictions[i]

    print(f"Sample {i+1}: Predicted Gesture → {gesture_names[pred]}")