# =========================================
# PART 1 - MODEL DEVELOPMENT AND TRAINING
# =========================================

import os
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# =========================================
# 1. FILE PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data")

CLEAN_FILE = os.path.join(DATA_PATH, "cleaned_emg.csv")

MODEL_OUTPUT = os.path.join(DATA_PATH, "best_emg_model.pkl")

# =========================================
# 2. LOAD CLEANED DATASET
# =========================================

if not os.path.exists(CLEAN_FILE):

    raise FileNotFoundError(
        f"cleaned_emg.csv not found at:\n{CLEAN_FILE}"
    )

df_clean = pd.read_csv(CLEAN_FILE)

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("\nDataset Shape:")
print(df_clean.shape)

# =========================================
# 3. REMOVE NON-MODEL COLUMNS
# =========================================

if "gesture_name" in df_clean.columns:

    df_clean = df_clean.drop(columns=["gesture_name"])

# =========================================
# 4. CLASS DISTRIBUTION
# =========================================

print("\nClass Distribution:\n")

print(df_clean["gesture"].value_counts().sort_index())

gesture_names = {
    0: "rock",
    1: "scissors",
    2: "paper",
    3: "ok"
}

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df_clean["gesture"].map(gesture_names)
)

plt.title("Gesture Class Distribution")

plt.xlabel("Gesture")

plt.ylabel("Count")

plt.show()

# =========================================
# 5. FEATURE / TARGET SEPARATION
# =========================================

X = df_clean.drop(columns=["gesture"])

y = df_clean["gesture"]

# Convert column names to strings
X.columns = X.columns.astype(str)

print("\nNumber Of Features:", X.shape[1])

print("Target Classes:", sorted(y.unique()))

# =========================================
# 6. TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)

print("Testing Set Shape:", X_test.shape)

# =========================================
# 7. MODEL DEFINITIONS
# =========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=42
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ])
}

# =========================================
# 8. CROSS VALIDATION
# =========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

print("\n" + "=" * 50)
print("CROSS VALIDATION RESULTS")
print("=" * 50)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    std_score = scores.std()

    cv_results.append({
        "Model": name,
        "Mean CV Accuracy": mean_score,
        "Standard Deviation": std_score
    })

    print(f"\n{name}")

    print(f"Mean Accuracy: {mean_score:.4f}")

    print(f"Standard Deviation: {std_score:.4f}")

# =========================================
# 9. MODEL COMPARISON
# =========================================

cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    by="Mean CV Accuracy",
    ascending=False
)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(cv_results_df)

plt.figure(figsize=(10, 5))

sns.barplot(
    x="Mean CV Accuracy",
    y="Model",
    data=cv_results_df
)

plt.title("Model Performance Comparison")

plt.xlim(0, 1)

plt.show()

# =========================================
# 10. SELECT BEST MODEL
# =========================================

best_model_name = cv_results_df.iloc[0]["Model"]

best_model = models[best_model_name]

print("\nBest Model Selected:")

print(best_model_name)

# =========================================
# 11. TRAIN BEST MODEL
# =========================================

best_model.fit(X_train, y_train)

print("\nBest model trained successfully.")

# =========================================
# 12. TEST SET EVALUATION
# =========================================

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("TEST SET EVALUATION")
print("=" * 50)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

target_names = [
    gesture_names[label]
    for label in sorted(y.unique())
]

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=target_names
))

# =========================================
# 13. CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.xlabel("Predicted Gesture")

plt.ylabel("Actual Gesture")

plt.tight_layout()

plt.show()

# =========================================
# 14. FEATURE IMPORTANCE
# =========================================

trained_model = best_model.named_steps["model"]

if hasattr(trained_model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": trained_model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 Important Features:\n")

    print(feature_importance.head(15))

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance.head(15)
    )

    plt.title(
        f"Top 15 Feature Importances - {best_model_name}"
    )

    plt.tight_layout()

    plt.show()

else:

    print("\nFeature importance not available for this model.")

# =========================================
# 15. SAVE TRAINED MODEL
# =========================================

joblib.dump(
    best_model,
    MODEL_OUTPUT
)

print("\nBest model saved successfully.")

print("\nSaved Path:")

print(MODEL_OUTPUT)

# =========================================
# 16. SAMPLE PREDICTION
# =========================================

sample = X_test.iloc[[0]]

actual = y_test.iloc[0]

predicted = best_model.predict(sample)[0]

print("\n" + "=" * 50)
print("SAMPLE PREDICTION")
print("=" * 50)

print("\nActual Gesture:")

print(gesture_names.get(actual))

print("\nPredicted Gesture:")

print(gesture_names.get(predicted))

# =========================================
# END OF SCRIPT
# =========================================