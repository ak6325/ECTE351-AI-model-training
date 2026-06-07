# =========================================
# EDA PART 2 - FEATURE RELATIONSHIPS
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================
# 1. LOAD CLEANED DATA
# =========================================

df_clean = pd.read_csv("data/cleaned_emg.csv")

print("Dataset Shape:", df_clean.shape)

# =========================================
# 2. PREPARE FEATURES
# =========================================

gesture_names = {
    0: "rock",
    1: "scissors",
    2: "paper",
    3: "ok"
}

df_clean["gesture_name"] = df_clean["gesture"].map(gesture_names)

numeric_cols = df_clean.select_dtypes(include=np.number).columns
numeric_cols = [col for col in numeric_cols if col != "gesture"]

# =========================================
# 3. FEATURE CORRELATION
# =========================================

plt.figure(figsize=(14,10))

corr = df_clean[numeric_cols].corr()

sns.heatmap(corr, cmap="coolwarm")

plt.title("Correlation Between EMG Features")

plt.show()

# =========================================
# 4. FEATURES RELATED TO GESTURE
# =========================================

gesture_corr = df_clean[numeric_cols + ["gesture"]].corr()["gesture"]

gesture_corr = gesture_corr.drop("gesture")

gesture_corr = gesture_corr.abs().sort_values(ascending=False)

print("\nTop Features Related To Gesture:\n")

print(gesture_corr.head(15))

plt.figure(figsize=(10,5))

gesture_corr.head(15).plot(kind="bar")

plt.title("Top Features Related To Gesture")

plt.xlabel("Feature")

plt.ylabel("Absolute Correlation")

plt.xticks(rotation=45)

plt.show()

# =========================================
# 5. AVERAGE FEATURE VALUES PER GESTURE
# =========================================

gesture_means = df_clean.groupby("gesture_name")[numeric_cols].mean()

print("\nAverage Feature Values Per Gesture:\n")

print(gesture_means)

plt.figure(figsize=(14,6))

sns.heatmap(gesture_means, cmap="viridis")

plt.title("Average EMG Feature Values For Each Gesture")

plt.xlabel("Features")

plt.ylabel("Gesture")

plt.show()

# =========================================
# 6. BOXPLOTS OF IMPORTANT FEATURES
# =========================================

top_features = gesture_corr.head(4).index

for col in top_features:

    plt.figure(figsize=(8,5))

    sns.boxplot(x="gesture_name", y=col, data=df_clean)

    plt.title(f"{col} Distribution Across Gestures")

    plt.xlabel("Gesture")

    plt.ylabel("Feature Value")

    plt.show()

# =========================================
# 7. PCA VISUALIZATION
# =========================================

X = df_clean[numeric_cols]

y = df_clean["gesture_name"]

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

pca_df["gesture"] = y.values

plt.figure(figsize=(9,6))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="gesture",
    data=pca_df
)

plt.title("PCA Visualization Of Gesture Separation")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()

print("\nPCA Explained Variance:")

print(pca.explained_variance_ratio_)

# =========================================
# 8. PAIRPLOT OF IMPORTANT FEATURES
# =========================================

sample_df = df_clean.sample(
    min(700, len(df_clean)),
    random_state=1
)

sns.pairplot(
    sample_df,
    vars=top_features,
    hue="gesture_name"
)

plt.suptitle(
    "Feature Relationships Between Gestures",
    y=1.02
)

plt.show()