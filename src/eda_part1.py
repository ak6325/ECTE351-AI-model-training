# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 2. LOAD AND MERGE DATA
# ================================
data_path = "../data"   # path relative to src folder

files = ["0.csv", "1.csv", "2.csv", "3.csv"]
dataframes = []

for file in files:
    df = pd.read_csv(os.path.join(data_path, file), header=None)
    df["gesture"] = int(file[0])  # label from filename (0,1,2,3)
    dataframes.append(df)

# Combine all into one dataset
df = pd.concat(dataframes, ignore_index=True)

print("Combined dataset shape:", df.shape)

# ================================
# 3. BASIC UNDERSTANDING
# ================================
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ================================
# 4. MISSING VALUES
# ================================
print("\nMissing Values:\n", df.isnull().sum())

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# ================================
# 5. DUPLICATES
# ================================
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ================================
# 6. UNIVARIATE ANALYSIS
# ================================
numeric_cols = df.select_dtypes(include=np.number).columns

# Remove target for plotting distributions
numeric_cols = [col for col in numeric_cols if col != "gesture"]

df[numeric_cols].hist(figsize=(12,10), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title("Outlier Detection")
plt.show()

# ================================
# 7. OUTLIER HANDLING (IQR)
# ================================
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | 
                  (data > (Q3 + 1.5 * IQR))).any(axis=1)]

df_clean = df.copy()
df_clean[numeric_cols] = remove_outliers_iqr(df_clean[numeric_cols])

# Drop rows that became NaN after filtering
df_clean = df_clean.dropna()

print("\nShape after outlier removal:", df_clean.shape)

# ================================
# 8. NORMALIZATION
# ================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

# ================================
# 9. FINAL CHECK
# ================================
print("\nFinal Cleaned Data Summary:")
print(df_clean.describe())

# ================================
# 10. SAVE CLEAN DATA
# ================================
output_path = "../data/cleaned_emg.csv"
df_clean.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)