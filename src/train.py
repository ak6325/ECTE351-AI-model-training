import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load & Scale
df = pd.read_csv('data/emg_data.csv')
X = df.iloc[:, 0:64].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train the Brain
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
model.fit(X_scaled, y)

# 3. Save for Phase 2
joblib.dump(model, 'models/refeel_brain.joblib')
joblib.dump(scaler, 'models/refeel_scaler.joblib')
print("Phase 1 Complete: Brain saved to /models/")