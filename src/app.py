
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "best_emg_model.pkl"
)

model = joblib.load(MODEL_PATH)

# =========================================
# APP SETUP
# =========================================

app = Flask(__name__)

gesture_names = {
    0: "Rock ✊",
    1: "Scissors ✌",
    2: "Paper ✋",
    3: "OK 👌"
}

# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():
    return render_template("index.html")

# =========================================
# PREDICTION ROUTE
# =========================================

@app.route("/predict", methods=["POST"])
def predict():

    gesture_names = {
        0: "rock",
        1: "scissors",
        2: "paper",
        3: "ok"
    }

    try:

        # Generate fake EMG sample
        sample = np.random.randn(1, 65)

        prediction = model.predict(sample)[0]

        predicted_gesture = gesture_names.get(
            prediction,
            "unknown"
        )

        return jsonify({
            "prediction": predicted_gesture
        })

    except Exception as e:

        return jsonify({
            "prediction": f"Error: {str(e)}"
        })

# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":
    app.run(debug=True)