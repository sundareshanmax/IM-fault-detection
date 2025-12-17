from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from scipy.stats import kurtosis, skew
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

labels = ["Normal", "Inner Fault", "Outer Fault", "Ball Fault"]

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal**2)),  # RMS
        kurtosis(signal),
        skew(signal)
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            signal = df.iloc[:, 0].values
            features = extract_features(signal)
            prediction = model.predict([features])[0]
            result = labels[prediction]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

