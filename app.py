
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

labels = ["Normal", "Inner Fault", "Outer Fault", "Ball Fault"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        values = [float(v) for v in request.form.values()]
        pred = model.predict([values])[0]
        result = labels[pred]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
