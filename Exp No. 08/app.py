from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Convert message into numerical features
            message_vector = vectorizer.transform([message])

            # Predict message category
            prediction = model.predict(message_vector)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)