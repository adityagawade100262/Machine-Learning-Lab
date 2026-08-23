from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
data = pickle.load(open("BCModel.pkl", "rb"))

model = data["model"]
scaler = data["scaler"]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read user inputs
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Create input data
    input_data = np.array([[cgpa, iq]])

    # Scale input using the SAME scaler used during training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)

    print("Input:", input_data)
    print("Scaled Input:", input_scaled)
    print("Prediction:", prediction)
    print("Probability:", probability)

    # Convert prediction to text
    if prediction[0] == 1:
        result = "Student is Likely to be PLACED"
    else:
        result = "Student is NOT Likely to be PLACED"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)