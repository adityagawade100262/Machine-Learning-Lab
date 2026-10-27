from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("StdModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read input values
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])
    assignment_marks = float(request.form['assignment_marks'])

    # Prediction
    prediction = model.predict([[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks
    ]])

    # Get predicted performance
    performance = prediction[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Performance : {performance}"
    )


if __name__ == "__main__":
    app.run(debug=True)