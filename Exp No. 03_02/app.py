from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("MedicalInsuranceModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])

    sex = request.form['sex']
    smoker = request.form['smoker']
    region = request.form['region']

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }])

    # Make prediction
    prediction = model.predict(input_data)

    prediction_text = (
        f"Predicted Medical Insurance Cost: "
        f"{prediction[0]:.2f}"
    )

    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)