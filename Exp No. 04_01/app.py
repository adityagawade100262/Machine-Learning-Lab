from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained Logistic Regression model
model = pickle.load(open('HeartDisease.pkl', 'rb'))

# Load StandardScaler
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from the form
    age = float(request.form['age'])
    glucose = float(request.form['glucose'])
    cholesterol = float(request.form['cholesterol'])
    systolic_bp = float(request.form['systolic_bp'])
    bmi = float(request.form['bmi'])

    # Keep the SAME order used during model training
    features = [[
        age,
        glucose,
        cholesterol,
        systolic_bp,
        bmi
    ]]

    # Scale the input using the saved scaler
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    # Display result
    if prediction == 'Yes':
        result = "❤️ Disease Detected"
    else:
        result = "✅ No Disease Detected"

    return render_template(
        'index.html',
        prediction=result
    )


if __name__ == '__main__':
    app.run(debug=True)