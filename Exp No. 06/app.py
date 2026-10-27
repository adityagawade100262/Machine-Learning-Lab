from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Decision Tree model
# Keep DTModel.pkl in the same folder as app.py
try:
    with open("DTModel.pkl", "rb") as f:
        model = pickle.load(f)
    print("Decision Tree model loaded successfully.")

except FileNotFoundError:
    raise FileNotFoundError(
        "DTModel.pkl not found. Place it in the same folder as app.py."
    )

except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")


# Feature names and order must match the training dataset
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read JSON sent by the webpage
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request. Please submit the form again."
            }), 400

        # Check that all required fields are present
        missing = [
            feature
            for feature in FEATURE_ORDER
            if feature not in data
            or data[feature] is None
            or str(data[feature]).strip() == ""
        ]

        if missing:
            return jsonify({
                "error": "Missing values for: " + ", ".join(missing)
            }), 400

        # Convert inputs to numbers
        try:
            features = [
                float(data[feature])
                for feature in FEATURE_ORDER
            ]
        except (ValueError, TypeError):
            return jsonify({
                "error": "All fields must contain valid numbers."
            }), 400

        # Reject NaN and infinity
        if not np.all(np.isfinite(features)):
            return jsonify({
                "error": "Inputs must be finite numbers."
            }), 400

        # Create a DataFrame with feature names.
        # This avoids the feature-name warning when the model
        # was trained using a DataFrame with these same columns.
        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_ORDER
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Calculate the highest predicted class probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = round(
                float(np.max(probabilities)) * 100,
                2
            )
        else:
            confidence = None

        # Convert NumPy scalar to a regular Python integer
        prediction = int(prediction)

        result = {
            "prediction": prediction,
            "label": (
                "Heart Disease Detected"
                if prediction == 1
                else "No Heart Disease Detected"
            ),
            "confidence": confidence
        }

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Please check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )