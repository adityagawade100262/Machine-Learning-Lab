
from flask import Flask, render_template, request
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "KNNModel.joblib"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Load the previously trained KNN model.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Place KNNModel.joblib in the same folder as app.py."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            values = {}

            for feature in FEATURES:
                raw_value = request.form.get(feature, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter {feature}."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"Enter a valid number for {feature}."
                    )

                if value < 0:
                    raise ValueError(
                        f"{feature} cannot be negative."
                    )

                values[feature] = value
                form_data[feature] = raw_value

            # Basic input validation.
            if not values["Pregnancies"].is_integer():
                raise ValueError(
                    "Pregnancies must be a whole number."
                )

            if values["Pregnancies"] > 20:
                raise ValueError(
                    "Please check the pregnancies value."
                )

            if not values["Age"].is_integer():
                raise ValueError("Age must be a whole number.")

            if not 1 <= values["Age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")

            if values["Glucose"] <= 0:
                raise ValueError("Glucose must be greater than zero.")

            if values["BloodPressure"] <= 0:
                raise ValueError(
                    "Blood pressure must be greater than zero."
                )

            if values["BMI"] <= 0:
                raise ValueError("BMI must be greater than zero.")

            # Keep feature names and order consistent.
            input_df = pd.DataFrame(
                [[values[f] for f in FEATURES]],
                columns=FEATURES
            )

            # Predict using the saved KNN model.
            predicted_class = int(model.predict(input_df)[0])

            if predicted_class == 1:
                prediction = "Higher likelihood of diabetes"
            elif predicted_class == 0:
                prediction = "Lower likelihood of diabetes"
            else:
                prediction = f"Predicted class: {predicted_class}"

            # Probability is optional; not all saved models support it.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                if predicted_class in classes:
                    class_index = classes.index(predicted_class)
                    probability = round(
                        float(probabilities[class_index]) * 100,
                        2
                    )

        except ValueError as exc:
            error = str(exc)

        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check that the saved model "
                "uses the expected eight input features and "
                "required preprocessing."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
