from flask import Flask, render_template, request
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "KNNModel.joblib"

# The uploaded model is a Pipeline containing StandardScaler + KNeighborsClassifier.
# It expects these 30 Wisconsin Breast Cancer dataset features in this exact order.
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "KNNModel.joblib was not found. Put the model file beside app.py "
        "and rename it to KNNModel.joblib."
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
            values = []
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                if raw == "":
                    raise ValueError(f"Please enter a value for {feature.replace('_', ' ')}.")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"Please enter a valid number for {feature.replace('_', ' ')}.")
                if value < 0:
                    raise ValueError(f"{feature.replace('_', ' ').title()} cannot be negative.")
                values.append(value)
                form_data[feature] = raw

            input_df = pd.DataFrame([values], columns=FEATURES)
            predicted_class = str(model.predict(input_df)[0])

            if predicted_class == "M":
                prediction = "Malignant pattern predicted"
            elif predicted_class == "B":
                prediction = "Benign pattern predicted"
            else:
                prediction = f"Predicted class: {predicted_class}"

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = [str(c) for c in model.classes_]
                if predicted_class in classes:
                    probability = round(float(probs[classes.index(predicted_class)]) * 100, 2)

        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Confirm that KNNModel.joblib is the supplied "
                "30-feature breast-cancer KNN pipeline and that its dependencies "
                "are compatible."
            )

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        probability=probability,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
