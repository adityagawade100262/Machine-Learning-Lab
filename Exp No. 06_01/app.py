
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained Decision Tree model
with open("Loan.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Collect input values from HTML form
            applicant = {
                "Gender": request.form["Gender"],
                "Married": request.form["Married"],
                "Education": request.form["Education"],
                "ApplicantIncome_INR_monthly": float(
                    request.form["ApplicantIncome_INR_monthly"]
                ),
                "CoapplicantIncome_INR_monthly": float(
                    request.form["CoapplicantIncome_INR_monthly"]
                ),
                "Credit_Score": float(
                    request.form["Credit_Score"]
                ),
                "LoanAmount_INR": float(
                    request.form["LoanAmount_INR"]
                ),
                "Loan_Term_Months": float(
                    request.form["Loan_Term_Months"]
                ),
                "Property_Area": request.form["Property_Area"]
            }

            input_df = pd.DataFrame([applicant])

            # Encode categorical columns
            input_df = pd.get_dummies(
                input_df,
                drop_first=True,
                dtype=int
            )

            # Get the exact feature names from the trained model
            feature_names = model.feature_names_in_

            # Match training columns and their order
            input_df = input_df.reindex(
                columns=feature_names,
                fill_value=0
            )

            # Predict loan status
            prediction = model.predict(input_df)[0]

            # Probability of predicted class
            probabilities = model.predict_proba(input_df)[0]
            predicted_index = list(model.classes_).index(prediction)
            probability = round(
                probabilities[predicted_index] * 100, 2
            )

        except (ValueError, KeyError) as e:
            error = "Please enter valid values in all fields."

        except AttributeError:
            error = (
                "Model feature names are unavailable. "
                "Please retrain and save the model using "
                "a pandas DataFrame with column names."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
