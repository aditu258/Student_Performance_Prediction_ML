import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Use relative paths for model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "final_model", "preprocessor.pkl")

# Load trained model and preprocessor
model = joblib.load(MODEL_PATH)  # Ensure this file exists
preprocessor = joblib.load(PREPROCESSOR_PATH)  # Ensure this file exists

@app.route("/")
def home():
    return render_template("index.html")  # A simple HTML form for input (optional)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data (supports both Form and JSON)
        if request.is_json:
            input_data = request.get_json()  # If JSON request
        else:
            input_data = {
                "gender": request.form.get("gender"),
                "race_ethnicity": request.form.get("race_ethnicity"),
                "parental_level_of_education": request.form.get("parental_level_of_education"),
                "lunch": request.form.get("lunch"),
                "test_preparation_course": request.form.get("test_preparation_course"),
                "writing_score": float(request.form.get("writing_score")),
                "reading_score": float(request.form.get("reading_score"))
            }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Apply the same transformations as training
        df_transformed = preprocessor.transform(df)

        # Convert to NumPy array and reshape for single prediction
        input_array = np.array(df_transformed).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]

        return jsonify({"maths_score": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)  # Runs on port 8000, accessible from any device