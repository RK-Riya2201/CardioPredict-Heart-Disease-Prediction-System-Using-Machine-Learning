from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np  # For numerical operations

app = Flask(__name__)

# --- Configuration for the Prediction Model (Heart Disease) ---
MODEL_PATH = 'cardio.pkl'  # Ensure this file is in the same folder

# List of input features received from the form
INPUT_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Columns that the model was trained on (MUST match training data)
TRAINED_COLUMNS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca',
    'cp_1', 'cp_2', 'cp_3',
    'restecg_1', 'restecg_2',
    'thal_1', 'thal_2', 'thal_3'
]

# Load the trained model
model = pickle.load(open(MODEL_PATH, 'rb'))


# --- Routing ---
@app.route("/")
def home_page():
    """Renders the home page."""
    return render_template('home.html')


@app.route("/about")
def about_page():
    """Renders the about page."""
    return render_template('about.html')


@app.route("/predict")
def predict_page():
    """Renders the prediction input form."""
    return render_template('predict.html')


@app.route("/submit", methods=['POST'])
def submit_prediction():
    """Handles form submission, preprocesses data, and makes a Heart Disease prediction."""

    try:
        # Collect data from the form
        input_data = [
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal']),
        ]

        # Create DataFrame
        input_df = pd.DataFrame([input_data], columns=INPUT_COLUMNS)

        # One-Hot Encode categorical features
        input_encoded = pd.get_dummies(input_df, columns=['cp', 'restecg', 'thal'], drop_first=False)

        # Align columns with the training model
        for col in TRAINED_COLUMNS:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns exactly as model expects
        final_input = input_encoded[TRAINED_COLUMNS]

        # Make prediction
        prediction_result = model.predict(final_input)[0]

        # Interpret result
        if prediction_result == 1:
            text = 'High Risk of Heart Disease'
            detail = 'Based on your input, the model predicts a higher likelihood of heart disease.'
        else:
            text = 'Low Risk of Heart Disease'
            detail = 'Based on your input, the model predicts a lower likelihood of heart disease.'

        return render_template('submit.html',
                               prediction_text=text,
                               prediction_detail=detail)

    except Exception as e:
        # Handle unexpected input or processing errors gracefully
        return render_template('submit.html',
                               prediction_text='Error in prediction!',
                               prediction_detail=str(e))


if __name__ == "__main__":
    app.run(debug=True)
