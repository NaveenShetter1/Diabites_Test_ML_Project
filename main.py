from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("save_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure form submission is not empty
        if not request.form:
            return "Error: No form data received!", 400

        # Extract input values and ensure all 8 fields exist
        expected_features = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']

        if any(feature not in request.form for feature in expected_features):
            return "Error: Missing one or more form fields!", 400

        # Convert input values to float
        input_data = [float(request.form[feature]) for feature in expected_features]

        # Scale input values
        input_scaled = scaler.transform([input_data])

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Display result
        message = "You have diabetes 😞" if prediction == 1 else "You don't have diabetes 😊"
        return render_template('home.html', prediction=message)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
