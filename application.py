from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles the prediction request.
    For GET requests, it renders the home page.
    For POST requests, it processes the form data and returns the prediction.
    """
    if request.method == 'GET':
        # If the user navigates to /predict directly, show the form
        return render_template('index.html')
    else:
        # This block runs when the form is submitted (POST request)
        
        # Create a CustomData object with data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert the custom data to a DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame for Prediction:")
        print(pred_df)
        
        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        # Get the raw prediction from the model
        results = predict_pipeline.predict(pred_df)
        
        # --- FIX: Cap the prediction score ---
        # Ensure the predicted score does not go above 100 or below 0
        predicted_score = results[0]
        if predicted_score > 100:
            final_result = 100.0
        elif predicted_score < 0:
            final_result = 0.0
        else:
            final_result = predicted_score
        # ------------------------------------

        # Render the index.html page again, passing the capped and rounded result
        return render_template('index.html', results=round(final_result, 2))

# This block allows the script to be run directly
if __name__ == "__main__":
    # Runs the Flask app on host 0.0.0.0, making it accessible on the local network
    # Set debug=True for development to see errors and auto-reload
    app.run(host="0.0.0.0")

