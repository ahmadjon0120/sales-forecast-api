import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the dictionary of trained models once when the app starts
try:
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    print(f"Successfully loaded {len(models)} models.")
except FileNotFoundError:
    models = None
    print("FATAL ERROR: 'models.pkl' file not found. The API cannot serve predictions.")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure models were loaded
    if models is None:
        return jsonify({"error": "Models are not loaded, check server logs."}), 500

    # Get JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

    # Validate input parameters
    sku = data.get('sku')
    days = data.get('days_to_forecast')

    if not sku or not days:
        return jsonify({"error": "Missing 'sku' or 'days_to_forecast' in request."}), 400

    if sku not in models:
        return jsonify({"error": f"Model for SKU '{sku}' not found."}), 404

    try:
        days = int(days)
    except (ValueError, TypeError):
        return jsonify({"error": "'days_to_forecast' must be an integer."}), 400

    # Select the correct model and make a forecast
    model = models[sku]
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Prepare the response
    # Select only the future dates for the response
    response_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    
    # Convert datetime to string for JSON compatibility
    response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
    
    # Convert dataframe to a list of dictionaries
    result = response_data.to_dict(orient='records')

    return jsonify(result)

# Add a root endpoint to easily check if the service is running
@app.route('/')
def index():
    return f"Sales Forecast API is running with {len(models) if models else 0} models loaded.", 200