import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import sys # Import the sys library to flush output

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

# --- DEBUG VERSION of get_forecast_data ---
def get_forecast_data(sku, days):
    """Selects a model and returns a future-only forecast DataFrame."""
    print(f"--- DEBUG: Getting forecast for SKU: {sku} for {days} days. ---", flush=True)
    model = models[sku]
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    last_history_date = model.history_dates.max()
    # --- CRITICAL DEBUG LINE ---
    print(f"--- DEBUG: Last history date for this model is: {last_history_date} ---", flush=True)

    future_forecast = forecast[forecast['ds'] > last_history_date]
    print(f"--- DEBUG: Shape of forecast BEFORE filtering: {forecast.shape} ---", flush=True)
    print(f"--- DEBUG: Shape of forecast AFTER filtering: {future_forecast.shape} ---", flush=True)
    
    return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


@app.route('/predict', methods=['POST'])
def predict():
    if models is None:
        return jsonify({"error": "Models are not loaded, check server logs."}), 500
    data = request.get_json()
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
    
    response_data = get_forecast_data(sku, days).copy()
    
    response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
    result = response_data.to_dict(orient='records')
    return jsonify(result)

# ... (the rest of your app.py file can remain the same) ...

@app.route('/view_forecast', methods=['GET', 'POST'])
def view_forecast():
    # ... (code for view_forecast) ...
    pass

@app.route('/')
def index():
    # ... (code for index) ...
    pass