import pickle
from flask import Flask, request, jsonify, render_template
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

# --- NEW HELPER FUNCTION ---
# This function contains the core prediction logic, used by both the API and the web page.
def get_forecast_data(sku, days):
    """Selects a model and returns a future-only forecast DataFrame."""
    model = models[sku]
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    # Get the last date from the model's own training history
    last_history_date = model.history_dates.max()
    
    # Filter the forecast to keep only the dates AFTER the last training date
    future_forecast = forecast[forecast['ds'] > last_history_date]

    # Return only the necessary columns from the future-only data
    return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# Define the prediction endpoint for machine-to-machine communication (e.g., Power Automate)
@app.route('/predict', methods=['POST'])
def predict():
    if models is None:
        return jsonify({"error": "Models are not loaded, check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

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

    # Call the helper function to get the forecast
    response_data = get_forecast_data(sku, days)
    
    # Convert datetime to string for JSON compatibility
    response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
    
    # Convert dataframe to a list of dictionaries
    result = response_data.to_dict(orient='records')

    return jsonify(result)

# Define the endpoint for the human-readable web page
@app.route('/view_forecast', methods=['GET', 'POST'])
def view_forecast():
    # Default values for the form
    sku_input = "DAN-0003"
    days_input = 7
    forecast_result = None

    if request.method == 'POST':
        sku_input = request.form.get('sku')
        days_input = int(request.form.get('days_to_forecast'))

        if sku_input and sku_input in models:
            # Call the helper function to get the forecast
            response_data = get_forecast_data(sku_input, days_input)
            
            # Format the data for the template
            response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
            forecast_result = response_data.to_dict(orient='records')

    # Render the HTML page, passing in the data
    return render_template('result.html', 
                           forecast_data=forecast_result, 
                           sku_in=sku_input, 
                           days_in=days_input)

# Add a root endpoint to easily check if the service is running
@app.route('/')
def index():
    return f"Sales Forecast API is running with {len(models) if models else 0} models loaded.", 200