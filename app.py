from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate synthetic stock data for training
np.random.seed(42)
X = np.random.rand(1000, 3) * 100  # Random Open, High, Low
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + np.random.normal(0, 5, 1000)  # Simulated Close Price with noise

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Generated {len(X)} synthetic data points for training")

# Define pipelines for multiple models
pipelines = [
    ('Linear Regression', Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])),
    ('Random Forest', Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])),
    ('Gradient Boosting', Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])),
    ('SVR', Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])),
    ('XGBoost', Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(n_estimators=100, random_state=42))
    ]))
]

# Train and evaluate models to select the best one
best_pipeline = None
best_mse = float('inf')
best_model_name = None
model_scores = []

print("Model Performance Metrics:")
print("-" * 50)
for name, pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_scores.append((name, mse, r2))
    print(f"{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("-" * 50)
    if mse < best_mse:
        best_mse = mse
        best_pipeline = pipeline
        best_model_name = name

print(f"Best Model: {best_model_name} with MSE: {best_mse:.4f}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data from request
        symbol = data["stock"]  # Get stock symbol (e.g., RELIANCE.NS)
        open_price = float(data["open"])
        high = float(data["high"])
        low = float(data["low"])

        # Log input data
        logger.info(f"Received prediction request for {symbol}: Open={open_price}, High={high}, Low={low}")

        # Prepare input for model
        features = np.array([[open_price, high, low]])
        predicted_close = best_pipeline.predict(features)[0]  # Use best pipeline to predict
        logger.info(f"Predicted close price for {symbol}: {predicted_close}")

        # Determine recommendation
        threshold = 0.02  # 2% threshold for Buy/Sell decision
        if predicted_close > open_price * (1 + threshold):
            recommendation = "Buy"
        elif predicted_close < open_price * (1 - threshold):
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        return jsonify({
            "stock": symbol,
            "predicted_close": round(predicted_close, 2),
            "recommendation": recommendation,
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)