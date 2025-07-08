STOCK PRICE PREDICTION PROJECT
=============================

PROJECT OVERVIEW:
-----------------
This project predicts stock closing prices based on input features such as Open, High, and Low prices using machine learning regression models. It provides an end-to-end pipeline from synthetic data generation to a web-based deployment using Flask. The goal is to create a user-friendly interface where users can select a stock and input daily price data to receive a predicted closing price and a Buy/Sell/Hold recommendation.

The system trains multiple regression models, evaluates their performance, and selects the best-performing model for predictions. The web application features a sleek, modern UI with a glass-effect design and interactive inputs for real-time predictions.

FILES INCLUDED:
---------------
1. app.py – Flask web application code for model training and prediction
2. requirements.txt – List of required Python libraries
3. Dockerfile – Docker configuration for containerizing the application
4. index.html – HTML template for the web interface
5. script.js – JavaScript code for handling user inputs and API calls
6. style.css – CSS styles for the web interface
7. README.txt – This file

DATASET DETAILS:
----------------
Source: Synthetic data generated within app.py
Description: The dataset is programmatically created with 1000 samples, consisting of three features (Open, High, Low prices) and a target variable (Close price). The Close price is simulated as a linear combination of the input features with added Gaussian noise for realism.

Key Features Used:
- Open: Stock opening price
- High: Stock highest price of the day
- Low: Stock lowest price of the day

Target Variable:
- Close: Stock closing price

MODEL INFORMATION:
------------------
Algorithms Evaluated:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor

Steps Involved:
- Data Generation: Synthetic dataset creation with random Open, High, Low prices and simulated Close prices
- Data Preprocessing: Feature scaling using StandardScaler
- Model Training: Training multiple models using scikit-learn pipelines
- Model Evaluation: Comparing models based on Mean Squared Error (MSE) and R-squared (R²) scores
- Model Selection: Automatically selecting the best-performing model based on the lowest MSE

Evaluation Metrics:
- Mean Squared Error (MSE)
- R-squared (R²) Score

The best model is chosen dynamically during runtime and used for predictions.

WEB APPLICATION FEATURES:
------------------------
Developed using Flask
Title: Stock Price Prediction
User Inputs:
- Stock Symbol: Dropdown with a list of major Indian stocks (e.g., RELIANCE, TCS, HDFCBANK)
- Open Price: Numeric input for the stock's opening price
- High Price: Numeric input for the stock's highest price
- Low Price: Numeric input for the stock's lowest price

Output:
- Predicted Close Price: Estimated closing price in INR (₹)
- Recommendation: Buy, Sell, or Hold based on a 2% threshold relative to the Open price
- Visual Feedback: Color-coded recommendation (Green for Buy, Red for Sell, Black for Hold)

UI Features:
- Glass-effect container with a blurred background
- Neon-styled input fields and button with glow effects
- Responsive design with a professional stock market-themed background image

SETUP INSTRUCTIONS:
-------------------
1. Install required Python libraries:
   - flask
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - requests
   - xgboost

   You can install them using pip:
   pip install -r requirements.txt

2. Ensure the following files are in the same directory:
   - app.py
   - requirements.txt
   - Dockerfile
   - index.html
   - script.js
   - style.css
   - A static folder containing the background image (stock1.jpg)

3. Run the Flask application:
   python app.py

4. Alternatively, build and run using Docker:
   docker build -t stock-prediction .
   docker run -p 5000:5000 stock-prediction

5. Access the application in your browser at:
   http://localhost:5000

6. Use the dropdown to select a stock, enter the Open, High, and Low prices, and click "Predict" to get the estimated closing price and recommendation.

PROJECT OUTPUT:
---------------
- Real-time stock price predictions based on user inputs
- A modern, visually appealing web interface with interactive inputs
- Accurate predictions using the best-performing machine learning model
- Clear Buy/Sell/Hold recommendations for trading decisions

CONCLUSION:
-----------
This project showcases the application of machine learning in financial markets by predicting stock closing prices using regression models. It integrates data science, model evaluation, and web development to deliver a practical tool for stock price estimation. The Flask-based web app provides an intuitive and visually appealing interface, making it accessible for users to explore stock predictions.

Developed as part of a Machine Learning portfolio project by Saanya Lakhani. Feel free to use it for any finance related projects. 
