# @title Load Data
ticker = "EGIE3.SA"  # @param {type:"string"}
start_date = "2019-05-15"  # @param {type:"date"}
end_date = "2024-05-15"  # @param {type:"date"}

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load historical stock data
def load_data(ticker, start_date, end_date):
    # Fetches historical stock data from yfinance API
    data = yf.download(ticker, start=start_date, end=end_date)
    # Returns only the 'Close' prices reshaped as a column vector
    return data['Close'].values.reshape(-1, 1)

# @title Prepare Data for LSTM
def prepare_data(data, look_back=60):
    # Scale the data using MinMaxScaler to the range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    # Create sequences of data for LSTM training
    for i in range(look_back, len(scaled_data)):
        # Append the previous 'look_back' data points to X
        X.append(scaled_data[i-look_back:i, 0])
        # Append the current data point to Y, this is what we want to predict
        Y.append(scaled_data[i, 0])
    
    # Convert lists to numpy arrays
    X, Y = np.array(X), np.array(Y)
    # Reshape X to be a 3D tensor with shape (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Return the prepared X, Y and the scaler for inverse transformation later
    return X, Y, scaler

# @title Build and Train LSTM Model
def build_and_train_model(X, Y, epochs=1000, batch_size=32):
    # Build the LSTM model with 2 LSTM layers and a dense output layer
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    # Compile the model with adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model with X, Y using the selected epochs and batch size
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    # Returns the trained model
    return model

# @title Make Predictions and Calculate Percentage Change
def predict_and_calculate_changes(model, data, scaler):
    # Gets the last batch of data
    last_batch = data[-1]
    # Make a prediction for the next day using the last batch
    next_day_prediction = model.predict(np.array([last_batch]))
    # Inverse transform the prediction back to the original scale
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    # Get the last known price of the stock in the original scale
    last_price = scaler.inverse_transform([data[-1][-1]])
    # Calculates the percentage change between the predicted price and last known price
    change_percent = ((next_day_prediction - last_price) / last_price) * 100
    # Returns the percentage change
    return change_percent[0][0]

# @title Initialization and Execution
# Load the stock data
data = load_data(ticker, start_date, end_date)
# Prepare the data for the LSTM model
X, Y, scaler = prepare_data(data)
# Build and train the LSTM model
model = build_and_train_model(X, Y)
# Predict the next day stock price variation and calculate the percentage
change_percent = predict_and_calculate_changes(model, X, scaler)
# Prints the predicted percentage change
print(f"Percentage change for the next trading day: {change_percent:.2f}%")
