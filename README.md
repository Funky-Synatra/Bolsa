# Bolsa


# Stock Price Prediction with LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict the percentage change in stock prices for the next trading day.

## Description

The code performs the following steps:

1.  **Data Loading**: Fetches historical stock data using the `yfinance` library.
2.  **Data Preparation**: Scales the data using `MinMaxScaler` and creates sequences of data for LSTM training.
3.  **Model Building**: Constructs an LSTM model using `tensorflow.keras`.
4.  **Model Training**: Trains the LSTM model with the prepared data.
5.  **Prediction**: Uses the trained model to predict the price for the next trading day and calculates the percentage change from the last known price.

## How to Use

1.  **Install Required Libraries**:

    ```bash
    pip install numpy pandas yfinance tensorflow scikit-learn
    ```
2.  **Run the script**:
    The script can be executed as is. You can change the `ticker`, `start_date` and `end_date` variables to change the stock you want to analyse.
    ```bash
        python your_script_name.py
    ```
    *   Replace `your_script_name.py` with the name you gave the python file.

## Parameters

The script uses the following parameters:
*   `ticker`: The stock ticker symbol to be analysed (default is EGIE3.SA)
*   `start_date`: Start date for historical stock data (default is 2019-05-15)
*   `end_date`: End date for historical stock data (default is 2024-05-15)

## Important Notes

*   This is a basic implementation of a stock price prediction model and may not provide accurate predictions for real trading.
*   The model's performance can be improved by tuning hyperparameters and adding more features.
*   The model currently only uses historical closing prices. Other data such as opening price, volume, and other indicators may improve the model's prediction capability.

## Disclaimer

This project is for educational purposes only. Do not use this model for actual financial investments.
