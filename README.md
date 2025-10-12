Volatility Forecasting with GARCH and LSTM

Overview
This project implements a hybrid model combining GARCH(1,1) and LSTM to forecast stock price volatility (( \sigma_t )) for financial assets like the Invesco QQQ Trust (QQQ). The GARCH.py script uses the GARCH(1,1) model with a constant mean and log returns to estimate conditional volatility, incorporating features such as High-Low Range and Volume Z-Score. The generated data is saved to a CSV file, which is then used by GARCH-LSTM.py to train an LSTM model for predicting the next day's conditional volatility. This approach is designed for short-term trading analysis, leveraging GARCH's volatility clustering and LSTM's ability to capture non-linear patterns.
For a Chinese version of this documentation, see README_zh.md.
Features

GARCH(1,1) Volatility Estimation:

Downloads historical stock data (e.g., QQQ) from 2021-01-01 to 2025-10-11 using yfinance.
Computes log returns, High-Low Range, Log Volume, and 20-day rolling Volume Z-Score.
Fits a GARCH(1,1) model with a constant mean and normal distribution.
Generates garch_data.csv with columns: Date, Returns, Conditional_Volatility, HL_Range, Log_Volume, Volume_ZScore.
Visualizes conditional volatility and 95% VaR.


LSTM Volatility Prediction:

Loads garch_data.csv and prepares features for LSTM training.
Trains a two-layer LSTM model to predict the next day's conditional volatility (( \sigma_{t+1} )).
Supports GPU acceleration with PyTorch (if available).
Visualizes training/validation loss and actual vs. predicted volatility.
Provides next-day volatility predictions for trading decisions.


