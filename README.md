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



Project Structure
volatility-forecasting/
├── GARCH.py                # GARCH(1,1) model for volatility estimation and CSV generation
├── GARCH-LSTM.py           # LSTM model for predicting next-day volatility
├── garch_data.csv          # Output CSV file with GARCH features
├── README.md               # Project documentation (English)
├── README_zh.md            # Project documentation (Chinese)
└── requirements.txt        # Required Python packages

Installation
Prerequisites

Python 3.8 or higher
Git (for cloning the repository)

Steps

Clone the repository:
git clone https://github.com/your-username/volatility-forecasting.git
cd volatility-forecasting


Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Ensure requirements.txt includes:
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
arch>=5.3.0
matplotlib>=3.7.0
torch>=2.0.0
scikit-learn>=1.2.0



Usage
Step 1: Run GARCH Model
Execute GARCH.py to download QQQ data, fit the GARCH(1,1) model, and generate garch_data.csv:
python GARCH.py

Output:

garch_data.csv with columns: Date, Returns, Conditional_Volatility, HL_Range, Log_Volume, Volume_ZScore.
Plots of conditional volatility and 95% VaR.

Step 2: Train and Predict with LSTM
Execute GARCH-LSTM.py to train the LSTM model and predict the next day's volatility:
python GARCH-LSTM.py

Output:

Training/validation loss plots.
Actual vs. predicted volatility plots.
Next-day volatility prediction (e.g., for the next trading day after 2025-10-11).

Example
# GARCH.py
garch_results, data, var_95, cond_mean, forecast_vol, forecast_var = calculate_garch(
    ticker='QQQ', start_date='2021-01-01', end_date='2025-10-11', csv_path='garch_data.csv'
)

# GARCH-LSTM.py
model, scaler, X_test, y_test, y_pred, test_dates = train_lstm_model(csv_path='garch_data.csv', seq_length=10)
next_day_pred, last_date = predict_next_day(model, scaler, csv_path='garch_data.csv', seq_length=10)
print(f"Predicted Conditional Volatility for {last_date + pd.tseries.offsets.BDay(1)}: {next_day_pred:.4f}%")

Future Improvements

Multi-Step Forecasting: Extend LSTM to predict volatility for multiple days ahead (e.g., ( \sigma_{t+1} ) to ( \sigma_{t+10} )).
Advanced GARCH Models: Incorporate EGARCH or GJR-GARCH to capture asymmetric volatility effects.
Additional Features: Include market indices (e.g., S&P 500) or technical indicators (e.g., RSI) as exogenous variables.
Hyperparameter Tuning: Optimize LSTM parameters (e.g., sequence length, hidden size) using grid search.
Backtesting: Integrate with trading strategies to evaluate performance in short-term trading.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, please open an issue or submit a pull request on GitHub.
