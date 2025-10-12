Volatility Forecasting with GARCH and LSTM
Overview
This project combines GARCH(1,1) and LSTM models to forecast stock price volatility (( \sigma_t )) for assets like the Invesco QQQ Trust (QQQ). The GARCH.py script estimates conditional volatility using a GARCH(1,1) model with log returns, incorporating features like High-Low Range and Volume Z-Score. The GARCH-LSTM.py script trains an LSTM model to predict the next day's volatility (( \sigma_{t+1} )) using features from garch_data.csv. This approach is ideal for short-term trading analysis.
GARCH(1,1) Model:

Mean Equation: ( y_t = \mu + \epsilon_t, \quad \epsilon_t = z_t \cdot \sigma_t, \quad z_t \sim N(0, 1) )
Variance Equation: ( \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 )

LSTM Objective: Predict ( \sigma_{t+1} ) using features { ( \sigma_t, HL_Range_t, Log_Volume_t, Volume_ZScore_t }_{t-n:t} ).
For Chinese documentation, see README_zh.md.
Features

GARCH(1,1) Volatility Estimation:

Downloads QQQ data (2021-01-01 to 2025-10-11) using yfinance.
Computes log returns: ( y_t = 100 \cdot \ln(P_t / P_{t-1}) ).
Calculates High-Low Range, Log Volume, and 20-day Volume Z-Score.
Fits GARCH(1,1) with constant mean and normal distribution.
Outputs garch_data.csv with columns: Date, Returns, Conditional_Volatility, HL_Range, Log_Volume, Volume_ZScore.
Visualizes ( \sigma_t ) and 95% VaR: ( VaR_{95%} = \mu - 1.96 \cdot \sigma_t ).


LSTM Volatility Prediction:

Trains a two-layer LSTM to predict ( \sigma_{t+1} ).
Uses features from garch_data.csv with sequence length 10.
Supports GPU acceleration via PyTorch.
Visualizes training/validation loss and actual vs. predicted ( \sigma_t ).
Outputs next-day volatility predictions.



Code Examples
GARCH(1,1) Model (GARCH.py)
import yfinance as yf
import pandas as pd
from arch import arch_model
import numpy as np

def calculate_garch(ticker='QQQ', start_date='2021-01-01', end_date='2025-10-11', csv_path='garch_data.csv'):
    # Download data
    start_date = pd.to_datetime(start_date)
    extended_start_date = start_date - pd.tseries.offsets.BDay(50)
    data = yf.download(ticker, start=extended_start_date, end=end_date)

    # Calculate log returns and features
    data['Returns'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
    data['HL_Range'] = 100 * np.log(data['High'] / data['Low'])
    data['Log_Volume'] = 100 * np.log((data['Volume'] / data['Volume'].shift(1)).replace(0, np.nan))
    data['Volume_ZScore'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()

    # Fit GARCH(1,1)
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    results = model.fit(disp='off')

    # Calculate VaR and save to CSV
    cond_vol = results.conditional_volatility
    cond_mean = pd.Series(results.params['mu'], index=cond_vol.index)
    var_95 = cond_mean - 1.96 * cond_vol
    output_df = pd.DataFrame({
        'Date': data.index,
        'Returns': data['Returns'],
        'Conditional_Volatility': cond_vol,
        'HL_Range': data['HL_Range'],
        'Log_Volume': data['Log_Volume'],
        'Volume_ZScore': data['Volume_ZScore']
    }).dropna()
    output_df.to_csv(csv_path, index=False)

    return results, data, var_95, cond_mean

LSTM Prediction (GARCH-LSTM.py)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(csv_path='garch_data.csv', seq_length=10, epochs=150):
    # Load and prepare data
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    features = df[['Conditional_Volatility', 'HL_Range', 'Log_Volume', 'Volume_ZScore']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - seq_length):
        X.append(scaled_features[i:i + seq_length])
        y.append(scaled_features[i + seq_length, 0])
    X, y = np.array(X), np.array(y)

    # Train LSTM
    model = LSTMModel().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.FloatTensor(X[:int(0.8 * len(X))])
    y_train = torch.FloatTensor(y[:int(0.8 * len(y))]).reshape(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

Mathematical Formulation
GARCH(1,1) Model
[\begin{cases}y_t = \mu + \epsilon_t \\epsilon_t = z_t \cdot \sigma_t, \quad z_t \sim N(0, 1) \\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2\end{cases}]

Long-term Volatility: ( \sigma^2 = \frac{\omega}{1 - \alpha - \beta} )
95% VaR: ( VaR_{95%} = \mu - 1.96 \cdot \sigma_t )

LSTM Prediction

Input: Sequence of { ( \sigma_t, HL_Range_t, Log_Volume_t, Volume_ZScore_t }_{t-9:t}
Output: Predicted ( \sigma_{t+1} )

Project Structure
volatility-forecasting/
├── GARCH.py                # GARCH(1,1) model implementation
├── GARCH-LSTM.py           # LSTM training and prediction
├── garch_data.csv          # Generated GARCH features
├── README.md               # English documentation
├── README_zh.md            # Chinese documentation
├── requirements.txt        # Dependencies
└── LICENSE                 # MIT License

Installation
Prerequisites

Python 3.8+
Git

Steps
git clone https://github.com/your-username/volatility-forecasting.git
cd volatility-forecasting
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Dependencies
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
arch>=5.3.0
matplotlib>=3.7.0
torch>=2.0.0
scikit-learn>=1.2.0

Usage
1. Run GARCH Model
python GARCH.py

2. Train and Predict with LSTM
python GARCH-LSTM.py

Example Results
GARCH Parameters



Parameter
Example Value
Description



( \mu )
0.0886%
Daily mean return


( \omega )
0.2090
Base variance


( \alpha )
0.0119
Short-term shock


( \beta )
0.9753
Long-term persistence


Long-term ( \sigma )
4.0380%
( \sqrt{\frac{\omega}{1-\alpha-\beta}} )


LSTM Performance

MSE Loss: Evaluates prediction accuracy for ( \sigma_{t+1} )
Next-Day Prediction: Forecast for trading decisions

Future Improvements

Multi-Step Forecasting: Predict ( \sigma_{t+1} ) to ( \sigma_{t+10} ).
Advanced GARCH: Use EGARCH: ( \ln(\sigma_t^2) = \omega + \alpha |z_{t-1}| + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2) ).
Features: Add RSI, MACD, or S&P 500 index.
Tuning: Optimize LSTM sequence length and hidden size.
Trading: Integrate with analyze_buy_signals.

License
MIT License - see LICENSE.
Contact
Open an issue or pull request on GitHub for contributions or questions.
