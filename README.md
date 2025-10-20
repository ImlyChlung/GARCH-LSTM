# Volatility Prediction Using GARCH and LSTM

## Project Overview

This project implements a hybrid model combining **GARCH(1,1)** and **LSTM** to predict the price volatility of financial assets, such as the Invesco QQQ Trust (QQQ), specifically the `Conditional_Volatility(t+1)`.

- **GARCH.py**: Uses the GARCH(1,1) model to estimate conditional volatility based on constant mean and index returns, incorporating features like high-low range (`HL_Range`) and volume Z-score (`Volume_ZScore`) to generate `garch_data.csv`.
- **GARCH-LSTM.py**: Trains an LSTM model using `garch_data.csv` to analyze non-linear relationships between `Conditional_Volatility`, trading volume, and high-low price differences, predicting the next trading day's conditional volatility.

This approach can be combined with volatility indicators like Historical Volatility (HV), Implied Volatility (IV), and VIX to help investors assess stock risk. For the Chinese version, see [README.md](README.md).

### GARCH(1,1) Model

The GARCH(1,1) model estimates daily conditional volatility, with the formula shown below:

![GARCH(1,1) Model Formula](figue/GARCH_equation.png)

**Visualization:**

![GARCH Graph](figue/GARCH_graph.png)

### LSTM Model

#### 1. Input Features

The LSTM model uses a sequence of 10 days (`seq_length=10`) with five features extracted from `garch_data.csv`:

- **Returns**: Daily index percentage return, reflecting price changes relative to the previous day.
- **Conditional_Volatility**: Conditional volatility estimated by the GARCH(1,1) model, representing the asset's daily price fluctuation (in percentage).
- **HL_Range**: Difference between daily high and low prices (in index percentage), indicating price fluctuation amplitude.
- **Log_Volume**: Log-transformed daily trading volume change (in percentage), capturing relative volume changes.
- **Volume_ZScore**: Daily trading volume Z-score relative to the past 20 days, measuring volume deviation from the recent average.

These features are normalized using `MinMaxScaler` to the range [0, 1] for training stability. The input data shape is `(batch_size, seq_length=10, input_size=5)`, representing 10 days of 5 features per sample.

#### 2. Model Architecture (Customizable)

The LSTM model architecture is as follows:

