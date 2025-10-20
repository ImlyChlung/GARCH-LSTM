import yfinance as yf
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.offsets import BDay


ticker = 'QQQ'
start_date = '2021-01-01'
end_date = '2025-10-21'

def calculate_garch(ticker, start_date, end_date, forecast_horizon=10,
                    csv_path='garch_data.csv'):
    """
    Downloads stock data using yfinance, including 50 extra trading days before start_date for rolling calculations,
    converts MultiIndex columns to single index, fits an optimized GARCH(1,1) model with a constant mean and log returns,
    includes High, Low, Volume, and Volume Z-Score (20-day rolling window) for analysis, and generates a CSV file.

    Parameters:
    - ticker: Stock ticker symbol (default: NVDA)
    - start_date: Start date for output data (format: YYYY-MM-DD)
    - end_date: End date for output data (format: YYYY-MM-DD)
    - forecast_horizon: Number of days to forecast (default: 10)
    - csv_path: Path for output CSV file (default: garch_data.csv)

    Returns:
    - GARCH model results, data, VaR, conditional mean, and multi-step forecasts
    """
    # Step 1: Calculate start date with 50 extra trading days
    start_date = pd.to_datetime(start_date)
    extended_start_date = start_date - BDay(50)  # Subtract 50 trading days
    end_date = pd.to_datetime(end_date)

    # Download data from yfinance (extended period)
    data = yf.download(ticker, start=extended_start_date, end=end_date, auto_adjust=False)
    if data.empty:
        raise ValueError(f"Failed to download data for {ticker}. Check ticker or date range.")

    # Convert MultiIndex columns to single index

    data.columns = [col[0] for col in data.columns]  # Keep first level (e.g., 'Volume' instead of ('Volume', 'AMZN'))

    # Verify required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Found: {data.columns}, Required: {required_columns}")

    # Step 2: Calculate log returns (in percentage)
    data['Returns'] = 100 * np.log(data['Close'] / data['Close'].shift(1))

    # Add High-Low Range and Log Volume
    data['HL_Range'] = 100 * np.log(data['High'] / data['Low'])  # Log High-Low Range
    data['Log_Volume'] = 100 * np.log((data['Volume']/data['Volume'].shift(1)).replace(0, np.nan))  # Avoid log(0)

    # Calculate Volume Z-Score (20-day rolling window)
    window_size = 20
    data['Rolling_Mean_Volume'] = data['Volume'].rolling(window=window_size, min_periods=window_size).mean()
    data['Rolling_Std_Volume'] = data['Volume'].rolling(window=window_size, min_periods=window_size).std()

    # Avoid division by zero in Z-Score
    data['Volume_ZScore'] = np.where(
        data['Rolling_Std_Volume'] > 0,
        (data['Volume'] - data['Rolling_Mean_Volume']) / data['Rolling_Std_Volume'],
        np.nan
    )

    # Drop NaN values
    data = data.dropna()

    # Trim data to original date range (start_date to end_date)
    data = data.loc[start_date:end_date]
    if data.empty:
        raise ValueError(f"No data available in the range {start_date} to {end_date} after processing.")

    # Step 3: Fit standard GARCH(1,1) model with constant mean
    returns = data['Returns']
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    results = model.fit(disp='off')
    print("\nStandard GARCH(1,1) Model Results (Log Returns):")
    print(results.summary())

    # Extract parameters
    mu = results.params.get('mu')
    omega = results.params['omega']
    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']

    print(f"\nExtracted Parameters (Standard GARCH):")
    print(f"mu (Constant Mean): {mu:.4f}%")
    print(f"omega (Base Variance): {omega:.4f}")
    print(f"alpha (Short-term Shock): {alpha:.4f}")
    print(f"beta (Long-term Persistence): {beta:.4f}")

    # Calculate long-term variance
    long_term_variance = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else float('inf')
    print(f"Long-term Standard Deviation: {np.sqrt(long_term_variance):.4f}%")

    # Calculate conditional mean and VaR
    cond_vol = results.conditional_volatility
    cond_mean = pd.Series(mu, index=cond_vol.index)  # Constant mean
    var_95 = cond_mean - 1.96 * cond_vol  # 95% VaR

    # Step 4: Multi-step forecasting
    forecast = results.forecast(horizon=forecast_horizon)
    forecast_vol = np.sqrt(forecast.variance.iloc[-1])  # Multi-step conditional volatility
    forecast_var = mu - 1.96 * forecast_vol  # Multi-step VaR
    print(f"\n{forecast_horizon}-Day Ahead Volatility Forecast:")
    print(forecast_vol)

    # Step 5: Generate CSV file with additional features
    output_df = pd.DataFrame({
        'Date': data.index,
        'Returns': data['Returns'],
        'Conditional_Volatility': cond_vol,
        'HL_Range': data['HL_Range'],
        'Log_Volume': data['Log_Volume'],
        'Volume_ZScore': data['Volume_ZScore']
    })
    output_df.to_csv(csv_path, index=False)
    print(f"\nCSV File Generated: {csv_path}")

    # Step 6: Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, cond_vol, label='Conditional Volatility (GARCH)', color='red')
    plt.plot(data.index, var_95, label='95% VaR', color='blue', linestyle='--')
    plt.plot(data.index, returns, label='Daily Log Returns', alpha=0.5)
    plt.title(f'{ticker} GARCH(1,1) Conditional Volatility and VaR (Constant Mean, Log Returns)')
    plt.xlabel('Date')
    plt.ylabel('Returns / Volatility / VaR (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return results, data, var_95, cond_mean, forecast_vol, forecast_var


# Example usage
if __name__ == "__main__":
    garch_results, data, var_95, cond_mean, forecast_vol, forecast_var = calculate_garch(ticker,
                                                                                         start_date,
                                                                                         end_date,
                                                                                         csv_path='garch_data.csv')
