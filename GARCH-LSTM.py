import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Step 1: Load and prepare data
def load_data(csv_path='garch_data.csv'):
    """
    Load garch_data.csv and prepare features for LSTM.

    Parameters:
    - csv_path: Path to CSV file (default: garch_data.csv)

    Returns:
    - df: DataFrame with selected features
    """
    df = pd.read_csv(csv_path)
    required_columns = ['Date', 'Conditional_Volatility', 'HL_Range', 'Log_Volume', 'Volume_ZScore']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Found: {df.columns}, Required: {required_columns}")

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    return df


# Step 2: Create sequences for LSTM
def create_sequences(data, seq_length=10):
    """
    Create sequences of length seq_length for LSTM input.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - seq_length: Number of time steps (default: 10)

    Returns:
    - X: Input sequences of shape (n_samples, seq_length, n_features)
    - y: Target values (next day's Conditional_Volatility)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict next day's Conditional_Volatility
    return np.array(X), np.array(y)


# Step 3: Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output from last time step
        return out


# Step 4: Predict next day's Conditional Volatility
def predict_next_day(model, scaler, csv_path='garch_data.csv', seq_length=10):
    """
    Predict the next day's Conditional Volatility using the trained LSTM model.

    Parameters:
    - model: Trained LSTM model
    - scaler: Fitted MinMaxScaler from training
    - csv_path: Path to CSV file (default: garch_data.csv)
    - seq_length: Number of time steps (default: 10)

    Returns:
    - next_day_pred: Predicted Conditional Volatility for the next trading day
    - last_date: Last date in the dataset
    """
    # Load data
    df = load_data(csv_path)

    # Check if enough data for prediction
    if len(df) < seq_length:
        raise ValueError(f"Dataset has {len(df)} rows, but {seq_length} are required for prediction.")

    # Get the last seq_length days
    last_sequence = df[['Conditional_Volatility', 'HL_Range', 'Log_Volume', 'Volume_ZScore']].values[-seq_length:]
    last_date = df.index[-1]

    # Scale the sequence
    scaled_sequence = scaler.transform(last_sequence)

    # Reshape for LSTM input: (1, seq_length, input_size)
    input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)  # Add batch dimension

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()

    # Inverse transform prediction
    pred_full = np.zeros((1, 4))  # Dummy array for inverse transform
    pred_full[:, 0] = pred.flatten()
    next_day_pred = scaler.inverse_transform(pred_full)[0, 0]

    return next_day_pred, last_date


# Step 5: Main LSTM pipeline
def train_lstm_model(csv_path='garch_data.csv', seq_length=10, epochs=150, batch_size=32, learning_rate=0.001):
    """
    Train LSTM model using PyTorch to predict next day's Conditional_Volatility.

    Parameters:
    - csv_path: Path to CSV file
    - seq_length: Number of time steps (default: 10)
    - epochs: Number of training epochs (default: 100)
    - batch_size: Batch size for training (default: 32)
    - learning_rate: Learning rate for optimizer (default: 0.001)

    Returns:
    - model: Trained LSTM model
    - scaler: Fitted MinMaxScaler
    - X_test, y_test: Test data for evaluation
    - y_pred: Predicted values for test set
    - test_dates: Dates corresponding to test set predictions
    """
    # Load data
    df = load_data(csv_path)
    features = df[['Conditional_Volatility', 'HL_Range', 'Log_Volume', 'Volume_ZScore']].values
    dates = df.index  # Store dates for plotting

    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Create sequences
    X, y = create_sequences(scaled_features, seq_length)

    # Split into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size + seq_length:]  # Dates for test set predictions

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")

    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_size=4, hidden_size=32, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        # Calculate training loss
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loss (use test set as proxy)
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(device)
            y_test_dev = y_test.to(device)
            val_output = model(X_test_dev)
            val_loss = criterion(val_output, y_test_dev).item()
        val_losses.append(val_loss)
        model.train()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        y_pred = model(X_test_dev).cpu().numpy()

    # Inverse transform predictions and actual values
    y_test_full = np.zeros((len(y_test), 4))  # Dummy array for inverse transform
    y_test_full[:, 0] = y_test.flatten()  # Place actual values in first column
    y_pred_full = np.zeros((len(y_pred), 4))  # Dummy array for predictions
    y_pred_full[:, 0] = y_pred.flatten()  # Place predictions in first column

    y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]  # Extract Conditional_Volatility
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]  # Extract Conditional_Volatility

    # Plot results with dates
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label='Actual Conditional Volatility')
    plt.plot(test_dates, y_pred_inv, label='Predicted Conditional Volatility', linestyle='--')
    plt.title('LSTM Prediction of Next-Day Conditional Volatility')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility (%)')
    plt.legend()
    plt.grid(True)
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate and align dates
    plt.show()

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, scaler, X_test, y_test_inv, y_pred_inv, test_dates


# Example usage
if __name__ == "__main__":
    # Train model and get predictions for test set
    model, scaler, X_test, y_test, y_pred, test_dates = train_lstm_model(csv_path='garch_data.csv', seq_length=10)
    print("Test MSE:", np.mean((y_test - y_pred) ** 2))


    # Predict next day's Conditional Volatility
    next_day_pred, last_date = predict_next_day(model, scaler, csv_path='garch_data.csv', seq_length=10)
    next_trading_day = last_date + pd.tseries.offsets.BDay(1)
    print(f"Predicted Conditional Volatility for {next_trading_day.date()}: {next_day_pred:.4f}%")