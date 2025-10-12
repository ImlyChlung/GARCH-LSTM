使用 GARCH 和 LSTM 進行波動性預測
專案概述
本專案實現了一個結合 GARCH(1,1) 和 LSTM 的混合模型，用於預測金融資產（如 Invesco QQQ Trust (QQQ)）的價格波動性 $ \sigma_t $。GARCH.py 使用 GARCH(1,1) 模型，基於常數均值和指數收益率估計條件波動性，結合 高低價範圍 和 成交量 Z 分數 等特徵，生成 garch_data.csv 文件。GARCH-LSTM.py 使用該 CSV 文件訓練 LSTM 模型，預測下一交易日的條件波動性 $ \sigma_{t+1} $。本方法適用於短線交易分析，利用 GARCH 的波動性集群特性和 LSTM 捕捉非線性模式的能力。
GARCH(1,1) 模型：

均值方程：$$ y_t = \mu + \epsilon_t, \quad \epsilon_t = z_t \cdot \sigma_t, \quad z_t \sim N(0, 1) $$
方差方程：$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

LSTM 目標：使用特徵 $ { \sigma_t, \text{HL_Range}_t, \text{Log_Volume}t, \text{Volume_ZScore}t }{t-9:t} $ 預測 $ \sigma{t+1} $。
如需英文版說明，請參見 README.md。
功能

GARCH(1,1) 波動性估計：

使用 yfinance 下載 2021-01-01 至 2025-10-11 的 QQQ 歷史股票數據（含 50 天額外數據用於滾動計算）。
計算指數收益率：$ y_t = 100 \cdot \ln(P_t / P_{t-1}) $。
計算 高低價範圍、對數成交量 和 20 天滾動 成交量 Z 分數。
擬合常數均值的 GARCH(1,1) 模型，假設正態分佈。
生成 garch_data.csv，包含欄位：Date、Returns、Conditional_Volatility、HL_Range、Log_Volume、Volume_ZScore。
可視化條件波動性 $ \sigma_t $ 和 95% VaR：$ \text{VaR}_{95%} = \mu - 1.96 \cdot \sigma_t $。


LSTM 波動性預測：

載入 garch_data.csv，準備 LSTM 訓練特徵。
訓練雙層 LSTM 模型，預測下一交易日的條件波動性 $ \sigma_{t+1} $。
支援 PyTorch 的 GPU 加速（若可用）。
可視化訓練/驗證損失及實際與預測波動性 $ \sigma_t $ 比較。
提供下一交易日的波動性預測，輔助交易決策。



程式碼範例
GARCH(1,1) 模型 (GARCH.py)
import yfinance as yf
import pandas as pd
from arch import arch_model
import numpy as np

def calculate_garch(ticker='QQQ', start_date='2021-01-01', end_date='2025-10-11', csv_path='garch_data.csv'):
    # 下載數據
    start_date = pd.to_datetime(start_date)
    extended_start_date = start_date - pd.tseries.offsets.BDay(50)
    data = yf.download(ticker, start=extended_start_date, end=end_date)

    # 計算特徵
    data['Returns'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
    data['HL_Range'] = 100 * np.log(data['High'] / data['Low'])
    data['Log_Volume'] = 100 * np.log((data['Volume'] / data['Volume'].shift(1)).replace(0, np.nan))
    data['Volume_ZScore'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()

    # 擬合 GARCH(1,1)
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    results = model.fit(disp='off')

    # 計算 VaR 並保存至 CSV
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

LSTM 預測 (GARCH-LSTM.py)
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
    # 載入並準備數據
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    features = df[['Conditional_Volatility', 'HL_Range', 'Log_Volume', 'Volume_ZScore']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # 創建序列
    X, y = [], []
    for i in range(len(scaled_features) - seq_length):
        X.append(scaled_features[i:i + seq_length])
        y.append(scaled_features[i + seq_length, 0])
    X, y = np.array(X), np.array(y)

    # 訓練 LSTM
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

數學公式
GARCH(1,1) 模型
$$\begin{cases}y_t = \mu + \epsilon_t \\epsilon_t = z_t \cdot \sigma_t, \quad z_t \sim N(0, 1) \\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2\end{cases}$$

長期波動性：$$ \sigma^2 = \frac{\omega}{1 - \alpha - \beta} $$
95% VaR：$$ \text{VaR}_{95%} = \mu - 1.96 \cdot \sigma_t $$

LSTM 預測

輸入：特徵序列 $ { \sigma_t, \text{HL_Range}_t, \text{Log_Volume}_t, \text{Volume_ZScore}t }{t-9:t} $
輸出：預測 $ \sigma_{t+1} $

文件結構
volatility-forecasting/
├── GARCH.py                # GARCH(1,1) 模型實現
├── GARCH-LSTM.py           # LSTM 訓練與預測
├── garch_data.csv          # 生成的 GARCH 特徵數據集
├── README.md               # 英文說明文件
├── README_zh.md            # 中文說明文件
├── requirements.txt        # 依賴套件
└── LICENSE                 # MIT 許可證

安裝
前提要求

Python 3.8+
Git

安裝步驟
git clone https://github.com/your-username/volatility-forecasting.git
cd volatility-forecasting
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

依賴套件
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
arch>=5.3.0
matplotlib>=3.7.0
torch>=2.0.0
scikit-learn>=1.2.0

使用方法
1. 運行 GARCH 模型
python GARCH.py

2. LSTM 訓練與預測
python GARCH-LSTM.py

範例結果
GARCH 參數



參數
範例數值
說明



$ \mu $
0.0886%
日均收益率


$ \omega $
0.2090
基礎方差


$ \alpha $
0.0119
短期衝擊


$ \beta $
0.9753
長期記憶


長期 $ \sigma $
4.0380%
$ \sqrt{\frac{\omega}{1-\alpha-\beta}} $


LSTM 表現

MSE 損失：評估 $ \sigma_{t+1} $ 預測準確性
下一交易日預測：用於交易決策的實時波動性預測

未來改進

多步預測：預測 $ \sigma_{t+1} $ 至 $ \sigma_{t+10} $。
高級 GARCH：使用 EGARCH：$$ \ln(\sigma_t^2) = \omega + \alpha |z_{t-1}| + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2) $$。
特徵工程：加入 RSI、MACD 或 S&P 500 指數。
參數調優：優化 LSTM 序列長度和隱藏層大小。
交易整合：連接預測至 analyze_buy_signals。

許可證
MIT 許可證 - 詳情見 LICENSE。
聯繫
請在 GitHub 上開啟 issue 或提交 pull request。
