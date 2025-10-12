## GARCH 和 LSTM 進行波動性預測

## 專案概述

本專案實現了一個結合 GARCH(1,1) 和 LSTM 的混合模型，用於預測金融資產的價格波動性 (( \sigma_t ))。GARCH.py 使用 GARCH(1,1) 模型，基於常數均值和指數收益率估計條件波動性，結合高低價範圍和成交量 Z 分數等特徵，生成 CSV 文件。GARCH-LSTM.py 使用該 CSV 文件訓練 LSTM 模型，預測下一交易日的條件波動性。本方法適用於短線交易分析，利用 GARCH 的波動性集群特性和 LSTM 捕捉非線性模式的能力。
如需英文版說明，請參見 README.md。
功能

GARCH(1,1) 波動性估計：

使用 yfinance 下載 2021-01-01 至 2025-10-11 的歷史股票數據（如 QQQ）。
計算指數收益率、高低價範圍、對數成交量和 20 天滾動成交量 Z 分數。
擬合常數均值的 GARCH(1,1) 模型，假設正態分佈。
生成 garch_data.csv，包含欄位：Date、Returns、Conditional_Volatility、HL_Range、Log_Volume、Volume_ZScore。
可視化條件波動性和 95% VaR。


LSTM 波動性預測：

載入 garch_data.csv，準備 LSTM 訓練特徵。
訓練雙層 LSTM 模型，預測下一交易日的條件波動性 (( \sigma_{t+1} ))。
支援 PyTorch 的 GPU 加速（若可用）。
可視化訓練/驗證損失及實際與預測波動性比較。
提供下一交易日的波動性預測，輔助交易決策。



文件結構
volatility-forecasting/
├── GARCH.py                # GARCH(1,1) 模型，估計波動性並生成 CSV 文件
├── GARCH-LSTM.py           # LSTM 模型，預測下一交易日波動性
├── garch_data.csv          # 輸出的 CSV 文件，包含 GARCH 特徵
├── README.md               # 專案說明文件（英文）
├── README_zh.md            # 專案說明文件（中文）
└── requirements.txt        # 所需 Python 套件

安裝
前提要求

Python 3.8 或更高版本
Git（用於克隆倉庫）

安裝步驟

克隆倉庫：
git clone https://github.com/your-username/volatility-forecasting.git
cd volatility-forecasting


創建虛擬環境（可選但推薦）：
python -m venv venv
source venv/bin/activate  # Windows 上：venv\Scripts\activate


安裝依賴：
pip install -r requirements.txt


確保 requirements.txt 包含：
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
arch>=5.3.0
matplotlib>=3.7.0
torch>=2.0.0
scikit-learn>=1.2.0



使用方法
步驟 1：運行 GARCH 模型
運行 GARCH.py 下載 QQQ 數據，擬合 GARCH(1,1) 模型，生成 garch_data.csv：
python GARCH.py

輸出：

garch_data.csv，包含欄位：Date、Returns、Conditional_Volatility、HL_Range、Log_Volume、Volume_ZScore。
條件波動性和 95% VaR 的圖表。

步驟 2：使用 LSTM 訓練與預測
運行 GARCH-LSTM.py 訓練 LSTM 模型並預測下一交易日波動性：
python GARCH-LSTM.py

輸出：

訓練/驗證損失圖表。
實際與預測波動性圖表。
下一交易日波動性預測（例如 2025-10-11 後的下一交易日）。

範例
# GARCH.py
garch_results, data, var_95, cond_mean, forecast_vol, forecast_var = calculate_garch(
    ticker='QQQ', start_date='2021-01-01', end_date='2025-10-11', csv_path='garch_data.csv'
)

# GARCH-LSTM.py
model, scaler, X_test, y_test, y_pred, test_dates = train_lstm_model(csv_path='garch_data.csv', seq_length=10)
next_day_pred, last_date = predict_next_day(model, scaler, csv_path='garch_data.csv', seq_length=10)
print(f"下一交易日 {last_date + pd.tseries.offsets.BDay(1)} 的預測條件波動性: {next_day_pred:.4f}%")

未來改進

多步預測：擴展 LSTM 預測多日波動性（如 ( \sigma_{t+1} ) 至 ( \sigma_{t+10} )）。
高級 GARCH 模型：加入 EGARCH 或 GJR-GARCH 捕捉非對稱波動效應。
額外特徵：加入市場指數（如 S&P 500）或技術指標（如 RSI）作為外生變量。
超參數調優：使用網格搜索優化 LSTM 參數（如序列長度、隱藏層大小）。
回測：與交易策略整合，評估短線交易表現。

許可證
本專案採用 MIT 許可證，詳情見 LICENSE 文件。
聯繫
如有問題或貢獻，請在 GitHub 上開啟 issue 或提交 pull request。
