# 使用 GARCH 和 LSTM 進行波動性預測

## 專案概述

本專案實現了一個結合 **GARCH(1,1)** 和 **LSTM** 的混合模型，用於預測金融資產（如 Invesco QQQ Trust (QQQ)）的價格波動性 Conditional_Volatility(t)。
- **GARCH.py** 使用 GARCH(1,1) 模型，基於常數均值和指數收益率估計條件波動性，結合 高低價範圍 和 成交量 Z 分數 等特徵，生成 **garch_data.csv** 文件。
- **GARCH-LSTM.py** 使用該 CSV 文件訓練 LSTM 模型，讓模型進一步分析Conditional_Volatility, 成交量, 最高和最低價之間的非線性關係，預測下一交易日的條件波動性 Conditional_Volatility(t+1)。此方法還可以結合HV, IV, VIX等波動性指標讓投資者評估一隻股票的風險。如需英文版說明，請參見 **README.md**。

### GARCH(1,1) 模型：

![GARCH(1,1) 模型公式](figue/GARCH_equation.png)


### LSTM 模型:

### 1. 輸入特徵

LSTM 模型以 10 天為一個序列（seq_length=10）的五個特徵作為輸入，這些特徵從 **garch_data.csv** 中提取，具體如下：

- **Returns**: 每日指數的百分比回報率，表示價格相對於前一日的變化。
- **Conditional_Volatility**: 由 GARCH(1,1) 模型估計的條件波動性，表示當日資產價格的波動程度（以百分比表示）。
- **HL_Range**: 每日最高價與最低價之間的差異（以指數百分比表示），反映價格波動幅度。
- **Log_Volume**: 當日成交量的對數變化（以百分比表示），捕捉成交量的相對變化。
- **Volume_ZScore**: 當日成交量相對於前 20 日的 Z 分數，衡量成交量相對於近期平均水平的偏差。

這些特徵經過 MinMaxScaler 正規化，確保數值範圍在 [0, 1] 之間，以提高模型訓練的穩定性。輸入數據的形狀為 (batch_size, seq_length=10, input_size=5)，表示每個樣本包含 10 天的 5 個特徵。

### 2. 模型結構 (可自行更改)

**LSTM 模型的架構如下：**

**LSTM 層:**

- 包含 2 層 LSTM（num_layers=2）。
- 每層有 64 個隱藏單元（hidden_size=64）。
- 啟用 dropout（比率 0.2），在層間應用以防止過擬合。

**全連接層 (Fully Connected Layer):**

- 將最後一個時間步的 LSTM 輸出（形狀為 (batch_size, hidden_size)）映射到單個輸出值。
- 輸出單個值，表示下一交易日的條件波動性 (Conditional_Volatility(t+1))。

**輸出:** 經過全連接層後，模型輸出一個標量值，表示預測的波動性（在正規化空間中）。此值通過 MinMaxScaler 的逆轉換還原為實際的波動性百分比。

### 3. 訓練過程

- 數據分割: 數據以 80% 訓練集和 20% 測試集的比例分割。
- 損失函數: 使用均方誤差 (Mean Squared Error, MSE) 作為損失函數，衡量預測值與實際波動性之間的差異。
- 優化器: 使用 Adam 優化器，學習率為 0.001（learning_rate=0.001）。
- 訓練週期: 默認訓練 100 個週期（epochs=100），每批次處理 32 個樣本（batch_size=32）。
- 硬體支持: 模型支持 GPU 加速（如果可用），否則使用 CPU。
- 損失監控: 訓練過程中記錄訓練損失和驗證損失（使用測試集作為代理），並繪製損失曲線以檢查模型收斂性。

### 4. 預測功能

下一交易日預測: predict_next_day 函數使用訓練好的模型，基於最近 10 天的特徵序列，預測下一交易日的 Conditional_Volatility。

### 5. 輸出與可視化

- 預測結果: 模型生成測試集的預測值 (y_pred_inv)，並與實際值 (y_test_inv) 進行比較，計算測試集的 MSE。

- 可視化:

繪製實際與預測條件波動性的折線圖，X 軸為日期，Y 軸為波動性百分比。


繪製訓練和驗證損失曲線，檢查模型訓練過程的穩定性和收斂性。

以下是以Invesco QQQ Trust (QQQ)的 2021-01-01 到 2025-10-21 的數據訓練的模型，它的效果如下:

![performance1](figue/performance1.png)

![performance2](figue/performance2.png)


