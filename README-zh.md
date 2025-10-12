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

