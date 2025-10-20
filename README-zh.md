# 使用 GARCH 和 LSTM 進行波動性預測

## 專案概述

本專案實現了一個結合 GARCH(1,1) 和 LSTM 的混合模型，用於預測金融資產（如 Invesco QQQ Trust (QQQ)）的價格波動性 Conditional_Volatility(t)。GARCH.py 使用 GARCH(1,1) 模型，基於常數均值和指數收益率估計條件波動性，結合 高低價範圍 和 成交量 Z 分數 等特徵，生成 garch_data.csv 文件。GARCH-LSTM.py 使用該 CSV 文件訓練 LSTM 模型，讓模型進一步分析Conditional_Volatility, 成交量, 最高和最低價之間的非線性關係，預測下一交易日的條件波動性 Conditional_Volatility(t+1)。此方法還可以結合HV, IV, VIX等波動性指標讓投資者評估一隻股票的風險。如需英文版說明，請參見 README.md。

### GARCH(1,1) 模型：

![GARCH(1,1) 模型公式](figue/GARCH_equation.png)


### LSTM 模型:

使用特徵features X, 
1. 相比前一日的指數百分比 Returns
2. 波動性 Conditional_Volatility
3. 最高價和最低價的指數百分比 High-Low Range
4. 單日成交量變化指數百分比 Log Volume
5. 成交量相比前20日的z-score水平

預測標籤labels Y
1. 明日的波動性 Conditional_Volatility
