# 數據驅動的目標函數估計與最佳化
## 專案簡介

此專案展示了一個數據驅動的目標函數估計與最佳化流程。透過結合數據模擬、表面估計及優化技術，針對複雜或無法明確表達的目標函數進行反應曲面的建模與估計，進一步實現基於該反應曲面的最佳化，求得使目標函數 $f(x)$ 最小化或最大化的最佳解 $x$。

## 專案動機

在現實生活中，$y$ 與 $x$ 之間的關係往往難以用單一函數形式 $y = f(x)$ 精確描述，或者該函數可能會隨時間動態變化為 $y = f(x, t)$。這種情況下，目標函數的不可明確性為最佳化過程帶來挑戰。為了解決此問題，我們引入機器學習技術，以近似方式估計函數 $f(x)$，並建立一個學習到的函數 $\hat{f}(x)$。基於此估計函數 $\hat{f}(x)$，我們可以進一步進行目標函數的最佳化，從而找到使目標值最小化或最大化的最佳解 $x$。這一方法有效地應對了目標函數不明確或動態變化所帶來的挑戰。

## 專案內容

### 方法模擬：
將我們的方法應用於模擬生成的數據上，假設數據來自以下目標函數分佈：

$y = f(x_1, x_2) = \frac{-\cos(\pi x_1) \cdot \cos(2\pi x_2)}{1 + x_1^2 + x_2^2} + \epsilon$;  
$\epsilon \sim N(0, 0.01)$;  
$x_1 \in [-1.75, 1.75]$;  
$x_2 \in [-1.75, 1.75]$; 
![Descent Plot](./pso.gif)

### 實際資料應用：
![Data Logo](./logo-xs.svg)

後將我們的方法應用於實際資料，該資料取自[內政部不動產交易實價查詢服務網](https://plvr.land.moi.gov.tw/DownloadOpenData)，取2024年第二季與第三季的預售屋成交資料進行模型 $\hat{f}(x)$ 建置，並透過 Differential evolution 最佳化。


## 使用方法