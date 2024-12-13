# ECG時間序列分群與異常點檢測

## 專案簡介

此專案展示了一個時間序列資料離群值檢測流程，使用公開數據集[ECG（心電圖）](https://www.timeseriesclassification.com/description.php?Dataset=ECG200)進行實作展示。首先觀察到數據有明顯分群跡象後，透過**K-medoids分群**將數據分群，並針對各個群體計算**Point-Wise Mahalanobis Distance**，接著根據所計算出來的距離進行異常點的判定，以下展示詳細流程。


## 專案動機
在持續訓練的過程中，離群值的出現往往影響模型估計，因此能夠及時辨別離群值的出現尤為重要。

## 專案內容

### 資料探索：
首先觀察數據分布，數據大致上分布如下：
![ECG Trend](./plots/raw_data_trend.png)

數據大致上分為三組，因此透過**K-medoids分群**將數據分為3群，分群後呈現如下：
![ECG Group Trend](./plots/raw_data_group_trend.png)
選擇**K-medoids**分群方法而不選擇**K-means**分群方法的理由在於，統計上若要估計中心值，中位數在有存在離群值的情況下較為穩定，而平均容易受到離群影響。而**K-medoids**分群方法恰好就是以實際數據作為群心。相較於使用群平均作為群心的**K-means**分群方法，比較穩健，較不受到離群值影響。

可以看到分群後，各個群體內部存在些許離群值，







