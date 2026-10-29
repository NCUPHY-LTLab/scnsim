---
title: "參數（Parameters）與最佳化（Optimization）"
lang: zh-TW
sidebar: testing-zh-tw
---


這些案例區分一般參數變更與在宣告支援範圍（support）內獲明確授權的仿射模型（affine model），也檢查有界物理座標如何轉換成最佳化器（optimizer）使用的座標。

## 仿射支援範圍與授權

仿射宣告會在可信支援區間（support interval）內，以固定係數將實體長度 `l` 對應到電容。案例的長度基準值為 1 m，支援區間（support interval）為 `[0.99, 1.01] m`。兩個並聯電容為 `C1 = (20 fF/m) l + 10 fF` 與 `C2 = (30 fF/m) l + 5 fF`，另有 8 nH 支路接地及 50 Ω 端口。1.05 m 位於支援範圍之外；測試也使用 `[1.05, 1.00, 1.05] m`，確認來源順序與重複點會保留。

1. 以未授權的 Operator（operator）宣告要求 1.05 m。宣告建立（authoring）階段必須引發 `SCNSimValidationError`，此時尚未建立請求或開始執行。
2. 明確授權候選政策後，透過 Memory 與 Workspace 執行 Direct、Operator 及導納扇出（fan-out）。確認有序支援邊及各自的授權來源仍附在原始來源列上。
3. 執行最佳化邊界 `[0.9, 1.1] m`、隨機種子（random seed）23、五次評估的 CMA 演化策略（CMA evolution strategy）案例。不受支援的候選必須記為 `invalid_candidate_physical_parameter`，stage 為 `affine_support`，cost 為正無限大；它不會成為公開最佳候選，也不會新增授權歷程。

![仿射案例：長度基準 1 m、支援範圍 [0.99, 1.01] m，以及需要明確授權的範圍外輸入 1.05 m；此圖不是最佳化軌跡。](figures/affine-domain.svg){#fig-affine-domain}

{{< askr-image-view target="fig-affine-domain" mode="inline" >}}

[開啟互動式 Plotly 圖](figures/affine-domain.html)。

來源：[test_affine_parameters.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_affine_parameters.py)。

## 區域座標尺度與 C1 案例

有界最佳化案例使用 110 fF 電容、5.8 nH 諧振器、6 fF 耦合電容與 50 Ω 端口，頻率提示（frequency hint）為 6 GHz，目標頻率（target frequency）為 6.2 GHz。參數範圍先為 80–140 fF，再由覆寫解析成 90–130 fF。區域座標尺度（local-coordinate scale）會把實體參數位移轉成最佳化器使用的座標。尺度依序優先採用明確座標尺度（explicit coordinate scale）、變數宣告尺度（declared variable scale），再採解析後的範圍寬度（resolved bounds width）；此案例使用最後一項，並將 `scale_source` 記為 `bounds_width`。

C1 案例依既有設定執行 L-BFGS-B（有界梯度式方法）與 Projected L-BFGS（以投影維持邊界）。

1. 解析參數範圍，確認哪一種尺度來源提供最佳化座標。
2. 執行兩種最佳化器，要求回報收斂且至少完成一次迭代，並讓獨立解析損失低於基準值。
3. 記錄最終頻率與剩餘目標距離；測試沒有設定目標距離容差。

來源：[test_local_coordinate_defaults.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_local_coordinate_defaults.py)。

## CMA 選擇器

另一個 CMA 演化策略（CMA evolution strategy）案例在 79.5–80.5 fF 範圍內取樣候選值，檢查七種宣告選擇器：Hybrid 頻率與線寬（linewidth）、TransferZero 頻率、耦合大小，以及響應大小、實部與虛部。seed 29 案例執行五次評估；獨立的匹配 RLC 零點案例使用 seed 31，也執行五次評估。

1. 確認每種選擇器都有宣告，且相應分支確實執行。
2. 確認每個 CMA 案例的零成本基準（zero-cost baseline）；不同後端不必產生相同候選軌跡。

來源：[test_optimization.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_optimization.py)。
