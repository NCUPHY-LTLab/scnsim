---
title: "測試（Testing）"
lang: zh-TW
sidebar: testing-zh-tw
---


SCNSim 的測試使用小型且明確宣告的電路案例，說明套件如何計算、已儲存結果如何讀回，以及無效輸入如何回報。本指南介紹測試案例與持續整合（CI）執行的檢查。

## 測試主題

::::{.askr-card}
::: {.askr-card-body}
### [CI 與 Pages](ci.md) {.askr-card-title .unlisted}

了解工作流程何時執行、產品工作會檢查什麼，以及文件建置與 Pages 部署的差異。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [電路輸入（Circuit inputs）與元件（Components）](circuit-data.md) {.askr-card-title .unlisted}

查看合成 RLGC／Q2D 載入案例，以及型錄宣告的小型元件模型和無效輸入案例。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [網路方程式（Network equations）與根（Roots）](network-roots.md) {.askr-card-title .unlisted}

了解 Direct 直接分析的網路物理量、導納（admittance）、傳輸零點，以及根證書如何與獨立案例比較。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [參數（Parameters）與最佳化（Optimization）](parameters-optimization.md) {.askr-card-title .unlisted}

追蹤仿射參數授權、區域座標尺度，以及小型最佳化案例；不同最佳化器不必走出相同路徑。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [證據（Evidence）與讀回（Readback）](evidence-readback.md) {.askr-card-title .unlisted}

從已完成的計算一路查看 v3 證據，再經過 Memory 或 Workspace 關閉與讀回。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [錯誤處理（Failure handling）與診斷（Diagnostics）](diagnostics.md) {.askr-card-title .unlisted}

了解原生錯誤如何保留原始內容，以及求解診斷如何區分彙總指標與逐右手邊結果。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [路徑（Paths）、電路圖（Diagrams）與教材輸出（Teaching outputs）](portability-diagrams.md) {.askr-card-title .unlisted}

查看檔案 URI 轉換、符合宣告的圖、可攜式筆記本，以及在求解前停止的課程準備流程。
:::
::::

::::{.askr-card}
::: {.askr-card-body}
### [數值檢查（Numerical checks）與本機執行（Local run）](numerical-checks.md) {.askr-card-title .unlisted}

了解前向界限、後向殘差、精確比較，以及預設本機 unittest 指令各自檢查什麼。
:::
::::
