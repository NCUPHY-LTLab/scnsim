---
title: "網路方程式（Network equations）與根（Roots）"
lang: zh-TW
sidebar: testing-zh-tw
---


這些案例從已宣告的電路支路出發，將指定的網路物理量與獨立計算比較。`Q(ω)` 是隨頻率變化的動態剛度矩陣（dynamic-stiffness matrix），與諧振器案例的品質因數（quality factor）`Q` 不同。本頁外觀相似的電路是不同案例，不能合併其元件值或用途。

## 雙節點 LC 方程式

進階 Direct 案例的節點 `a` 以 80 fF 與 8 nH 並聯接地，節點 `b` 以 95 fF 與 7 nH 並聯接地；兩節點之間另有 2 fF 與 500 nH 並聯支路。電容矩陣 `C` 與電感剛度矩陣 `K` 為：

```text
C = [[82, -2], [-2, 97]] fF
K11 = 1/(8 nH) + 1/(500 nH)
K22 = 1/(7 nH) + 1/(500 nH)
K12 = K21 = -1/(500 nH)
Q(ω) = K - ω² C
```

![上圖是進階 Direct 計畫的耦合 LC；下圖是獨立的匹配零點 RLC 計畫，具有各自宣告的 50 Ω 電阻與端口。](figures/lc-networks.svg){#fig-lc-networks}

{{< askr-image-view target="fig-lc-networks" mode="inline" >}}

[開啟互動式 Plotly 圖](figures/lc-networks.html)。

1. 在 5、5.5 GHz 要求 `Q(ω)`，並在 5.5 GHz 要求 `a` 到 `b` 的導納（admittance）元素。
2. 將運算子（operator）與響應（response）和獨立案例計算比較，包含形狀與物理量。

這會抓出矩陣被化為純量、支路順序改變或耦合符號組裝錯誤。來源：[test_direct_analysis.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_direct_analysis.py)。

## 分佈線路與有序端口

另一個兩導體 readout/filter 多導體傳輸線（multiconductor transmission line，MTL）長 1.6 mm，使用兩段 π 型網路（π section）。每條導體有三個站點，共六個網路節點：四個端點與兩個中間點。四個 50 Ω 端口（ports）依序為 `head_readout`、`head_filter`、`tail_readout`、`tail_filter`。

1. 在 1、2 GHz 要求散射矩陣（scattering matrix）`S`、導納矩陣（admittance matrix）`Y` 與阻抗矩陣（impedance matrix）`Z`。
2. 確認各結果形狀為 `(2, 4, 4)`，並保留非對角電容與端口順序。

測試把整條線路當成矩陣網路，不會拆成四個互不相干的純量通道。來源：[test_direct_analysis.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_direct_analysis.py)。

## 導納（admittance）、耗散（dissipation）與匹配散射零點（matched scattering zero）

導納（admittance）案例包含 200 fF 電容、3 nH 電感與電阻，並透過 1 fF 接至 50 Ω 端口。電阻為 `R = Q/(ω₀ C)`；四個品質因數（quality factors）從 10³ 到 10⁹，取樣頻率為 `[0.9, 0.999, 1, 1.001, 1.1] × f₀`。

1. 在 50 Ω 與 100 Ω 參考阻抗（reference impedance）下計算導納（admittance）`Y`、阻抗（impedance）`Z` 的實部。
2. 將結果與獨立基準及 NumPy Schur 縮減（Schur reduction）比較；改變參考阻抗不能憑空產生元件耗散（dissipation）。
3. 在另一個匹配散射零點（matched scattering zero）案例中，評估 80 fF／8 nH 支路、50 Ω 分流與端口在 6.3 GHz 附近的 `Spp`。比對獨立 LC 零點，並確認分子與分母斜率都非零。

散射零點（scattering zero）是所選響應等於零的頻率。導數案例也會比較指定響應（response）與解析頻率導數。來源：[test_direct_admittance.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_direct_admittance.py) 與 [test_direct_analysis.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_direct_analysis.py)。

## 根（root）、極點（pole）與根證書（root certificate）

根（root）是符合指定網路方程式的頻率；極點（pole）是網路運算子（operator）成為奇異的複數頻率；零點（zero）是所選響應（response）消失的頻率。根證書（root certificate）記錄候選是否符合單根條件。高 Q 案例使用 200 fF／3 nH 諧振器，經由 `Cc = sqrt(C/(Q ω₀ R))` 耦合至 50 Ω 端口。品質因數依序為 10⁵、10⁶、2×10⁶、4×10⁶、6×10⁶、8×10⁶、10⁷、3×10⁷、10⁸、10⁹，並以兩種座標順序測試。

1. 將計算極點與獨立四乘四全域二次特徵值問題（global quadratic eigenvalue problem）及 60 位 Decimal 行列式精修比較。
2. 依各觀測量的條件界限縮放極點誤差，檢查 `error/(κ ε) ≤ 100`；這不是未條件化的線寬相對誤差。
3. 對另一個無損錨點（lossless anchor）案例，比較 Hybrid、Diagonal 與耦合（coupling）支路由 6 GHz 實數錨點（anchor）及兩種尺度的正負虛數擾動得到的頻率。五種輸入必須產生完全相同的頻率位元；複數根與 Newton 軌跡不必相同。
4. 確認根證書（root certificate）只對符合單根條件的候選發出。重複模態（repeated modes）、頻率斜率為零的案例，以及無法解析的留數耦合（residue coupling）都維持未解析狀態。

例如，兩個相同且未耦合的 80 fF／8 nH 諧振器在 6.3 GHz 錨點無法唯一決定單根。其他案例使用非零交叉支路但頻率斜率為零，或以 1e-300 F 電容搭配 1e300 H 與 2e300 H 電感；後者的留數耦合無法解析。來源：[test_root_certificates.py](https://github.com/NCUPHY-LTLab/scnsim/blob/codex/r2-jax-only-surface/tests/test_root_certificates.py)。
