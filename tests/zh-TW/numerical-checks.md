---
title: "數值檢查（Numerical checks）與本機執行（Local run）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../numerical-checks.md) · [繁體中文](numerical-checks.md)

SCNSim 數值測試會依問題採用不同檢查。前向誤差（forward error）以計算值（`actual`）和獨立參考值（`oracle`）比較，判斷指定物理量的差異。後向殘差（backward residual）則檢查計算結果代回原方程式後的誤差；精確檢查比對形狀、順序、位元與已存結果。這些檢查回答不同問題。

## 前向比較

對已宣告的案例，正規化前向誤差為：

```text
relative_error = max(abs(actual - oracle)) / max(abs(oracle))
relative_error <= 100 * condition_bound * float64_epsilon
```

`κ`（`condition_bound`，條件界限）估計方程式對該物理量前向誤差的放大程度；`float64_epsilon` 是 64 位元浮點數（binary64）可分辨的相對間距尺度。每個界限都依案例方程式與該觀測量的獨立基準推導。報告列出計算值、參考值、單位、形狀、`κ` 與正規化誤差比例。極點使用可觀測量專屬的條件界限，不使用通用線寬相對誤差。

## 後向殘差與精確比較

實際運算子（operator）與線性求解的後向殘差使用 `256 * (n + 1) * float64_epsilon`。精確檢查則分別比對形狀、順序、編碼值、單位投影、binary64 位元與已存結果欄位。基準正規化為零時不額外加 epsilon 下限。

這些檢查適用於主題頁介紹的小型宣告案例。目前套件沒有 `StaticCapacitance` 專用數值回歸測試；120 組 Golden 比較與效能執行不屬於預設 CI。

## 驗證測試家族

另有五個測試家族檢查單一解析案例無法涵蓋的性質。每個前向斷言都記錄其 `κ`，以及基準端如何估計它。

| 家族 | 檢查內容 | 基準 | 原始碼 |
| --- | --- | --- | --- |
| 網路不變量 | 含一個消去節點的雙埠網路，其 Direct `Y`、`Z`、`S` 的互易性、被動性、無損性、頻率與阻抗縮放律，以及節點與埠的排列 | 精確有理數節點解；`κ` 對每個分支值、角頻率與參考阻抗加總各元素的精確一階變化 | [`test_network_invariants.py`](../test_network_invariants.py)、[`exact_network.py`](../exact_network.py) |
| 微分規則 | 已安裝套件中所有自訂 JVP 與轉置規則的清單；每條受測規則都有 Taylor 測試與 JVP/VJP 內積恆等式 | 精確或 60 位數的原值與導數；Taylor 餘項與恆等式使用由各部分前向界限推得的界限 | [`test_differentiation_rules.py`](../test_differentiation_rules.py) |
| 全域極點基準 | 三模態有損網路的所有極點，由描述子矩陣束 `A = [[0, I], [-K, -G]]`、`B = diag(I, C)` 求得，並由錨定的混成極點一對一對應；以及保留 `a`、`b` 之 View 中由 `b` 到 `a` 的阻抗零點。另以建構的封包直接測試傳輸零點最後一次牛頓修正後的選擇：修正點閉合則回傳修正點，未閉合則退回先前已接受的點，修正點的評估失敗則以該失敗回傳 | 以 60 位數 Decimal 精修的廣義特徵值；零點從其錨點以相同算術，分別在加邊 Rosenbrock 系統矩陣與 `Q` 的餘因子兩種形式上精修；由伴隨矩陣求得逐分量的根 `κ` | [`test_pole_oracle.py`](../test_pole_oracle.py) |
| 電報方程式傳輸線 | 有損傳輸線的 π 梯形網路與同一梯形網路的鏈矩陣解比較（至 16 段）；另外檢查段數加倍時，相對精確電報方程式解的誤差嚴格下降 | 60 位數鏈矩陣與雙曲函數 | [`test_telegrapher_line.py`](../test_telegrapher_line.py) |
| 無誤差轉換 | 各平台 wheel 內核的 TwoSum 與 TwoProduct，涵蓋 binary32 與 binary64、正規範圍與次正規範圍；JAX 補償算術只測正規範圍 | 精確 `Fraction` 運算，不涉及容許誤差 | [`test_probe_eft.py`](../test_probe_eft.py) |

新增微分規則而未附測試或記錄原因時，清單測試會失敗。收斂檢查不斷言收斂階數：只要求誤差下降，而且基準端先確認精確梯形網路的改善量大於兩個前向界限。

## 執行預設 unittest 套件

在儲存庫根目錄使用鎖定的課程產生依賴群組：

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

預設套件使用 JAX，不會啟動 Julia。CI 也會在已安裝 wheel 的工作中執行這套測試。
