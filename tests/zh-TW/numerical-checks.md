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

## 執行預設 unittest 套件

在儲存庫根目錄使用鎖定的課程產生依賴群組：

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

預設套件使用 JAX，不會啟動 Julia。CI 也會在已安裝 wheel 的工作中執行這套測試。
