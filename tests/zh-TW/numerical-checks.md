---
title: "數值檢查與本機執行"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../numerical-checks.md) · [繁體中文](numerical-checks.md)

## 前向比較

對宣告的前向比較案例，界限為 `relative_error <= 100 * condition_bound * float64_epsilon`。測試會從案例方程式獨立推導每個條件界限，並回報實際值、參考值、單位、界限及正規化誤差比例。

## 後向與精確檢查

實際運算子與求解的後向殘差使用 `256 * (n + 1) * float64_epsilon`。這些檢查和精確的形狀、順序、編碼值、單位投影、binary64 位元及已儲存結果比較分開處理。套件不會在零相對正規化上額外加 epsilon 下限。目前的測試套件沒有 `StaticCapacitance` 專用數值回歸測試。

## 執行預設套件

在儲存庫根目錄使用鎖定的課程產生套件群組（course-generation），自動搜尋並執行預設 unittest 測試：

```sh
uv sync --locked --group course-generation
uv run --locked --group course-generation python -m unittest discover -s tests -t . -v
```

預設套件會使用 JAX，不會啟動 Julia。120 組 Golden 比較與效能執行另有指派，不屬於預設 CI。
