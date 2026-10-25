---
title: "參數與最佳化"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../parameters-optimization.md) · [繁體中文](parameters-optimization.md)

## 仿射授權

仿射案例透過 Memory 與 Workspace 提交一般未授權點及已授權的扇出點。未授權外插必須在建立請求前失敗；已授權列會保留支援邊與授權來源，重複點保留原始列，CMA 也會記錄拒絕與授權候選。這些檢查會抓出證據缺漏或歸屬錯誤。

來源：[test_affine_parameters.py](../test_affine_parameters.py)。

## 區域座標尺度

有界區域座標案例涵蓋明確指定尺度、參數範圍尺度、解析後的邊界寬度，以及既有的 signed-log 對應。C1 案例會執行兩種區域最佳化器，要求至少完成迭代且結果收斂，並回報獨立解析損失的改善幅度與剩餘目標距離；它不設定目標距離容差。這些檢查會抓出尺度優先順序及最佳化器接線錯誤。

來源：[test_local_coordinate_defaults.py](../test_local_coordinate_defaults.py)。

## CMA 選擇器

CMA 案例確認七種已宣告選擇器都出現在 JAX 說明中，且各選擇器分支（包括零成本基準案例）都會執行。測試不要求不同後端產生相同的候選軌跡。

來源：[test_optimization.py](../test_optimization.py)。
