---
title: "電路輸入（Circuit inputs）與元件（Components）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../circuit-data.md) · [繁體中文](circuit-data.md)

## RLGC 與 Q2D 輸入（Inputs）

測試會建立有序的 RLGC 矩陣輸入並載入 Q2D 來源資料。數值必須保持不可變且順序不變，矩陣須符合物理要求，Q2D 對應須為雙射且符合內容摘要；這些檢查會抓出資料遭修改、矩陣不符合物理條件、對應不完整或來源檔不相符的問題。

來源：[test_rlgc.py](../test_rlgc.py)。

## 型錄元件（Catalog components）與耦合（Coupling）

元件工廠（Component factories）依型錄宣告組成接面、電容、SQUID、共振器與耦合電感（coupled inductors），並檢查電感耦合（Inductive coupling）。來源資訊（provenance）只擷取一次，展開過程（expansions）保持明確；非正定（non-positive-definite）的電感圖會被拒絕，有效耦合則與獨立的剛度矩陣（stiffness matrix）組裝結果比較。這些檢查會抓出工廠函式被重複呼叫及展開錯誤。

來源：[test_component_models.py](../test_component_models.py)。
