---
title: "電路輸入與元件"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../circuit-data.md) · [繁體中文](circuit-data.md)

## RLGC 與 Q2D 輸入

測試會建立有序的 RLGC 矩陣輸入並載入 Q2D 來源資料。數值必須保持不可變且順序不變，矩陣須符合物理要求，Q2D 對應須為雙射且符合內容摘要；這些檢查會抓出資料遭修改、矩陣不符合物理條件、對應不完整或來源檔不相符的問題。

來源：[test_rlgc.py](../test_rlgc.py)。

## 型錄元件與耦合

元件案例依型錄宣告組成接面、電容、SQUID、共振器與耦合電感。來源資訊只擷取一次，展開過程保持明確；非正定的電感圖會被拒絕，有效耦合則與獨立的剛度矩陣組裝結果比較。這些檢查會抓出工廠函式被重複呼叫及展開錯誤。

來源：[test_component_models.py](../test_component_models.py)。
