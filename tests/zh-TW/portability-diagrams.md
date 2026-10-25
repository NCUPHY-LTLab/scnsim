---
title: "路徑、電路圖與教材輸出"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../portability-diagrams.md) · [繁體中文](portability-diagrams.md)

## 跨平台與儲存模式的路徑

路徑案例涵蓋原生檔案系統路徑、file URI、Windows 磁碟代號與 UNC 轉換、套件來源資訊，以及 Memory／Workspace 路徑生命週期。必要路徑無法讀取時，必須保留嚴格失敗原因；測試會抓出重複解碼及平台特有的路徑解讀錯誤。

來源：[test_path_portability.py](../test_path_portability.py)。

## 手動畫電路圖

手動畫圖會依目前電路宣告檢查。呈現設定不得改變連線；`show` 必須使用目前宣告。錯誤的導線端點、MTL 接腳、缺少的交越跳線，或來自另一個 plan 的配置都會遭拒，藉此抓出與宣告電路不一致的圖。

來源：[test_manual_diagrams.py](../test_manual_diagrams.py)。

## 筆記本與課程輔助程式

筆記本與課程輔助程式案例保留儲存格和核心身分，同時移除機器本機的核心路徑；它們會準備宣告的章節請求，但不進行求解，並為作者繪製的電路圖匯出證書。發布綁定案例保留外部命名空間，並避免清理錯誤取代主要錯誤。這些檢查會抓出不可攜的筆記本、文件準備時意外求解，以及不安全的清理流程。

來源：[test_portable_notebook.py](../test_portable_notebook.py)、[test_engineer_chapter3_generation.py](../test_engineer_chapter3_generation.py)、[test_engineer_chapter4_generation.py](../test_engineer_chapter4_generation.py) 與 [test_engineer_publication_binding.py](../test_engineer_publication_binding.py)。
