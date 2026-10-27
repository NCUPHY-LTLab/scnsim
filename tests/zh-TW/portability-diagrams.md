---
title: "路徑（Paths）、電路圖（Diagrams）與教材輸出（Teaching outputs）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../portability-diagrams.md) · [繁體中文](portability-diagrams.md)

這組測試檢查可攜式路徑，以及依宣告電路建立的呈現資料。圖面或筆記本準備測試本身不會驗證 CAD 幾何，也不會執行物理求解器。

## 跨平台檔案路徑

路徑案例會建立並讀取暫存檔，資料夾名稱包含空格、繁體中文字、字面百分號，以及字串 `%20`。測試檢查本機檔案路徑，以及 authority（URI 主機欄位）為空或 `localhost` 的 `file:` URI（file URI）。URI 路徑只解碼一次：檔名中原有的 `%20` 必須保留為字面文字，解析後的路徑及檔案內容都須相符。

1. 在暫存位置建立並讀取檔名為 `內容 %20.txt` 的檔案，檢查原生路徑與 `file:` URI。
2. 在 Windows 檢查磁碟代號與 UNC（Universal Naming Convention）路徑轉換；其他平台略過這些平台專屬案例。測試只檢查路徑字串，不會連線到網路分享。
3. 讓必要路徑無法讀取，確認原始失敗原因保留。來源：[test_path_portability.py](../test_path_portability.py)。

## 圖面必須符合電路宣告

單一諧振器圖面使用 110 fF、5.8 nH、6 fF 耦合電容及 50 Ω 端口，以自動配置（automatic layout）與深色模式（dark mode）繪製指定的 120 fF 點。圖面稽核（schematic audit）會檢查參數與呈現選項不會改變計畫宣告的連線；`show` 必須使用目前宣告。

多導體傳輸線（multiconductor transmission line, MTL）圖面使用 1.6 mm、兩段的 readout/filter 線路，head 與 tail 端口依序分開。測試會調換電容的物理端點或把 MTL 接腳接到錯誤端點；圖面稽核必須拒絕這些圖，且後續無效配置不能取代先前已宣告的圖。缺少跨線跳接（crossing jumper）也會遭拒。這些是連線檢查，不是導體尺寸量測。來源：[test_manual_diagrams.py](../test_manual_diagrams.py)。

## 筆記本與課程準備

筆記本可攜性轉換器（notebook portability converter）只移除機器本機的 `kernelspec.path`，保留儲存格（cell）內容、可攜的核心（kernel）身分與其他解析欄位。課程準備案例會執行宣告章節輸入，以及指定的 Direct trace，接著在求解前停止；作者繪製的電路圖可以輸出稽核證書（audit certificate）。

發布綁定案例保留外部命名空間（namespace），並確保清理錯誤不會取代原始錯誤。這些測試可抓出綁定單一機器的筆記本、文件準備時意外啟動求解器，以及清理流程掩蓋主要失敗。來源：[test_portable_notebook.py](../test_portable_notebook.py)、[test_engineer_chapter3_generation.py](../test_engineer_chapter3_generation.py)、[test_engineer_chapter4_generation.py](../test_engineer_chapter4_generation.py) 與 [test_engineer_publication_binding.py](../test_engineer_publication_binding.py)。
