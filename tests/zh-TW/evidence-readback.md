---
title: "證據（Evidence）與讀回（Readback）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../evidence-readback.md) · [繁體中文](evidence-readback.md)

Memory 會在執行程序中保留結果；Workspace 則將計畫與結果持續儲存在 SQLite 資料庫。證據（evidence）測試追蹤小型計算經過編碼、完成、關閉與讀回的流程，確認讀取時使用已儲存的值與失敗，而非重新計算。SCNSim 軟體版本（software version）識別套件發布版本；證據格式（evidence format）版本 `v4` 則描述陣列欄位的編碼方式，兩者是不同的版本識別。

## Evidence v4 儲存什麼

`v4` 證據格式會以型別化陣列欄（typed array columns）編碼精確的 64 位元浮點值（binary64）。案例包含負零位元 `0x8000000000000000`，以及帶有指定 NaN 有效位元內容（payload）的 `0x7ff8000000000042`；解碼後必須保留原位元。

1. 編碼並解碼型別化欄位，再比對精確的 binary64 表示。
2. 翻轉標頭或 payload 位元。讀取器必須引發 `EvidenceIntegrityError`，不能回傳看似正常的數值。
3. 截短區塊、移除物件，或中斷成功指標（success pointer）交易。未完成的寫入不得發布為成功，較早完成的區塊仍須可讀。不支援的證據格式必須在 Workspace 變更前遭拒。
4. 取一個已存有一筆 Direct 結果的完整 Workspace，分別在根封套（root envelope）、葉節點封套或資料庫中宣告證據版本 3。開啟時必須引發 `UnsupportedEvidenceVersionError`，且不得變更任何位元組。版本 3 的區塊、標頭與清單（manifest）同樣不會被讀取。

來源：[test_evidence_v4.py](../test_evidence_v4.py)。

## Direct 摘要與 Direct 呼叫留下的內容

Direct 結果把 S、Y、Z 陣列存在資料分塊（chunk），並把同一次呼叫的方程式證書（certificate）存在分塊摘要（chunk summary）。摘要對每個證書欄位只存一次，也就是一個涵蓋該分塊所有工作列（work row）的陣列；一個工作列對應一個已求解的頻率。

1. 把帶有負零、NaN 有效位元內容、無限大與向量值欄位的工作列彙整成欄，儲存後再讀回各列。每個位元都必須原樣回來；證書不同的列必須遭拒。
2. 以分塊大小 3 與 8 求解八個頻率。涵蓋 3、2 與 8 個工作列的摘要必須儲存相同的欄：每個工作計數一欄，每個證書欄位一欄。
3. 逐一比對每個已存的欄在各工作列的切片，與求解器在同一次呼叫回傳的證書值。關閉後與在新程序中會再讀一次已存的值，而且求解器的值在 Memory 與 Workspace 之間必須一致。
4. 分別執行一次完成的 Direct 呼叫、一次在最後一個頻率失敗的呼叫、一次在求解途中被中斷的呼叫，以及一次在寫入結果途中被中斷的呼叫。之後操作資料庫不得留有暫存物件（staging object），也不得留有未完成呼叫的結果物件。唯一沒有任何參照的物件，是操作列與嘗試（attempt）的較早狀態；儲存庫更新這兩筆記錄時只移動指標。

來源：[test_evidence_v4.py](../test_evidence_v4.py)。

## 部分掃描（partial sweep）保留順序與失敗

三點掃描以 `[110, 1e308, 110]` fF 要求 6 GHz 諧振器的純量阻抗響應。

1. 分別在 Memory 與 Workspace 評估此網格。中間列在導納轉換（transform）為阻抗時失敗，前後兩列成功。
2. 確認儲存列序為成功／失敗／成功，來源索引依序為 `(0,)`、`(1,)`、`(2,)`。兩個相同端點值仍須是不同列。
3. 關閉並重新開啟 Workspace，再於新程序讀回部分結果；已存數值與失敗必須對應原始列。

來源：[test_evidence_v4.py](../test_evidence_v4.py)。

## Workspace 身分與新程序讀回

1. 升級套件版本並保留原計畫葉節點；變更計畫後確認建立新節點；要求禁止降級時，確認它在變更前遭拒。
2. 重用共享計畫身分，並在關閉、重新開啟 SQLite 資料庫後解析已完成結果。
3. 最佳化讀回先執行一次 CMA 或 L-BFGS-B，再檢查暖快取（warm cache）重用，關閉 Workspace，並在新程序解析已存歷程。快取值與身分必須完全相同，之後呼叫 `optimize` 或 `resolve` 不得增加原生工作。

這些步驟可抓出過期身分、最佳化歷程遺失，以及讀回時意外重新計算。來源：[test_workspace_readback.py](../test_workspace_readback.py)、[test_workspace_optimization_readback.py](../test_workspace_optimization_readback.py) 與 [test_evidence_v4.py](../test_evidence_v4.py)。
