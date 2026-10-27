---
title: "證據（Evidence）與讀回（Readback）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../evidence-readback.md) · [繁體中文](evidence-readback.md)

Memory 會在執行程序中保留結果；Workspace 則將計畫與結果持續儲存在 SQLite 資料庫。證據（evidence）測試追蹤小型計算經過編碼、完成、關閉與讀回的流程，確認讀取時使用已儲存的值與失敗，而非重新計算。SCNSim 軟體版本（software version）識別套件發布版本；證據格式（evidence format）版本 `v3` 則描述陣列欄位的編碼方式，兩者是不同的版本識別。

## Evidence v3 儲存什麼

`v3` 證據格式會以型別化陣列欄（typed array columns）編碼精確的 64 位元浮點值（binary64）。案例包含負零位元 `0x8000000000000000`，以及帶有指定 NaN 有效位元內容（payload）的 `0x7ff8000000000042`；解碼後必須保留原位元。

1. 編碼並解碼型別化欄位，再比對精確的 binary64 表示。
2. 翻轉標頭或 payload 位元。讀取器必須引發 `EvidenceIntegrityError`，不能回傳看似正常的數值。
3. 截短區塊、移除物件，或中斷成功指標（success pointer）交易。未完成的寫入不得發布為成功，較早完成的區塊仍須可讀。不支援的證據格式必須在 Workspace 變更前遭拒。

來源：[test_evidence_v3.py](../test_evidence_v3.py)。

## 部分掃描（partial sweep）保留順序與失敗

三點掃描以 `[110, 1e308, 110]` fF 要求 6 GHz 諧振器的純量阻抗響應。

1. 分別在 Memory 與 Workspace 評估此網格。中間列在導納轉換（transform）為阻抗時失敗，前後兩列成功。
2. 確認儲存列序為成功／失敗／成功，來源索引依序為 `(0,)`、`(1,)`、`(2,)`。兩個相同端點值仍須是不同列。
3. 關閉並重新開啟 Workspace，再於新程序讀回部分結果；已存數值與失敗必須對應原始列。

來源：[test_evidence_v3.py](../test_evidence_v3.py)。

## Workspace 身分與新程序讀回

1. 升級套件版本並保留原計畫葉節點；變更計畫後確認建立新節點；要求禁止降級時，確認它在變更前遭拒。
2. 重用共享計畫身分，並在關閉、重新開啟 SQLite 資料庫後解析已完成結果。
3. 最佳化讀回先執行一次 CMA 或 L-BFGS-B，再檢查暖快取（warm cache）重用，關閉 Workspace，並在新程序解析已存歷程。快取值與身分必須完全相同，之後呼叫 `optimize` 或 `resolve` 不得增加原生工作。

這些步驟可抓出過期身分、最佳化歷程遺失，以及讀回時意外重新計算。來源：[test_workspace_readback.py](../test_workspace_readback.py)、[test_workspace_optimization_readback.py](../test_workspace_optimization_readback.py) 與 [test_evidence_v3.py](../test_evidence_v3.py)。
