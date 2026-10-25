---
title: "證據與讀回"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../evidence-readback.md) · [繁體中文](evidence-readback.md)

## Evidence v3

Evidence-v3 案例保留 binary64 原始值並偵測位元組損毀。短寫入、缺漏或中斷的寫入不得發布成功；不支援的證據須在修改 Workspace 前遭拒；部分參數掃描會保留失敗及來源順序。Direct、Operator、根、敏感度與最佳化結果在關閉後及新程序中解析時，必須精確保留已儲存的位元。

來源：[test_evidence_v3.py](../test_evidence_v3.py)。

## Workspace 身分與解析

Workspace 案例涵蓋版本升級、禁止降級、共享 plan 身分、暖快取重用，以及受控的新程序解析。最佳化歷程在關閉後仍可解析；重複呼叫 optimize／resolve 不得在原始執行後增加原生工作。這些檢查會抓出身分過期、意外重新計算及結果遺失。

來源：[test_workspace_readback.py](../test_workspace_readback.py) 與 [test_workspace_optimization_readback.py](../test_workspace_optimization_readback.py)。
