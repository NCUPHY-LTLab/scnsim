---
title: "錯誤處理與診斷"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../diagnostics.md) · [繁體中文](diagnostics.md)

## 原生錯誤路由

錯誤路由案例會讓已歸屬的純量、批次及封存方程式錯誤通過各自包裝層。原始錯誤內容必須保留；未知或混合錯誤會重新拋出，不合法的包裝層則遭拒。這些檢查會抓出被隱藏的原生錯誤及不支援的分類。

來源：[test_native_failure_routing.py](../test_native_failure_routing.py)。

## 求解診斷

求解診斷會比較逐右手邊的分量指標與彙總指標，並涵蓋密集、稀疏與補償求解路徑。診斷資訊必須描述實際求解並保留其輔助資料，可抓出掩蓋失敗右手邊的彙總方式，以及與求解脫節的診斷資料。

來源：[test_solve_diagnostics.py](../test_solve_diagnostics.py)。
