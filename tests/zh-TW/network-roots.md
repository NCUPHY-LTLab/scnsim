---
title: "網路方程式與根"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../network-roots.md) · [繁體中文](network-roots.md)

## Direct 網路與物理量

Direct 分析案例保留宣告順序，並在選取的縮減、轉換與探測下，保留矩陣值的多導體與 n-port 網路。運算子、響應、極點、留數、耦合與傳輸零點會和獨立案例計算比較，可抓出純量化、順序錯誤及方程式組裝錯誤。

來源：[test_direct_analysis.py](../test_direct_analysis.py)。

## 導納與導數

導納案例把耗散網路與獨立參考計算及 NumPy Schur 縮減結果比較，並改變參考阻抗以確認它不會憑空造成元件損耗。導數與匹配散射零點案例會檢查指定響應及解析頻率導數組，可抓出損耗計算錯誤和保留座標錯誤。

來源：[test_direct_admittance.py](../test_direct_admittance.py)。

## 根與耦合證書

根案例在兩種座標順序下，將高 Q 極點與全域二次特徵值計算比較。無損極點使用五種錨點：實數錨點，以及兩種尺度下正負虛數擾動；頻率位元必須完全相同。重複模態、零斜率或不可用留數不得取得單根證書。

這些案例會抓出符號捨入及錯誤認證。來源：[test_root_certificates.py](../test_root_certificates.py)。
