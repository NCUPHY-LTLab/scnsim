---
title: "網路方程式（Network equations）與根（Roots）"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../network-roots.md) · [繁體中文](network-roots.md)

## Direct 網路與物理量（Physical quantities）

Direct 分析案例保留宣告順序，並在選取的縮減（Reduction）、轉換（Transform）與探測（Probing）下，保留矩陣值的多導體（Multiconductor）與 n-port 網路。運算子（Operator）、響應（Response）、極點（Poles）、留數（Residues）、耦合（Coupling）與傳輸零點（Transfer zeros）會和獨立案例計算比較，可抓出純量化、順序錯誤及方程式組裝錯誤。

來源：[test_direct_analysis.py](../test_direct_analysis.py)。

## 導納（Admittance）與導數（Derivatives）

導納案例把耗散（Dissipation）網路與獨立基準（independent oracle）及 NumPy Schur 縮減（Schur reduction）結果比較，並改變參考阻抗（Reference impedance）以確認它不會憑空造成元件損耗。導數（Derivatives）與匹配散射零點（matched-scattering-zero）案例會檢查指定響應（requested response）及解析頻率導數組（analytic frequency jet），可抓出損耗計算（loss accounting）錯誤和保留座標（retained-coordinate）錯誤。

來源：[test_direct_admittance.py](../test_direct_admittance.py)。

## 根（Roots）與耦合（Coupling）證書（Certificates）

根案例（Root cases）在兩種座標順序（coordinate orders）下，將高 Q 極點（high-Q poles）與全域二次特徵值（Global quadratic eigenvalue）計算比較。無損（Lossless）極點使用五種錨點（Anchor choices）：實數錨點（real anchor），以及兩種尺度下正負虛數擾動（positive/negative imaginary perturbations）；頻率位元（frequency bits）必須完全相同。重複模態（Repeated modes）、零斜率（zero slopes）或不可用留數（unusable residues）不得取得單根證書（simple-root certificates）。

這些案例會抓出符號捨入及錯誤認證。來源：[test_root_certificates.py](../test_root_certificates.py)。
