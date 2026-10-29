# SCNSim

**SCNSim 正在密集開發中。**

**描述超導電路、選擇觀測網路，並計算所需的響應或物理量。**

[![Apache-2.0 授權](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 需求：>=3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](pyproject.toml)
[![工程課程](https://img.shields.io/badge/tutorial-Engineer%20course-blue.svg)](https://ncuphy-ltlab.github.io/scnsim/)

![SCNSim 虎鯨與企鵝識別圖。](docs/assets/readme-hero-orca-penguin.png)

SCNSim 是用於超導電路網路分析的 Python 套件。先建立明確的實體 `CircuitPlan`，再選擇觀測邊界 `View`，並由 `CircuitRun` 計算響應或物理量。

## 功能入口

- [組合實體 Plan](docs/concepts/physical-authority-and-reusable-composition.qmd) — 在 `CircuitPlan` 中連接元件，並以巢狀 `Subsystem` 整理可重用的實體單元。
- [選擇 View 並保留座標](docs/concepts/compilation-coordinates-and-network-views.qmd) — 用 `View` 選定分析邊界，再以 `retain()` 選擇該計算保留的座標。
- [計算響應、根、靈敏度與最佳化](docs/guides/roots-and-branch-selection.qmd) — 請求 S、Y 或 Z 響應，評估特徵量與根，計算局部靈敏度，或最佳化已宣告的目標。
- [將結果保留在記憶體或 Workspace](docs/concepts/requests-execution-and-result-ownership.qmd) — 互動計算可使用記憶體結果；需要支援的保存、精確讀回與延續時，請使用 Workspace。
- [非線性 Josephson 電路仍在設計中](docs/component-authoring.qmd) — 目前的 Josephson 接面原始元件採用線性化模型；完整的非線性電路流程仍在設計中。

## 安裝

安裝方式與支援環境請參閱[環境設定指南](examples/engineer/setup.qmd)；從來源建置的方法請參閱[貢獻指南](docs/contribute.qmd#build-from-source)。

[查看文件總覽與範例](https://ncuphy-ltlab.github.io/scnsim/)。

## 從第 1 個電路開始

第 1 章先在 `Subsystem` 中建立接地 LC 諧振器，再由父層連接耦合電容與終端量測 Port。同一 Plan 可用來檢視單埠響應與負載根。

![第 1 章 Plan：父層耦合器與終端 Port 連接至接地 LC 諧振器 Subsystem。](examples/engineer/figures/chapter1-plan.light.svg)

`Subsystem` 也可再包含子系統。邊界顯示元件的所有權，父層透過子層公開的 Pin 連接電路。

![父層模組包含具有電容、電感與局部接地的 LC 單元。](examples/engineer/figures/nested-subsystems.light.svg)

## 用 View 看差模（Differential mode）

兩個節點磁通座標的差模座標為 `Phi_a - Phi_b`。`transform_pair()` 轉換座標基底，再由 `retain()` 選擇差模座標作為 View。保留的 View 透過 Schur 補數（Schur complement）消去共模座標，並在差模座標上計算所請求的根。

![兩節點浮接 LC；Ca、Cb 接地，Cm 與 L 並聯連接 Phi_a 和 Phi_b。](examples/engineer/figures/floating-simple.light.svg)

此延伸範例建立 View，並請求差模座標的頻率；完整程式碼請見[文件總覽](https://ncuphy-ltlab.github.io/scnsim/)：

完整內容請見[浮接模型](examples/engineer/figures/floating_model_simple.py)與 [View 範例](examples/engineer/figures/floating_view_simple.py)。

## 第 1 章 S11 響應

Port 1 是 `signal_in`；S11 表示同一量測 Port 的反射。此理想無損單埠模型在指定頻率網格上的振幅接近 0 dB，相位則呈現響應特徵。

![Port 1 `signal_in` 的 S11 振幅（Magnitude，dB）。](examples/engineer/figures/chapter1-s11-magnitude.svg)

![Port 1 `signal_in` 的 S11 相位（Phase，度）。](examples/engineer/figures/chapter1-s11-phase.svg)

互動式 S11 [振幅圖](https://ncuphy-ltlab.github.io/scnsim/)與[相位圖](https://ncuphy-ltlab.github.io/scnsim/)可在文件中開啟。
完整的 S11 模型、求解與繪圖程式碼收合在[英文 README 的第 1 章 S11 範例](../README.md#chapter-1-s11-response)。

## 套件內建知識入口

<!-- agent-knowledge-start:begin -->
SCNSim 以實體 `CircuitPlan` 描述電路；`View` 選擇分析邊界，`Spec` 指定所需物理量。請先閱讀[工程課程](docs/index.qmd)，再用[概念總覽](docs/concepts/index.qmd)了解電路組成與分析 View。
<!-- agent-knowledge-start:end -->

## 接續閱讀

閱讀[輸入與單位](docs/tutorials/inputs-units.qmd)、瀏覽[工程課程](https://ncuphy-ltlab.github.io/scnsim/)，或前往[貢獻指南](docs/contribute.qmd)查找來源建置與開發說明。

## 授權條款

Apache-2.0；請參閱 [LICENSE](LICENSE) 與 [NOTICE](NOTICE)。
