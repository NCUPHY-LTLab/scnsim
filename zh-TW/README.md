# SCNSim

**建立超導電路模型、計算響應（Response），並探索參數如何影響你關心的物理量（Physical quantity）。**

[![CI 狀態](https://github.com/OrPenStrike/scnsim/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/OrPenStrike/scnsim/actions/workflows/ci.yml?query=branch%3Adevelop)
[![授權：Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 需求：>=3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](pyproject.toml)
[![工程課程（Tutorial）](https://img.shields.io/badge/tutorial-Engineer%20course-blue.svg)](docs/index.qmd)

![SCNSim 虎鯨與企鵝品牌圖。](docs/assets/readme-hero-orca-penguin.png)

SCNSim 是以 Notebook 為主的超導電路網路分析 Python 套件。請在 `CircuitPlan` 描述元件與連線，透過 View 選擇觀測的網路，再使用 `CircuitRun` 計算具型別且帶有物理單位的 Result。同一模型可用於響應、特徵量、導數與參數搜尋。

<!-- agent-knowledge-start:begin -->
## 套件內建知識入口

SCNSim 以一個 `CircuitPlan` 描述實體電路。View 選擇分析邊界；Spec 指定所需物理量。先宣告模型，再選擇 `CircuitRun` 操作。本節是已安裝知識套件的 `getting-started` 入口。

### 選擇入門任務

| 任務 | 說明與資源識別碼 |
|---|---|
| 宣告有限電容並檢視端點 | [電容（英文）](/en/docs/reference/agent-catalogue/components/capacitor.md) — `catalogue/components/capacitor` |
| 宣告電感與電感支路 | [電感（英文）](/en/docs/reference/agent-catalogue/components/inductor.md) — `catalogue/components/inductor` |
| 使用 LJ0 與獨立 Cj 宣告 Josephson 接面 | [Josephson 接面（英文）](/en/docs/reference/agent-catalogue/components/josephson_junction.md) — `catalogue/components/josephson_junction` |
| 定義、綁定並替換實體輸入 | [參數系統（英文）](/en/docs/contracts/parameter-system.html) — `contracts/parameter-system` |
| 登錄元件本體並組裝連線 | [電路 Plan 組裝（英文）](/en/docs/contracts/circuit-plan-assembly.html#structured-authoring-grammar) — `contracts/circuit-plan-assembly` |
| 選取有序保留座標 | [View 與 retain](docs/concepts/compilation-coordinates-and-network-views.qmd#network-view-lineage) — `compilation-coordinates-and-network-views` |
| 請求相同 View 的靜態矩陣或元素 | [靜態電容](docs/guides/static-capacitance.qmd) — `static-capacitance` |
| 提供每單位長度數值矩陣與導體順序 | [RLGC（英文）](/en/docs/reference/agent-catalogue/api/rlgc.md) — `catalogue/api/rlgc` |
| 選擇單位並解讀頻率與 Result 轉換 | [單位](docs/concepts/simulation-conventions.qmd#phasors-and-units) — `simulation-conventions` |

英文[公開目錄](/en/docs/reference/agent-catalogue/index.md)（`catalogue/index`）列出匯出的名稱與內建工廠；其建立觀察只適用於記錄中的輸入。[元件建立指南](docs/component-authoring.qmd)（`component-authoring`）說明 Library 所有權與物理模型。英文[此套件以外的已知公開文件](/en/docs/reference/agent-catalogue/known-public-documents.md)（`catalogue/known-public-documents`）依據產生端宣告的來源清單；單一未解析連結不能證明文件存在。

### 輸入、單位與常見錯誤

物理 Quantity 請使用 `from scnsim import units as u`，例如 `110 * u.fF`；單獨的數字並不是電容。固定值直接使用常數。只有在實體元件欄位確實使用輸入時，才使用 `ParameterDefinitions` 與 `ParameterSpec`；`ParameterSet` 只為這些參照提供值，不會建立綁定。一般頻率使用 Hz，角頻率使用 rad/s。請轉換 Result 的 Quantity 投影，例如 `root_result.frequency.to("GHz")` 或 `matrix_result.matrix.to("farad")`，不要只替數值改標籤。

連通關係由明確宣告的 Pin、Bus 與接地定義；電路圖不會建立電氣連線。傳輸線需要實際數值 RLGC 矩陣、導體順序與參考導體，以及物理長度；圖面上的線長不是這個長度。Q2D 載入器接受文件定義的分佈式 RLGC 格式，不接受任意集中電容匯出。

`retain()` 會在 View 中選擇座標。靜態 C 被消去的有限補集滿足零電荷條件；動態請求則保留各自的方程式。`StaticCapacitanceSpec()` 沒有 `zero_charge` 輸入。Root Hint 是局部起始值，不是目標值，也不代表搜尋所有根。

### 執行環境與 Result 所有權

請安裝與 Python 版本及平台相符的精確套件版本。數值執行使用 JAX CPU；Notebook 呈現另有選用相依套件。執行前請閱讀[環境設定指南](examples/engineer/setup.qmd)（`engineer-setup`）。閱讀此內容不會載入產生端套件，也不會執行模擬。

`CircuitRun(plan=plan, storage="memory")` 回傳獨立的記憶體（In-memory）Result；`CircuitRun(plan=plan, workspace=path)` 則儲存支援的操作，供精確讀回。這是兩種不同的保存方式。讀取已保存證據或繼續最佳化前，請先看[請求、執行與 Result 所有權](docs/concepts/requests-execution-and-result-ownership.qmd)（`requests-execution-and-result-ownership`）。`run.close()` 後仍可讀取已回傳的公開值；記憶體 Result 不會取得已保存的請求／嘗試識別碼。

SCNSim 負責電路宣告與網路計算。電磁或材料輸入的萃取證據應由產生該資料的套件提供，SCNSim 無法從矩陣推知來源。量子 Hamiltonian 或脈衝動力學應使用相應的電路量子化／動力學工具，不要將它們偽裝成 SCNSim 網路結果。使用已安裝知識套件時，請引用套件版本、bundle ID、資源 ID 與來源資訊；若內容缺失，請透過套件的 Issues 連結回報，不要虛構 API。
<!-- agent-knowledge-start:end -->

## 你想計算什麼？

| 計算內容 | 從這裡開始 |
|---|---|
| **電路響應** — S、Y、Z 矩陣，或具名反射／傳輸軌跡 | `run.solve(view, DirectSolveSpec(...))` |
| **特徵量** — 選定運算子、根、極點、傳輸零點或耦合 | `run.evaluate(view, spec, parameters=point)` |
| **靜態電容** — 編譯後有限電容矩陣、View 選取的縮減結果或座標元素 | `run.evaluate(view, StaticCapacitanceSpec(), parameters=point)` |
| **靈敏度** — 物理量如何隨電路參數變化 | `run.sensitivity(SensitivitySpec(...))`，可計算 Jacobian、JVP 或 VJP |
| **參數探索** — 比較宣告的掃描或搜尋純量目標 | 將 `ParameterSpace` 傳給 solve／evaluate，或使用 `run.optimize(view, OptimizationSpec(...))` |

選擇 API 前，請閱讀[概念總覽](docs/concepts/index.qmd)，了解物理模型、View 與量值的關係。[根與分支選擇指南](docs/guides/roots-and-branch-selection.qmd)說明取樣凹口、複數零點與局部 Hint 回答的是不同問題；[靈敏度指南](docs/guides/sensitivity.qmd)說明單位與局部導數。最佳化方法包含 CMA-ES、L-BFGS-B、先評估目標值的投影式 L-BFGS，以及依序執行的 Hybrid 階段，請依目標與可用導數選擇。[靜態電容指南](docs/guides/static-capacitance.qmd)說明有限矩陣、View 選取的零電荷縮減，以及支援的靈敏度。

## 選擇記憶體（In-memory）或 Workspace

兩種模式都使用相同的 JAX 物理程式。互動式計算可使用 `CircuitRun(plan=plan, storage="memory")`，將 Result 保留在 RAM 而不建立 Workspace。若要保存計算、再次讀回或稍後繼續支援的最佳化，請使用 `CircuitRun(plan=plan, workspace=path)`。預設使用 JAX CPU 與 Float64／Complex128。

每個操作結束時都會自動釋放暫時數值資源。`run.close()` 後仍可使用已回傳的 Result，因此繪圖、比較與報告不必保持 Run 開啟。儲存模式專屬的讀取、復原與繼續能力，請參閱[請求、執行與 Result 所有權](docs/concepts/requests-execution-and-result-ownership.qmd)。

重複計算時，`run.warmup(...)` 可預先準備已知的可執行工作，而不求解模型。熟悉 JAX 的使用者可透過 `run.prepare(...)` 取得分離的可呼叫物件。英文[效能分析指南](docs/guides/benchmark.qmd)可協助理解計算時間花在哪裡。這些工具支援上述計算，但不會改變所請求的物理量。

## 第一個 Direct S11 結果

第 1 章範例建立接地並聯 LC 諧振器，透過耦合器連接一個 50 Ω 終端量測 Port，並請求具名反射軌跡。物理模型與 110 fF／120 fF 數值來自工程課程。

{{< include _generated/canonical-code/README.md/block-001.qmd >}}

程式使用 JAX 執行環境。下方圖表是此電路案例與通道的歷史第 1 章 Julia 輸出；新的課程圖表會在 JAX 執行後另行發布。

![110 fF 基準點在 5.75–6.25 GHz、401 點網格上的已發布 S11。](examples/engineer/figures/01-s11-baseline.svg)

*已保存的第 1 章基準 Result：110 fF。振幅軸固定顯示 ±0.05 dB；包裹相位的不連續是呈現方式，不是求解器失敗。[精確複數取樣值](examples/engineer/figures/01-s11-data.csv)。*

![同一網格上 120 fF 選定點的已發布 S11。](examples/engineer/figures/01-s11-selected.svg)

*已保存的第 1 章選定 Result：120 fF；電感仍為 5.8 nH，耦合器為 6 fF，Port 為 50 Ω，並使用相同的 401 點網格。本圖不推斷凹陷或諧振。[精確複數取樣值](examples/engineer/figures/01-s11-data.csv)。*

請繼續閱讀[工程課程](docs/index.qmd)。

## 安裝

請使用 Python 3.12 與符合平台的精確版本 SCNSim 輪子。目前開發版本為 `1.0.0.dev42`；1.0 beta 尚未發布。基本相依套件已包含 JAX CPU，不需要 backend 關鍵字、Julia 安裝或 Julia extra。

在使用 SCNSim 的專案中，請設定精確的 SCNSim 輪子來源，並將此選擇保留於 `uv.lock`。接著安裝鎖定環境：

{{< include _generated/canonical-code/README.md/block-002.qmd >}}

Windows 請使用 CPython 3.12 AMD64 與相符的 Windows 輪子；同一命令也可在 PowerShell 執行。Linux 與 macOS 同樣需要符合 Python ABI 與處理器架構的輪子。

安裝相符的 SCNSim 預建輪子不需要 C++ 編譯器；從來源安裝 SCNSim 則需要 C++17 工具鏈。依輪子所需的 MSVC 執行階段，Windows 也可能需要相符的 Microsoft Visual C++ Redistributable；這與編譯器工具不同，請參閱[Microsoft 支援的 Redistributable 指引](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)。CPU JAX 請參閱[JAX CPU 安裝說明](https://docs.jax.dev/en/latest/installation.html#cpu)。Notebook 環境與來源安裝方式請參閱[環境設定指南](examples/engineer/setup.qmd)。

## 安裝歷史

先前交付的 dev39 輪子及其 Windows CPU-JAX 觀察屬於 dev39 歷史證據，不是 dev42 安裝。

## 先理解模型，再選擇計算

先閱讀[概念總覽](docs/concepts/index.qmd)建立心智模型，或依照[工程課程](docs/index.qmd)建立並計算範例。遇到相關問題時，請參閱英文[根與分支選擇](docs/guides/roots-and-branch-selection.qmd)、[靈敏度](docs/guides/sensitivity.qmd)與[效能分析](docs/guides/benchmark.qmd)指南。

## 授權條款

Apache-2.0；請參閱 [LICENSE](LICENSE) 與 [NOTICE](NOTICE)。
