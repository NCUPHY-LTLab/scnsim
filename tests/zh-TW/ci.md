---
title: "CI 與 Pages"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../ci.md) · [繁體中文](ci.md)

推送至 `develop` 或 `main`、Pull Request 開啟／更新／重新開啟／標記為可審查，以及手動派送時，CI 都會執行。Draft Pull Request 只執行變更分類與 `ci-status`；產品和文件工作會略過。

## 變更分類（Change classification）

只有變更限於根目錄的 `README.md` 或 `docs/` 下的一般 Markdown，且每個變更行都分類為純文字時，才會進入僅適用於這些純文字變更的快速流程。程式碼、受保護標記、可執行內容、無法辨識或混合的路徑，以及不確定的分類，都會選擇產品流程。`tests/README.md` 不在允許清單內，因此變更 Testing 文件會進入產品流程。

判斷方式定義於 [.github/workflows/ci.yml](../../.github/workflows/ci.yml) 與 [.github/scripts/change_scope.py](../../.github/scripts/change_scope.py)。

## 產品工作（Product jobs）

產品流程分為五類工作，Linux／macOS 矩陣展開後共七個工作：

1. **`package-and-docs`** 同步鎖定環境、檢查版本與公開 API 契約，並自動搜尋及執行用途導向的 unittest 測試。它會建置並檢查 wheel 與原始碼散佈套件、確認最低 Python 版本和可攜式 wheel 身分、驗證筆記本衛生與生成的教材圖，並在不執行筆記本程式碼下渲染和檢查文件。
2. **`native-platform-wheels`** 建置並稽核 CPython 3.12 Linux x86_64 manylinux 與 macOS 12 Apple ARM wheel，再檢查已安裝的原生擴充模組是否對應封裝的 C++ 原始碼。macOS 工作使用 macOS 15 主機及 macOS 12 ARM 目標。
3. **`windows-native-wheel`** 使用嚴格 MSVC 浮點設定建置 CPython 3.12 Windows AMD64 wheel，並稽核 wheel 資源。
4. **`windows-jax-wheel-smoke`** 在消費端環境安裝實際 Windows wheel，測試原生核心與公開 Memory／Workspace JAX 路徑、自動搜尋並執行預設測試，並執行 Windows Path Portability Set。
5. **`runtime-preparation-smoke`** 在安裝平台 wheel 的 Linux 與 macOS 環境執行。它涵蓋 Direct 與根計算、敏感度導數、CMA 最佳化、已存結果解析、預設套件及該平台的 Path Portability Set。

## 文件工作與狀態（Documentation jobs and status）

獨立的 **`documentation`** 工作服務於純文字流程。它執行 `uv sync --locked --no-install-project --no-build`，再以 Quarto 1.10.18 在不執行程式碼下渲染。來源檢查會確認本機 include 與 Markdown／QMD 連結目標存在，並確認生成輸出未被追蹤；它不檢查渲染後的 HTML 連結。

`ci-status` 會檢查所選流程要求的工作是否成功，並確認另一流程的工作按預期略過。失敗、取消或非預期略過都不算通過。

## Pages

CI 文件工作只渲染目前來源版本。Pages 工作流程會以展示來源、即時 `develop` 來源與固定版本來源組成版本化網站，再以 Quarto 1.10.18 分別渲染各來源。非 Draft Pull Request 會建置供審查的網站但不部署；Draft Pull Request 會略過建置。推送及符合條件的手動派送可部署組合後的 Pages 網站。規則定義於 [.github/workflows/pages.yml](../../.github/workflows/pages.yml)。
