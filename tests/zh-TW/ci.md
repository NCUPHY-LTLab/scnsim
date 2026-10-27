---
title: "CI 與 Pages"
lang: zh-TW
sidebar: testing-zh-tw
---

[English](../ci.md) · [繁體中文](ci.md)

持續整合（CI）會檢查宣告的 unittest 套件、已安裝的平台 wheel 套件、套件內容與文件。推送到 `develop` 或 `main`、開啟／更新／重新開啟／標記為可審查的 Pull Request，以及手動派送都會啟動工作流程；草稿 Pull Request（draft pull request）只執行變更分類與 `ci-status`。

## 變更如何選擇工作流程

變更分類器（change classifier）提供範圍很窄的純文字快速流程（prose-only lane）。變更必須只涉及根目錄 `README.md` 或 `docs/` 下的一般 Markdown，而且每一行變更都必須分類為純文字。程式碼、QMD 或 IPYNB 檔案、共用設定、依賴變更、受保護標記、可執行內容、不明路徑、混合變更及不確定分類都會選擇產品流程。`tests/README.md` 不在純文字允許清單內，因此修改 Testing 指南會進入產品流程。

產品流程共有七個工作，由五類工作組成。`package-and-docs` 會同步鎖定環境、檢查版本與公開 API 契約、自動搜尋並執行用途導向的 unittest 套件、檢查 wheel 與原始碼散佈套件、確認筆記本格式與教材電路圖，並在不執行筆記本程式碼下渲染文件。

`native-platform-wheels` 會建置並稽核 CPython 3.12 Linux x86_64 manylinux 與 macOS 12 Apple ARM wheel，再比對已安裝原生擴充模組（native extension）與 wheel 中的 C++ 原始碼。macOS 工作使用 macOS 15 主機及 macOS 12 ARM 目標。`windows-native-wheel` 使用嚴格 MSVC 浮點設定建置 CPython 3.12 Windows AMD64 wheel，並稽核資源。

`windows-jax-wheel-smoke` 會在消費端環境安裝真正的 Windows wheel，執行原生核心、公開 Memory／Workspace JAX 路徑、預設 unittest 搜尋與 Windows Path Portability Set。`runtime-preparation-smoke` 則在已安裝 Linux 與 macOS wheel 的環境執行 Direct 與根計算、敏感度導數、CMA 最佳化、已存結果解析、預設 unittest 搜尋，以及該平台的 Path Portability Set。

Draft Pull Request 會略過產品與文件工作。較小的 `ci-status` 工作仍會檢查所選流程要求的工作是否成功，並確認另一流程的工作依預期略過。必要工作失敗、取消或非預期略過時，整體流程不會成功。

## 文件建置與 Pages 發布

獨立的 `documentation` 工作服務純文字流程。它會同步鎖定依賴，但不安裝或建置套件，再以 Quarto 1.10.18 在不執行程式碼下渲染。

產品流程與純文字流程共用連結檢查器：檢查本機 include 與相對來源連結，並確認以 `/` 開頭的網站連結能對應到渲染後的頁面或下載檔。

文件工作只渲染一個目前來源版本。Pages 則依 `scripts/pages_source.json` 固定的文件提交與版本設定，組合版本化網站，再以 Quarto 1.10.18 分別渲染各來源。非 Draft Pull Request 會建置審查網站但不部署；Draft Pull Request 會略過該建置。推送及符合條件的手動派送可部署組合後的 Pages 網站。

工作流程定義：[CI](../../.github/workflows/ci.yml)、[Pages](../../.github/workflows/pages.yml) 與[變更分類器](../../.github/scripts/change_scope.py)。
