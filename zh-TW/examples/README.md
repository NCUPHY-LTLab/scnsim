# SCNSim 工程課程來源

`engineer/setup.qmd` 是共用環境入口。`engineer/chapter-01/` 至 `engineer/chapter-08/` 是現行課程章節。各章的標準課程文字與程式碼存放於不含 YAML 的 QMD 片段。精簡 QMD 外層檔案將各課呈現為獨立 HTML 頁面；每個 `chapter.qmd` 則依序納入片段，並產生完整的 `chapter.ipynb`。

網頁課程不能獨立執行；每一課都會清楚標示所需的前置狀態。整章是以乾淨核心執行的單位。第 1–7 章請開啟 `chapter.ipynb`、重新啟動 Python 核心並按 **Run All**。第 8 章則明確使用兩個獨立核心，並說明各核心要執行哪一部分。提交的 Notebook 都是零輸出傳遞檔，不是執行紀錄。

章節中的相對路徑 `workspace="workspaces/..."` 會依程序目前的工作目錄解析。因此，從儲存庫根目錄啟動 Jupyter 時，資料會寫入根目錄的 `workspaces/`；從其他位置啟動則會寫在該位置。儲存庫各層名為 `workspaces/` 的資料夾都會被忽略，以降低意外發布的風險，但忽略不代表刪除或備份。若之後需要精確 `resolve()`，請保留或備份 Workspace 證據；移動或刪除資料可能使該 Result 無法讀回。

QMD 是唯一手動編輯的課程權威來源。請勿編輯衍生 Notebook，也不要在外層檔案重複課程程式碼。網站設定 `execute.enabled: false`；瀏覽或建置文件不會啟動核心、求解器、執行期準備或最佳化。

產生課程數值證據前，請選用儲存庫本機 Jupyter 工具群組與獨立的靜態匯出 extra：

{{< include ../_generated/canonical-code/examples/README.md/block-001.qmd >}}

兩者都不是數值執行期相依套件，也不是已發布輪子的 extra。

## 明確產生第 1 章證據

第 1 章電路圖、數值圖、複數資料與物理量表由一個有界工具根據精確課程片段產生：

{{< include ../_generated/canonical-code/examples/README.md/block-002.qmd >}}

Workspace 必須位於儲存庫外，且為新建或空資料夾。產生流程會實際執行本章 Direct 與對角根請求，這不是 Quarto 步驟。確認已提交來源與成品的身分時，不要載入 SCNSim 或執行求解器：

{{< include ../_generated/canonical-code/examples/README.md/block-003.qmd >}}

`engineer/figures/chapter-01-artifacts.json` 會綁定章節匯總、片段、外層檔案、套件來源樹、環境、精確 Result 身分及下列公開成品：

- `01-authoring.svg`：基準 Plan 的建立投影證明；
- `01-s11-baseline.svg` 與 `01-s11-selected.svg`：公開 `.s.plot()` 的振幅與相位圖，振幅顯示範圍固定為 ±0.05 dB，僅影響呈現；
- `01-s11-data.csv`：精確 401 點網格上未捨入的基準與選定單埠複數響應；以及
- `01-quantities.csv`：兩個參數點的電容、獨立未載入 LC 計算、負載根頻率與負載線寬。

理想無損單埠振幅約為 0 dB。相位仍可判讀；包裹相位的不連續是一般角度呈現，不是數值失敗。Plotly 以 GHz 顯示水平軸，Result 則保留宣告的頻率單位。數值圖呈現具型別的 Result，不是 ASCDLS 證明。負載根是獨立的 Result，不會從凹陷或未載入 LC 公式推導。

## 明確產生第 2 章證據

第 2 章參數點表、參數場圖與選定候選解電路圖由精確課程片段產生：

{{< include ../_generated/canonical-code/examples/README.md/block-004.qmd >}}

Workspace 同樣必須位於儲存庫外，且為新建或空資料夾。確認已提交的綁定時，不要載入 SCNSim 或執行求解器：

{{< include ../_generated/canonical-code/examples/README.md/block-005.qmd >}}

`engineer/figures/chapter-02-artifacts.json` 會綁定章節匯總、四個片段、四個外層檔案、套件來源樹、環境、精確 Result 身分、候選解證明及下列公開成品：

- `02-capacitance-sweep.svg`、`.csv` 與 `.md`：L 固定為 5.8 nH 時依序排列的三個 C 點，包含具型別狀態與負載根值；
- `02-optimized-winner.svg`、`.csv` 與 `.md`：實際回傳的候選解、其獨立負載根讀回，以及使用精確 `ParameterSet` 繪製的同一 Plan；以及
- `02-optional-grid.svg`、`02-optional-spaces.csv` 與 `.md`：完整 3 × 2 笛卡兒參數場及另外列舉的兩個 C/L 點。

數值圖呈現具型別 Result；只有候選解電路圖帶有 ASCDLS 電路圖證明。最佳化器成本較低或成品已產生，都不代表科學或 Human 已接受。

## 明確產生第 3 章證據

第 3 章具型別的完整 Plan 配置結果、兩張獨立子電路插圖與具名雙埠響應，都來自精確課程片段：

{{< include ../_generated/canonical-code/examples/README.md/block-006.qmd >}}

`engineer/figures/chapter-03-artifacts.json` 會綁定章節來源、套件來源、環境、請求、Result 與公開成品身分。較早的完整 Plan 自動電路圖回報具型別的 `schematic_layout` 失敗 `fixed connector intersects occupied geometry`，因此該歷史發布沒有 SVG 或證明。目前章節來源改為明確宣告完整手動配置；新產生作業會有自己的結果與身分。`03-feedline-illustration.svg` 與 `03-readout-illustration.svg` 是分別宣告之示意 Plan 的證明投影，不是從失敗的完整 Plan 擷取的片段，也不是其替代品。`03-direct-s21.svg`、`.csv` 與 `.md` 呈現 `transmission` 軌跡在 5.5、6.0、6.5 GHz 的精確取樣。三個取樣點不能定義插值曲線或已解析諧振。

## 明確產生第 4 章證據

第 4 章明確電路圖與數值 Result 會在網站之外獨立產生：

{{< include ../_generated/canonical-code/examples/README.md/block-007.qmd >}}

`engineer/figures/chapter-04-artifacts.json` 會綁定六個片段、六個外層檔案、章節匯總、套件來源、環境、電路圖證明、View 來源脈絡、請求、Result 與公開成品。必要主線包含 `04-explicit-capstone.svg`、原始四埠 `04-raw-direct-s21.*` Result，以及另行標示的 `04-ptc-direct-s21.*` 表格與圖。選讀證據包含精確 6.2 GHz 的 `04-response-element.*` 查詢與獨立的 `04-optional-direct-s21.*` Result。`04-pump-off-hb.*` 是退役 HB 路徑保留的歷史資料，不由目前的 JAX 產生器執行。歷史 Direct 與 HB 資料各自保留原先宣告的網格；摘要不會疊合或插值。

## 明確產生第 5–8 章證據

其餘各章都有自己的有界產生器與僅檢查來源的檢查器：

{{< include ../_generated/canonical-code/examples/README.md/block-008.qmd >}}

第 5 章建立可重用的 Composite 電路圖，並只公開 C、L 參照、兩個等效一般 Pin 及一個 Coordinate。父層直接耦合其中一個 Pin，另一個則保持開路。

第 6 章會明確宣告配置、Port 姿態、T 形與十字接點，並在完全相同的 Plan 上重用配置。已保存的 Default、T 形、十字與重用圖表沿用各自原有的發布身分。

第 7 章宣告手動建立幾何，並評估完整十六通道 Direct 結果。早期的有限 π 電路圖編譯曾產生具型別的 `schematic_layout` 失敗，沒有對應 SVG；目前課程來源採用手動配置。三點批次另有三張標示為 readout-head 反射範例的圖。

第 8 章在全新執行目錄中使用兩個獨立 Jupyter 核心：第一個核心保存 Result 與獨立 Report，第二個核心重建宣告，並只對未變更的共用 Workspace 呼叫 `resolve()`。清單記錄兩個核心與精確相符的 Result 身分，也會分別記錄原始匯出產生器及後續來源檢查器的重新綁定。來源檢查流程依據經審查的轉換檔及既有發布清單，保留歷史執行與重新繪圖身分，並逐一比對公開成品的位元組。轉換檔記錄新舊產生器、套件來源、教材來源與程式區塊的精確雜湊，以及新增或移除項目；新雜湊本身不授權變更。八份候選清單須全部通過驗證後才會逐檔原子取代；若驗證失敗，發布器不會取代任何清單。這八次逐檔取代並非單一檔案系統交易。未來重新繪圖須另有經審查輸出，並綁定原始已驗證輸入 Result、資料與產生器身分；目前來源轉換不授權重新繪圖。

第 8 章產生器另有一條以收據綁定的延續流程，供已封存兩個 Result、但在呈現階段停止的經審查執行使用。維護者先執行 `--verify-resume-only`，再傳入相同的 `--workspace` 與精確 `--resume-receipt`。此流程會重建相同 Plan，並在綁定前拒絕任何變更的目錄／執行期／程式區塊／請求／Result 身分；接著在兩個新的獨立核心中使用公開 `resolve()`，絕不改用 `solve()` 或 `evaluate()`。已發布清單會將失敗核心中不可取得的程序遙測，與延續及重啟核心紀錄分開。

{{< include ../_generated/canonical-code/examples/README.md/block-009.qmd >}}

`FAILED_RUN_CAPTURE_SHA256` 是交接失敗執行擷取資料時獨立保存的 SHA-256。產生器不會從正在驗證的收據推導或猜測此信任錨點。

## 舊章節網址導覽

`tutorials/01_*.qmd` 至 `tutorials/16_*.qmd` 是通往現行工程課程的簡短無程式碼遷移指南。它們保留公開網址，但不構成第二套課程。被取代的可執行內容、零輸出 Notebook、檢查點圖與電路圖產生器仍保留於 Git 歷史。英文[遷移圖](/en/docs/implementation/engineer-course-migration-map.html)記錄所有現行目的地。

`tutorials/fixtures/` 中的歷史 Python 固定範例是**保留**的實作基準證據。現行課程產生器不會載入它們；CI 也將既有固定範例使用者與工程課程分開。來源綁定圖表、產生的 Notebook 與通過的靜態檢查，個別都不代表 Human 已接受、完整課程已執行、已發布或科學有效。
