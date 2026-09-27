# Google 官方 Colab MCP 狀態與 Codex 相容性報告

本文件記錄 Google 官方 Colab MCP 在本專案中的定位與相容性限制。此處不記錄任何使用者帳號、授權狀態或本機 Codex 設定位置。

---

## 一、官方 Colab MCP 設定

Colab MCP 是可選互動除錯工具，請依 [Google 官方 Colab MCP 說明](https://github.com/googlecolab/colab-mcp) 設定。不同 Codex 版本與 session 可能不會即時顯示 notebook 動態工具；每次要用前先確認工具已載入。

---

## 二、Codex 與 Colab MCP 的動態工具清單相容性問題

### 1. 相容性標記
`COLAB_MCP_FULL` 是執行環境狀態，不應硬編碼成專案能力保證。

### 2. 技術原因分析
1. Google 官方 Colab MCP 採用 MCP 規範中的 `notifications/tools/list_changed` 通知機制，在使用者連線至特定 Colab Notebook 後，動態載入該 Notebook 的操作工具（如 `get_cells`, `update_cell`, `run_code_cell`, `move_cell`, `delete_cell`）。
2. 如果 notebook tools 沒有出現，改用官方 Colab CLI 進行批次執行，不要反覆重連或安裝第三方 fork。

---

## 三、調度分工原則（嚴格遵守）

1. **Primary Execution Transport（主要自動化通道）**：
   - **Google 官方 Colab CLI** (`google-colab-cli` 0.7.4)。
   - 所有批次化 AI 運算（如 Whisper 轉錄、特徵抽取）皆經由 CLI 啟動、執行與回收。
   - 具有高度確定性、完整的超時與錯誤回滾機制。

2. **Optional Interactive Transport（可選互動通道）**：
   - **Google 官方 Colab MCP**。
   - 僅用於互動式 Notebook 檢查、除錯或人類觀察輸出。

3. **禁止事項**：
   - **嚴格禁止擅自安裝未經審查與批准的第三方非官方 Colab MCP Fork**。
   - 當 MCP 動態工具未出現時，自動無縫回退至 Colab CLI，保證剪輯流程不受阻礙。
