# Video Factory 運算成本與 Compute Units (CU) 控制策略

本文件規範系統在運用雲端 GPU 時的成本控制原則與審計記錄。

---

## 一、硬體與運算階層原則 (Compute Hierarchy)

| 任務種類 | 指定運算位置 | 成本標準 | 說明 |
|---|---|---|---|
| 素材盤點、雜湊計算 (Ingest) | Mac mini M4 (Local CPU) | 0 CU | 零雲端成本 |
| 720p Proxy 製作 | Mac mini M4 (Apple VideoToolbox) | 0 CU | 硬體即時編碼，節省大量雲端頻寬 |
| 場景切換偵測 (Scene Detect) | Mac mini M4 (Local CPU) | 0 CU | 輕量快速，無須調度 GPU |
| 音訊抽取與格式轉換 | Mac mini M4 (Local FFmpeg) | 0 CU | 本機高效抽取 FLAC/WAV |
| 代表影格與接觸表 (Contact Sheets) | Mac mini M4 (Local FFmpeg) | 0 CU | 本機影像縮圖拼貼 |
| 故事決策與剪輯計畫 | Antigravity/Gemini director；Codex orchestration | 0 CU | 導演決策與驗證不需要 Colab GPU |
| **Speech intelligence (VAD + faster-whisper)** | **Google Colab (T4 預設)** | 依 live usage | **僅上傳所選 derived audio，完畢即停** |
| Visual semantic index / clustering | Google Colab T4 | 依 live usage | 只處理 representative frames / low-resolution proxy，依 hash cache 重用 |
| Shortlist temporal analysis | Google Colab L4 only when justified | 依 live usage | MAX_QUALITY、少量 720p proxy、backend 可插拔 |
| 最終影像合成與編碼 (Render) | Mac mini M4 (VideoToolbox) | 0 CU | 高品質無損輸出，不燒雲端點數 |

---

## 二、四大鐵律 (Golden Rules)

1. **Rule 1：Never allocate a premium GPU when the task can reasonably be completed on CPU or T4.**
   - 嚴格禁止系統自動申請或指派 `A100`、`H100` 或 `G4` 頂級運算資源。
   - 預設一律使用 `T4`。唯有在有量測數據支持且使用者明確批准時，方可使用 `L4`。

2. **Rule 2：Always release Colab compute after the GPU stage, including failure paths.**
   - 任何啟動 Colab session 的程式碼必須具備 `finally: stop` 防護網，確保程式崩潰、逾時或被使用者中斷時，VM 不會在背景持續燃燒點數。

3. **Rule 3：Proxy-first & Data Minimization.**
   - 4K / 原畫質大型影像檔案**絕不上傳** Colab。
   - 依 privacy mode 僅上傳 derived audio、代表影格與低解析 proxy；原始 4K 永不上傳。

4. **Rule 4：Strict Content-Addressable Caching.**
   - 凡轉錄過的音訊與設定組合，依 SHA-256 雜湊儲存本機快取；快取命中時**絕不重複發起雲端請求**。

---

## 三、執行成本審計紀錄格式 (Cost Audit Log)

所有 Colab 任務執行時，會在專案目錄 `work/cost_log.jsonl` 中留存完整的結構化紀錄：

```json
{
  "timestamp": "2026-09-27T06:30:00Z",
  "task": "transcribe",
  "accelerator": "T4",
  "start_time": "2026-09-27T06:30:05Z",
  "end_time": "2026-09-27T06:31:15Z",
  "duration_seconds": 70.0,
  "input_file": "work/transcripts/audio/sample.flac",
  "input_bytes": 15420100,
  "cu_balance_before": "100.0",
  "cu_balance_after": "99.8",
  "status": "success"
}
```

> **注意**：除非 Google Colab API 或 CLI `colab usage` 有明確回傳數值，系統不會自行推測或虛構法幣金錢金額。
