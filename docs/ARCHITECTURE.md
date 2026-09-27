# Video Factory 混合式架構設計 (Hybrid Video Editing Architecture)

## 一、系統架構概觀

本系統將影片製作流程拆分為「導演決策層」、「本機高性能處理層」與「雲端短週期 GPU 運算層」，徹底避免 Mac mini 長時間進行耗電的 GPU 模型推論，同時極大化降低 Google Colab Compute Units (CCU) 的消耗。

```
                    Antigravity/Gemini
                    Editorial Director
                       │
                       │
                 Video Editing Skill
                       │
          ┌────────────┴─────────────┐
          │                          │
          ▼                          ▼
      Mac mini M4                Google Colab
                                  Cloud worker
          │                          │
          │                          ├─ faster-whisper (large-v3-turbo)
          │                          ├─ GPU inference (CUDA)
          │                          ├─ speech intelligence (VAD + Whisper)
          │                          ├─ SigLIP2 visual index / duplicate clusters
          │                          ├─ shortlist-only temporal backend
          │                          └─ perception_index.json + automatic release
          │
          ├─ ffprobe (metadata, hash cache)
          ├─ VideoToolbox 720p proxy generation
          ├─ FFmpeg (scene detection, stills)
          ├─ audio extraction (FLAC/WAV)
          ├─ subtitle cleaning & SRT export
          ├─ final assembly & ducking
          └─ VideoToolbox H.264/HEVC encoding
                       │
                       ▼
                 final_video.mp4
```

---

## 二、角色與職責分離

### 1. Antigravity/Gemini（總導演 / Editorial Director）
- **核心職責**：理解專案目標、理解生活與衛教內容、規劃故事結構、決定剪輯片段、設定情緒節奏與文字卡。
- **原則**：導演掌握所有的創作與編輯決策；Codex 負責 orchestration、validation 與工具整合，不做無意義的底層轉碼，Colab 也不部署取代導演的大型 LLM。

### 2. Mac mini M4（本機高性能編解碼層）
- **核心職責**：
  - 高速讀取原始素材（支援 ProRes、4K H.264/HEVC）。
  - 利用 Apple VideoToolbox 硬體加速，迅速產出 720p 輕量 proxy。
  - 本機 FFmpeg 進行快速 CPU 場景切換偵測（Scene Detection）。
  - 音訊抽取、雙向字幕渲染、配樂自動降音（Audio Ducking）。
  - 最終成品影片輸出（Apple VideoToolbox H.264/HEVC 加速），保證最高畫質與最高本機效率。

### 3. Google Colab（COLAB PERCEPTION WORKER）
- **核心職責**：
  - Speech Intelligence：VAD、faster-whisper `large-v3-turbo`、word timestamps、匿名 ECAPA speaker turns。說話者分群目前是單一音檔內的 utterance heuristic，不處理重疊語音且 confidence 未校準；audio events 與 speech importance 尚未配置。
  - Visual Semantic Index：從 Mac 的 representative frames 建立 SigLIP2 embeddings、相似度與 duplicate clusters。Live GPU 尚待驗證；routine worker 目前不傳 proxy，也不宣稱從 embedding 直接決定事件或保留片段。
  - Temporal Deep Analysis：只處理經導演 shortlist 的 derived proxy；目前註冊 SmolVLM2 adapter，backend registry 保留插拔點，live GPU 尚待驗證。
  - 產生 `work/perception_index.json`（兼容鏡像位於 `outputs/work/perception_index.json`），供導演讀取候選證據。
  - **預設使用最經濟實惠的 T4 GPU**；L4 只供 MAX_QUALITY shortlist temporal workload，A100/H100/G4 禁止自動使用。
  - **嚴格執行任務完畢自動釋放**：無論成功或發生異常（finally 區塊），一律確保停止該次分配的 session，絕不留置背景空轉燒點數。

---

## 三、三大核心優化策略

### 1. Proxy-First 工作流程
- 原始素材（如 4K 60fps 數十 GB 檔案）絕對不直接上傳雲端。
- 第一時間在 Mac 本機製作 720p 低碼率 proxy。
- 所有的場景辨識、影格取樣、接觸表（Contact Sheet）與語音辨識均從 Proxy 或抽取音訊出發。
- 僅在最後的「Render」階段，才由本機合成器讀取原始無損素材進行高品質輸出。

### 2. Pipeline 狀態機與斷點續跑（Resume Machine）
系統將流程劃分為 12 個明確且具快取的階段：
1. `ingest`：素材盤點與 SHA-256 快取。
2. `proxy`：720p VideoToolbox proxy 製作。
3. `scenes`：本機場景切換邊界偵測。
4. `audio`：純音訊無損擷取。
5. `transcription`：Whisper GPU 轉錄（快取命中時免跑）。
6. `contact_sheets`：5x5 代表影格接觸表建立。
7. `perception`：整合 speech、visual semantic、duplicate/event clusters 與 temporal candidate evidence。
8. `editorial_analysis`：故事弧線與素材評分。
9. `edit_plan`：產生 `edit_plan.json` 與人類可讀 `edit_summary.md`。
10. `validate`：嚴格驗證時間軸、素材路徑與衛教引用。
11. `render`：Mac 本機 Remotion / VideoToolbox 渲染。
12. `qa`：黑畫面、音量電平與技術指標自動 QA。

所有進度存入 `work/pipeline_state.json`。Render 前的 `REVIEW` gate 會真正停下來，不會自動確認；使用者確認後以 `--approve-review` 或 GUI application API 繼續。中途若在任何步驟中斷或調整，重新執行時自動略過已完成項目。

### 3. 人機協同審核閘門（Human Review Gate）
提供 `AUTO`、`REVIEW`（預設）、`MANUAL` 三種閘門模式：
- 在 Render 之前自動生成繁體中文 `edit_summary.md`，列出預計片長、採用素材、刪除素材、故事開場/重點/結尾、字幕與需注意項目。
- 讓人類創作者在最終耗時輸出前有明確的視覺依據進行把關。

### 4. Perception 與導演決策的邊界

`perception_index.json` 是證據資料，不是剪輯決策。它保存 transcript、speech turns、匿名 speaker、audio events、scene metadata、embedding reference、duplicate cluster、temporal result、confidence 與 editorial candidate signals；`keep_decision` 一律由 Antigravity/Gemini 或人工導演填入。Claude Opus 僅在需要時做選定片段的深度 review。

### 5. 隱私與資料最小化

- `LOCAL_ONLY`：音訊、影格、影片均不上傳。
- `BALANCED`（預設）：可上傳抽取音訊、代表影格與 360p/480p derived proxy；原始 4K 永不上傳。
- `MAX_QUALITY`：只有 shortlist 後的少數片段可產生 720p proxy 供 temporal backend 使用。
- 每次 Colab 工作都是 allocate → run → download → verify → release；失敗路徑也必須 release。
- 所有 perception cache 都以 source hash + model/version + parameters 建 key，Family short/standard/full 共用。

### 6. Application / GUI boundary

GUI 使用 `video_editor.application.VideoFactoryApplication` 與結構化 progress events，不解析 stdout。未來 Tauri 2 shell 只負責資料夾選擇、profile、duration、privacy、music 與 review；素材仍留在原位置，Mac processing layer 負責 deterministic media work。
