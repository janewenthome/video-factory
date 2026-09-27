# Video Factory 本機優先架構

## 系統分工

目前的工作路徑以 Mac mini M4 為中心。Antigravity/Gemini 是 editorial director，最終決定故事、片段、節奏與情緒走向；Codex 協調可重跑流程、把導演決策轉成可驗證的計畫並執行本機工具；Claude Opus 可作選擇性深度複核。Mac 執行確定性媒體工作、本機可用的語音推論與最終渲染。Colab adapter 留作未來架構擴充，目前不分配 Colab runtime、不上傳素材。

```text
使用者目標與素材
        │
        ▼
Antigravity / Gemini ── 最終故事、片段、節奏與情緒決策
        │
        ▼
Codex ── 導演決策轉成 edit plan、驗證並協調本機執行
        │
        ▼
Mac mini M4
  ├─ manifest、hash cache、metadata
  ├─ FFmpeg / VideoToolbox proxies、scene samples、audio extraction
  ├─ local perception 與支援 Apple Silicon 時的 MLX Whisper transcription
  ├─ timestamp 對齊、SRT 驗證
  └─ Remotion render、FFmpeg QA、final encoding
        │
        ▼
本機 outputs/
```

## 角色

### Antigravity / Gemini（Editorial Director）

- 根據素材資料、使用者目標、profile 與 transcript，決定 `story_plan.json` 中的故事、片段、節奏與情緒走向。
- 對感知模型提供的候選作最後編輯判斷；embedding 或重要度分數只作線索，不直接等於保留決策。
- 檢查字幕候選是否有意義。無意義、噪音、幻覺或無法辨識片段不燒錄；疑義內容標記人工確認，不補寫、不猜測。

### Codex

- 協調可重跑流程、把已決定的剪輯方向落成 `story_plan.json` / `edit_plan.json`，並驗證結構、時間範圍、隱私和 QA 證據。
- 執行本機 CLI、管理快取與產物；不以 perception 分數自行取代 editorial director 的最終選擇。

### Claude Opus（選擇性深度複核）

- 只在使用者選擇時複核複雜故事、字幕或 QA 判斷；不是必要執行依賴。

### Mac mini M4

- 原始照片與影片留在本機；所有衍生物存於 project `work/` 或 `outputs/`。
- 執行檔案雜湊、metadata、proxy、scene sampling、representative frames、音訊擷取等確定性工作。
- 在已配置且相容的 Apple Silicon 環境以 MLX Whisper 執行本機轉錄。本專案已在一台 M4 上對少量真實保留音訊完成 live inference；這只證明該環境與快取模型可執行，不是通用效能或辨識品質 benchmark。可指定既有模型 revision，並以 `--local-only` 禁止下載。
- 依 edit plan 做 Remotion 合成、字幕、配樂 ducking、FFmpeg/VideoToolbox 輸出與技術 QA。

### Colab（可選、目前 deferred）

- 程式庫仍保留未來 perception worker 的設計位置，供 CUDA 工作日後評估。
- 目前 Codex 工作階段無法可靠刷新 Colab MCP 動態新增的 notebook tools，因此 Colab MCP 不納入可執行路徑。不要因此改用 GUI/另一個 agent 來繞過本專案目前的決定。
- 本專案尚未完成可引用的 live Colab model inference、GPU/Compute Units benchmark 或素材品質比較；SigLIP2、diarization 與 temporal adapters 不代表已跑過真實 Colab job。
- 未來重新啟用需重新驗證工具 refresh、資料最小化與逐次明確授權；原始 4K 和整個素材庫不得上傳。

## 字幕工作流程

字幕在剪輯決策之後產生，讓轉錄時間能對齊最後保留的內容：

1. 檢查 edit plan 保留且有現場音的片段。
2. 若檢出語音，使用本機 ASR 產生帶時間戳的字幕候選；不自動改送雲端服務。
3. 將來源時間戳按 source in/out 映射至最終時間軸，切分長句並避免 cue 重疊。
4. 指定導演模型或人工檢查內容是否有語意價值；Codex 記錄 keep/drop。被判定無意義的段落不放字幕；保留原始 transcript 證據，讓排除可追溯。低信心或不確定內容標為 review，不臆測。
5. 驗證 SRT 時間範圍、排序、重疊與輸出長度，再由 Mac renderer 燒錄。

字幕語意審閱不得改寫說話者原意。衛教影片字幕中的醫療主張必須回指使用者提供的內容或 references；找不到來源時標記待確認。

## 自動配樂與現場聲

- 新 job 預設啟用 `auto_open_licensed` 配樂。Codex 依 job 主題和 story plan 搜尋 Openverse、挑選曲目並加入剪輯計畫；不詢問使用者先選歌。
- Openverse 是聚合索引，不代表授權已核實。加入前檢查原始來源頁的曲目、授權與署名條件；只用可以核實的 Public Domain、CC0 或 CC BY 曲目，並保留 attribution/credits。查不到來源或授權時跳過，不猜測。
- 配樂預設由片頭播放到片尾。保留片段的自然收音維持原始增益（預設 1.0）；只在自然聲期間把配樂平滑降到基準增益的 24%，約 0.25 秒下降、自然聲結束後約 0.25 秒恢復。這是 renderer 已實作的行為。
- 只有使用者或 job 明確指定才關閉配樂或使用指定曲目。不要把自動配樂理解成可以降低自然收音音量。

## 隱私、快取與重跑

- 預設保持本機處理；`LOCAL_ONLY` 禁止任何資料上傳。其他隱私模式目前不會自動啟動 Colab。
- 不修改、移動或覆寫 `assets/` 來源檔；衍生檔只寫入 `work/` 或 `outputs/`。
- 可快取的結果以 source hash、model/version 與 parameters 區分。素材沒有改變時避免重做昂貴推論。
- `perception_index.json` 是觀察證據，不是剪輯 plan；尚未配置的欄位明確標成 `pending` / `not_configured`。
- CLI/application API 負責確定性處理和結構化進度；GUI 不解析 CLI stdout，也不以滑鼠操作 timeline。

## 狀態與限制

- 確定性本機媒體流程與 Remotion renderer 可獨立驗證；合成 smoke test 不代表真實家庭素材的故事、裁切或字幕視覺已通過 QA。
- MLX Whisper adapter 已完成一次有限的 M4 live inference；其他主機、模型與長片效能仍需各自驗證。每次產生的字幕候選仍要語意審閱；mock tests 不能代替真實執行。
- Colab SigLIP2、匿名 diarization、temporal analysis 尚無本專案 live GPU/Compute Units benchmark。
- 有音訊不一定有值得呈現的語音；需經語音辨識與語意檢查後再決定字幕。沒有 meaningful cue 時可以不燒錄字幕。
- 只有播放並檢視成品後才可宣稱視覺與聽覺 QA 完成。
- 新對話的 `start`／`開始`／`開工` 或一般「怎麼開始剪片」詢問使用引導式 intake；旁白不列為起始問題，配樂自動開啟。
- 已發現的專案踩坑與決策記錄於 [DECISIONS.md](DECISIONS.md)。
