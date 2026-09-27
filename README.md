# Video Factory (Codex + Colab + Mac mini M4 混合式 AI 剪輯系統)

這是一套設計給長期重複使用的 AI 智慧影片剪輯系統。
由 **Antigravity/Gemini** 擔任 editorial director，**Mac mini M4** 負責 proxy、FFmpeg、檔案處理與最終渲染，**Google Colab** 是隨選即用的 **COLAB PERCEPTION WORKER**，只處理需要 CUDA 的語音、視覺語意與 shortlist temporal evidence。

---

## 目前可用範圍

目前提供 Python CLI、確定性媒體處理工具、Colab CLI adapter、音樂搜尋命令，以及供桌面程式呼叫的 application service。`apps/ai-video-studio/` 是未來 Tauri GUI 的介面規格；完整桌面 GUI 尚未實作。故事和剪輯決策仍需由導演模型或使用者審閱。

## 開始使用

需求：Python 3.11+、FFmpeg/ffprobe；本機 render 需要 Node.js、pnpm 和 Remotion runtime。Colab inference 另外需要 Google Colab CLI、已授權帳戶與本次衍生檔上傳同意。Colab MCP 是可選的互動除錯工具。

```bash
# 建立新專案，先檢查 job.draft.yaml，再核准為 job.yaml
python3 skills/video-factory/scripts/video_factory.py init projects/my-family-edit --profile memory

# 加入自己的素材後，盤點 metadata、建立 proxies / scenes / 音訊 / 接觸表
python3 -m video_editor prepare projects/my-family-edit --mode family

# 建立本機 perception index；此步驟不會上傳檔案
python3 -m video_editor perception projects/my-family-edit --privacy-mode BALANCED

# 驗證 edit plan；依 skill 建立並審閱 edit_plan.json 後才能 render
python3 skills/video-factory/scripts/video_factory.py validate-plan projects/my-family-edit
```

每個 project 的來源素材放在 `assets/`，本機 cache 和衍生物放在 `work/`，交付物放在 `outputs/`。私人 project 資料不得提交到公開 repository。

### Colab 與隱私

Colab 預設用 T4 處理語音和視覺 embedding；L4 僅供 MAX_QUALITY 下明確指定的 temporal shortlist，A100/H100 不會自動申請。`LOCAL_ONLY` 不上傳；`BALANCED` 可允許選定衍生音訊、代表影格及低解析 proxy，目前的第一階段實際只傳音訊和代表影格；`MAX_QUALITY` 只允許 shortlist 後的少量 720p proxy。原始 4K 和整個素材庫不得上傳。

先檢查 `work/perception_index.json` 的 eligible upload 清單。雲端分析前，必須就該次實際衍生檔取得明確同意，再以 `colab-perception ... --allow-upload` 執行。詳見 [Colab setup](docs/COLAB_SETUP.md) 和 [Colab privacy contract](skills/video-factory/references/colab.md)。

### 音樂搜尋

Openverse 收錄來自多個來源的大量音訊和影像，可用不同曲風／情緒關鍵字重複搜尋並切換候選；搜尋範圍與結果會隨上游資料改變。系統不會自動挑選或下載歌曲。[Openverse](https://openverse.org/) 不替個別作品確認授權，因此下載前必須回原始來源頁核對授權與歸因。

`music search` 只列出 Openverse 索引中通過基本 CC0、PDM 或 CC BY metadata 檢查的音訊，不會自動下載或選曲。使用者明確選曲後，可用 `music add <ID>` 下載至專案 `work/`，並保存來源和授權證據。Openverse 是聚合搜尋服務，使用前仍要開啟原始來源頁檢查授權、歸因及同步到影片的條件。

```bash
python3 -m video_editor music search --project projects/my-family-edit warm acoustic instrumental
python3 -m video_editor music add <TRACK_ID> --project projects/my-family-edit
```

### 片長預設

- Family：SHORT 約 90 秒、STANDARD 約 180 秒、FULL 約 300 秒。
- Health education：SHORT 約 30 秒、STANDARD 約 90 秒、FULL 約 180 秒。
- `LOCAL_ONLY`、`BALANCED`、`MAX_QUALITY` 控制衍生資料能否離開本機；project `job.yaml` 可覆寫預設。

---
## 系統如何製作影片


1. 本機 Python 和 ffprobe 盤點每個檔案、計算 SHA-256，並快取 metadata。
2. FFmpeg/VideoToolbox 產生場景候選、proxy、代表影格、接觸表與抽取音訊；素材本身保持不變。
3. 建立 `work/perception_index.json`。`LOCAL_ONLY` 完全本機；`BALANCED`（預設）允許選定音訊、代表影格和低解析 proxy，目前 routine worker 只傳音訊與代表影格；`MAX_QUALITY` 只把 shortlist 後的少數 720p proxy 交給 temporal backend。原始 4K 永不上傳。
4. 若已授權，Colab Perception Worker 執行 VAD、faster-whisper `large-v3-turbo`、必要 word timestamps、匿名 speaker metadata、SigLIP2 embeddings、相似度、duplicate clusters 和候選分組；結果下載回 Mac，session 於成功與失敗路徑都釋放。
5. Antigravity/Gemini 依 profile、transcript、perception evidence 與素材資料決定故事、片段、節奏，建立 `story_plan.json` 和 `edit_plan.json`。
6. Antigravity/Gemini 把導演決策交給 pipeline 寫入 `work/edit-plan/edit_plan.json`；Codex 負責 orchestration 與驗證。驗證器檢查路徑、時間範圍和公衛 claim references。
7. 本機 Remotion 依 edit plan 合成畫面、燒錄字幕、調整配樂音量並輸出 MP4；FFmpeg/ffprobe 執行媒體檢查。
8. 音樂流程先寫 `work/music_requirements.json`（同步鏡像至 `outputs/work/`）；SAFE_AUTO 只接受 Public Domain、CC0、CC BY 且有上游證據的曲目。搜尋或授權失敗時無音樂完成，並產生 `work/music_attribution.json`、`outputs/work/music_attribution.json`、`outputs/final/music_attribution.json`、`outputs/MUSIC_CREDITS.txt`、`outputs/PUBLISHING_CREDITS.txt`。
9. Skill 匯出 SRT、QA 報告與少量人工 review notes。修改時沿用 hash 未變的檢查、影格、perception 與分析結果，只重做受影響的步驟。

`edit_plan.json` 是 render 的剪輯決策來源。Renderer 不會自行挑素材、改故事或補寫醫療資訊。16:9 與 9:16 需分別做構圖決定，並各自 render。

編輯計畫選用 HEIC 照片時，`prepare-render` 會以 libheif 優先、FFmpeg 解碼驗證的流程建立最長邊 4096px 的 JPEG 衍生檔，存放在 `work/render-public/assets/` 並以來源雜湊快取；Remotion 使用 JPEG，HEIC 原檔不會被修改。

9:16 編輯若將團照設為 `contain`，renderer 會保留完整照片，並以同一張照片的柔焦暗化版本延伸直式畫布，避免人物被裁掉或留下純色空帶。

每次 render 會在 `outputs/versions/` 保存 edit plan、review/music metadata 與該版本影片；重新修改不會只覆蓋上一版的決策記錄。

## 範例專案

- `projects/sample-family-travel/`：memory profile 家庭旅行設定，請加入自己的素材和已授權配樂。
- `projects/sample-public-health-60s/`：public-health 60 秒直式短片設定。製作前必須把實際衛教來源放入 `assets/references/`，並更新 job 和 edit plan 的 source/claim 對照。
- `projects/sample-project/`：15 秒 synthetic smoke-test job 設定。來源媒體和 render outputs 是本機衍生檔，刻意不隨 GitHub repository 發布；合成音不可當作授權音樂。

## 本機與外部服務

| 功能 | 執行位置 | 備註 |
|---|---|---|
| metadata、SHA-256 cache、proxy、scene detection、影格和接觸表 | Mac M4 | 不上傳原始素材 |
| Speech/visual/temporal perception | Google Colab CLI | 只傳 derived data；預設 T4；L4 僅 shortlist temporal；必須先取得本次同意並以 `--allow-upload` 執行 |
| 故事／剪輯決策、edit plan | Antigravity/Gemini director；Codex orchestration | 以 transcript、metadata、perception evidence 和 contact sheet 作依據 |
| Remotion render、FFmpeg、SRT 匯出與 QA | Mac M4 | 最終編碼不使用 Colab GPU |
| Colab notebook 互動除錯 | Google Colab MCP | 只在目前 Codex 工作階段確實出現 notebook tools 時使用；否則回到 CLI |
| OAuth、Colab GUI 檢查與視覺 QA | Computer Use | GUI 需要時才使用；不以滑鼠操作 timeline |
| Editorial director | Antigravity/Gemini；必要時 Claude Opus review | perception score 不等於 keep；GUI 呼叫 `video_editor.application.VideoFactoryApplication`，不解析 stdout |
| OpenAI timestamp transcription | 使用者明確選用時 | 可保留作另一個 provider；和 Colab 分開授權，不會在 Colab 失敗時自動改送 OpenAI。參數見 [Transcriptions API](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) 與 [Speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text)。 |
| TTS 旁白 | 尚未接入 | 預設關閉；建立旁白文稿後才考慮接入與計費 |

Colab CLI 與 MCP 是否可用取決於各自的本機安裝；可用 `colab version` 和 `codex mcp list` 檢查。SigLIP2、ECAPA 匿名說話者分群與 SmolVLM2 temporal adapter 已有實作與 pinned model revision，但 live GPU 執行、CU 消耗、模型輸出品質仍須分開驗證。SpeechBrain 的 speaker label 僅在單一音檔內分群，不識別真實身分、不處理重疊語音，信心值未校準。Audio event detection 與 speech importance scoring 仍標記為未配置，不應當作已完成能力。真實素材雲端分析仍需 OAuth、usage 查驗及每次精確 derived-data 授權。沒使用雲端時仍可完成素材盤點、perception index 的本機證據、故事規劃、edit plan、Remotion render、SRT 和技術 QA。OpenAI key 只能由 `OPENAI_API_KEY` 環境變數提供；範例檔 `.env.example` 不含真實金鑰。TTS 尚未接入。

不要自動下載音樂。`music-library/manifest.yaml` 只列出使用者擁有或明確有權使用的曲目。

## 安裝與主要檔案

- `skills/video-factory/SKILL.md`：Codex 入口與工作流程。
- `skills/video-factory/profiles/`：`memory` 和 `public-health` 導演偏好。
- `skills/video-factory/templates/`：job、edit plan、素材分析與 story plan 格式。
- `skills/video-factory/scripts/`：本機確定性工具及受同意閘門保護的 Colab worker adapter。
- `video_editor/application.py`：GUI／桌面應用呼叫的 application boundary 與結構化 progress events。
- `skills/video-factory/scripts/perception.py`、`colab_perception.py`：Perception Index 與 derived-data Colab worker。
- `skills/video-factory/references/colab.md`：Colab CLI、MCP、Computer Use 的分工與授權流程。
- `skills/video-factory/runtime/remotion/`：鎖定版本的 Remotion composition 與 dependencies。
- `.agents/skills/`：此專案可用的 Remotion 官方 Agent Skills。
- `projects/<project>/assets/`：原始素材；不修改、不搬移、不覆寫。
- `projects/<project>/work/`：cache、manifest、分析、轉檔與 edit plan。
- `projects/<project>/outputs/`：交付的 MP4、QA 和 review notes；SRT/逐字稿留在 `work/transcripts/`。

主要本機命令可從 Skill 所在目錄呼叫：

```bash
python3 <SKILL_DIR>/scripts/video_factory.py inspect <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py extract-scenes <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py extract-frames <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py build-contact-sheets <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py validate-plan <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py prepare-render <PROJECT> --ratio 16:9
python3 <SKILL_DIR>/scripts/video_factory.py export-srt <PROJECT>
python3 <SKILL_DIR>/scripts/video_factory.py colab-transcribe <PROJECT> --source work/transcripts/audio/<hash>.flac --language zh --gpu T4 --dry-run
python3 <SKILL_DIR>/scripts/video_factory.py colab-transcribe <PROJECT> --source work/transcripts/audio/<hash>.flac --language zh --gpu T4 --allow-upload
python3 <SKILL_DIR>/scripts/video_factory.py transcribe <PROJECT> --source work/transcripts/audio/<hash>.flac --language zh --allow-upload
```

`prepare-render` 產生唯一的 Remotion props 與 hash-verified 素材副本。要直接 render，從 Remotion runtime 目錄呼叫：

```bash
cd <SKILL_DIR>/runtime/remotion
VIDEO_FACTORY_PUBLIC_DIR="<PROJECT>/work/render-public" \
  ./node_modules/.bin/remotion render src/index.ts VideoFactory \
  "<PROJECT>/outputs/master.mp4" \
  --props "<PROJECT>/work/render-input.json"
python3 <SKILL_DIR>/scripts/video_factory.py qa <PROJECT> --video outputs/master.mp4
```

若 runtime 尚無 `node_modules/`，先在該 runtime 執行 `pnpm install`；不需安裝全域 Remotion。

兩種 transcription CLI 都只接受 `assets/` 或 `work/transcripts/audio/` 下的支援音訊格式，在 `work/transcripts/` 產生依來源 hash 和 inference 設定命名的 timestamped JSON/SRT。Colab 版本先用 `--dry-run` 顯示傳送計畫，再由 Skill 說明指定音訊和 Google Colab、取得本次同意後才加 `--allow-upload`；worker 會關閉自己建立的 runtime。OpenAI 版本需要獨立的本次同意並使用 `OPENAI_API_KEY`。同一來源/config 的快取命中不會再次上傳。範例中的 `<hash>` 需換成 `extract-audio` 輸出 index 記錄的實際檔名。

Remotion runtime 版本由 `skills/video-factory/runtime/remotion/package.json` 和 lockfile 固定。若 runtime 尚無 `node_modules/`，在該目錄執行 `pnpm install`。專案提供的 Remotion Agent Skills 可放在 `.agents/skills/`；此目錄是本機開發輔助檔，不是執行核心程式的必要依賴。

安裝完成後重開 Codex 工作階段，使用 `$video-factory` 觸發共用 Skill。Remotion 官方 Agent Skills 來源與 Codex 支援見 [Remotion Agent Skills](https://github.com/remotion-dev/remotion/blob/main/packages/skills/README.md)；目前的 Remotion CLI 與 composition 依官方文件設定。

## 驗證狀態與限制

合成 smoke test 曾輸出 640×360 H.264/AAC 影片並檢視標題、繁體中文字幕、結尾和混音 ducking。這只驗證 renderer，不代表真實家庭素材的敘事、人物裁切或字幕視覺已通過 QA。真實影片只有實際播放與檢視後才能標示視覺 QA 完成。

近似照片自動去重、TTS、完整多比例自動產出、短暫黑閃／字幕溢出偵測、audio event detection、speech importance scoring 與 FFmpeg loudness normalization 尚未完成。不得把文件、mock 測試或 CLI dry-run 當成 live GPU job 成功。public-health validator 能要求 claim reference IDs，但仍要由人核實來源是否真的支持該主張。

## 授權

本專案自行撰寫的程式碼以 MIT License 發布。第三方元件仍依各自授權使用。Remotion 原始碼公開可讀，但依 Remotion 自有授權條款使用，並非 OSI 定義的開源授權；請在使用、散布或部署前閱讀 [Remotion License FAQ](https://convert.remotion.dev/docs/license/faq) 與 [Remotion License](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md)。
