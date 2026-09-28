# Video Factory (Codex + Mac mini M4 local-first; optional Colab perception)

這是一套可重複使用的 AI 影片剪輯系統：**Codex** 依需求自動建立第一版故事與剪輯計畫、協調工具並驗證輸出；**Mac mini M4** 負責素材處理、本機語音推論、proxy、FFmpeg 與最終渲染；**Antigravity/Gemini** 和 **Claude Opus** 可作選擇性複核。使用者看完初版後再提出修改。Colab perception adapter 保留為未來擴充，目前不在執行路徑。

---

## 目前可用範圍

目前提供 Python CLI、本機確定性媒體處理工具、可選雲端 adapter、音樂搜尋命令，以及供桌面程式呼叫的 application service。`apps/ai-video-studio/` 是未來 Tauri GUI 的介面規格；完整桌面 GUI 尚未實作。Codex 會自動建立首版故事／剪輯計畫並判斷字幕候選；renderer 不自行推斷故事或改寫語音。

## 開始使用

需求：Python 3.11+、FFmpeg/ffprobe；本機 render 需要 Node.js、pnpm 和 Remotion runtime。Apple Silicon 語音推論使用 `mlx-whisper==0.4.3`；預設重用固定版 `mlx-community/whisper-large-v3-mlx` snapshot，並且只查本機快取，不自動下載權重。只有明確加入 `--allow-model-download` 才會在 cache miss 時下載。

在新 Codex 對話輸入 `start`、`開始`、`開工`，或詢問怎麼開始剪片時，Video Factory Skill 會先給引導：家庭／旅遊／生活回憶選 `memory`，醫療／衛教選 `public-health`；不會在起始訪談詢問是否需要旁白。家庭影片的建專案範例：

```bash
cd /Volumes/2TB/program/video_factory
python3 skills/video-factory/scripts/video_factory.py init projects/first-edit --profile memory
```

衛教影片把 `memory` 改成 `public-health`，並備妥可引用的來源。素材可放在 `assets/videos/`、`assets/photos/`，也可直接放在專案根目錄；Codex 會盤點專案內全部支援的照片與影片，原始檔保持不變。接著提供影片類型、片長、比例、主題／要留下的時刻與專案路徑。需求足夠時，Codex 會直接建立故事計畫、剪輯計畫、選擇可核實授權的配樂並輸出本機初版；看完後再告訴 Codex 要怎麼修改。

```bash
# 建立新專案；Codex 會依完整需求建立 job 並產生第一版
python3 skills/video-factory/scripts/video_factory.py init projects/my-family-edit --profile memory

# 加入素材後，盤點 metadata、建立 proxies / scenes / 音訊 / 接觸表
python3 -m video_editor prepare projects/my-family-edit --mode family

# 建立本機 perception index；此步驟不會上傳檔案
python3 -m video_editor perception projects/my-family-edit --privacy-mode LOCAL_ONLY

# 驗證 edit plan
python3 skills/video-factory/scripts/video_factory.py validate-plan projects/my-family-edit
```

每個 project 的來源素材放在 `assets/`，本機 cache 和衍生物放在 `work/`，交付物放在 `outputs/`。私人 project 資料不得提交到公開 repository。

### 多專案序列佇列

可以先把多個已完成故事與剪輯計畫的 project 加入佇列，再啟動一次 runner。影片分析、Whisper、proxy、FFmpeg 和 render 共用一把重型工作鎖；同一時間只處理一個 project。`queue add` 需要 `job.yaml`、`work/analysis/story_plan.json` 和 `work/edit-plan/edit_plan.json`，並在加入前驗證 edit plan。Codex 仍負責建立故事、字幕與配樂決策；runner 按保存的計畫執行本機 pipeline。

```bash
python3 -m video_editor queue add projects/project-a
python3 -m video_editor queue add projects/project-b
python3 -m video_editor queue list
python3 -m video_editor queue run
```

支援 `queue pause`、`queue resume`、`queue cancel JOB_ID` 與 `queue reorder JOB_ID...`。取消執行中的工作會停止其 process group 並確認退出，再開始下一支。失敗工作預設最多執行兩次，之後標記 failed 並繼續其餘項目。狀態、lock 和完整 log 保存在 `.video-factory/`；該目錄已排除於 Git。

Runner 在開始前取兩次唯讀 `memory_pressure` 樣本；之後每支影片間冷卻 45 秒並比較前後 free-memory 與 Swapouts。資源不穩時每 30 秒重查，最多三次，仍無法確認安全便持久暫停。低於 10% free-memory、Swapouts 持續增加或讀值不可用都會阻止下一支。此 10% 是保守 heuristic，不是 macOS 正式 pressure level。需要批次剪輯時，請在一個 Codex 工作中提供多個專案路徑，逐案準備計畫後加入 queue；不要同時啟動多個重型 Codex 工作。queue 只限制 Video Factory 媒體程序，不能限制其他 Codex/ChatGPT 工作階段或 Ollama 的記憶體使用。若在不同 clone/worktree 共用佇列，為它們設定相同的 `VIDEO_FACTORY_QUEUE_FILE`。

桌面 GUI 尚未實作；`apps/ai-video-studio/gui_contract.json` 已列出未來 GUI 的 queue 狀態、控制項與結構化事件。

### 本機分析、字幕與隱私

先完成剪輯計畫，再對保留片段中偵測到語音的音訊產生本機逐字稿候選與時間戳，讓字幕與最後時間軸對齊。Codex 自動判斷每句在上下文中的意義並記錄 keep/drop：有意義的語音保留字幕；無意義、噪音、幻覺或無法辨識的片段不燒字幕；不確定內容先省略並記錄，不自行補寫。輸出前驗證 SRT 時間範圍與重疊。這是自動首版流程，不等待逐句人工審核。

安裝本機語音辨識 backend：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-local.txt
```

建立 `edit_plan.json` 後，只對保留現場聲的影片片段產生字幕候選：

```bash
.venv/bin/python skills/video-factory/scripts/video_factory.py draft-captions projects/my-family-edit --language zh
```

若你已有其他 MLX Whisper 模型快取，可用 `--model` 與 `--model-revision` 指定該快照；CLI 預設不下載模型，只有加 `--allow-model-download` 才允許取得缺少的權重。revision 可從 Hugging Face cache 的 snapshot 目錄取得。

自動首版流程中，Codex 依音訊和逐字稿候選直接寫入 `work/transcripts/caption_review.json`，記錄每句 `keep` 或 `drop` 及理由，不等待人工逐句核准。若使用者想自行覆核，也可手動執行以下命令：

```bash
.venv/bin/python skills/video-factory/scripts/video_factory.py apply-caption-review projects/my-family-edit --review work/transcripts/caption_review.json
.venv/bin/python skills/video-factory/scripts/video_factory.py validate-plan projects/my-family-edit
.venv/bin/python skills/video-factory/scripts/video_factory.py export-srt projects/my-family-edit
```

若候選都是噪音、幻覺、純語助詞或無法辨識片段，標記 `drop`；有意義內容保留原意，不自行改寫。SRT 通過時間、重疊和片長驗證後交給 renderer。

目前剪輯工作流程在 Mac 本機執行，不會把音訊、影格或影片交給 Colab。Colab worker code 保留為未來可選擴充；[Google Colab MCP](https://github.com/googlecolab/colab-mcp) 要求 client 支援動態 `notifications/tools/list_changed`，而 [Codex issue #43642](https://github.com/openai/codex/issues/43642) 記錄了非同步啟動時工具清單未刷新的情形，因此先暫緩使用。更多已確認的架構決定與踩坑記錄見 [docs/DECISIONS.md](docs/DECISIONS.md)。

### 音樂搜尋

Codex 的標準剪輯流程預設自動配樂。它會依照影片主題、故事和情緒產生搜尋詞，在 Openverse 搜尋後挑選並套用合適候選，不會要求使用者先挑一首歌。背景音樂預設覆蓋全片；保留現場聲片段時，只將配樂在約 0.25 秒內平滑壓低至基準音量的 24%，現場聲維持原始音量，片段結束後配樂在約 0.25 秒內恢復。使用者可在 job 或明確要求中關閉配樂或指定歌曲。

Openverse 是聚合搜尋服務，不替個別作品確認授權。下載和使用前，Codex 必須檢查原始來源頁的作品、授權與署名要求；只使用可核實的 Public Domain、CC0 或 CC BY 曲目，並產生 credits。若原始來源或授權無法核實，就略過該曲；找不到合格歌曲時不套用音樂。單獨使用 CLI `music search` / `music add` 時仍是手動搜尋／下載；自動依影片主題搜尋、選曲和加入是 Skill 編排的剪輯流程。

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


1. 本機 Python 和 ffprobe 盤點 `assets/` 及專案根目錄中的來源照片／影片，計算 SHA-256 並快取 metadata；內容完全相同的檔案只保留一筆索引，其他路徑記錄為 duplicate sources。JPEG/HEIC/HEIF 另外由 Apple ImageIO 讀 EXIF 拍攝時間、時區、GPS 和 IPTC 內嵌地名；manifest 依可解析的拍攝時間排序，時區缺失會標成 approximate。
2. FFmpeg/VideoToolbox 產生場景候選、proxy、代表影格、接觸表與抽取音訊；素材本身保持不變。
3. 建立 `work/perception_index.json`，加入照片／代表影格的本機 Apple Vision OCR、拍攝時間與 GPS 證據、語音逐字稿及信心訊號；缺少的元件標記為 pending/not configured。OCR 文字不是字幕或核准文案，GPS 不做外部反查。
4. Codex 依 profile、transcript、perception evidence 與素材資料建立 `story_plan.json` 和 `edit_plan.json`，自動完成第一版故事、片段、節奏和情緒走向，不等待計畫核准；使用者看過初版後再提出修改。Antigravity/Gemini 可作選擇性複核。
5. 依 edit plan，對保留且含語音的片段執行本機轉錄，再將時間戳映射到輸出時間軸；Codex 判斷字幕是否有意義並記錄 keep/drop。只省略無意義或不確定的字幕，不改寫或編造語音。
6. 驗證器檢查素材路徑、時間範圍、公衛 claim references 與字幕 cue；Codex 協調和審閱結果。
7. 本機 Remotion 依 edit plan 合成畫面、燒錄語意判斷後保留的字幕、調整配樂音量並輸出 MP4；FFmpeg/ffprobe 執行媒體檢查。
8. 音樂流程先寫 `work/music_requirements.json`（同步鏡像至 `outputs/work/`）；SAFE_AUTO 只接受 Public Domain、CC0、CC BY 且有上游證據的曲目。搜尋或授權失敗時無音樂完成，並產生 `work/music_attribution.json`、`outputs/work/music_attribution.json`、`outputs/final/music_attribution.json`、`outputs/MUSIC_CREDITS.txt`、`outputs/PUBLISHING_CREDITS.txt`。
9. Skill 匯出 SRT、QA 報告與 review notes，直接交付第一版。修改時沿用 hash 未變的檢查、影格、perception 與分析結果，只重做受影響的步驟。

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
| EXIF/GPS、metadata chronology、SHA-256 cache、proxy、scene detection、frames/contact sheets、Apple Vision OCR、speech transcription | Mac mini M4 | 照片時間／GPS與圖片文字留在本機；字幕候選保留 Whisper 信心訊號並須聽音審閱；不自動上傳 |
| 故事／剪輯決策 | Codex 依使用者需求與素材證據建立首版 | 可由 Antigravity/Gemini 或 Claude Opus 選擇性複核；perception 分數只作線索 |
| 字幕語意審閱 | 指定導演模型／使用者；Codex 記錄決定 | 有意義內容保留，無意義 cue 可省略，不改寫原話 |
| Remotion render、FFmpeg、SRT 匯出與 QA | Mac M4 | 最終編碼不使用 Colab GPU |
| Colab notebook / GPU perception | Optional, currently deferred | Codex dynamic MCP tool refresh support is not available reliably; no current inference/upload path |
| 視覺 QA | 人工檢視 / Computer Use when needed | 不以滑鼠操作 timeline |
| Optional external transcription | 使用者明確選用時 | 需要單獨、逐次授權；不作為本機或 Colab 失敗時的自動 fallback。 |
| TTS 旁白 | 尚未接入 | 旁白預設不加入；起始訪談不詢問旁白需求，只有使用者提出時才討論 |

Apple Silicon local Whisper supports cut-first transcription and word-level timestamps. A live run on one M4 processed a few short retained clips using an already-cached MLX Whisper large-v3 snapshot; this is not a general speed or accuracy benchmark. Recording conditions and model version affect transcription quality, so review every candidate. The existing SigLIP2, anonymous speaker and temporal adapters have not completed live Colab inference or GPU/CU benchmarks and are not used by the current workflow. Audio event detection and speech-importance scoring remain unconfigured. Local-only mode supports asset inventory, local perception evidence, story/edit planning, Remotion rendering, SRT export and technical QA. OpenAI keys must only be supplied via `OPENAI_API_KEY`; `.env.example` contains no real key. TTS is not connected.

The local EXIF/GPS and Apple Vision OCR path is now wired into inspection and perception, but this revision was validated statically rather than against a real JPEG/HEIC file. Confirm its metadata/OCR output during the next local project run before treating a particular photo's time, location or visible text as verified.

`music-library/manifest.yaml` 只列出使用者擁有或明確有權使用的本機曲目。Openverse 自動選曲的授權來源與 credits 必須一併記錄。

## 安裝與主要檔案

- `skills/video-factory/SKILL.md`：Codex 入口與工作流程。
- `skills/video-factory/profiles/`：`memory` 和 `public-health` 導演偏好。
- `skills/video-factory/templates/`：job、edit plan、素材分析與 story plan 格式。
- `skills/video-factory/scripts/`：本機確定性工具及保留的可選雲端 adapters。
- `video_editor/application.py`、`video_editor/queue_runner.py`：GUI／桌面應用呼叫的 application boundary、序列 queue service 與結構化 progress events。
- `video_editor/job_queue.py`、`video_editor/system_resources.py`、`video_editor/process_lock.py`：persistent queue、Mac memory/swap 閘門及跨程序重型工作鎖。
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

字幕只分析 edit plan 已保留現場音的片段。用 `draft-captions` 產生候選後，Codex 自動判斷有意義的語音並以 `apply-caption-review` 套用 keep/drop，再匯出 SRT。預設沿用本機快取，找不到模型時不下載、不改送外部服務；不會把 API 或 Colab 當成 fallback。清楚有意義的原話按詞級時間戳映射到最終 timeline；意義或轉錄不確定時先不放字幕，並在 review notes 說明。

Remotion runtime 版本由 `skills/video-factory/runtime/remotion/package.json` 和 lockfile 固定。若 runtime 尚無 `node_modules/`，在該目錄執行 `pnpm install`。專案提供的 Remotion Agent Skills 可放在 `.agents/skills/`；此目錄是本機開發輔助檔，不是執行核心程式的必要依賴。

安裝完成後重開 Codex 工作階段，使用 `$video-factory` 觸發共用 Skill。Remotion 官方 Agent Skills 來源與 Codex 支援見 [Remotion Agent Skills](https://github.com/remotion-dev/remotion/blob/main/packages/skills/README.md)；目前的 Remotion CLI 與 composition 依官方文件設定。

## 驗證狀態與限制

合成 smoke test 曾輸出 640×360 H.264/AAC 影片並檢視標題、繁體中文字幕、結尾和混音 ducking。這只驗證 renderer，不代表真實家庭素材的敘事、人物裁切或字幕視覺已通過 QA。真實影片只有實際播放與檢視後才能標示視覺 QA 完成。

近似照片自動去重、TTS、完整多比例自動產出、短暫黑閃／字幕溢出偵測、audio event detection、speech importance scoring 與 FFmpeg loudness normalization 尚未完成。不得把文件、mock 測試或 CLI dry-run 當成 live GPU job 成功。public-health validator 能要求 claim reference IDs，但仍要由人核實來源是否真的支持該主張。

## 授權

本專案自行撰寫的程式碼以 MIT License 發布。第三方元件仍依各自授權使用。Remotion 原始碼公開可讀，但依 Remotion 自有授權條款使用，並非 OSI 定義的開源授權；請在使用、散布或部署前閱讀 [Remotion License FAQ](https://convert.remotion.dev/docs/license/faq) 與 [Remotion License](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md)。
