# Google Colab CLI 設定與驗證

本文件說明 Google Colab CLI 的一般安裝、登入授權與驗證流程。每台電腦的安裝狀態與版本可能不同。

---

## 一、安裝與版本確認

依照 Google Colab CLI 上游專案提供的安裝方式安裝；使用 `uv` 時可執行：

```bash
uv tool install google-colab-cli
colab --logtostderr version
```

---

## 二、Google OAuth 授權流程

Colab CLI 首次使用時會發起標準的 Google Cloud / Colab OAuth 流程：
1. 執行 `colab --logtostderr --auth=oauth2 sessions`，並依 CLI 指示開啟授權頁面。
2. 在瀏覽器中選取你預定用來執行 Colab 的 Google 帳戶並確認授權。
3. 頁面顯示授權碼（Authorization Code）。
4. 只在發起登入的本機終端 `Enter the authorization code:` 提示處貼上授權碼。
5. 憑證存放於本機環境，**不要複製到專案目錄、版本控制或對話紀錄**。

---

## 三、常用驗證與管理指令

### 1. 檢查帳戶 Compute Units (CU) 與即時額度
```bash
colab --logtostderr usage
```
- 檢查當前剩餘 Compute Units。
- 檢查當前每小時消耗速率（Usage rate/hr）。
- 檢查進行中的指派任務（Active assignments）。

### 2. 檢查進行中的 Colab 執行個體 (Sessions)
```bash
colab --logtostderr sessions
```
- 列出目前仍在運行的 VM 與 Session 名稱。

### 3. 手動停止特定 Session（釋放 GPU）
```bash
colab --logtostderr stop -s <session_name>
```
- 確保未使用的 GPU 即時終止，避免 Compute Units 耗盡。

---

## 四、安全與隱私規範

1. **禁止自動分配高價 GPU**：預設僅使用 `T4`。只有在明確經使用者核可且大工作量驗證時才可升級 `L4`；`A100` / `H100` / `G4` 嚴禁自動指派。
2. **自動釋放保證**：所有的 Colab 自動化指令皆包覆在 Python 的 `try ... finally` 區塊中，無論正常結束或異常中斷，皆會主動執行 `stop` 指令。
3. **資料最小化**：`LOCAL_ONLY` 不上傳；`BALANCED` 只上傳選定抽取音訊、representative frames 與 360p/480p proxy；`MAX_QUALITY` 只對 shortlist 產生 720p proxy。絕不將 4K 原始影像、照片圖庫或整個專案打包上傳。

## 五、Colab Perception Worker

先在 Mac 建立本機 index，不會產生雲端流量：

```bash
python3 skills/video-factory/scripts/video_factory.py perception PROJECT --privacy-mode BALANCED
```

確認 `work/perception_index.json` 的 eligible uploads 後，才可明確授權本次 derived-data transfer：

```bash
python3 skills/video-factory/scripts/video_factory.py colab-perception PROJECT \
  --privacy-mode BALANCED --gpu T4 --allow-upload
```

Worker 實作 VAD、faster-whisper `large-v3-turbo`、word timestamps、匿名 ECAPA speaker clustering、SigLIP2 image embeddings、相似度與 duplicate clustering，並將 index 下載回 Mac。這些模型尚需 live Colab benchmark；mock/unit tests 不代表 GPU 成功。Temporal backend 使用已註冊的 SmolVLM2 adapter，僅接收 shortlist proxy，亦需 live benchmark。Speaker labels 只代表單一音檔內的匿名聲紋群集，短或不確定片段保留 `unknown`；不代表真實身分、不處理重疊語音，confidence 未校準。Audio event detection 和 speech importance 尚未配置，index 必須明確保留未配置狀態。

原始 4K 永不上傳。`work/perception-cache/` 以 source hash、model/version 與參數重用結果；Family short/standard/full 共用該 cache。任何錯誤或中斷都會嘗試停止本次唯一擁有的 session。
