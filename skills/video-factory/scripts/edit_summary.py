"""Generate human-readable edit summary markdown from edit_plan.json.

Acts as the human review gate before final render. Summarizes duration,
used assets, omitted sections, story structure, highlights, subtitles, music,
and items requiring manual verification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from video_factory import (
    MANIFEST_RELATIVE_PATH,
    PLAN_RELATIVE_PATH,
    UserFacingError,
    load_json,
    parse_float,
    project_path,
    resolve_project,
    utc_now,
    work_path,
)


def generate_edit_summary(project_dir: Path) -> Path:
    project = resolve_project(project_dir)
    plan_file = project_path(project, PLAN_RELATIVE_PATH)
    if not plan_file.is_file():
        raise UserFacingError(f"Edit plan is missing: {plan_file}. Run plan generation first.")

    plan = load_json(plan_file, "edit plan")
    if not isinstance(plan, dict):
        raise UserFacingError("Edit plan must be a JSON object.")

    # Load manifest to compare used vs unused assets
    manifest_file = project_path(project, MANIFEST_RELATIVE_PATH)
    all_assets: dict[str, Any] = {}
    if manifest_file.is_file():
        manifest = load_json(manifest_file, "media manifest")
        for asset in manifest.get("assets", []):
            if isinstance(asset, dict) and asset.get("source"):
                all_assets[asset["source"]] = asset

    title = plan.get("title") or "未命名剪輯專案"
    timeline = plan.get("timeline", [])
    fps = parse_float(plan.get("fps")) or 30.0
    duration = parse_float(plan.get("duration_seconds"))
    if duration is None:
        duration = max((parse_float(item.get("timeline_end")) or 0.0 for item in timeline if isinstance(item, dict)), default=0.0)
    target_duration = parse_float(plan.get("target_duration_seconds"))
    duration_policy = plan.get("duration_policy") if isinstance(plan.get("duration_policy"), dict) else {}

    used_sources: set[str] = set()
    clips_summary: list[dict[str, Any]] = []
    subtitles_count = 0
    music_tracks: list[str] = []
    text_cards: list[str] = []
    medical_claims: list[dict[str, Any]] = []

    for index, seg in enumerate(timeline):
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type", "unknown")
        src = seg.get("source")
        if src:
            used_sources.add(src)

        t_start = parse_float(seg.get("timeline_start")) or 0.0
        t_end = parse_float(seg.get("timeline_end")) or 0.0

        if seg_type in {"video", "photo"}:
            clips_summary.append({
                "type": seg_type,
                "source": src,
                "range": f"{t_start:.1f}s - {t_end:.1f}s (片長 {t_end - t_start:.1f}s)",
                "notes": seg.get("notes") or seg.get("reason") or "無特殊說明",
            })
        elif seg_type == "subtitle":
            subtitles_count += 1
        elif seg_type == "music":
            if src:
                music_tracks.append(f"{src} ({t_start:.1f}s - {t_end:.1f}s)")
        elif seg_type in {"title", "text_card"}:
            text_cards.append(f"[{seg_type}] {seg.get('text', '')} ({t_start:.1f}s - {t_end:.1f}s)")

        claims = seg.get("medical_claims", [])
        if isinstance(claims, list):
            for c in claims:
                if isinstance(c, dict):
                    medical_claims.append(c)

    # Calculate omitted / unused assets
    unused_sources = sorted(set(all_assets.keys()) - used_sources)

    # Detect opening hook, highlights, and ending
    opening = clips_summary[0] if clips_summary else None
    ending = clips_summary[-1] if clips_summary else None

    lines = [
        f"# 剪輯計畫審核摘要：{title}",
        "",
        f"- **產生時間**：{utc_now()}",
        f"- **預計影片長度**：`{duration:.1f} 秒` (約 {duration/60:.1f} 分鐘)",
        f"- **畫面格率**：`{fps} fps`",
        f"- **剪輯片段數量**：{len(clips_summary)} 個鏡頭",
        f"- **字幕數量**：{subtitles_count} 條",
        "",
        "---",
        "",
        "## 一、故事架構與節奏",
        "",
    ]
    if target_duration is not None:
        lines.insert(4, f"- **目標片長**：`{target_duration:.1f} 秒`（{plan.get('profile', '未指定')} / {duration_policy.get('preset', 'custom')}）")
        minimum = parse_float(duration_policy.get("minimum_seconds"))
        maximum = parse_float(duration_policy.get("maximum_seconds"))
        if minimum is not None and maximum is not None and not (minimum <= duration <= maximum):
            if plan.get("profile") in {"health_education", "public-health"} and duration > maximum:
                lines.append(f"- **片長超出衛教範圍**：目前 `{duration:.1f} 秒` 超過上限 `{maximum:.1f} 秒`；醫療內容完整性優先，請在 review notes 說明保留超時的事實理由。")
            else:
                lines.append(f"- **片長需要確認**：目前 `{duration:.1f} 秒` 不在 `{minimum:.1f}–{maximum:.1f} 秒` 建議範圍內。")

    if opening:
        lines.append(f"- **開場鉤子 (Opening Hook)**：`{opening['source']}` ({opening['range']}) - {opening['notes']}")
    lines.append(f"- **主體段落**：共 {max(0, len(clips_summary) - 2)} 個過渡與主活動鏡頭")
    if ending and ending != opening:
        lines.append(f"- **結尾收尾 (Ending)**：`{ending['source']}` ({ending['range']}) - {ending['notes']}")

    lines.extend([
        "",
        "## 二、使用的素材清單",
        "",
    ])
    for c in clips_summary:
        lines.append(f"- [{c['type'].upper()}] `{c['source']}`: {c['range']} | {c['notes']}")

    lines.extend([
        "",
        "## 三、未採用的素材 (已略過/剪除)",
        "",
    ])
    if unused_sources:
        for u in unused_sources:
            lines.append(f"- `{u}` (未進入最終剪輯線)")
    else:
        lines.append("- （無，全部可用素材皆已被納入）")

    lines.extend([
        "",
        "## 四、標題與字幕",
        "",
        f"- 總字幕段落數：{subtitles_count} 段",
    ])
    for tc in text_cards:
        lines.append(f"- 文字卡片：{tc}")

    lines.extend([
        "",
        "## 五、配樂與音訊設定",
        "",
    ])
    if music_tracks:
        for m in music_tracks:
            lines.append(f"- 配樂軌：`{m}`")
    else:
        lines.append("- 配樂：未使用背景音樂或僅保留現場原音 (Natural Audio)")

    lines.extend([
        "",
        "## 六、需要人工審核與確認的事項 (Human Review Notes)",
        "",
    ])
    if medical_claims:
        lines.append(f"- **公衛/醫療主張審核**：本片包含 {len(medical_claims)} 條醫療事實主張，請確認皆已核對 source/references。")
    lines.append("- **構圖與人物臉部裁切**：請在 Render 後檢查人臉是否被邊界裁切。")
    lines.append("- **字幕可讀性與時間對齊**：請檢查重要語音是否與字幕同步。")
    lines.append("- **音量平衡**：請確認人聲對話與背景音樂音量比例適中 (已套用 ducking)。")
    lines.append("")

    summary_content = "\n".join(lines)
    summary_path = work_path(project, "edit_summary.md", create_dir=False)
    summary_path.write_text(summary_content, encoding="utf-8")
    return summary_path


def command_edit_summary(args: Any) -> int:
    project = resolve_project(args.project)
    out_path = generate_edit_summary(project)
    print(f"Edit summary generated: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate human-readable edit_summary.md from edit plan")
    parser.add_argument("project", help="project directory")
    args = parser.parse_args()
    return command_edit_summary(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
