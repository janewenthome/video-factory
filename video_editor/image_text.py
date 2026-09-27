"""Local OCR for text visible in project photos.

On macOS this uses the system Vision framework through Swift. It has no model
download path and never turns recognized text into approved editorial captions;
results are evidence for a later editorial review. Unsupported hosts return an
explicit pending result instead of installing an OCR dependency.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_VISION_SWIFT = r'''import Foundation
import NaturalLanguage
import Vision

func emit(_ value: [String: Any], exitCode: Int32 = 0) -> Never {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("Could not encode Vision result as JSON.\n".utf8))
        exit(exitCode == 0 ? 2 : exitCode)
    }
    exit(exitCode)
}

guard CommandLine.arguments.count == 2 else {
    emit(["status": "error", "reason": "invalid_input"] , exitCode: 2)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
do {
    let availableLanguages = try VNRecognizeTextRequest.supportedRecognitionLanguages(
        for: request.recognitionLevel,
        revision: request.revision
    )
    let preferredLanguages = ["zh-Hant", "zh-Hans", "en-US", "ja-JP"]
    let configuredLanguages = preferredLanguages.filter { availableLanguages.contains($0) }
    if !configuredLanguages.isEmpty {
        request.recognitionLanguages = configuredLanguages
    }
    // URL-backed loading handles the source image's orientation metadata and
    // ImageIO-supported formats such as HEIC without rewriting the source.
    try VNImageRequestHandler(url: imageURL, options: [:]).perform([request])
} catch {
    emit(["status": "error", "reason": "vision_request_failed", "detail": String(describing: error)], exitCode: 2)
}

let observations = request.results ?? []
let items: [[String: Any]] = observations.compactMap { observation in
    guard let candidate = observation.topCandidates(1).first else { return nil }
    let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return nil }

    let box = observation.boundingBox
    let tagger = NLTagger(tagSchemes: [.language])
    tagger.string = text
    let language = tagger.dominantLanguage?.rawValue ?? "und"

    return [
        "text": text,
        "bounding_box": [
            "x": Double(box.origin.x),
            "y": Double(1.0 - box.origin.y - box.height),
            "width": Double(box.width),
            "height": Double(box.height),
            "coordinate_space": "normalized_top_left"
        ],
        "language": language,
        "confidence": Double(candidate.confidence)
    ]
}

emit([
    "status": "completed",
    "items": items,
    "engine": "Apple Vision VNRecognizeTextRequest",
    "engine_revision": request.revision,
    "operating_system": ProcessInfo.processInfo.operatingSystemVersionString
])
'''


def _pending(source: Path, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pending",
        "availability": "unavailable",
        "reason": reason,
        "source": source.name,
        "local_only": True,
        "items": [],
        "provenance": {"backend": None, "source_sha256": None},
    }


def recognize_image_text(image_path: str | Path, *, timeout_seconds: int = 90) -> dict[str, Any]:
    """Recognize visible text using an already available local macOS backend.

    Bounding boxes use normalized top-left coordinates in the range [0, 1].
    This helper only returns OCR evidence; callers must not treat text as an
    approved subtitle, caption, or factual assertion without editorial review.
    """
    source = Path(image_path)
    if not source.is_file():
        return {
            "schema_version": 1,
            "status": "error",
            "availability": "available",
            "reason": "image_file_not_found",
            "source": source.name,
            "local_only": True,
            "items": [],
            "provenance": {"backend": "Apple Vision", "source_sha256": None},
        }

    if platform.system() != "Darwin":
        return _pending(source, "macos_vision_unavailable_on_this_platform")

    swift = shutil.which("swift")
    if not swift:
        return _pending(source, "swift_runtime_unavailable")

    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return {
            "schema_version": 1,
            "status": "error",
            "availability": "available",
            "reason": "image_read_failed",
            "source": source.name,
            "local_only": True,
            "items": [],
            "provenance": {"backend": "Apple Vision", "source_sha256": None},
        }

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".swift", encoding="utf-8") as script:
            script.write(_VISION_SWIFT)
            script.flush()
            result = subprocess.run(
                [swift, script.name, str(source.resolve())],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "schema_version": 1,
            "status": "pending",
            "availability": "available",
            "reason": "vision_request_timed_out",
            "source": source.name,
            "local_only": True,
            "items": [],
            "provenance": {"backend": "Apple Vision", "source_sha256": digest.hexdigest()},
        }
    except OSError:
        return _pending(source, "vision_process_unavailable")

    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        payload = {"status": "error", "reason": "vision_result_invalid"}

    if result.returncode != 0 or payload.get("status") != "completed":
        return {
            "schema_version": 1,
            "status": "error",
            "availability": "available",
            "reason": payload.get("reason", "vision_request_failed"),
            "source": source.name,
            "local_only": True,
            "items": [],
            "provenance": {
                "backend": "Apple Vision VNRecognizeTextRequest",
                "source_sha256": digest.hexdigest(),
                "detail": payload.get("detail"),
            },
        }

    recognized_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "status": "completed",
        "availability": "available",
        "source": source.name,
        "local_only": True,
        "items": payload.get("items", []),
        "provenance": {
            "backend": payload.get("engine"),
            "engine_revision": payload.get("engine_revision"),
            "operating_system": payload.get("operating_system"),
            "source_sha256": digest.hexdigest(),
            "recognized_at": recognized_at,
            "editorial_review_required": True,
        },
    }
