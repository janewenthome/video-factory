"""Read embedded still-photo capture time, GPS, and place labels locally.

The helper uses macOS ImageIO through the system Swift runtime. It never
rewrites the source image, performs network lookups, or installs dependencies.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".heic", ".heif"}
_SWIFT_PROGRAM = r'''import Foundation
import ImageIO
import CoreFoundation

func asDictionary(_ value: Any?) -> NSDictionary {
    value as? NSDictionary ?? NSDictionary()
}

func field(_ dictionary: NSDictionary, _ key: String) -> Any? {
    if let value = dictionary.object(forKey: key) {
        return value
    }
    for (candidate, value) in dictionary {
        if String(describing: candidate).hasSuffix(key) {
            return value
        }
    }
    return nil
}

func text(_ dictionary: NSDictionary, _ names: [String]) -> Any {
    for name in names {
        if let value = field(dictionary, name) {
            return String(describing: value)
        }
    }
    return NSNull()
}

func number(_ dictionary: NSDictionary, _ names: [String]) -> Any {
    for name in names {
        if let value = field(dictionary, name), let number = value as? NSNumber {
            return number.doubleValue
        }
    }
    return NSNull()
}

func dictionary(_ properties: NSDictionary, _ key: CFString) -> NSDictionary {
    asDictionary(properties.object(forKey: key))
}

let path = ProcessInfo.processInfo.environment["VIDEO_FACTORY_PHOTO_PATH"] ?? ""
let url = URL(fileURLWithPath: path)
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let rawProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) else {
    let error = ["error": "ImageIO could not read the image or its properties"]
    let data = try! JSONSerialization.data(withJSONObject: error)
    print(String(data: data, encoding: .utf8)!)
    exit(0)
}

let properties = rawProperties as NSDictionary
let exif = dictionary(properties, kCGImagePropertyExifDictionary)
let tiff = dictionary(properties, kCGImagePropertyTIFFDictionary)
let gps = dictionary(properties, kCGImagePropertyGPSDictionary)
let iptc = dictionary(properties, kCGImagePropertyIPTCDictionary)
let result: [String: Any] = [
    "width": number(properties, ["PixelWidth"]),
    "height": number(properties, ["PixelHeight"]),
    "datetime_original": text(exif, ["DateTimeOriginal"]),
    "offset_time_original": text(exif, ["OffsetTimeOriginal"]),
    "datetime_digitized": text(exif, ["DateTimeDigitized"]),
    "offset_time_digitized": text(exif, ["OffsetTimeDigitized"]),
    "datetime_exif": text(exif, ["DateTime"]),
    "offset_time": text(exif, ["OffsetTime"]),
    "datetime_tiff": text(tiff, ["DateTime"]),
    "latitude": number(gps, ["Latitude"]),
    "latitude_ref": text(gps, ["LatitudeRef"]),
    "longitude": number(gps, ["Longitude"]),
    "longitude_ref": text(gps, ["LongitudeRef"]),
    "horizontal_positioning_error": number(gps, ["HPositioningError", "GPSHPositioningError"]),
    "iptc_sublocation": text(iptc, ["SubLocation", "Sublocation"]),
    "iptc_city": text(iptc, ["City"]),
    "iptc_region": text(iptc, ["ProvinceState"]),
    "iptc_country": text(iptc, ["CountryName"]),
]
let data = try! JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
'''


def _optional_text(value: Any) -> str | None:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned and cleaned.lower() not in {"null", "(null)"} else None
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalized_offset(value: str | None) -> str | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*([+-])(\d{2}):?(\d{2})\s*", value)
    if not match:
        return None
    sign, hours, minutes = match.groups()
    if int(hours) > 14 or int(minutes) > 59 or (int(hours) == 14 and int(minutes) != 0):
        return None
    return f"{sign}{hours}:{minutes}"


def _capture_datetime(raw: str | None, offset: str | None) -> tuple[str | None, str | None]:
    """Return ISO-8601 capture time and explicit offset, without guessing timezone."""
    if not raw:
        return None, None
    candidate = raw.strip()
    if re.match(r"^\d{4}:\d{2}:\d{2} ", candidate):
        candidate = f"{candidate[:4]}-{candidate[5:7]}-{candidate[8:]}"
        candidate = candidate.replace(" ", "T", 1)
    candidate = candidate.replace("Z", "+00:00") if candidate.endswith("Z") else candidate

    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return raw.strip(), None

    embedded_offset = parsed.utcoffset()
    if embedded_offset is not None:
        timezone = parsed.strftime("%z")
        normalized = f"{timezone[:3]}:{timezone[3:]}" if len(timezone) == 5 else timezone
        return parsed.isoformat(), normalized

    normalized_offset = _normalized_offset(offset)
    if normalized_offset:
        try:
            parsed = parsed.replace(tzinfo=dt.timezone(dt.timedelta(
                hours=int(normalized_offset[1:3]) * (1 if normalized_offset[0] == "+" else -1),
                minutes=int(normalized_offset[4:6]) * (1 if normalized_offset[0] == "+" else -1),
            )))
            return parsed.isoformat(), normalized_offset
        except ValueError:
            pass
    return parsed.isoformat(), None


def _run_imageio(path: Path) -> dict[str, Any] | None:
    swift = shutil.which("swift")
    if not swift:
        return None

    # Swift's compiler module cache is redirected to a writable temporary path;
    # no generated code or metadata is placed beside the source image.
    cache_root = Path("/private/tmp/video-factory-swift-module-cache")
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_root = Path(tempfile.gettempdir()) / "video-factory-swift-module-cache"
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None

    environment = os.environ.copy()
    environment["CLANG_MODULE_CACHE_PATH"] = str(cache_root)
    environment["TMPDIR"] = "/private/tmp" if Path("/private/tmp").is_dir() else tempfile.gettempdir()
    environment["VIDEO_FACTORY_PHOTO_PATH"] = str(path.resolve())
    try:
        completed = subprocess.run(
            [swift, "-e", _SWIFT_PROGRAM],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _signed_coordinate(
    value: Any,
    reference: Any,
    *,
    negative_ref: str,
    positive_ref: str,
    lower: float,
    upper: float,
) -> float | None:
    number = _optional_number(value)
    if number is None or abs(number) > upper:
        return None
    ref = (_optional_text(reference) or "").upper()
    if ref == negative_ref:
        return -abs(number)
    if ref == positive_ref:
        return abs(number)
    return number if lower <= number <= upper else None


def read_photo_metadata(path: str | Path) -> dict[str, Any]:
    """Read local EXIF/IPTC capture time and GPS from JPEG, HEIC, or HEIF.

    Returned time remains timezone-naive when the source does not store an
    offset. Coordinates are never reverse-geocoded; only embedded labels are
    returned. The source file is opened read-only and is never modified.
    """
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    result: dict[str, Any] = {
        "status": "unavailable",
        "format": suffix.lstrip(".") or None,
        "metadata_sources": [],
        "width": None,
        "height": None,
        "captured_at": None,
        "capture_time": {
            "value": None,
            "timezone": None,
            "source": None,
            "uncertainty": "No embedded capture timestamp was found.",
        },
        "gps": {
            "latitude": None,
            "longitude": None,
            "accuracy_m": None,
            "source": None,
            "uncertainty": "No embedded GPS coordinates were found.",
        },
        "place_label": None,
        "place_labels": {},
        "place_label_source": None,
        "unknown_fields": [],
    }
    if suffix not in SUPPORTED_SUFFIXES:
        result["status"] = "unsupported"
        result["unknown_fields"].append("format: only JPEG, HEIC, and HEIF are supported")
        return result
    if not source.is_file():
        result["status"] = "unavailable"
        result["unknown_fields"].append("source: file does not exist or is not readable")
        return result

    raw = _run_imageio(source)
    if raw is None or raw.get("error"):
        result["unknown_fields"].append("embedded_metadata: ImageIO could not read the image properties")
        return result

    result["metadata_sources"] = ["Apple ImageIO"]
    result["width"] = _optional_number(raw.get("width"))
    result["height"] = _optional_number(raw.get("height"))

    capture_candidates = (
        ("datetime_original", "offset_time_original", "EXIF.DateTimeOriginal"),
        ("datetime_digitized", "offset_time_digitized", "EXIF.DateTimeDigitized"),
        ("datetime_exif", "offset_time", "EXIF.DateTime"),
        ("datetime_tiff", "offset_time", "TIFF.DateTime"),
    )
    for datetime_key, offset_key, provenance in capture_candidates:
        raw_datetime = _optional_text(raw.get(datetime_key))
        if not raw_datetime:
            continue
        captured_at, timezone = _capture_datetime(raw_datetime, _optional_text(raw.get(offset_key)))
        result["captured_at"] = captured_at
        if timezone:
            uncertainty = "Timestamp precision and camera clock accuracy are not stated in EXIF."
        elif captured_at == raw_datetime:
            uncertainty = (
                "Timestamp format is unrecognized; original EXIF text is retained "
                "without timezone inference."
            )
        else:
            uncertainty = (
                "EXIF does not record a timezone offset; this is camera-local time "
                "and is not converted to UTC."
            )
        result["capture_time"] = {
            "value": captured_at,
            "timezone": timezone,
            "source": provenance,
            "uncertainty": uncertainty,
        }
        if timezone is None:
            result["unknown_fields"].append("capture_time.timezone")
        break
    if result["captured_at"] is None:
        result["unknown_fields"].append("capture_time: not embedded")

    latitude = _signed_coordinate(
        raw.get("latitude"), raw.get("latitude_ref"),
        negative_ref="S", positive_ref="N", lower=-90, upper=90,
    )
    longitude = _signed_coordinate(
        raw.get("longitude"), raw.get("longitude_ref"),
        negative_ref="W", positive_ref="E", lower=-180, upper=180,
    )
    accuracy = _optional_number(raw.get("horizontal_positioning_error"))
    gps_present = latitude is not None and longitude is not None
    if gps_present:
        gps_uncertainty = (
            f"Embedded horizontal positioning error is {accuracy:g} m; device GPS "
            "and capture context can still be inaccurate."
            if accuracy is not None and accuracy >= 0
            else "Coordinates are embedded, but GPS accuracy/uncertainty is not recorded."
        )
        result["gps"] = {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_m": accuracy if accuracy is not None and accuracy >= 0 else None,
            "source": "EXIF.GPSLatitude + EXIF.GPSLongitude",
            "uncertainty": gps_uncertainty,
        }
    else:
        result["unknown_fields"].append("gps: latitude and longitude are not both available")

    labels = {
        "sublocation": _optional_text(raw.get("iptc_sublocation")),
        "city": _optional_text(raw.get("iptc_city")),
        "region": _optional_text(raw.get("iptc_region")),
        "country": _optional_text(raw.get("iptc_country")),
    }
    result["place_labels"] = {key: value for key, value in labels.items() if value}
    if result["place_labels"]:
        result["place_label"] = ", ".join(value for value in labels.values() if value)
        result["place_label_source"] = "IPTC embedded place fields"
    else:
        result["unknown_fields"].append("place_label: no embedded IPTC place label")

    result["status"] = "ok" if result["captured_at"] or gps_present or result["place_label"] else "partial"
    return result


__all__ = ["read_photo_metadata"]
