from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import plistlib
import shutil
import subprocess
from typing import Any


SUPPORTED_EXPORT_FORMATS = {"jpeg", "png", "webp", "avif", "tiff", "jxl"}
FORMAT_SUFFIXES = {
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "avif": ".avif",
    "tiff": ".tiff",
    "jxl": ".jxl",
}


class RapidRawError(RuntimeError):
    """Raised when a staged RapidRAW workflow cannot continue safely."""


@dataclass(frozen=True, slots=True)
class RapidRawStage:
    root: Path
    input_dir: Path
    output_dir: Path
    manifest_path: Path
    approval_template_path: Path
    photos: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class RapidRawExportResult:
    manifest_path: Path
    approved: int
    exported: int
    resumed: int


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
PreviewExtractor = Callable[[Path], bytes]


def rapidraw_install_info(binary: Path) -> dict[str, object]:
    resolved = binary.expanduser().resolve()
    info: dict[str, object] = {
        "binary": str(resolved),
        "exists": resolved.is_file(),
        "executable": resolved.is_file() and resolved.stat().st_mode & 0o111 != 0,
    }
    plist_path = resolved.parent.parent / "Info.plist"
    if plist_path.is_file():
        with plist_path.open("rb") as handle:
            payload = plistlib.load(handle)
        info["version"] = payload.get("CFBundleShortVersionString")
        info["build"] = payload.get("CFBundleVersion")
        info["bundle_id"] = payload.get("CFBundleIdentifier")
    return info


def rapidraw_render_environment(binary: Path) -> dict[str, object]:
    """Snapshot render-affecting macOS preferences, excluding private UI state."""
    if rapidraw_install_info(binary).get("bundle_id") != "io.github.CyberTimon.RapidRAW":
        return {}
    path = Path.home() / "Library/Application Support/io.github.CyberTimon.RapidRAW/settings.json"
    settings = json.loads(path.read_text()) if path.is_file() else {}
    return {key: settings.get(key) for key in (
        "tonemapperOverrideEnabled", "defaultRawTonemapper", "defaultNonRawTonemapper",
        "linearRawMode", "rawHighlightCompression", "rawPreprocessingColorNr",
        "rawPreprocessingSharpening", "applyPreprocessingToNonRaws", "processingBackend",
    )}


def rapidraw_adjustments_from_suggestion(
    suggestion: dict[str, object],
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    exposure = _bounded_float(suggestion, "exposure", -5.0, 5.0)
    brightness = _bounded_float(suggestion, "brightness", -5.0, 5.0)
    contrast = _bounded_int(suggestion, "contrast", -100, 100)
    highlights = _bounded_int(suggestion, "highlights", -100, 100)
    shadows = _bounded_int(suggestion, "shadows", -100, 100)
    whites = _bounded_int(suggestion, "whites", -100, 100)
    blacks = _bounded_int(suggestion, "blacks", -100, 100)
    temperature = _bounded_int(suggestion, "temperature", -100, 100)
    tint = _bounded_int(suggestion, "tint", -100, 100)
    vibrance = _bounded_int(suggestion, "vibrance", -100, 100)
    saturation = _bounded_int(suggestion, "saturation", -100, 100)
    clarity = _bounded_int(suggestion, "clarity", -100, 100)
    dehaze = _bounded_int(suggestion, "dehaze", -100, 100)
    structure = _bounded_int(suggestion, "structure", -100, 100)
    sharpness = _bounded_int(suggestion, "sharpness", -100, 100)
    luma_noise_reduction = _bounded_int(
        suggestion, "luma_noise_reduction", 0, 100
    )
    color_noise_reduction = _bounded_int(
        suggestion, "color_noise_reduction", 0, 100
    )
    vignette_amount = _bounded_int(suggestion, "vignette_amount", -100, 100)
    rotation = _bounded_float(suggestion, "crop_angle", -45.0, 45.0)
    has_crop = bool(suggestion.get("has_crop", False))
    if rotation != 0.0 and not has_crop:
        raise ValueError("rotation requires a crop that removes rotated edges")
    adjustments: dict[str, object] = {
        "exposure": exposure,
        "brightness": brightness,
        "contrast": contrast,
        "highlights": highlights,
        "shadows": shadows,
        "whites": whites,
        "blacks": blacks,
        "temperature": temperature,
        "tint": tint,
        "vibrance": vibrance,
        "saturation": saturation,
        "clarity": clarity,
        "dehaze": dehaze,
        "structure": structure,
        "sharpness": sharpness,
        "lumaNoiseReduction": luma_noise_reduction,
        "colorNoiseReduction": color_noise_reduction,
        "vignetteAmount": vignette_amount,
        "rotation": rotation,
        "masks": [],
        "sectionVisibility": {
            "basic": True,
            "color": True,
            "curves": True,
            "details": True,
            "effects": True,
        },
    }
    if has_crop:
        if image_size is None:
            raise ValueError("crop conversion requires the source image dimensions")
        image_width, image_height = image_size
        if image_width < 1 or image_height < 1:
            raise ValueError("crop conversion requires positive image dimensions")
        left = _bounded_float(suggestion, "crop_left", 0.0, 1.0)
        top = _bounded_float(suggestion, "crop_top", 0.0, 1.0)
        right = _bounded_float(suggestion, "crop_right", 0.0, 1.0)
        bottom = _bounded_float(suggestion, "crop_bottom", 0.0, 1.0)
        if left >= right or top >= bottom:
            raise ValueError("crop coordinates must describe a normalized non-empty rectangle")
        if (
            rotation != 0.0
            and left == 0.0
            and top == 0.0
            and right == 1.0
            and bottom == 1.0
        ):
            raise ValueError("rotation requires non-default crop bounds")
        if rotation != 0.0:
            safe_left, safe_top, safe_right, safe_bottom = (
                _rotation_safe_centered_bounds(image_width, image_height, rotation)
            )
            left = max(left, safe_left)
            top = max(top, safe_top)
            right = min(right, safe_right)
            bottom = min(bottom, safe_bottom)
            if left >= right or top >= bottom:
                raise ValueError("crop and rotation leave no safe image area")
        retained_area = (right - left) * (bottom - top)
        if retained_area < 0.10:
            raise ValueError("crop must retain at least 10% of the source image")
        adjustments["crop"] = {
            # RapidRAW's Rust export path deserializes Crop as pixel coordinates;
            # its React canvas alone understands the optional percent unit.
            "x": round(left * image_width, 6),
            "y": round(top * image_height, 6),
            "width": round((right - left) * image_width, 6),
            "height": round((bottom - top) * image_height, 6),
        }
    return adjustments


def stage_rapidraw_develop(
    suggestions_path: Path,
    root: Path,
    *,
    limit: int | None = None,
    cull_sh_commit: str | None = None,
) -> RapidRawStage:
    """Copy suggested RAWs into an isolated RapidRAW stage and write sidecars."""
    suggestions_path = suggestions_path.expanduser().resolve()
    root = root.expanduser().resolve()
    if limit is not None and limit < 1:
        raise ValueError("stage limit must be at least 1")
    if root.exists():
        raise FileExistsError(f"RapidRAW stage already exists: {root}")

    header, records = load_edit_suggestions(suggestions_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("edit suggestions contain no usable photo records")

    sources = [Path(str(record["source"])).expanduser().resolve() for record in records]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"suggested RAW is missing: {missing[0]}")

    root.mkdir(parents=True)
    input_dir = root / "input"
    output_dir = root / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    duplicate_names = _duplicate_names(sources)
    manifest_records: list[dict[str, object]] = []
    for index, (record, source) in enumerate(zip(records, sources), start=1):
        record_id = str(record.get("id") or f"edit-{index:04d}")
        staged_name = (
            f"{index:04d}-{source.name}"
            if source.name.casefold() in duplicate_names
            else source.name
        )
        staged = input_dir / staged_name
        shutil.copy2(source, staged)
        suggestion = record["suggestion"]
        assert isinstance(suggestion, dict)
        sidecar = staged.with_name(staged.name + ".rrdata")
        sidecar_payload = {
            "version": 1,
            "rating": 0,
            "adjustments": rapidraw_adjustments_from_suggestion(
                suggestion,
                image_size=(
                    _raw_image_dimensions(source)
                    if bool(suggestion.get("has_crop", False))
                    else None
                ),
            ),
            "tags": ["Cull.sh", "AI edit suggestion", "Approval pending"],
        }
        sidecar.write_text(
            json.dumps(sidecar_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest_records.append(
            {
                "id": record_id,
                "filename": source.name,
                "source": str(source),
                "staged_raw": str(staged),
                "rapidraw_sidecar": str(sidecar),
                "suggestion": suggestion,
                "rapidraw_adjustments": sidecar_payload["adjustments"],
            }
        )

    manifest_path = root / "rapidraw-manifest.json"
    manifest_payload = {
        "schema_version": 2,
        "kind": "cull-sh-rapidraw-stage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_suggestions": str(suggestions_path),
        "suggestion_header": header,
        "cull_sh_commit": cull_sh_commit,
        "photos": len(manifest_records),
        "originals_modified": False,
        "approval_required": True,
        "records": manifest_records,
    }
    _write_json(manifest_path, manifest_payload)
    digest = _file_sha256(manifest_path)
    approval_template_path = root / "rapidraw-approvals.template.json"
    _write_json(
        approval_template_path,
        {
            "schema_version": 1,
            "manifest_sha256": digest,
            "decisions": {str(record["id"]): False for record in manifest_records},
        },
    )
    return RapidRawStage(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        approval_template_path=approval_template_path,
        photos=len(manifest_records),
        manifest_sha256=digest,
    )


def render_rapidraw_review(
    root: Path,
    binary: Path,
    *,
    quality: int = 82,
    command_runner: CommandRunner | None = None,
    preview_extractor: PreviewExtractor | None = None,
) -> Path:
    """Render disposable RapidRAW JPEGs and build a local approval page."""
    if not 1 <= quality <= 100:
        raise ValueError("preview quality must be between 1 and 100")
    root, manifest, digest = _load_stage(root)
    install = rapidraw_install_info(binary)
    if not install["exists"] or not install["executable"]:
        raise FileNotFoundError(f"RapidRAW binary is not executable: {binary}")

    runner = command_runner or _run_command
    extractor = preview_extractor or _default_preview_extractor
    render_dir = root / "preview-renders"
    thumbnail_dir = root / "review-thumbnails"
    render_dir.mkdir(exist_ok=True)
    thumbnail_dir.mkdir(exist_ok=True)
    records = _manifest_records(manifest)
    for record in records:
        record_id = str(record["id"])
        staged_raw = Path(str(record["staged_raw"]))
        before_thumb = thumbnail_dir / f"{record_id}-before.jpg"
        after_thumb = thumbnail_dir / f"{record_id}-after.jpg"
        rendered = render_dir / f"{record_id}.jpg"
        if not before_thumb.is_file():
            _thumbnail_bytes(extractor(staged_raw), before_thumb)
        if not rendered.is_file():
            command = _export_command(
                binary,
                staged_raw,
                rendered,
                output_format="jpeg",
                quality=quality,
                keep_metadata=False,
            )
            _run_checked(runner, command)
        if not rendered.is_file():
            raise RapidRawError(f"RapidRAW reported success but did not create {rendered}")
        if not after_thumb.is_file():
            _thumbnail_path(rendered, after_thumb)

    page = _write_review_page(root, records, digest, install)
    _write_json(
        root / "rapidraw-preview.json",
        {
            "schema_version": 1,
            "manifest_sha256": digest,
            "rapidraw": install,
            "quality": quality,
            "photos": len(records),
            "review_page": str(page),
        },
    )
    return page


def export_rapidraw_stage(
    root: Path,
    binary: Path,
    *,
    approvals_path: Path | None = None,
    approve_all: bool = False,
    output_format: str = "jpeg",
    quality: int = 92,
    keep_metadata: bool = True,
    command_runner: CommandRunner | None = None,
) -> RapidRawExportResult:
    """Export only explicitly approved staged photographs through RapidRAW."""
    if output_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(
            f"unsupported RapidRAW export format {output_format!r}; "
            f"choose one of {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}"
        )
    if not 1 <= quality <= 100:
        raise ValueError("export quality must be between 1 and 100")
    root, manifest, digest = _load_stage(root)
    records = _manifest_records(manifest)
    approved_ids = (
        {str(record["id"]) for record in records}
        if approve_all
        else _load_approvals(approvals_path, digest)
    )
    approved_records = [record for record in records if str(record["id"]) in approved_ids]
    if not approved_records:
        raise RapidRawError("approval file contains no approved photographs")

    install = rapidraw_install_info(binary)
    if not install["exists"] or not install["executable"]:
        raise FileNotFoundError(f"RapidRAW binary is not executable: {binary}")
    runner = command_runner or _run_command
    export_manifest_path = root / "rapidraw-export.json"
    previous = (
        json.loads(export_manifest_path.read_text(encoding="utf-8"))
        if export_manifest_path.is_file()
        else {}
    )
    if previous and previous.get("manifest_sha256") != digest:
        raise RapidRawError("existing export manifest belongs to a different stage")
    completed_records = {
        str(record["id"]): record
        for record in previous.get("records", [])
        if record.get("status") == "exported"
    }
    export_records: list[dict[str, object]] = list(completed_records.values())
    suffix = FORMAT_SUFFIXES[output_format]
    resumed = 0
    exported = 0
    for record in approved_records:
        record_id = str(record["id"])
        staged_raw = Path(str(record["staged_raw"]))
        target = root / "output" / f"{Path(staged_raw.name).stem}_edited{suffix}"
        completed = completed_records.get(record_id)
        if completed is not None and target.is_file():
            resumed += 1
            continue
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite output not recorded as complete: {target}"
            )
        command = _export_command(
            binary,
            staged_raw,
            target,
            output_format=output_format,
            quality=quality,
            keep_metadata=keep_metadata,
        )
        try:
            _run_checked(runner, command)
            if not target.is_file():
                raise RapidRawError(
                    f"RapidRAW reported success but did not create {target}"
                )
        except Exception as exc:
            export_records.append(
                {
                    "id": record_id,
                    "source": record["source"],
                    "output": str(target),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            _write_export_manifest(
                export_manifest_path,
                digest,
                install,
                output_format,
                quality,
                keep_metadata,
                approved_ids,
                export_records,
            )
            raise
        export_records.append(
            {
                "id": record_id,
                "source": record["source"],
                "output": str(target),
                "status": "exported",
                "bytes": target.stat().st_size,
            }
        )
        exported += 1
        _write_export_manifest(
            export_manifest_path,
            digest,
            install,
            output_format,
            quality,
            keep_metadata,
            approved_ids,
            export_records,
        )
    return RapidRawExportResult(
        manifest_path=export_manifest_path,
        approved=len(approved_records),
        exported=exported,
        resumed=resumed,
    )


def load_edit_suggestions(
    suggestions_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load both production flat JSONL suggestions and benchmark records."""
    header: dict[str, object] = {}
    records: list[dict[str, object]] = []
    with suggestions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"suggestion line {line_number} is not a JSON object")
            nested = payload.get("suggestion")
            if isinstance(nested, dict):
                if payload.get("status") not in (None, "ok"):
                    continue
                suggestion = dict(nested)
                source_value = payload.get("raw_path") or suggestion.get("asset_id")
            elif payload.get("raw_path") or payload.get("asset_id"):
                suggestion = dict(payload)
                source_value = payload.get("raw_path") or payload.get("asset_id")
            else:
                header.update(payload)
                continue
            if not source_value:
                continue
            records.append(
                {
                    "id": payload.get("id"),
                    "source": str(source_value),
                    "suggestion": suggestion,
                }
            )
    return header, records


def _write_review_page(
    root: Path,
    records: list[dict[str, object]],
    digest: str,
    install: dict[str, object],
) -> Path:
    cards: list[str] = []
    for record in records:
        record_id = str(record["id"])
        filename = str(record["filename"])
        suggestion = record.get("suggestion", {})
        adjustments = record.get("rapidraw_adjustments", {})
        assert isinstance(suggestion, dict)
        assert isinstance(adjustments, dict)
        recipe_fields = (
            "exposure",
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "vibrance",
            "saturation",
            "clarity",
            "dehaze",
            "structure",
            "sharpness",
            "lumaNoiseReduction",
            "colorNoiseReduction",
            "vignetteAmount",
            "rotation",
        )
        changed = [
            f"{name} {adjustments.get(name)}"
            for name in recipe_fields
            if adjustments.get(name, 0) != 0
        ]
        if adjustments.get("crop") is not None:
            changed.append("crop")
        recipe = " · ".join(changed) if changed else "No executable adjustment"
        additional = suggestion.get("additional_edits")
        additional_text = (
            "; ".join(str(item) for item in additional)
            if isinstance(additional, list) and additional
            else ""
        )
        cards.append(
            f"<article data-id='{escape(record_id)}'>"
            f"<div class='heading'><h2>{escape(filename)}</h2>"
            f"<label><input class='decision' type='checkbox' data-id='{escape(record_id)}'> Approve final export</label></div>"
            "<div class='pair'>"
            f"<figure><img src='review-thumbnails/{escape(record_id)}-before.jpg'><figcaption>Before: embedded RAW preview</figcaption></figure>"
            f"<figure><img src='review-thumbnails/{escape(record_id)}-after.jpg'><figcaption>After: RapidRAW preview render</figcaption></figure>"
            "</div>"
            f"<p class='recipe'>{escape(recipe)}</p>"
            f"<p>{escape(str(suggestion.get('summary', '')))}</p>"
            + (
                f"<p><strong>Additional edit intents:</strong> {escape(additional_text)}</p>"
                if additional_text
                else ""
            )
            + "</article>"
        )
    ids_json = json.dumps([str(record["id"]) for record in records])
    version = escape(str(install.get("version") or "unknown"))
    page = root / "review.html"
    page.write_text(
        """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Cull.sh RapidRAW review</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101010;color:#eee}
body{max-width:1700px;margin:auto;padding:28px}h1{margin-bottom:4px}.note{color:#bbb;margin-bottom:18px}
.toolbar{position:sticky;top:0;z-index:2;background:#101010ee;padding:12px 0;display:flex;gap:10px;align-items:center}
button{padding:9px 13px;border-radius:8px;border:1px solid #555;background:#292929;color:#eee;cursor:pointer}
article{background:#191919;border:1px solid #333;border-radius:14px;padding:16px;margin:18px 0}
.heading{display:flex;justify-content:space-between;gap:12px;align-items:center}h2{font-size:1rem;margin:0 0 12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}
img{display:block;width:100%;height:520px;object-fit:contain;background:#080808;border-radius:8px}
figcaption{color:#aaa;padding-top:6px}.recipe{color:#e9c46a}article.approved{border-color:#2a9d8f}
@media(max-width:800px){.pair{grid-template-columns:1fr}img{height:auto}.heading{display:block}}
</style></head><body><h1>Cull.sh RapidRAW review</h1>
"""
        + f"<p class='note'>{len(records)} staged photographs rendered with RapidRAW {version}. Originals were not modified.</p>"
        + "<div class='toolbar'><button id='approve-all'>Approve all</button><button id='reject-all'>Reject all</button><button id='download'>Download approvals</button><strong id='count'></strong></div>"
        + "".join(cards)
        + f"""<script>
const ids={ids_json}; const manifestHash={json.dumps(digest)};
const storageKey='cull-sh-rapidraw-'+manifestHash; const saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
function refresh(){{let count=0; document.querySelectorAll('.decision').forEach(box=>{{box.checked=!!saved[box.dataset.id]; box.closest('article').classList.toggle('approved',box.checked); if(box.checked) count++;}}); document.getElementById('count').textContent=count+' / '+ids.length+' approved'; localStorage.setItem(storageKey,JSON.stringify(saved));}}
document.querySelectorAll('.decision').forEach(box=>box.addEventListener('change',()=>{{saved[box.dataset.id]=box.checked;refresh();}}));
document.getElementById('approve-all').onclick=()=>{{ids.forEach(id=>saved[id]=true);refresh();}};
document.getElementById('reject-all').onclick=()=>{{ids.forEach(id=>saved[id]=false);refresh();}};
document.getElementById('download').onclick=()=>{{const payload={{schema_version:1,manifest_sha256:manifestHash,decisions:saved}};const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));link.download='rapidraw-approvals.json';link.click();URL.revokeObjectURL(link.href);}};
refresh();</script></body></html>""",
        encoding="utf-8",
    )
    return page


def _load_approvals(path: Path | None, digest: str) -> set[str]:
    if path is None:
        raise RapidRawError(
            "final export requires --approvals from review.html or explicit --approve-all"
        )
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != digest:
        raise RapidRawError("approval file belongs to a different RapidRAW stage")
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict):
        raise RapidRawError("approval file decisions must be an object keyed by photo id")
    return {str(record_id) for record_id, approved in decisions.items() if approved is True}


def _load_stage(root: Path) -> tuple[Path, dict[str, object], str]:
    root = root.expanduser().resolve()
    manifest_path = root / "rapidraw-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RapidRAW stage manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "cull-sh-rapidraw-stage":
        raise RapidRawError("not a Cull.sh RapidRAW stage manifest")
    return root, manifest, _file_sha256(manifest_path)


def _manifest_records(manifest: dict[str, object]) -> list[dict[str, object]]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RapidRawError("RapidRAW stage manifest contains no records")
    if not all(isinstance(record, dict) for record in records):
        raise RapidRawError("RapidRAW stage manifest has an invalid record")
    return records  # type: ignore[return-value]


def _write_export_manifest(
    path: Path,
    digest: str,
    install: dict[str, object],
    output_format: str,
    quality: int,
    keep_metadata: bool,
    approved_ids: set[str],
    records: list[dict[str, object]],
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "cull-sh-rapidraw-export",
            "manifest_sha256": digest,
            "rapidraw": install,
            "format": output_format,
            "quality": quality,
            "keep_metadata": keep_metadata,
            "approved_ids": sorted(approved_ids),
            "records": records,
        },
    )


def _export_command(
    binary: Path,
    source: Path,
    output: Path,
    *,
    output_format: str,
    quality: int,
    keep_metadata: bool,
) -> list[str]:
    command = [
        str(binary.expanduser().resolve()),
        "export",
        str(source),
        "--output",
        str(output),
        "--format",
        output_format,
        "--quality",
        str(quality),
    ]
    if keep_metadata:
        command.append("--keep-metadata")
    return command


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _run_checked(runner: CommandRunner, command: list[str]) -> None:
    result = runner(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown RapidRAW error").strip()
        raise RapidRawError(f"RapidRAW export failed ({result.returncode}): {detail}")


def _default_preview_extractor(path: Path) -> bytes:
    from cull_sh.extractors import build_default_extractor

    return build_default_extractor().extract_preview_bytes(path)


def _thumbnail_bytes(data: bytes, target: Path) -> None:
    from PIL import Image, ImageOps

    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=88, optimize=True)


def _thumbnail_path(source: Path, target: Path) -> None:
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=88, optimize=True)


def _bounded_float(
    payload: dict[str, object], key: str, minimum: float, maximum: float
) -> float:
    value = float(payload.get(key, 0.0))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")
    return value


def _rotation_safe_centered_bounds(
    image_width: int, image_height: int, rotation_degrees: float
) -> tuple[float, float, float, float]:
    """Return a centered same-aspect crop whose corners contain no rotation fill."""
    if rotation_degrees == 0.0:
        return 0.0, 0.0, 1.0, 1.0

    angle = math.radians(rotation_degrees)
    cosine = abs(math.cos(angle))
    sine = abs(math.sin(angle))
    # Scaling the original canvas about its center gives four symmetric crop
    # corners.  The inverse-rotated corners remain inside the source while both
    # of these constraints hold.
    scale = min(
        image_width / (image_width * cosine + image_height * sine),
        image_height / (image_width * sine + image_height * cosine),
    )
    inset = (1.0 - scale) / 2.0
    return inset, inset, 1.0 - inset, 1.0 - inset


def _bounded_int(
    payload: dict[str, object], key: str, minimum: int, maximum: int
) -> int:
    raw = payload.get(key, 0)
    if isinstance(raw, bool) or int(raw) != float(raw):
        raise ValueError(f"{key} must be an integer")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _raw_image_dimensions(path: Path) -> tuple[int, int]:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("exiftool is required to map normalized crops to RapidRAW pixels")
    result = subprocess.run(
        [exiftool, "-j", "-ImageWidth", "-ImageHeight", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"failed to read RAW dimensions for crop: {detail}")
    payload = json.loads(result.stdout)
    if not payload:
        raise RuntimeError(f"failed to read RAW dimensions for crop: {path}")
    width = int(payload[0].get("ImageWidth", 0))
    height = int(payload[0].get("ImageHeight", 0))
    if width < 1 or height < 1:
        raise RuntimeError(f"invalid RAW dimensions for crop: {path}")
    return width, height


def _duplicate_names(paths: list[Path]) -> set[str]:
    counts: dict[str, int] = {}
    for path in paths:
        key = path.name.casefold()
        counts[key] = counts.get(key, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
