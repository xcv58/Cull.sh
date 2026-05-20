from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from cull_sh.models import LightroomEditScope
from cull_sh.scanner import discover_jpeg_assets
from cull_sh.scanner import discover_raw_assets
from cull_sh.xmp import jpeg_is_rejected
from cull_sh.xmp import sidecar_has_adaptive_color_payload
from cull_sh.xmp import sidecar_has_lens_corrections
from cull_sh.xmp import sidecar_is_rejected


@dataclass(slots=True)
class LightroomAdaptiveColorStage:
    path: Path
    edit_scope: LightroomEditScope
    total: int
    rejected: int
    non_rejected: int
    candidates: int
    already_adaptive_color: int
    pending_adaptive_color: int
    lens_corrections_enabled: int
    rejected_filenames: list[str]
    pending_filenames: list[str]
    adaptive_color_filenames: list[str]
    lightroom_acr_filenames: list[str]
    embedded_dng_filenames: list[str]
    dng_candidate_filenames: list[str]
    errors: list[dict[str, str]]


@dataclass(slots=True)
class LightroomJpegAutoStage:
    path: Path
    edit_scope: LightroomEditScope
    total: int
    rejected: int
    non_rejected: int
    candidates: int
    rejected_filenames: list[str]
    candidate_filenames: list[str]
    errors: list[dict[str, str]]


@dataclass(slots=True)
class _EmbeddedDngState:
    rejected: bool = False
    has_adaptive_color: bool = False
    has_lens_corrections: bool = False


def build_adaptive_color_stage(
    path: Path,
    extensions: tuple[str, ...],
    limit: int | None = None,
    edit_scope: LightroomEditScope = LightroomEditScope.ALL,
) -> LightroomAdaptiveColorStage:
    assets = discover_raw_assets(path, extensions)
    if limit is not None:
        assets = assets[:limit]

    rejected_filenames: list[str] = []
    pending_filenames: list[str] = []
    adaptive_color_filenames: list[str] = []
    lightroom_acr_filenames: list[str] = []
    embedded_dng_filenames: list[str] = []
    dng_candidate_filenames: list[str] = []
    errors: list[dict[str, str]] = []
    lens_corrections_enabled = 0
    non_rejected = 0

    for asset in assets:
        embedded_dng_state = _read_embedded_dng_state(asset.raw_path)
        try:
            is_rejected = sidecar_is_rejected(asset.xmp_path)
            has_adaptive_color_xmp = sidecar_has_adaptive_color_payload(asset.xmp_path)
            has_lens_corrections = sidecar_has_lens_corrections(asset.xmp_path)
        except Exception as exc:
            errors.append(
                {
                    "filename": asset.filename,
                    "xmp_path": str(asset.xmp_path),
                    "error": str(exc),
                }
            )
            continue

        has_embedded_adaptive_color = bool(
            embedded_dng_state and embedded_dng_state.has_adaptive_color
        )
        if embedded_dng_state is not None:
            is_rejected = is_rejected or embedded_dng_state.rejected
            has_lens_corrections = (
                has_lens_corrections or embedded_dng_state.has_lens_corrections
            )

        has_lightroom_acr = asset.raw_path.with_suffix(".acr").exists()
        if is_rejected:
            rejected_filenames.append(asset.filename)
        else:
            non_rejected += 1

        if edit_scope == LightroomEditScope.KEPT and is_rejected:
            continue

        if asset.raw_path.suffix.lower() == ".dng":
            dng_candidate_filenames.append(asset.filename)
        if has_lens_corrections:
            lens_corrections_enabled += 1
        if has_adaptive_color_xmp or has_lightroom_acr or has_embedded_adaptive_color:
            adaptive_color_filenames.append(asset.filename)
            if has_lightroom_acr:
                lightroom_acr_filenames.append(asset.filename)
            if has_embedded_adaptive_color:
                embedded_dng_filenames.append(asset.filename)
        else:
            pending_filenames.append(asset.filename)

    candidates = len(pending_filenames) + len(adaptive_color_filenames)
    return LightroomAdaptiveColorStage(
        path=path,
        edit_scope=edit_scope,
        total=len(assets),
        rejected=len(rejected_filenames),
        non_rejected=non_rejected,
        candidates=candidates,
        already_adaptive_color=len(adaptive_color_filenames),
        pending_adaptive_color=len(pending_filenames),
        lens_corrections_enabled=lens_corrections_enabled,
        rejected_filenames=rejected_filenames,
        pending_filenames=pending_filenames,
        adaptive_color_filenames=adaptive_color_filenames,
        lightroom_acr_filenames=lightroom_acr_filenames,
        embedded_dng_filenames=embedded_dng_filenames,
        dng_candidate_filenames=dng_candidate_filenames,
        errors=errors,
    )


def write_adaptive_color_handoff(
    stage: LightroomAdaptiveColorStage,
    run_dir: Path,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = run_dir / "lightroom-adaptive-color.json"
    checklist_path = run_dir / "lightroom-adaptive-color.md"

    payload_path.write_text(
        json.dumps(_stage_payload(stage), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checklist_path.write_text(_stage_checklist(stage), encoding="utf-8")
    return payload_path, checklist_path


def build_jpeg_auto_stage(
    path: Path,
    jpeg_extensions: tuple[str, ...],
    limit: int | None = None,
    edit_scope: LightroomEditScope = LightroomEditScope.ALL,
) -> LightroomJpegAutoStage:
    assets = discover_jpeg_assets(path, jpeg_extensions, mirror_paired_jpegs=False)
    if limit is not None:
        assets = assets[:limit]

    rejected_filenames: list[str] = []
    candidate_filenames: list[str] = []
    errors: list[dict[str, str]] = []
    non_rejected = 0

    for asset in assets:
        try:
            is_rejected = jpeg_is_rejected(asset.raw_path)
        except Exception as exc:
            errors.append(
                {
                    "filename": asset.filename,
                    "path": str(asset.raw_path),
                    "error": str(exc),
                }
            )
            continue

        if is_rejected:
            rejected_filenames.append(asset.filename)
        else:
            non_rejected += 1

        if edit_scope == LightroomEditScope.KEPT and is_rejected:
            continue
        candidate_filenames.append(asset.filename)

    return LightroomJpegAutoStage(
        path=path,
        edit_scope=edit_scope,
        total=len(assets),
        rejected=len(rejected_filenames),
        non_rejected=non_rejected,
        candidates=len(candidate_filenames),
        rejected_filenames=rejected_filenames,
        candidate_filenames=candidate_filenames,
        errors=errors,
    )


def write_jpeg_auto_handoff(
    stage: LightroomJpegAutoStage,
    run_dir: Path,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = run_dir / "lightroom-jpeg-auto.json"
    checklist_path = run_dir / "lightroom-jpeg-auto.md"

    payload_path.write_text(
        json.dumps(_jpeg_stage_payload(stage), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checklist_path.write_text(_jpeg_stage_checklist(stage), encoding="utf-8")
    return payload_path, checklist_path


def _stage_payload(stage: LightroomAdaptiveColorStage) -> dict[str, object]:
    return {
        "path": str(stage.path),
        "edit_scope": stage.edit_scope.value,
        "total": stage.total,
        "rejected": stage.rejected,
        "non_rejected_candidates": stage.non_rejected,
        "edit_candidates": stage.candidates,
        "already_adaptive_color": stage.already_adaptive_color,
        "pending_adaptive_color": stage.pending_adaptive_color,
        "lens_corrections_enabled": stage.lens_corrections_enabled,
        "seed_filename": _seed_filename(stage),
        "verification_filenames": _verification_filenames(stage),
        "rejected_filenames": stage.rejected_filenames,
        "pending_filenames": stage.pending_filenames,
        "adaptive_color_filenames": stage.adaptive_color_filenames,
        "lightroom_acr_filenames": stage.lightroom_acr_filenames,
        "embedded_dng_filenames": stage.embedded_dng_filenames,
        "dng_candidate_filenames": stage.dng_candidate_filenames,
        "dng_update_ai_settings_required": bool(stage.dng_candidate_filenames),
        "automation_mode": "seed_profile_selective_copy_paste",
        "errors": stage.errors,
        "computer_use_instruction": _computer_use_instruction(stage),
    }


def _jpeg_stage_payload(stage: LightroomJpegAutoStage) -> dict[str, object]:
    return {
        "path": str(stage.path),
        "edit_scope": stage.edit_scope.value,
        "total": stage.total,
        "rejected": stage.rejected,
        "non_rejected_candidates": stage.non_rejected,
        "edit_candidates": stage.candidates,
        "candidate_filenames": stage.candidate_filenames,
        "rejected_filenames": stage.rejected_filenames,
        "automation_mode": "jpeg_auto_settings_batch",
        "errors": stage.errors,
        "computer_use_instruction": _jpeg_computer_use_instruction(stage),
    }


def _stage_checklist(stage: LightroomAdaptiveColorStage) -> str:
    pending_preview = "\n".join(f"- {name}" for name in stage.pending_filenames[:20])
    if stage.pending_adaptive_color > 20:
        pending_preview += f"\n- ... {stage.pending_adaptive_color - 20} more"
    if not pending_preview:
        pending_preview = "- None"

    return f"""# Lightroom Adaptive Color Computer Use Handoff

Folder: `{stage.path}`

Counts:

- RAW files discovered: {stage.total}
- Edit scope: {_scope_label(stage)}
- Edit candidates: {stage.candidates}
- Non-rejected RAW files: {stage.non_rejected}
- Rejected RAW files: {stage.rejected} ({_rejected_scope_note(stage)})
- Already Adaptive Color / Lightroom AI payload: {stage.already_adaptive_color}
- Lightroom `.acr` AI payload files: {len(stage.lightroom_acr_filenames)}
- Embedded DNG AI payload files: {len(stage.embedded_dng_filenames)}
- In-scope DNG files for batch AI refresh after paste: {len(stage.dng_candidate_filenames)}
- Pending Adaptive Color: {stage.pending_adaptive_color}
- Lens corrections enabled: {stage.lens_corrections_enabled}
- Seed photo: {_seed_filename(stage) or "None"}
- Verification photos: {", ".join(_verification_filenames(stage)) or "None"}

Computer Use instruction:

{_computer_use_instruction(stage)}

Pending sample:

{pending_preview}
"""


def _jpeg_stage_checklist(stage: LightroomJpegAutoStage) -> str:
    candidate_preview = "\n".join(f"- {name}" for name in stage.candidate_filenames[:20])
    if stage.candidates > 20:
        candidate_preview += f"\n- ... {stage.candidates - 20} more"
    if not candidate_preview:
        candidate_preview = "- None"

    return f"""# Lightroom JPEG Auto Settings Computer Use Handoff

Folder: `{stage.path}`

Counts:

- JPEG files discovered: {stage.total}
- Edit scope: {_jpeg_scope_label(stage)}
- Edit candidates: {stage.candidates}
- Non-rejected JPEG files: {stage.non_rejected}
- Rejected JPEG files: {stage.rejected} ({_rejected_scope_note(stage)})

Computer Use instruction:

{_jpeg_computer_use_instruction(stage)}

Candidate sample:

{candidate_preview}
"""


def _computer_use_instruction(stage: LightroomAdaptiveColorStage) -> str:
    folder_name = stage.path.name
    seed = _seed_filename(stage)
    verification = ", ".join(_verification_filenames(stage))
    if stage.edit_scope == LightroomEditScope.ALL:
        scope_instruction = (
            "Use the search/filter controls to show RAW photos in the folder, "
            "including rejected photos; do not filter rejected photos out. "
        )
        seed_label = "one RAW"
        selection_label = "visible RAW photos"
    else:
        scope_instruction = (
            "Use the filter/refine controls to show non-rejected photos only "
            "(picked plus unflagged; rejected excluded). "
        )
        seed_label = "one non-rejected RAW"
        selection_label = "visible non-rejected photos"
    dng_instruction = _dng_update_instruction(stage)
    return (
        "Use Adobe Lightroom, not Lightroom Classic. "
        f"Open the Local folder named '{folder_name}' at {stage.path}. "
        "Clear the search box and enable Include subfolders when Lightroom shows it. "
        f"{scope_instruction}"
        f"Before applying anything, verify the visible count is {stage.candidates}. "
        f"Use {seed or seed_label} as the seed photo. In Detail view, "
        "choose Adaptive Color directly from the Edit panel's Profile dropdown and "
        "verify the seed Profile field shows Adaptive Color. Open Copy Edit "
        "Settings, clear every setting, then enable only the profile/treatment "
        "setting that carries the selected Profile. Do not copy exposure, color "
        "mixers, curves, masks, crop, geometry, detail, or lens correction settings. "
        f"Return to Grid view, select all {selection_label}, and paste the "
        "copied edit settings to the entire selection. Wait for Lightroom's paste "
        "and AI settings update to finish. "
        f"{dng_instruction}"
        "Then refresh each verification photo before reading the Profile field. "
        "Verify Adaptive Color on these photos: "
        f"{verification or 'first, middle, and last visible photos'}. If any "
        "verification photo remains Adobe Standard after the update finishes, stop "
        "and report the failed filename."
    )


def _jpeg_computer_use_instruction(stage: LightroomJpegAutoStage) -> str:
    folder_name = stage.path.name
    if stage.edit_scope == LightroomEditScope.ALL:
        scope_instruction = (
            "Use the search/filter controls to show JPEG files in the folder, "
            "including rejected photos; do not filter rejected photos out. "
        )
        selection_label = "visible JPEG photos"
    else:
        scope_instruction = (
            "Use the filter/refine controls to show non-rejected JPEG photos only "
            "(picked plus unflagged; rejected excluded). "
        )
        selection_label = "visible non-rejected JPEG photos"
    return (
        "Use Adobe Lightroom, not Lightroom Classic. "
        f"Open the Local folder named '{folder_name}' at {stage.path}. "
        "Clear the search box and enable Include subfolders when Lightroom shows it. "
        f"{scope_instruction}"
        "Filter/search for JPEG files only, using Lightroom's type/extension "
        "controls when available or a jpg/jpeg search token otherwise. "
        f"Before applying anything, verify the visible count is {stage.candidates}. "
        f"Return to Grid view, select all {selection_label}, and choose "
        f"Photo > Apply Auto Settings to {stage.candidates} Photos. Do not paste "
        "RAW edit settings and do not apply Adaptive Color/Profile to JPEGs. Wait "
        "for Lightroom's Auto Settings batch to finish. If Lightroom reports a "
        "different affected count, stop and report the mismatch."
    )


def _dng_update_instruction(stage: LightroomAdaptiveColorStage) -> str:
    if not stage.dng_candidate_filenames:
        return ""

    dng_count = len(stage.dng_candidate_filenames)
    return (
        f"Because this run includes {dng_count} in-scope DNG file(s), run the "
        "DNG AI refresh after the paste: filter/search the current folder to "
        f"show only DNG files, verify the visible count is {dng_count}, select "
        "all visible DNGs, choose Photo > Update AI Settings, and wait for "
        f"Lightroom to finish updating all {dng_count} photo(s). If you repair "
        "or rewrite cull/lens metadata after Lightroom applies Adaptive Color, "
        "repeat this DNG-only Update AI Settings step afterward. "
    )


def _scope_label(stage: LightroomAdaptiveColorStage) -> str:
    if stage.edit_scope == LightroomEditScope.ALL:
        return "all RAW files"
    return "kept RAW files only"


def _jpeg_scope_label(stage: LightroomJpegAutoStage) -> str:
    if stage.edit_scope == LightroomEditScope.ALL:
        return "all JPEG files"
    return "kept JPEG files only"


def _rejected_scope_note(stage: LightroomAdaptiveColorStage | LightroomJpegAutoStage) -> str:
    if stage.edit_scope == LightroomEditScope.ALL:
        return "included"
    return "skipped"


def _seed_filename(stage: LightroomAdaptiveColorStage) -> str | None:
    if stage.adaptive_color_filenames:
        return stage.adaptive_color_filenames[0]
    if stage.pending_filenames:
        return stage.pending_filenames[0]
    return None


def _verification_filenames(stage: LightroomAdaptiveColorStage) -> list[str]:
    filenames = stage.pending_filenames or stage.adaptive_color_filenames
    if not filenames:
        return []

    indexes = [0, len(filenames) // 2, len(filenames) - 1]
    verification: list[str] = []
    for index in indexes:
        filename = filenames[index]
        if filename not in verification:
            verification.append(filename)
    return verification


def _read_embedded_dng_state(raw_path: Path) -> _EmbeddedDngState | None:
    if raw_path.suffix.lower() != ".dng":
        return None

    exiftool = shutil.which("exiftool")
    if exiftool is None:
        return None

    try:
        result = subprocess.run(
            [
                exiftool,
                "-j",
                "-XMP-crs:AILookActive",
                "-XMP-crs:LookName",
                "-XMP-crs:LensProfileEnable",
                "-XMP-xmp:Rating",
                "-XMP-xmpDM:Pick",
                str(raw_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not payload:
        return None

    metadata = payload[0]
    return _EmbeddedDngState(
        rejected=metadata.get("Rating") == -1 or metadata.get("Pick") == -1,
        has_adaptive_color=(
            metadata.get("AILookActive") is True
            and metadata.get("LookName") == "Adaptive Color"
        ),
        has_lens_corrections=metadata.get("LensProfileEnable") == 1,
    )
