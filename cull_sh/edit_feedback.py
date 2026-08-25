from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
from PIL import ImageStat

from cull_sh.backends.base import VisionBackend
from cull_sh.manifests import decision_from_manifest_record
from cull_sh.manifests import load_manifest_records
from cull_sh.models import AssetKind
from cull_sh.models import DecisionBucket
from cull_sh.models import EditReviewPair
from cull_sh.models import EditSuggestion
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.rapidraw import rapidraw_install_info
from cull_sh.rapidraw import rapidraw_adjustments_from_suggestion
from cull_sh.xmp import sidecar_is_picked


ProgressCallback = Callable[[str], None]
CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class FeedbackPilotResult:
    root: Path
    manifest_path: Path
    review_page: Path
    photos: int
    accepted: int
    refined: int
    reverted: int


@dataclass(frozen=True, slots=True)
class FeedbackExportResult:
    manifest_path: Path
    output_dir: Path
    photos: int
    exported: int
    resumed: int


def select_machine_picks(
    cull_run: Path,
    human_baseline: Path,
    source_root: Path,
) -> list[RawAsset]:
    """Select manifest picks that were not already human-picked in the backup."""
    return select_feedback_picks(
        cull_run,
        human_baseline,
        source_root,
        include_existing_picks=False,
    )


def select_feedback_picks(
    cull_run: Path,
    human_baseline: Path,
    source_root: Path,
    *,
    include_existing_picks: bool,
    ignore_baseline_picks: bool = False,
) -> list[RawAsset]:
    """Select machine picks, optionally unioned with protected baseline picks."""
    if include_existing_picks and ignore_baseline_picks:
        raise ValueError(
            "include_existing_picks and ignore_baseline_picks are mutually exclusive"
        )
    source_root = source_root.expanduser().resolve()
    human_baseline = human_baseline.expanduser().resolve()
    selected: list[RawAsset] = []
    matched_baseline_sidecars: set[str] = set()
    protected_sidecars = (
        set()
        if ignore_baseline_picks
        else {
            path.name.casefold()
            for path in human_baseline.glob("*.xmp")
            if not path.name.startswith("._") and sidecar_is_picked(path)
        }
    )
    for record in load_manifest_records(cull_run.expanduser().resolve()):
        decision = decision_from_manifest_record(record)
        raw_path = Path(str(record["raw_path"])).expanduser().resolve()
        if raw_path.parent != source_root:
            continue
        xmp_path = Path(str(record.get("xmp_path", raw_path.with_suffix(".xmp"))))
        baseline_sidecar = human_baseline / xmp_path.name
        baseline_picked = (
            False if ignore_baseline_picks else sidecar_is_picked(baseline_sidecar)
        )
        if baseline_picked:
            matched_baseline_sidecars.add(xmp_path.name.casefold())
        machine_picked = (
            decision is not None and decision.bucket == DecisionBucket.PICK
        )
        if ignore_baseline_picks:
            selected_by_scope = machine_picked
        else:
            selected_by_scope = machine_picked or (
                include_existing_picks and baseline_picked
            )
        if not selected_by_scope:
            continue
        if baseline_picked and not include_existing_picks and not ignore_baseline_picks:
            continue
        if not raw_path.is_file():
            raise FileNotFoundError(f"selected RAW is missing: {raw_path}")
        selected.append(
            RawAsset(
                raw_path=raw_path,
                xmp_path=xmp_path,
                kind=AssetKind(str(record.get("asset_kind", AssetKind.RAW.value))),
            )
        )
    if include_existing_picks:
        missing = sorted(protected_sidecars - matched_baseline_sidecars)
        if missing:
            raise ValueError(
                "protected baseline pick is missing from the frozen cull manifest: "
                f"{missing[0]}"
            )
    return sorted(selected, key=lambda asset: asset.filename)


def run_feedback_pilot(
    assets: list[RawAsset],
    root: Path,
    binary: Path,
    backend: VisionBackend,
    *,
    prompt: str,
    model: str,
    cull_run: Path,
    human_baseline: Path,
    suggestion_batch_size: int = 1,
    review_batch_size: int = 1,
    include_crop: bool = True,
    selection_scope: str = "machine-picks-only",
    quality: int = 88,
    progress: ProgressCallback | None = None,
    command_runner: CommandRunner | None = None,
) -> FeedbackPilotResult:
    """Run a resumable neutral-render, suggest, render, and one-review pilot."""
    if not assets:
        raise ValueError("feedback pilot requires at least one selected photo")
    if suggestion_batch_size < 1 or review_batch_size < 1:
        raise ValueError("feedback pilot batch sizes must be at least one")
    if not 1 <= quality <= 100:
        raise ValueError("feedback pilot quality must be between 1 and 100")
    binary = binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"RapidRAW binary is missing: {binary}")
    root = root.expanduser().resolve()
    progress = progress or (lambda _message: None)
    runner = command_runner or _run_command

    input_dir = root / "input"
    baseline_dir = root / "baseline"
    first_dir = root / "first-pass"
    final_dir = root / "final"
    for directory in (root, input_dir, baseline_dir, first_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "feedback-manifest.json"
    manifest = _load_or_create_manifest(
        manifest_path,
        assets,
        model=model,
        prompt=prompt,
        cull_run=cull_run,
        human_baseline=human_baseline,
        include_crop=include_crop,
        selection_scope=selection_scope,
        suggestion_batch_size=suggestion_batch_size,
        review_batch_size=review_batch_size,
    )
    records = _records(manifest)
    _migrate_crop_coordinate_space(root, manifest_path, manifest, records)

    progress(f"Preparing {len(records)} isolated RAW copies and neutral renders.")
    for index, (asset, record) in enumerate(zip(assets, records), start=1):
        staged_raw = input_dir / asset.filename
        if not staged_raw.is_file():
            shutil.copy2(asset.raw_path, staged_raw)
        neutral = EditSuggestion(filename=asset.filename, asset_id=str(asset.raw_path))
        baseline = baseline_dir / f"{asset.raw_path.stem}.jpg"
        if not baseline.is_file():
            _write_rrdata(staged_raw, neutral, "Neutral baseline")
            _render(binary, staged_raw, baseline, quality, runner)
        record["staged_raw"] = str(staged_raw)
        record["baseline_render"] = str(baseline)
        record["baseline_metrics"] = _image_metrics(baseline)
        _write_json(manifest_path, manifest)
        progress(f"Baseline {index}/{len(records)}: {asset.filename}")

    pending_initial = [record for record in records if "initial_suggestion" not in record]
    for cohort_index, cohort in enumerate(
        _chunked(pending_initial, suggestion_batch_size), start=1
    ):
        previews = [
            PreviewImage(
                asset=_asset_from_record(record),
                image_bytes=Path(str(record["baseline_render"])).read_bytes(),
            )
            for record in cohort
        ]
        progress(
            f"Qwen initial suggestions cohort {cohort_index}: "
            + ", ".join(preview.asset.filename for preview in previews)
        )
        suggestions = backend.suggest_edits(
            prompt,
            previews,
            include_crop=include_crop,
        )
        if len(suggestions) != len(cohort):
            raise RuntimeError("edit backend returned the wrong suggestion count")
        for record, suggestion in zip(cohort, suggestions):
            record["initial_suggestion"] = _suggestion_payload(suggestion)
        _write_json(manifest_path, manifest)

    progress("Rendering the first-pass Qwen adjustments through RapidRAW.")
    for index, record in enumerate(records, start=1):
        staged_raw = Path(str(record["staged_raw"]))
        suggestion = _suggestion_from_payload(record["initial_suggestion"])
        first_render = first_dir / f"{staged_raw.stem}.jpg"
        if not first_render.is_file():
            baseline_size = _image_size(Path(str(record["baseline_render"])))
            _write_rrdata(
                staged_raw,
                suggestion,
                "Qwen first pass",
                image_size=baseline_size,
            )
            _render(binary, staged_raw, first_render, quality, runner)
            _validate_render_dimensions(
                Path(str(record["baseline_render"])),
                first_render,
                suggestion,
            )
        record["first_render"] = str(first_render)
        record["first_metrics"] = _image_metrics(first_render)
        _write_json(manifest_path, manifest)
        progress(f"First render {index}/{len(records)}: {staged_raw.name}")

    pending_reviews = [record for record in records if "review" not in record]
    for cohort_index, cohort in enumerate(
        _chunked(pending_reviews, review_batch_size), start=1
    ):
        pairs = [
            EditReviewPair(
                asset=_asset_from_record(record),
                baseline_bytes=Path(str(record["baseline_render"])).read_bytes(),
                edited_bytes=Path(str(record["first_render"])).read_bytes(),
                suggestion=_suggestion_from_payload(record["initial_suggestion"]),
            )
            for record in cohort
        ]
        progress(
            f"Qwen rendered-edit review cohort {cohort_index}: "
            + ", ".join(pair.asset.filename for pair in pairs)
        )
        reviews = backend.review_edits(prompt, pairs)
        if len(reviews) != len(cohort):
            raise RuntimeError("edit backend returned the wrong review count")
        for record, review in zip(cohort, reviews):
            record["review"] = {
                "verdict": review.verdict.value,
                "summary": review.summary,
            }
            record["final_suggestion"] = _suggestion_payload(
                review.final_suggestion
            )
        _write_json(manifest_path, manifest)

    progress("Rendering accepted, reverted, or once-refined final recipes.")
    for index, record in enumerate(records, start=1):
        staged_raw = Path(str(record["staged_raw"]))
        final_render = final_dir / f"{staged_raw.stem}.jpg"
        initial = _suggestion_from_payload(record["initial_suggestion"])
        final = _suggestion_from_payload(record["final_suggestion"])
        baseline_path = Path(str(record["baseline_render"]))
        _write_rrdata(
            staged_raw,
            final,
            "Qwen validated final",
            image_size=_image_size(baseline_path),
        )
        if not final_render.is_file():
            if _suggestion_payload(initial) == _suggestion_payload(final):
                shutil.copy2(Path(str(record["first_render"])), final_render)
            else:
                _render(binary, staged_raw, final_render, quality, runner)
                _validate_render_dimensions(baseline_path, final_render, final)
        record["final_render"] = str(final_render)
        record["final_metrics"] = _image_metrics(final_render)
        _write_json(manifest_path, manifest)
        progress(f"Final render {index}/{len(records)}: {staged_raw.name}")

    review_page = _write_review_page(root, records)
    verdicts = [str(record["review"]["verdict"]) for record in records]
    return FeedbackPilotResult(
        root=root,
        manifest_path=manifest_path,
        review_page=review_page,
        photos=len(records),
        accepted=verdicts.count("accept"),
        refined=verdicts.count("refine"),
        reverted=verdicts.count("reject"),
    )


def export_feedback_stage(
    root: Path,
    binary: Path,
    output_dir: Path,
    *,
    quality: int = 95,
    keep_metadata: bool = True,
    command_runner: CommandRunner | None = None,
) -> FeedbackExportResult:
    """Export every completed Qwen-validated recipe as a delivery JPEG."""
    if not 1 <= quality <= 100:
        raise ValueError("feedback export quality must be between 1 and 100")
    root = root.expanduser().resolve()
    binary = binary.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest_path = root / "feedback-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feedback manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "cull-sh-rendered-edit-feedback-pilot":
        raise ValueError("not a Cull.sh rendered feedback stage")
    records = _records(manifest)
    if not records:
        raise ValueError("feedback manifest contains no records")

    required = {
        "id",
        "filename",
        "source",
        "staged_raw",
        "baseline_render",
        "final_render",
        "final_suggestion",
        "review",
    }
    for record in records:
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"feedback record {record.get('filename', record.get('id'))} is incomplete: "
                f"missing {missing[0]}"
            )
        if not isinstance(record["review"], dict):
            raise ValueError(
                f"feedback record {record['filename']} has an invalid review"
            )
        if not isinstance(record["final_suggestion"], dict):
            raise ValueError(
                f"feedback record {record['filename']} has an invalid final suggestion"
            )
        if not Path(str(record["final_render"])).is_file():
            raise FileNotFoundError(
                f"validated final render is missing: {record['final_render']}"
            )

    record_ids = [str(record["id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("feedback manifest contains duplicate record ids")
    source_roots = {Path(str(record["source"])).resolve().parent for record in records}
    if any(
        output_dir == source_root or source_root in output_dir.parents
        for source_root in source_roots
    ):
        raise ValueError("feedback delivery output must be outside the source photo folder")
    if output_dir == root or root in output_dir.parents:
        raise ValueError("feedback delivery output must be outside the feedback stage")

    install = rapidraw_install_info(binary)
    if not install["exists"] or not install["executable"]:
        raise FileNotFoundError(f"RapidRAW binary is not executable: {binary}")
    runner = command_runner or _run_command
    feedback_digest = _file_sha256(manifest_path)
    export_manifest_path = root / "feedback-export.json"
    previous = (
        json.loads(export_manifest_path.read_text(encoding="utf-8"))
        if export_manifest_path.is_file()
        else {}
    )
    if previous:
        expected = {
            "feedback_manifest_sha256": feedback_digest,
            "output_dir": str(output_dir),
            "quality": quality,
            "keep_metadata": keep_metadata,
        }
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"existing feedback export uses a different {key.replace('_', ' ')}"
                )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to use non-empty untracked delivery folder: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = {
        str(record["id"]): record
        for record in previous.get("records", [])
        if isinstance(record, dict) and record.get("status") == "exported"
    }
    export_records: list[dict[str, object]] = list(completed.values())
    output_names = _feedback_output_names(records)
    exported = 0
    resumed = 0
    for record, output_name in zip(records, output_names):
        record_id = str(record["id"])
        staged_raw = Path(str(record["staged_raw"]))
        if not staged_raw.is_file():
            raise FileNotFoundError(f"staged RAW is missing: {staged_raw}")
        target = output_dir / output_name
        prior = completed.get(record_id)
        if prior is not None and target.is_file():
            expected_sha = prior.get("sha256")
            if expected_sha and _file_sha256(target) != expected_sha:
                raise ValueError(f"completed delivery JPEG changed after export: {target}")
            resumed += 1
            continue
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite output not recorded as complete: {target}"
            )

        baseline = Path(str(record["baseline_render"]))
        final = _suggestion_from_payload(record["final_suggestion"])
        _write_rrdata(
            staged_raw,
            final,
            "Qwen validated unattended final",
            image_size=_image_size(baseline),
        )
        command = [
            str(binary),
            "export",
            str(staged_raw),
            "--output",
            str(target),
            "--format",
            "jpeg",
            "--quality",
            str(quality),
        ]
        if keep_metadata:
            command.append("--keep-metadata")
        try:
            result = runner(command)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "unknown error").strip()
                raise RuntimeError(
                    f"RapidRAW delivery export failed for {staged_raw.name}: {detail}"
                )
            if not target.is_file():
                raise RuntimeError(
                    f"RapidRAW reported success but did not create {target}"
                )
            _validate_render_dimensions(baseline, target, final)
        except Exception as exc:
            export_records.append(
                {
                    "id": record_id,
                    "filename": record["filename"],
                    "source": record["source"],
                    "output": str(target),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            _write_feedback_export_manifest(
                export_manifest_path,
                feedback_digest,
                output_dir,
                install,
                quality,
                keep_metadata,
                len(records),
                export_records,
            )
            raise
        width, height = _image_size(target)
        review = record["review"]
        assert isinstance(review, dict)
        export_records.append(
            {
                "id": record_id,
                "filename": record["filename"],
                "source": record["source"],
                "selection": record.get("selection"),
                "qwen_verdict": review.get("verdict"),
                "output": str(target),
                "status": "exported",
                "bytes": target.stat().st_size,
                "width": width,
                "height": height,
                "sha256": _file_sha256(target),
            }
        )
        exported += 1
        _write_feedback_export_manifest(
            export_manifest_path,
            feedback_digest,
            output_dir,
            install,
            quality,
            keep_metadata,
            len(records),
            export_records,
        )
    return FeedbackExportResult(
        manifest_path=export_manifest_path,
        output_dir=output_dir,
        photos=len(records),
        exported=exported,
        resumed=resumed,
    )


def _feedback_output_names(records: list[dict[str, object]]) -> list[str]:
    stems = [Path(str(record["filename"])).stem for record in records]
    duplicate_stems = {
        stem.casefold() for stem in stems if sum(s.casefold() == stem.casefold() for s in stems) > 1
    }
    return [
        f"{record['id']}-{stem}.jpg" if stem.casefold() in duplicate_stems else f"{stem}.jpg"
        for record, stem in zip(records, stems)
    ]


def _write_feedback_export_manifest(
    path: Path,
    feedback_digest: str,
    output_dir: Path,
    install: dict[str, object],
    quality: int,
    keep_metadata: bool,
    total_photos: int,
    records: list[dict[str, object]],
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "cull-sh-rapidraw-feedback-export",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "unattended",
            "feedback_manifest_sha256": feedback_digest,
            "output_dir": str(output_dir),
            "rapidraw": install,
            "format": "jpeg",
            "quality": quality,
            "keep_metadata": keep_metadata,
            "photos": total_photos,
            "completed_photos": sum(
                1 for record in records if record.get("status") == "exported"
            ),
            "records": records,
        },
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_create_manifest(
    path: Path,
    assets: list[RawAsset],
    *,
    model: str,
    prompt: str,
    cull_run: Path,
    human_baseline: Path,
    include_crop: bool,
    selection_scope: str,
    suggestion_batch_size: int,
    review_batch_size: int,
) -> dict[str, object]:
    filenames = [asset.filename for asset in assets]
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = [str(record["filename"]) for record in _records(payload)]
        if existing != filenames:
            raise ValueError("existing feedback stage has a different selection")
        return payload
    payload: dict[str, object] = {
        "schema_version": 3,
        "kind": "cull-sh-rendered-edit-feedback-pilot",
        "model": model,
        "prompt": prompt,
        "cull_run": str(cull_run.expanduser().resolve()),
        "human_baseline": str(human_baseline.expanduser().resolve()),
        "include_crop": include_crop,
        "selection_scope": selection_scope,
        "suggestion_batch_size": suggestion_batch_size,
        "review_batch_size": review_batch_size,
        "originals_modified": False,
        "maximum_refinements": 1,
        "crop_coordinate_space": "rapidraw-pixels-v2",
        "records": [
            {
                "id": f"pilot-{index:04d}",
                "filename": asset.filename,
                "source": str(asset.raw_path),
                "xmp_path": str(asset.xmp_path),
                "asset_kind": asset.kind.value,
                "selection": (
                    "frozen-machine-pick"
                    if selection_scope == "frozen-machine-picks"
                    else "protected-existing-pick"
                    if sidecar_is_picked(
                        human_baseline.expanduser().resolve() / asset.xmp_path.name
                    )
                    else "machine-pick-not-in-human-baseline"
                ),
            }
            for index, asset in enumerate(assets, start=1)
        ],
    }
    _write_json(path, payload)
    return payload


def _records(payload: dict[str, object]) -> list[dict[str, object]]:
    records = payload.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("feedback manifest records are invalid")
    return records  # type: ignore[return-value]


def _asset_from_record(record: dict[str, object]) -> RawAsset:
    source = Path(str(record["source"]))
    return RawAsset(
        raw_path=source,
        xmp_path=Path(str(record["xmp_path"])),
        kind=AssetKind(str(record.get("asset_kind", AssetKind.RAW.value))),
    )


def _suggestion_payload(suggestion: EditSuggestion) -> dict[str, object]:
    return asdict(suggestion)


def _suggestion_from_payload(payload: object) -> EditSuggestion:
    if not isinstance(payload, dict):
        raise ValueError("invalid edit suggestion in feedback manifest")
    return EditSuggestion(**payload)


def _write_rrdata(
    raw_path: Path,
    suggestion: EditSuggestion,
    tag: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> None:
    payload = {
        "version": 1,
        "rating": 0,
        "adjustments": rapidraw_adjustments_from_suggestion(
            _suggestion_payload(suggestion),
            image_size=image_size,
        ),
        "tags": ["Cull.sh", tag],
    }
    _write_json(raw_path.with_name(raw_path.name + ".rrdata"), payload)


def _render(
    binary: Path,
    source: Path,
    output: Path,
    quality: int,
    runner: CommandRunner,
) -> None:
    command = [
        str(binary),
        "export",
        str(source),
        "--output",
        str(output),
        "--format",
        "jpeg",
        "--quality",
        str(quality),
    ]
    result = runner(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"RapidRAW render failed for {source.name}: {detail}")
    if not output.is_file():
        raise RuntimeError(f"RapidRAW reported success but did not create {output}")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _image_metrics(path: Path) -> dict[str, float | int]:
    with Image.open(path) as image:
        gray = image.convert("L")
        histogram = gray.histogram()
        pixels = max(1, sum(histogram))
        return {
            "width": image.width,
            "height": image.height,
            "mean_luma": round(float(ImageStat.Stat(gray).mean[0]), 3),
            "shadow_clip_fraction": round(sum(histogram[:4]) / pixels, 6),
            "highlight_clip_fraction": round(sum(histogram[252:]) / pixels, 6),
        }


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _validate_render_dimensions(
    baseline: Path,
    rendered: Path,
    suggestion: EditSuggestion,
) -> None:
    base_width, base_height = _image_size(baseline)
    render_width, render_height = _image_size(rendered)
    base_area = base_width * base_height
    rendered_fraction = (render_width * render_height) / max(1, base_area)
    if suggestion.has_crop:
        expected_fraction = (
            (suggestion.crop_right - suggestion.crop_left)
            * (suggestion.crop_bottom - suggestion.crop_top)
        )
        if rendered_fraction < expected_fraction * 0.5:
            raise RuntimeError(
                f"RapidRAW crop render is unexpectedly small: {rendered.name} "
                f"is {render_width}x{render_height}, expected roughly "
                f"{expected_fraction:.0%} of {base_width}x{base_height}"
            )
    elif rendered_fraction < 0.5:
        raise RuntimeError(
            f"RapidRAW non-crop render is unexpectedly small: {rendered.name} "
            f"is {render_width}x{render_height} versus {base_width}x{base_height}"
        )


def _migrate_crop_coordinate_space(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    if manifest.get("crop_coordinate_space") == "rapidraw-pixels-v2":
        return
    invalid_dir = root / "invalid-percent-crop-renders"
    affected = 0
    for record in records:
        suggestion = record.get("initial_suggestion")
        if not isinstance(suggestion, dict) or not suggestion.get("has_crop"):
            continue
        affected += 1
        for key, prefix in (("first_render", "first"), ("final_render", "final")):
            value = record.get(key)
            if value:
                source = Path(str(value))
                if source.is_file():
                    invalid_dir.mkdir(exist_ok=True)
                    target = invalid_dir / f"{prefix}-{source.name}"
                    if not target.exists():
                        shutil.move(source, target)
        for key in (
            "first_render",
            "first_metrics",
            "review",
            "final_suggestion",
            "final_render",
            "final_metrics",
        ):
            record.pop(key, None)
    manifest["crop_coordinate_space"] = "rapidraw-pixels-v2"
    manifest["invalidated_legacy_crop_records"] = affected
    _write_json(manifest_path, manifest)


def _write_review_page(root: Path, records: list[dict[str, object]]) -> Path:
    cards: list[str] = []
    for record in records:
        record_id = str(record["id"])
        initial = record["initial_suggestion"]
        final = record["final_suggestion"]
        review = record["review"]
        assert isinstance(initial, dict)
        assert isinstance(final, dict)
        assert isinstance(review, dict)
        final_matches_first = initial == final
        figures = (
            _figure(root, Path(str(record["baseline_render"])), "Baseline")
            + _figure(
                root,
                Path(str(record["first_render"])),
                "Edited (validated final)" if final_matches_first else "First edit",
            )
        )
        choices = (
            (("baseline", "Baseline"), ("edited", "Edited"), ("tie", "Tie"))
            if final_matches_first
            else (
                ("baseline", "Baseline"),
                ("first", "First edit"),
                ("final", "Validated final"),
                ("tie", "Tie"),
            )
        )
        if not final_matches_first:
            figures += _figure(
                root,
                Path(str(record["final_render"])),
                "Validated final",
            )
        cards.append(
            f"<article data-id='{escape(record_id)}'>"
            f"<h2>{escape(str(record['filename']))}</h2>"
            + f"<div class='images {'two' if final_matches_first else 'three'}'>"
            + figures
            + "</div>"
            + f"<p><strong>Qwen verdict:</strong> {escape(str(review['verdict']))} — {escape(str(review.get('summary', '')))}</p>"
            + f"<p class='recipe'>First: {escape(_recipe(initial))}<br>Final: {escape(_recipe(final))}</p>"
            + "<div class='choices'>Human preference: "
            + " ".join(
                f"<label><input type='radio' name='{escape(record_id)}' value='{choice}'> {label}</label>"
                for choice, label in choices
            )
            + "</div></article>"
        )
    ids = json.dumps([str(record["id"]) for record in records])
    protected = sum(
        1 for record in records if record.get("selection") == "protected-existing-pick"
    )
    frozen = sum(
        1 for record in records if record.get("selection") == "frozen-machine-pick"
    )
    machine_added = len(records) - protected
    selection_note = (
        f"{frozen} frozen machine pick(s); pre-existing flags were ignored."
        if frozen
        else f"{protected} protected existing pick(s) and {machine_added} machine-added pick(s)."
        if protected
        else "Machine-added picks only."
    )
    page = root / "review.html"
    page.write_text(
        """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Cull.sh rendered edit feedback pilot</title><style>
:root{color-scheme:dark;font-family:system-ui;background:#101010;color:#eee}body{max-width:1900px;margin:auto;padding:24px}
.toolbar{position:sticky;top:0;background:#101010ee;padding:12px 0;z-index:2}button{padding:9px 14px}article{border:1px solid #333;border-radius:12px;padding:14px;margin:18px 0;background:#181818}
.images{display:grid;gap:10px}.images.two{grid-template-columns:repeat(2,1fr)}.images.three{grid-template-columns:repeat(3,1fr)}figure{margin:0}img{width:100%;height:460px;object-fit:contain;background:#080808}figcaption{color:#aaa}.recipe{color:#e9c46a}.choices{display:flex;gap:18px;flex-wrap:wrap}label{padding:8px;border:1px solid #555;border-radius:8px}label:has(input:checked){background:#14532d;border-color:#4ade80}@media(max-width:900px){.images{grid-template-columns:1fr!important}img{height:auto}}
</style></head><body><h1>Rendered edit feedback pilot</h1><p>"""
        + escape(selection_note)
        + """ Originals were not modified.</p>
<div class='toolbar'><button id='download'>Download choices CSV</button> <strong id='count'></strong></div>"""
        + "".join(cards)
        + f"""<script>const ids={ids};const key='cull-sh-edit-feedback-'+location.pathname;const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
function refresh(){{let n=0;ids.forEach(id=>{{const v=saved[id];if(v){{n++;const el=document.querySelector(`input[name="${{id}}"]`+`[value="${{v}}"]`);if(el)el.checked=true;}}}});document.getElementById('count').textContent=n+' / '+ids.length+' reviewed';localStorage.setItem(key,JSON.stringify(saved));}}
document.querySelectorAll('input[type=radio]').forEach(el=>el.addEventListener('change',()=>{{saved[el.name]=el.value;refresh();}}));
document.getElementById('download').onclick=()=>{{let csv='id,choice\\n'+ids.map(id=>id+','+(saved[id]||'')).join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='edit-feedback-choices.csv';a.click();}};refresh();</script></body></html>""",
        encoding="utf-8",
    )
    return page


def _figure(root: Path, path: Path, label: str) -> str:
    relative = path.relative_to(root).as_posix()
    return f"<figure><img src='{escape(relative)}'><figcaption>{escape(label)}</figcaption></figure>"


def _recipe(payload: dict[str, object]) -> str:
    fields = (
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
        "luma_noise_reduction",
        "color_noise_reduction",
        "vignette_amount",
    )
    changed = [
        f"{field} {payload.get(field)}"
        for field in fields
        if payload.get(field, 0) != 0
    ]
    text = " · ".join(changed) if changed else "no executable global adjustment"
    if payload.get("has_crop"):
        text += " · crop"
    if payload.get("crop_angle", 0) != 0:
        text += f" · rotation {payload.get('crop_angle')}"
    additional = payload.get("additional_edits")
    if isinstance(additional, list) and additional:
        text += " · additional: " + "; ".join(str(item) for item in additional)
    return text


def _chunked(items: list[dict[str, object]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
