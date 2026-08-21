from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from cull_sh.config import PipelineConfig
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import WorkItem


def create_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        suffix = "" if attempt == 0 else f"-{attempt}"
        run_dir = root / f"{timestamp}{suffix}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("failed to create a unique run directory")


def find_latest_run_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no run directories found under {root}")
    return candidates[-1]


def write_manifest(run_dir: Path, items: list[WorkItem]) -> Path:
    manifest_path = run_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in items:
            payload = {
                "filename": item.filename,
                "asset_kind": item.asset.kind.value,
                "raw_path": str(item.asset.raw_path),
                "xmp_path": str(item.asset.xmp_path),
                "paired_raw_path": (
                    str(item.asset.paired_raw_path)
                    if item.asset.paired_raw_path is not None
                    else None
                ),
                "scene_id": item.scene_id,
                "scene_index": item.scene_index,
                "status": item.status.value,
                "blur_score": item.metrics.blur_score if item.metrics else None,
                "tenengrad_score": item.metrics.tenengrad_score if item.metrics else None,
                "musiq_score": item.metrics.musiq_score if item.metrics else None,
                "nima_score": item.metrics.nima_score if item.metrics else None,
                "topiq_score": item.metrics.topiq_score if item.metrics else None,
                "brisque_score": item.metrics.brisque_score if item.metrics else None,
                "cpbd_score": item.metrics.cpbd_score if item.metrics else None,
                "brightness_mean": item.metrics.brightness_mean if item.metrics else None,
                "contrast_stddev": item.metrics.contrast_stddev if item.metrics else None,
                "face_count": item.metrics.face_count if item.metrics else None,
                "eye_count": item.metrics.eye_count if item.metrics else None,
                "local_rank_score": item.metrics.local_rank_score if item.metrics else None,
                "combined_rank_score": (
                    item.metrics.combined_rank_score if item.metrics else None
                ),
                "perceptual_hash": item.metrics.perceptual_hash if item.metrics else None,
                "decision": (
                    {
                        "rating": item.decision.rating,
                        "label": item.decision.label.value
                        if item.decision and item.decision.label
                        else None,
                        "bucket": item.decision.bucket.value,
                        "keep": item.decision.keep,
                        "source": item.decision.source.value,
                        "summary": item.decision.summary,
                    }
                    if item.decision
                    else None
                ),
                "preview_cache_path": (
                    str(item.preview.cache_path)
                    if item.preview and item.preview.cache_path is not None
                    else None
                ),
                "local_trace": item.local_trace,
                "vision_trace": item.vision_trace,
                "sidecar_written": item.sidecar_written,
                "lightroom_edit_written": item.lightroom_edit_written,
                "error": item.error,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    return manifest_path


def load_manifest_records(run_dir: Path) -> list[dict[str, object]]:
    manifest_path = run_dir / "manifest.jsonl"
    records: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def decision_from_manifest_record(record: dict[str, object]) -> FinalDecision | None:
    decision_payload = record.get("decision")
    if not isinstance(decision_payload, dict):
        return None

    label_value = decision_payload.get("label")
    label = ColorLabel(label_value) if isinstance(label_value, str) else None
    bucket_value = decision_payload.get("bucket")
    if isinstance(bucket_value, str):
        bucket = DecisionBucket(bucket_value)
    else:
        keep_value = decision_payload.get("keep")
        bucket = DecisionBucket.PICK if bool(keep_value) else DecisionBucket.REJECT
    return FinalDecision(
        filename=str(record["filename"]),
        rating=int(decision_payload["rating"]),
        label=label,
        bucket=bucket,
        source=DecisionSource(str(decision_payload["source"])),
        summary=str(decision_payload.get("summary", "")),
    )


def write_run_config(run_dir: Path, config: PipelineConfig) -> Path:
    config_path = run_dir / "config.json"
    payload = {
        "path": str(config.path),
        "prompt": config.prompt,
        "genre": config.genre,
        "prefer": config.prefer,
        "prompt_source": config.prompt_source,
        "limit": config.limit,
        "limit_kind": "scene",
        "provider": config.backend.provider,
        "model": config.backend.model,
        "backend_url": config.backend.base_url,
        "backend_timeout_seconds": config.backend.timeout_seconds,
        "backend_max_attempts": config.backend.max_attempts,
        "backend_max_output_tokens": config.backend.max_output_tokens,
        "batch_size": config.batch_size,
        "extract_workers": config.extract_workers,
        "score_workers": config.score_workers,
        "scene_gap_seconds": config.scene_gap_seconds,
        "scene_max_sequence_gap": config.scene_max_sequence_gap,
        "min_blur_score": config.min_blur_score,
        "min_tenengrad_score": config.min_tenengrad_score,
        "enable_learned_iqa": config.enable_learned_iqa,
        "enable_topiq_ranking": config.enable_topiq_ranking,
        "topiq_rank_weight": config.topiq_rank_weight,
        "enable_topiq_shadow": config.enable_topiq_shadow,
        "topiq_shadow_workers": config.topiq_shadow_workers,
        "topiq_shadow_low_percentile": config.topiq_shadow_low_percentile,
        "topiq_shadow_high_percentile": config.topiq_shadow_high_percentile,
        "topiq_shadow_max_items": config.topiq_shadow_max_items,
        "min_musiq_score": config.min_musiq_score,
        "min_nima_score": config.min_nima_score,
        "enable_brisque": config.enable_brisque,
        "enable_cpbd": config.enable_cpbd,
        "max_brisque_score": config.max_brisque_score,
        "min_cpbd_score": config.min_cpbd_score,
        "use_brisque_for_reject": config.use_brisque_for_reject,
        "use_cpbd_for_reject": config.use_cpbd_for_reject,
        "local_reject_required_support_votes": config.local_reject_required_support_votes,
        "duplicate_hamming_threshold": config.duplicate_hamming_threshold,
        "max_scene_candidates": config.max_scene_candidates,
        "include_jpegs": config.include_jpegs,
        "mirror_paired_jpegs": config.mirror_paired_jpegs,
        "dry_run": config.dry_run,
        "cache_previews": config.cache_previews,
        "lightroom_auto_edit": config.lightroom_auto_edit,
        "lightroom_edit_scope": config.lightroom_edit_scope.value,
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config_path
