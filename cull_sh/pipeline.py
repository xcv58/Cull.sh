from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from cull_sh.backends import VisionBackendError
from cull_sh.backends import build_backend
from cull_sh.config import PipelineConfig
from cull_sh.extractors import PreviewExtractionError
from cull_sh.extractors import PreviewExtractor
from cull_sh.extractors import build_default_extractor
from cull_sh.grouping import assign_scene_groups
from cull_sh.manifests import create_run_dir
from cull_sh.manifests import write_manifest
from cull_sh.manifests import write_run_config
from cull_sh.models import (
    ColorLabel,
    DecisionBucket,
    DecisionSource,
    FinalDecision,
    LightroomEditScope,
    LocalQualityMetrics,
    PipelineSummary,
    PreviewImage,
    WorkItem,
    WorkStatus,
)
from cull_sh.quality import analyze_local_quality
from cull_sh.quality import build_local_decision_trace
from cull_sh.quality import support_metric_import_errors
from cull_sh.quality import should_reject_for_local_quality
from cull_sh.ranking import build_scene_cohorts
from cull_sh.ranking import rescue_scene_review_candidates
from cull_sh.ranking import sort_candidates_for_vision
from cull_sh.ranking import suppress_duplicates_and_rerank
from cull_sh.reporting import NullReporter
from cull_sh.reporting import PipelineReporter
from cull_sh.scanner import discover_photo_assets
from cull_sh.xmp import write_photo_metadata


def build_work_items(config: PipelineConfig) -> list[WorkItem]:
    assets = discover_photo_assets(
        config.path,
        config.extensions,
        include_jpegs=config.include_jpegs,
        jpeg_extensions=config.jpeg_extensions,
        mirror_paired_jpegs=config.mirror_paired_jpegs,
    )
    return [WorkItem(asset=asset) for asset in assets]


def run_pipeline(
    config: PipelineConfig,
    reporter: PipelineReporter | None = None,
) -> tuple[list[WorkItem], PipelineSummary, Path]:
    """Run discovery plus the local extraction and blur filtering phases."""
    reporter = reporter or NullReporter()
    items = build_work_items(config)
    total_discovered = len(items)
    run_dir = create_run_dir(config.runs_dir)
    write_run_config(run_dir, config)
    scene_count = assign_scene_groups(
        items,
        max_gap_seconds=config.scene_gap_seconds,
        max_sequence_gap=config.scene_max_sequence_gap,
    )
    items, processed_scene_count = limit_items_to_scenes(items, config.limit)
    summary = PipelineSummary(discovered=len(items))
    reporter.message(f"Run initialized: discovered {total_discovered} photo file(s)")
    if scene_count:
        reporter.message(f"Grouped discovery into {scene_count} provisional scene(s).")
    if config.limit is not None:
        reporter.message(
            "Scene limit active: "
            f"processing first {processed_scene_count} scene(s) spanning {len(items)} photo file(s)."
        )

    if not items:
        write_manifest(run_dir, items)
        return items, summary, run_dir

    extractor = build_default_extractor()
    extract_previews(items, extractor, config, run_dir, reporter)
    write_manifest(run_dir, items)
    score_previews(items, config, reporter)
    optimize_scene_candidates(items, config, reporter)
    rescue_scene_candidates(items, config, reporter)
    local_sidecars_written = persist_decisions(items, config)
    write_manifest(run_dir, items)
    if local_sidecars_written:
        reporter.message(
            f"Persisted local decisions and wrote {local_sidecars_written} metadata record(s)."
        )
    score_with_backend(items, config, run_dir, reporter)
    mirrored_jpegs = mirror_paired_jpeg_decisions(items)
    mirrored_metadata_written = persist_decisions(items, config)
    if mirrored_jpegs:
        reporter.message(f"Mirrored cull decisions to {mirrored_jpegs} paired JPEG file(s).")
    if mirrored_metadata_written:
        reporter.message(
            f"Persisted mirrored JPEG decisions and wrote {mirrored_metadata_written} metadata record(s)."
        )
    write_manifest(run_dir, items)
    summary = summarize_items(items)
    return items, summary, run_dir


def extract_previews(
    items: list[WorkItem],
    extractor: PreviewExtractor,
    config: PipelineConfig,
    run_dir: Path,
    reporter: PipelineReporter,
) -> None:
    preview_dir = run_dir / "previews"
    if config.cache_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    extract_items = [item for item in items if not item.asset.mirrors_paired_raw]
    reporter.start_phase("extract", "Extracting previews", len(extract_items))
    with ThreadPoolExecutor(max_workers=config.extract_workers) as executor:
        futures = {
            executor.submit(
                _extract_preview,
                item,
                extractor,
                preview_dir if config.cache_previews else None,
            ): item
            for item in extract_items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                item.preview = future.result()
            except PreviewExtractionError as exc:
                item.status = WorkStatus.FAILED
                item.error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive fallback
                item.status = WorkStatus.FAILED
                item.error = f"unexpected preview extraction error: {exc}"
            reporter.advance_phase("extract")
    reporter.complete_phase("extract", "Extracting previews")


def score_previews(
    items: list[WorkItem],
    config: PipelineConfig,
    reporter: PipelineReporter,
) -> None:
    ready_items = [item for item in items if item.preview is not None]
    if not ready_items:
        reporter.message("No previews were extracted successfully; skipping local quality analysis.")
        return

    include_portrait_metrics = config.genre in {"portrait", "event"}
    support_errors_reported = False
    reporter.start_phase("quality", "Scoring local quality", len(ready_items))
    with ProcessPoolExecutor(max_workers=config.score_workers) as executor:
        futures = {
            executor.submit(
                analyze_local_quality,
                item.preview.image_bytes,
                include_portrait_metrics,
                config.enable_learned_iqa,
                config.enable_brisque,
                config.enable_cpbd,
            ): item
            for item in ready_items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                metrics = future.result()
            except Exception as exc:
                item.status = WorkStatus.FAILED
                item.error = f"local quality scoring failed: {exc}"
            else:
                item.local_trace = build_local_decision_trace(
                    metrics,
                    min_blur_score=config.min_blur_score,
                    min_tenengrad_score=config.min_tenengrad_score,
                    min_musiq_score=config.min_musiq_score,
                    min_nima_score=config.min_nima_score,
                    max_brisque_score=config.max_brisque_score,
                    min_cpbd_score=config.min_cpbd_score,
                    use_brisque_for_reject=config.use_brisque_for_reject,
                    use_cpbd_for_reject=config.use_cpbd_for_reject,
                    local_reject_required_support_votes=config.local_reject_required_support_votes,
                )
                if should_reject_for_local_quality(
                    metrics,
                    min_blur_score=config.min_blur_score,
                    min_tenengrad_score=config.min_tenengrad_score,
                    min_musiq_score=config.min_musiq_score,
                    min_nima_score=config.min_nima_score,
                    max_brisque_score=config.max_brisque_score,
                    min_cpbd_score=config.min_cpbd_score,
                    use_brisque_for_reject=config.use_brisque_for_reject,
                    use_cpbd_for_reject=config.use_cpbd_for_reject,
                    local_reject_required_support_votes=config.local_reject_required_support_votes,
                ):
                    apply_local_rejection(item, metrics, trace=item.local_trace)
                else:
                    item.metrics = metrics
                    item.status = WorkStatus.READY_FOR_VISION
                    if item.local_trace is not None:
                        item.local_trace["final_local_action"] = "ready_for_vision"
                        item.local_trace["final_local_reason"] = item.local_trace["quality_explanation"]
                if not support_errors_reported:
                    errors = support_metric_import_errors()
                    if errors:
                        reporter.message(
                            "Optional support metric errors: "
                            + ", ".join(f"{name}={error}" for name, error in sorted(errors.items()))
                        )
                    support_errors_reported = True
            reporter.advance_phase("quality")
    reporter.complete_phase("quality", "Scoring local quality")


def summarize_items(items: list[WorkItem]) -> PipelineSummary:
    summary = PipelineSummary(discovered=len(items))
    for item in items:
        if item.asset.is_jpeg:
            summary.jpeg_discovered += 1
        else:
            summary.raw_discovered += 1
        if item.asset.mirrors_paired_raw and item.decision is not None:
            summary.mirrored_jpegs += 1
        if item.status == WorkStatus.REJECTED_LOCAL:
            summary.locally_rejected += 1
        elif item.status in {WorkStatus.READY_FOR_VISION, WorkStatus.SCORED}:
            summary.queued_for_vision += 1
        if item.status == WorkStatus.SCORED:
            summary.scored += 1
        if item.decision is not None:
            if item.decision.bucket == DecisionBucket.REJECT:
                summary.rejected_total += 1
            elif item.decision.bucket == DecisionBucket.REVIEW:
                summary.reviewed += 1
            elif item.decision.bucket == DecisionBucket.PICK:
                summary.picked += 1
        if item.sidecar_written:
            summary.sidecars_written += 1
        if item.lightroom_edit_written:
            summary.lightroom_edits_written += 1
        if item.status == WorkStatus.FAILED and not item.sidecar_written:
            summary.failed += 1
    return summary


def score_with_backend(
    items: list[WorkItem],
    config: PipelineConfig,
    run_dir: Path,
    reporter: PipelineReporter,
) -> None:
    ready_items = sort_candidates_for_vision(items)
    if not ready_items:
        reporter.message("No items survived local filtering; skipping vision scoring.")
        return

    try:
        backend = build_backend(config.backend)
    except ValueError as exc:
        for item in ready_items:
            item.status = WorkStatus.FAILED
            item.error = f"vision backend setup failed: {exc}"
        return

    cohorts = build_scene_cohorts(items, config.batch_size)
    total_cohorts = len(cohorts)
    reporter.start_phase("vision", "Scoring with vision backend", len(ready_items))
    for index, cohort in enumerate(cohorts, start=1):
        previews = [item.preview for item in cohort if item.preview is not None]
        scene_id = cohort[0].scene_id or "ungrouped"
        if len(previews) != len(cohort):
            for item in cohort:
                item.status = WorkStatus.FAILED
                item.error = "missing preview before vision scoring"
            write_manifest(run_dir, items)
            reporter.advance_phase("vision", advance=len(cohort))
            reporter.message(
                "Persisted failed vision cohort "
                f"{index}/{total_cohorts} for {scene_id} with {len(cohort)} item(s)."
            )
            continue

        try:
            decisions = backend.score_batch(config.prompt, previews)
        except (VisionBackendError, ValueError) as exc:
            for item in cohort:
                item.status = WorkStatus.FAILED
                item.error = f"vision scoring failed: {exc}"
            write_manifest(run_dir, items)
            reporter.advance_phase("vision", advance=len(cohort))
            reporter.message(
                "Persisted failed vision cohort "
                f"{index}/{total_cohorts} for {scene_id} with {len(cohort)} item(s)."
            )
            continue

        if len(decisions) != len(cohort):
            for item in cohort:
                item.status = WorkStatus.FAILED
                item.error = "vision backend returned an unexpected number of results"
            write_manifest(run_dir, items)
            reporter.advance_phase("vision", advance=len(cohort))
            reporter.message(
                "Persisted failed vision cohort "
                f"{index}/{total_cohorts} for {scene_id} with {len(cohort)} item(s)."
            )
            continue

        for item, decision in zip(cohort, decisions):
            apply_vision_decision(item, decision)
        sidecars_written = persist_decisions(cohort, config)
        write_manifest(run_dir, items)
        reporter.advance_phase("vision", advance=len(cohort))
        reporter.message(
            f"Persisted vision cohort {index}/{total_cohorts} for {scene_id}: "
            f"{len(cohort)} item(s), {sidecars_written} metadata record(s) written."
        )
    reporter.complete_phase("vision", "Scoring with vision backend")


def _extract_preview(
    item: WorkItem,
    extractor: PreviewExtractor,
    preview_dir: Path | None,
) -> PreviewImage:
    if item.asset.is_jpeg:
        image_bytes = item.asset.raw_path.read_bytes()
    else:
        image_bytes = extractor.extract_preview_bytes(item.asset.raw_path)
    cache_path = None
    if preview_dir is not None:
        cache_path = preview_dir / f"{item.asset.raw_path.stem}.jpg"
        cache_path.write_bytes(image_bytes)

    return PreviewImage(
        asset=item.asset,
        image_bytes=image_bytes,
        cache_path=cache_path,
    )


def apply_local_rejection(
    item: WorkItem,
    metrics: LocalQualityMetrics,
    reason: str | None = None,
    trace: dict[str, object] | None = None,
) -> None:
    item.metrics = metrics
    item.status = WorkStatus.REJECTED_LOCAL
    item.local_trace = trace or item.local_trace
    if item.local_trace is not None:
        item.local_trace["final_local_action"] = "reject"
        item.local_trace["final_local_reason"] = reason or item.local_trace.get("quality_explanation")
    item.decision = FinalDecision(
        filename=item.asset.filename,
        rating=-1,
        label=ColorLabel.RED,
        bucket=DecisionBucket.REJECT,
        source=DecisionSource.LOCAL,
        summary=reason
        or (
            "Rejected locally due to sharpness thresholds "
            f"(laplacian={_format_metric(metrics.blur_score)}, "
            f"tenengrad={_format_metric(metrics.tenengrad_score)}, "
            f"musiq={_format_metric(metrics.musiq_score)}, "
            f"nima={_format_metric(metrics.nima_score)})"
        ),
    )


def apply_vision_decision(item: WorkItem, decision: FinalDecision) -> None:
    if decision.bucket == DecisionBucket.REJECT:
        decision.rating = -1
        decision.label = ColorLabel.YELLOW
    elif decision.bucket == DecisionBucket.REVIEW:
        decision.rating = 0
        decision.label = None
    elif decision.bucket == DecisionBucket.PICK:
        if decision.rating < 1:
            decision.rating = 4
        if decision.label is None:
            decision.label = ColorLabel.GREEN
    item.decision = decision
    item.vision_trace = {
        "bucket": decision.bucket.value,
        "label": decision.label.value if decision.label else None,
        "rating": decision.rating,
        "source": decision.source.value,
        "summary": decision.summary,
    }
    item.status = WorkStatus.SCORED


def persist_decisions(items: list[WorkItem], config: PipelineConfig) -> int:
    if config.dry_run:
        return 0

    written = 0
    for item in items:
        if item.decision is None or item.sidecar_written:
            continue
        try:
            write_photo_metadata(
                item.asset,
                item.decision,
                apply_lightroom_edit=config.lightroom_auto_edit,
                lightroom_edit_scope=config.lightroom_edit_scope,
            )
        except Exception as exc:
            item.status = WorkStatus.FAILED
            item.error = f"sidecar write failed: {exc}"
            continue
        item.sidecar_written = True
        item.lightroom_edit_written = (not item.asset.is_jpeg) and config.lightroom_auto_edit and (
            config.lightroom_edit_scope == LightroomEditScope.ALL or item.decision.keep
        )
        written += 1
    return written


def mirror_paired_jpeg_decisions(items: list[WorkItem]) -> int:
    items_by_path = {item.asset.raw_path: item for item in items}
    mirrored = 0
    for item in items:
        if not item.asset.mirrors_paired_raw or item.decision is not None:
            continue
        paired_raw_path = item.asset.paired_raw_path
        if paired_raw_path is None:
            continue
        raw_item = items_by_path.get(paired_raw_path)
        if raw_item is None or raw_item.decision is None:
            continue
        item.decision = FinalDecision(
            filename=item.filename,
            rating=raw_item.decision.rating,
            label=raw_item.decision.label,
            bucket=raw_item.decision.bucket,
            source=raw_item.decision.source,
            summary=f"Mirrored from paired RAW {raw_item.filename}.",
        )
        item.status = raw_item.status
        item.local_trace = {
            "mirrored_from_raw": str(paired_raw_path),
            "raw_decision": raw_item.local_trace,
        }
        item.vision_trace = {
            "mirrored_from_raw": str(paired_raw_path),
            "raw_decision": raw_item.vision_trace,
        }
        mirrored += 1
    return mirrored


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def optimize_scene_candidates(
    items: list[WorkItem],
    config: PipelineConfig,
    reporter: PipelineReporter,
) -> None:
    ready_items = [item for item in items if item.status == WorkStatus.READY_FOR_VISION]
    if not ready_items:
        return

    duplicates_rejected, rerank_rejected = suppress_duplicates_and_rerank(
        items,
        duplicate_hamming_threshold=config.duplicate_hamming_threshold,
        max_scene_candidates=config.max_scene_candidates,
        apply_local_rejection=apply_local_rejection,
    )
    if duplicates_rejected or rerank_rejected:
        reporter.message(
            "Optimized scene candidates before vision scoring: "
            f"{duplicates_rejected} near-duplicate(s) rejected, "
            f"{rerank_rejected} scene-reranked rejection(s)."
        )


def rescue_scene_candidates(
    items: list[WorkItem],
    config: PipelineConfig,
    reporter: PipelineReporter,
) -> None:
    rescued_items = rescue_scene_review_candidates(
        items,
        min_blur_score=config.min_blur_score,
        min_tenengrad_score=config.min_tenengrad_score,
    )
    if not rescued_items:
        return

    rescued_filenames = ", ".join(item.filename for item in rescued_items[:5])
    if len(rescued_items) > 5:
        rescued_filenames += ", ..."
    reporter.message(
        "Scene safeguard restored "
        f"{len(rescued_items)} fallback candidate(s) for review: {rescued_filenames}"
    )


def limit_items_to_scenes(
    items: list[WorkItem],
    scene_limit: int | None,
) -> tuple[list[WorkItem], int]:
    if scene_limit is None:
        return items, len({item.scene_id for item in items if item.scene_id is not None})

    ordered_scene_ids = sorted({item.scene_id for item in items if item.scene_id is not None})
    selected_scene_ids = set(ordered_scene_ids[:scene_limit])
    if not selected_scene_ids:
        return [], 0

    limited_items = [item for item in items if item.scene_id in selected_scene_ids]
    return limited_items, len(selected_scene_ids)
