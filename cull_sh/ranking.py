from __future__ import annotations

from collections import defaultdict
from typing import Callable

from cull_sh.models import LocalQualityMetrics
from cull_sh.models import WorkItem
from cull_sh.models import WorkStatus


ApplyLocalRejection = Callable[[WorkItem, LocalQualityMetrics, str], None]


def suppress_duplicates_and_rerank(
    items: list[WorkItem],
    duplicate_hamming_threshold: int,
    max_scene_candidates: int | None,
    apply_local_rejection: ApplyLocalRejection,
) -> tuple[int, int]:
    candidates = [item for item in items if item.status == WorkStatus.READY_FOR_VISION and item.metrics]
    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in candidates:
        grouped[item.scene_id or "ungrouped"].append(item)

    duplicates_rejected = 0
    rerank_rejected = 0
    for scene_id, scene_items in grouped.items():
        ranked = sorted(scene_items, key=_scene_rank_key, reverse=True)
        retained: list[WorkItem] = []

        for item in ranked:
            duplicate_anchor = _find_duplicate_anchor(
                item,
                retained,
                duplicate_hamming_threshold=duplicate_hamming_threshold,
            )
            if duplicate_anchor is not None:
                apply_local_rejection(
                    item,
                    item.metrics,
                    (
                        "Rejected locally as a near-duplicate of "
                        f"{duplicate_anchor.filename} in {scene_id}"
                    ),
                    trace={
                        **(item.local_trace or {}),
                        "duplicate": {
                            "rejected": True,
                            "anchor_filename": duplicate_anchor.filename,
                            "scene_id": scene_id,
                        },
                    },
                )
                duplicates_rejected += 1
                continue
            retained.append(item)

        if max_scene_candidates is None:
            continue

        for rank_index, item in enumerate(retained[max_scene_candidates:], start=max_scene_candidates + 1):
            apply_local_rejection(
                item,
                item.metrics,
                f"Rejected locally after scene reranking in {scene_id} (rank {rank_index})",
                trace={
                    **(item.local_trace or {}),
                    "scene_rerank": {
                        "rejected": True,
                        "scene_id": scene_id,
                        "rank_index": rank_index,
                    },
                },
            )
            rerank_rejected += 1

    return duplicates_rejected, rerank_rejected


def sort_candidates_for_vision(items: list[WorkItem]) -> list[WorkItem]:
    candidates = [item for item in items if item.status == WorkStatus.READY_FOR_VISION]
    return sorted(
        candidates,
        key=lambda item: (
            item.scene_id or "ungrouped",
            -(item.metrics.local_rank_score if item.metrics and item.metrics.local_rank_score else 0.0),
            item.scene_index or 0,
            item.filename,
        ),
    )


def build_scene_cohorts(
    items: list[WorkItem],
    cohort_size: int,
) -> list[list[WorkItem]]:
    ordered = sort_candidates_for_vision(items)
    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in ordered:
        grouped[item.scene_id or "ungrouped"].append(item)

    cohorts: list[list[WorkItem]] = []
    for scene_id in sorted(grouped):
        scene_items = grouped[scene_id]
        for index in range(0, len(scene_items), cohort_size):
            cohorts.append(scene_items[index : index + cohort_size])
    return cohorts


def rescue_scene_review_candidates(
    items: list[WorkItem],
    min_blur_score: float,
    min_tenengrad_score: float,
) -> list[WorkItem]:
    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[item.scene_id or "ungrouped"].append(item)

    rescued: list[WorkItem] = []
    for scene_items in grouped.values():
        if any(item.status == WorkStatus.READY_FOR_VISION for item in scene_items):
            continue

        rejected_items = [
            item
            for item in scene_items
            if item.status == WorkStatus.REJECTED_LOCAL and item.metrics is not None
        ]
        if not rejected_items:
            continue

        candidate = max(rejected_items, key=_scene_rank_key)
        if not _is_viable_scene_fallback(
            candidate.metrics,
            min_blur_score=min_blur_score,
            min_tenengrad_score=min_tenengrad_score,
        ):
            continue

        candidate.status = WorkStatus.READY_FOR_VISION
        candidate.decision = None
        candidate.sidecar_written = False
        candidate.error = None
        if candidate.local_trace is None:
            candidate.local_trace = {}
        candidate.local_trace["scene_safeguard"] = {
            "rescued": True,
            "scene_id": candidate.scene_id,
            "reason": "Best remaining candidate in an otherwise locally rejected scene.",
        }
        candidate.local_trace["final_local_action"] = "ready_for_vision"
        candidate.local_trace["final_local_reason"] = (
            "Scene safeguard restored this candidate for human/model review."
        )
        rescued.append(candidate)

    return rescued


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _find_duplicate_anchor(
    candidate: WorkItem,
    retained: list[WorkItem],
    duplicate_hamming_threshold: int,
) -> WorkItem | None:
    candidate_hash = candidate.metrics.perceptual_hash if candidate.metrics else None
    if candidate_hash is None:
        return None

    for item in retained:
        retained_hash = item.metrics.perceptual_hash if item.metrics else None
        if retained_hash is None:
            continue
        if hamming_distance(candidate_hash, retained_hash) <= duplicate_hamming_threshold:
            return item
    return None


def _scene_rank_key(item: WorkItem) -> tuple[float, float, float, float, str]:
    metrics = item.metrics or LocalQualityMetrics()
    return (
        metrics.local_rank_score or 0.0,
        metrics.blur_score or 0.0,
        metrics.tenengrad_score or 0.0,
        metrics.contrast_stddev or 0.0,
        item.filename,
    )


def _is_viable_scene_fallback(
    metrics: LocalQualityMetrics,
    min_blur_score: float,
    min_tenengrad_score: float,
) -> bool:
    blur_score = metrics.blur_score or 0.0
    tenengrad_score = metrics.tenengrad_score or 0.0
    return (
        blur_score >= (min_blur_score * 0.75)
        or tenengrad_score >= (min_tenengrad_score * 0.75)
    )
