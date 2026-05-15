from __future__ import annotations

from pathlib import Path
import re

from cull_sh.models import WorkItem


TRAILING_DIGITS = re.compile(r"(\d+)$")


def assign_scene_groups(
    items: list[WorkItem],
    max_gap_seconds: float = 4.0,
    max_sequence_gap: int = 3,
) -> int:
    if not items:
        return 0

    ordered = sorted(items, key=_scene_sort_key)
    scene_number = 1
    scene_size = 0
    previous: WorkItem | None = None

    for item in ordered:
        if previous is None or not _same_scene(
            previous,
            item,
            max_gap_seconds=max_gap_seconds,
            max_sequence_gap=max_sequence_gap,
        ):
            scene_number += 0 if previous is None else 1
            scene_size = 0

        scene_size += 1
        item.scene_id = f"scene-{scene_number:04d}"
        item.scene_index = scene_size
        previous = item

    return scene_number


def _scene_sort_key(item: WorkItem) -> tuple[str, float, int]:
    raw_path = item.asset.raw_path
    try:
        modified_time = raw_path.stat().st_mtime
    except OSError:
        modified_time = 0.0
    return (str(raw_path.parent), modified_time, _numeric_suffix(raw_path))


def _same_scene(
    previous: WorkItem,
    current: WorkItem,
    max_gap_seconds: float,
    max_sequence_gap: int,
) -> bool:
    if previous.asset.raw_path.parent != current.asset.raw_path.parent:
        return False

    time_gap = _modified_time_gap(previous.asset.raw_path, current.asset.raw_path)
    if time_gap is not None and time_gap > max_gap_seconds:
        return False

    previous_sequence = _numeric_suffix(previous.asset.raw_path)
    current_sequence = _numeric_suffix(current.asset.raw_path)
    if previous_sequence and current_sequence:
        return (current_sequence - previous_sequence) <= max_sequence_gap

    return True


def _modified_time_gap(left: Path, right: Path) -> float | None:
    try:
        return abs(right.stat().st_mtime - left.stat().st_mtime)
    except OSError:
        return None


def _numeric_suffix(path: Path) -> int:
    match = TRAILING_DIGITS.search(path.stem)
    if match is None:
        return 0
    return int(match.group(1))
