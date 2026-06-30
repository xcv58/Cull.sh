from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ColorLabel(str, Enum):
    RED = "Red"
    YELLOW = "Yellow"
    GREEN = "Green"
    BLUE = "Blue"
    PURPLE = "Purple"


class DecisionSource(str, Enum):
    LOCAL = "local"
    VISION = "vision"


class DecisionBucket(str, Enum):
    REJECT = "reject"
    REVIEW = "review"
    PICK = "pick"


class WorkStatus(str, Enum):
    PENDING = "pending"
    REJECTED_LOCAL = "rejected_local"
    READY_FOR_VISION = "ready_for_vision"
    SCORED = "scored"
    FAILED = "failed"


class LightroomEditScope(str, Enum):
    ALL = "all"
    KEPT = "kept"


class AssetKind(str, Enum):
    RAW = "raw"
    JPEG = "jpeg"


@dataclass(slots=True)
class RawAsset:
    raw_path: Path
    xmp_path: Path
    kind: AssetKind = AssetKind.RAW
    paired_raw_path: Path | None = None

    @property
    def filename(self) -> str:
        return self.raw_path.name

    @property
    def is_jpeg(self) -> bool:
        return self.kind == AssetKind.JPEG

    @property
    def mirrors_paired_raw(self) -> bool:
        return self.is_jpeg and self.paired_raw_path is not None


@dataclass(slots=True)
class PreviewImage:
    asset: RawAsset
    image_bytes: bytes
    mime_type: str = "image/jpeg"
    cache_path: Path | None = None


@dataclass(slots=True)
class LocalQualityMetrics:
    blur_score: float | None = None
    tenengrad_score: float | None = None
    musiq_score: float | None = None
    nima_score: float | None = None
    brisque_score: float | None = None
    cpbd_score: float | None = None
    brightness_mean: float | None = None
    contrast_stddev: float | None = None
    face_count: int | None = None
    eye_count: int | None = None
    local_rank_score: float | None = None
    perceptual_hash: str | None = None


@dataclass(slots=True)
class FinalDecision:
    filename: str
    rating: int
    label: ColorLabel | None
    bucket: DecisionBucket
    source: DecisionSource
    summary: str = ""

    @property
    def keep(self) -> bool:
        return self.bucket != DecisionBucket.REJECT

    @property
    def picked(self) -> bool:
        return self.bucket == DecisionBucket.PICK

    @property
    def reviewed(self) -> bool:
        return self.bucket == DecisionBucket.REVIEW


@dataclass(slots=True)
class EditSuggestion:
    """A set of gentle global Lightroom develop adjustments for one photo."""

    filename: str
    asset_id: str = ""
    exposure: float = 0.0  # EV, roughly -5..+5
    contrast: int = 0  # -100..100
    highlights: int = 0  # -100..100
    shadows: int = 0  # -100..100
    vibrance: int = 0  # -100..100
    has_crop: bool = False
    crop_left: float = 0.0  # normalized 0..1
    crop_top: float = 0.0  # normalized 0..1
    crop_right: float = 1.0  # normalized 0..1
    crop_bottom: float = 1.0  # normalized 0..1
    crop_angle: float = 0.0  # degrees
    summary: str = ""

    @property
    def is_noop(self) -> bool:
        return (
            self.exposure == 0.0
            and self.contrast == 0
            and self.highlights == 0
            and self.shadows == 0
            and self.vibrance == 0
            and not self.has_crop
        )


@dataclass(slots=True)
class WorkItem:
    asset: RawAsset
    status: WorkStatus = WorkStatus.PENDING
    scene_id: str | None = None
    scene_index: int | None = None
    metrics: LocalQualityMetrics | None = None
    local_trace: dict[str, object] | None = None
    vision_trace: dict[str, object] | None = None
    preview: PreviewImage | None = None
    decision: FinalDecision | None = None
    sidecar_written: bool = False
    lightroom_edit_written: bool = False
    error: str | None = None

    @property
    def filename(self) -> str:
        return self.asset.filename


@dataclass(slots=True)
class PipelineSummary:
    discovered: int = 0
    raw_discovered: int = 0
    jpeg_discovered: int = 0
    mirrored_jpegs: int = 0
    locally_rejected: int = 0
    queued_for_vision: int = 0
    scored: int = 0
    reviewed: int = 0
    picked: int = 0
    rejected_total: int = 0
    sidecars_written: int = 0
    lightroom_edits_written: int = 0
    failed: int = 0
