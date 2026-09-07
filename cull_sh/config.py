from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cull_sh.models import LightroomEditScope


DEFAULT_EXTENSIONS = (
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".orf",
    ".raf",
    ".rw2",
)

JPEG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
)

DEFAULT_PRODUCTION_MODEL = "orcarouter/Qwen3.8-27B-Uncensored"


@dataclass(slots=True)
class BackendConfig:
    provider: str = "ollama"
    model: str = DEFAULT_PRODUCTION_MODEL
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 300.0
    max_attempts: int = 1
    think: bool | str | None = True
    fail_fast: bool = True
    max_output_tokens: int = 2048
    context_tokens: int | None = None


@dataclass(slots=True)
class PipelineConfig:
    path: Path
    prompt: str
    genre: str = "auto"
    prefer: str | None = None
    prompt_source: str = "custom"
    backend: BackendConfig = field(default_factory=BackendConfig)
    runs_dir: Path = Path("runs")
    limit: int | None = None
    batch_size: int = 4
    extract_workers: int = 6
    score_workers: int = 6
    scene_gap_seconds: float = 4.0
    scene_max_sequence_gap: int = 3
    min_blur_score: float = 110.0
    min_tenengrad_score: float = 45.0
    enable_learned_iqa: bool = True
    enable_topiq_ranking: bool = True
    topiq_rank_weight: float = 0.25
    enable_topiq_shadow: bool = True
    topiq_shadow_workers: int = 2
    topiq_shadow_low_percentile: float = 0.2
    topiq_shadow_high_percentile: float = 0.8
    topiq_shadow_max_items: int = 80
    min_musiq_score: float = 50.0
    min_nima_score: float = 4.6
    enable_brisque: bool = True
    enable_cpbd: bool = True
    max_brisque_score: float = 55.0
    min_cpbd_score: float = 0.3
    use_brisque_for_reject: bool = False
    use_cpbd_for_reject: bool = False
    local_reject_required_support_votes: int = 2
    duplicate_hamming_threshold: int = 6
    max_scene_candidates: int | None = None
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    jpeg_extensions: tuple[str, ...] = JPEG_EXTENSIONS
    include_jpegs: bool = True
    mirror_paired_jpegs: bool = True
    dry_run: bool = True
    cache_previews: bool = False
    lightroom_auto_edit: bool = False
    lightroom_edit_scope: LightroomEditScope = LightroomEditScope.ALL
