from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import typer
from rich.console import Console
from rich.table import Table

from cull_sh.backends import build_backend
from cull_sh.backends import VisionBackendError
from cull_sh.benchmark import run_internal_benchmark
from cull_sh.config import BackendConfig, DEFAULT_EXTENSIONS, JPEG_EXTENSIONS, PipelineConfig
from cull_sh.extractors import PreviewExtractionError
from cull_sh.extractors import build_default_extractor
from cull_sh.edit_benchmark import EditDatasetSpec
from cull_sh.edit_benchmark import EditModelSpec
from cull_sh.edit_benchmark import run_edit_benchmark
from cull_sh.edit_benchmark import stage_lightroom_blind_review
from cull_sh.edit_benchmark import write_blind_review_page
from cull_sh.lightroom_ui import build_adaptive_color_stage
from cull_sh.lightroom_ui import build_jpeg_auto_stage
from cull_sh.lightroom_ui import write_adaptive_color_handoff
from cull_sh.lightroom_ui import write_jpeg_auto_handoff
from cull_sh.manifests import create_run_dir
from cull_sh.manifests import decision_from_manifest_record
from cull_sh.manifests import find_latest_run_dir
from cull_sh.manifests import load_manifest_records
from cull_sh.models import AssetKind
from cull_sh.models import EditSuggestion
from cull_sh.models import LightroomEditScope
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.pipeline import run_pipeline
from cull_sh.prompting import GenrePreset
from cull_sh.prompting import parse_genre
from cull_sh.prompting import resolve_prompt
from cull_sh.prompting import supported_genre_labels
from cull_sh.quality import learned_iqa_available
from cull_sh.quality import learned_iqa_import_error
from cull_sh.quality import support_metric_import_errors
from cull_sh.rapidraw import export_rapidraw_stage
from cull_sh.rapidraw import RapidRawError
from cull_sh.rapidraw import rapidraw_install_info
from cull_sh.rapidraw import render_rapidraw_review
from cull_sh.rapidraw import stage_rapidraw_develop
from cull_sh.reporting import RichPipelineReporter
from cull_sh.scanner import discover_raw_assets
from cull_sh.xmp import sidecar_is_rejected
from cull_sh.xmp import write_develop_sidecar
from cull_sh.xmp import write_photo_metadata
from cull_sh.xmp import write_lightroom_edit_sidecar
from cull_sh.vlm_benchmark import VLMModelSpec
from cull_sh.vlm_benchmark import run_vlm_benchmark


DEFAULT_EDIT_PROMPT = (
    "Suggest natural, balanced global edits that improve each photo while keeping "
    "a realistic look."
)
DEFAULT_EDIT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored"
DEFAULT_RAPIDRAW_BINARY = (
    Path.home() / "Applications/RapidRAW.app/Contents/MacOS/RapidRAW"
)


app = typer.Typer(
    help="Cull.sh: natural-language photo culling for RAW and JPEG files."
)
console = Console()


@app.command("benchmark")
def benchmark(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    include_subfolders: bool = typer.Option(
        False,
        "--include-subfolders/--top-level-only",
        help="Include nested folders; the default matches Lightroom's current-folder view.",
    ),
    facet_db: Path | None = typer.Option(
        None,
        "--facet-db",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional read-only Facet SQLite database containing scores for this folder.",
    ),
    shadow_run: Path | None = typer.Option(
        None,
        "--shadow-run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help=(
            "Frozen Cull.sh cull run to compare with final human XMP; its cached "
            "signals are reused instead of rescored."
        ),
    ),
    current: bool = typer.Option(
        True,
        "--current/--no-current",
        help="Compute and compare the current Cull.sh MUSIQ/NIMA/local signals.",
    ),
    current_workers: int = typer.Option(
        2,
        min=1,
        help="Worker processes for current Cull.sh scoring; each worker loads the IQA models.",
    ),
    current_cache: Path | None = typer.Option(
        None,
        help="Optional resumable JSONL cache for current Cull.sh signals.",
    ),
    runs_root: Path = typer.Option(
        Path("runs"),
        file_okay=False,
        dir_okay=True,
        help="Root directory for benchmark artifacts.",
    ),
) -> None:
    """Compare model signals against human Lightroom pick/reject decisions."""
    if facet_db is None and shadow_run is None and not current:
        raise typer.BadParameter(
            "enable --current, provide --facet-db, or provide --shadow-run"
        )

    run_dir = create_run_dir(runs_root)
    console.print("Benchmark is read-only for photos and XMP sidecars.")
    console.print(f"Human ground truth: {path}")
    console.print(f"Run artifacts: {run_dir}")
    payload = run_internal_benchmark(
        path,
        facet_db,
        run_dir,
        include_subfolders=include_subfolders,
        shadow_run=shadow_run,
        compute_current=current and shadow_run is None,
        current_workers=current_workers,
        current_cache=current_cache,
        progress=console.print,
    )

    truth = payload["ground_truth"]
    signals = payload["signals"]
    assert isinstance(truth, dict)
    assert isinstance(signals, list)
    counts = Table(title="Human Ground Truth")
    counts.add_column("Class")
    counts.add_column("Count", justify="right")
    counts.add_row("Pick", str(truth["picks"]))
    counts.add_row("Reject", str(truth["rejects"]))
    counts.add_row("Neutral (excluded)", str(truth["neutral"]))
    console.print(counts)

    if signals:
        comparison = Table(title="Internal Benchmark")
        comparison.add_column("Signal")
        comparison.add_column("AUC", justify="right")
        comparison.add_column("Burst pairs", justify="right")
        comparison.add_column("Top-1", justify="right")
        comparison.add_column("Top-3", justify="right")
        comparison.add_column("Zero-FR coverage", justify="right")
        for signal in signals:
            assert isinstance(signal, dict)
            comparison.add_row(
                str(signal["display_name"]),
                f"{float(signal['auc']):.3f}",
                _format_cli_percent(signal["pairwise_accuracy"]),
                _format_cli_percent(signal["top1_recall"]),
                _format_cli_percent(signal["top3_recall"]),
                f"{float(signal['zero_false_reject_coverage']):.1%}",
            )
        console.print(comparison)
    else:
        console.print(
            "No cached signal covered both a human pick and reject; "
            "ranking metrics were skipped."
        )
    comparisons = payload.get("comparisons", [])
    if isinstance(comparisons, list) and comparisons:
        direct = comparisons[0]
        assert isinstance(direct, dict)
        console.print(
            f"Direct comparison: {direct['left_display_name']} minus "
            f"{direct['right_display_name']} AUC={float(direct['auc_delta']):+.3f}, "
            f"paired bootstrap 95% interval "
            f"{float(direct['delta_ci95_low']):+.3f} to "
            f"{float(direct['delta_ci95_high']):+.3f}."
        )
    hard_gates = payload.get("hard_gates", [])
    if isinstance(hard_gates, list) and hard_gates:
        gate_table = Table(title="Current Auto-Reject Gate")
        gate_table.add_column("Gate")
        gate_table.add_column("Coverage", justify="right")
        gate_table.add_column("Reject precision", justify="right")
        gate_table.add_column("False-reject rate", justify="right")
        gate_table.add_column("False rejects", justify="right")
        gate_table.add_column("Reject recall", justify="right")
        for gate in hard_gates:
            assert isinstance(gate, dict)
            gate_table.add_row(
                str(gate["display_name"]),
                f"{float(gate['coverage']):.1%}",
                f"{float(gate['reject_precision']):.1%}",
                f"{float(gate['false_reject_rate']):.1%}",
                str(gate["false_rejects"]),
                f"{float(gate['reject_recall']):.1%}",
            )
        console.print(gate_table)
    console.print(f"Report: {run_dir / 'benchmark.md'}")
    console.print(f"Per-photo scores: {run_dir / 'photo-scores.csv'}")


@app.command("model-benchmark")
def model_benchmark(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    prompt: str | None = typer.Option(
        None,
        help="Optional frozen culling prompt; overrides the genre preset.",
    ),
    genre: GenrePreset = typer.Option(
        GenrePreset.AUTO,
        case_sensitive=False,
        help="Frozen genre prompt used for both models.",
    ),
    gemma_model: str = typer.Option("gemma4:12b"),
    qwen_model: str = typer.Option("orcarouter/Qwen3.8-27B-Uncensored"),
    qwen_thinking: bool = typer.Option(
        False,
        "--qwen-thinking/--qwen-no-thinking",
        help="Enable Qwen thinking mode; non-thinking is the deterministic canary default.",
    ),
    qwen_only: bool = typer.Option(
        False,
        "--qwen-only/--compare-models",
        help="Run only Qwen while tuning on the calibration folder.",
    ),
    full: bool = typer.Option(
        False,
        "--full/--canary",
        help="Run every cohort instead of a small stratified canary.",
    ),
    canary_cohorts: int = typer.Option(8, min=1),
    cohort_id: list[str] | None = typer.Option(
        None,
        "--cohort-id",
        help="Run only an exact cohort id; repeat this option for multiple cohorts.",
    ),
    batch_size: int = typer.Option(4, min=1),
    include_subfolders: bool = typer.Option(
        False,
        "--include-subfolders/--top-level-only",
    ),
    timeout_seconds: float = typer.Option(600.0, min=1.0),
    max_attempts: int = typer.Option(3, min=1),
    frozen_baseline_run: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Existing frozen Cull.sh run used as the production baseline.",
    ),
    resume_run: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    runs_root: Path = typer.Option(
        Path("runs/vlm-benchmarks"),
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Compare Gemma and Qwen culling decisions without writing photo metadata."""
    prompt_selection = resolve_prompt(prompt, genre)
    qwen_spec = VLMModelSpec(
        label="qwen3-8-27b-nothink" if not qwen_thinking else "qwen3-8-27b-thinking",
        model=qwen_model,
        think=qwen_thinking,
    )
    specs = (
        [qwen_spec]
        if qwen_only
        else [VLMModelSpec(label="gemma4-12b", model=gemma_model), qwen_spec]
    )
    console.print("Vision-model benchmark is read-only for photos and XMP sidecars.")
    console.print(f"Ground truth: {path}")
    if cohort_id:
        console.print(f"Mode: targeted ({len(cohort_id)} cohorts)")
    else:
        console.print(f"Mode: {'full' if full else f'canary ({canary_cohorts} cohorts)'}")
    payload, run_dir = run_vlm_benchmark(
        path,
        prompt_selection.prompt,
        specs,
        runs_root,
        include_subfolders=include_subfolders,
        batch_size=batch_size,
        canary_cohorts=None if full else canary_cohorts,
        cohort_ids=cohort_id,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        frozen_baseline_run=frozen_baseline_run,
        resume_run=resume_run,
        progress=console.print,
    )

    truth = payload["ground_truth"]
    models = payload["models"]
    assert isinstance(truth, dict)
    assert isinstance(models, list)
    console.print(
        f"Evaluated {truth['photos']} photos: {truth['picks']} pick, "
        f"{truth['rejects']} reject, {truth['neutral']} neutral."
    )
    table = Table(title="Vision-model benchmark")
    table.add_column("Model")
    table.add_column("Covered", justify="right")
    table.add_column("First-pass", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Top-1", justify="right")
    table.add_column("False rejects", justify="right")
    table.add_column("Photos/min", justify="right")
    for result in models:
        assert isinstance(result, dict)
        signal = result.get("signal")
        signal = signal if isinstance(signal, dict) else {}
        table.add_row(
            str(result["label"]),
            str(result["photos_covered"]),
            _format_cli_percent(result["first_pass_success_rate"]),
            f"{float(signal['auc']):.3f}" if signal.get("auc") is not None else "-",
            _format_cli_percent(signal.get("top1_recall")),
            f"{result['false_rejects']} ({_format_cli_percent(result['false_reject_rate'])})",
            (
                f"{float(result['photos_per_minute']):.2f}"
                if result["photos_per_minute"] is not None
                else "-"
            ),
        )
    console.print(table)
    console.print(f"Run artifacts: {run_dir}")
    console.print(f"Report: {run_dir / 'vlm-benchmark.md'}")


@app.command("edit-model-benchmark")
def edit_model_benchmark(
    xiuling_path: Path = typer.Option(
        Path("~/Pictures/Photos/Xiuling").expanduser(),
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    longwood_path: Path = typer.Option(
        Path("/Volumes/Sandisk 4T/RAW Photos/2026-08-01 Longwood Gardens DONE EXPORTED"),
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    per_dataset: int = typer.Option(24, min=1),
    prompt: str = typer.Option(DEFAULT_EDIT_PROMPT),
    gemma_model: str = typer.Option("gemma4:12b"),
    qwen_model: str = typer.Option("orcarouter/Qwen3.8-27B-Uncensored"),
    timeout_seconds: float = typer.Option(600.0, min=1.0),
    max_attempts: int = typer.Option(2, min=1),
    seed: int = typer.Option(20260820),
    resume_run: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    runs_root: Path = typer.Option(
        Path("runs/edit-benchmarks"),
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Compare Gemma and Qwen edit suggestions on a locked human-pick cohort."""
    datasets = [
        EditDatasetSpec("Xiuling", xiuling_path, per_dataset),
        EditDatasetSpec("Longwood", longwood_path, per_dataset),
    ]
    models = [
        EditModelSpec("gemma4-12b", gemma_model),
        EditModelSpec("qwen3-8-27b-nothink", qwen_model, think=False),
    ]
    console.print(
        "Edit benchmark is read-only for original photos and XMP sidecars; "
        "Qwen thinking is disabled and both models use temperature 0."
    )
    payload, run_dir = run_edit_benchmark(
        datasets,
        models,
        prompt,
        runs_root,
        seed=seed,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        resume_run=resume_run,
        progress=console.print,
    )
    table = Table(title="Edit-suggestion benchmark diagnostics")
    table.add_column("Model")
    table.add_column("Valid", justify="right")
    table.add_column("Mean sec/photo", justify="right")
    table.add_column("Unique vectors", justify="right")
    table.add_column("Duplicate rate", justify="right")
    for result in payload["models"]:
        duplicate_rate = result["duplicate_vector_rate"]
        table.add_row(
            str(result["label"]),
            f"{result['covered']}/{payload['cohort_photos']}",
            f"{float(result['mean_seconds']):.2f}" if result["mean_seconds"] is not None else "-",
            str(result["unique_vectors"]),
            f"{float(duplicate_rate):.1%}" if duplicate_rate is not None else "-",
        )
    console.print(table)
    console.print(f"Run artifacts: {run_dir}")
    console.print(f"Report: {run_dir / 'edit-benchmark.md'}")
    console.print(f"Blind review sheet: {run_dir / 'blind-review.csv'}")


@app.command("stage-edit-review")
def stage_edit_review(
    run_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Make disposable blinded RAW/XMP variants for a Lightroom A/B render."""
    console.print("Original photos and sidecars will not be modified.")
    render_input = stage_lightroom_blind_review(run_dir, progress=console.print)
    console.print(f"Lightroom render input: {render_input}")
    console.print(f"Lightroom export target: {run_dir / 'lightroom-render-output'}")


@app.command("build-edit-review-page")
def build_edit_review_page(
    run_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Build a private local review page from the blinded Lightroom JPEGs."""
    page = write_blind_review_page(run_dir)
    console.print(f"Blind review page: {page}")


@app.command()
def doctor(
    backend_url: str = typer.Option(
        "http://localhost:11434",
        help="Backend URL to probe for Ollama health.",
    ),
    rapidraw_binary: Path = typer.Option(
        DEFAULT_RAPIDRAW_BINARY,
        help="RapidRAW headless exporter binary to inspect.",
    ),
) -> None:
    """Report key local dependencies used by the scaffold."""
    table = Table(title="Cull.sh Doctor")
    table.add_column("Dependency")
    table.add_column("Location")

    for tool in ("exiftool", "sips", "ollama"):
        location = shutil.which(tool) or "not found"
        table.add_row(tool, location)

    rapidraw = rapidraw_install_info(rapidraw_binary)
    rapidraw_location = str(rapidraw["binary"])
    if rapidraw.get("version"):
        rapidraw_location += f" (version {rapidraw['version']})"
    if not rapidraw["exists"]:
        rapidraw_location += " (not found)"
    elif not rapidraw["executable"]:
        rapidraw_location += " (not executable)"
    table.add_row("RapidRAW", rapidraw_location)

    console.print(table)

    health = Table(title="Backend Health")
    health.add_column("Check")
    health.add_column("Result")
    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        health.add_row("Ollama API", f"reachable ({len(models)} model(s) listed)")
        installed_names = {
            str(model.get("name") or model.get("model"))
            for model in models
            if isinstance(model, dict)
        }
        edit_model_ready = DEFAULT_EDIT_MODEL in installed_names or any(
            name.startswith(f"{DEFAULT_EDIT_MODEL}:") for name in installed_names
        )
        health.add_row(
            "Edit model",
            f"ready ({DEFAULT_EDIT_MODEL})"
            if edit_model_ready
            else f"missing ({DEFAULT_EDIT_MODEL})",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        health.add_row("Ollama API", f"unreachable: {exc}")

    console.print(health)


@app.command()
def cull(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    prompt: str | None = typer.Option(
        None,
        help="Explicit culling instructions. Overrides genre presets when provided.",
    ),
    genre: GenrePreset = typer.Option(
        GenrePreset.AUTO,
        case_sensitive=False,
        help="Genre preset used when --prompt is omitted.",
    ),
    prefer: str | None = typer.Option(
        None,
        help="Optional short preference appended to the genre preset prompt.",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Ask for genre and preferences when no prompt is provided.",
    ),
    provider: str = typer.Option("ollama", help="Vision backend provider."),
    model: str = typer.Option("gemma4:12b", help="Vision backend model name."),
    backend_url: str = typer.Option(
        "http://localhost:11434",
        help="Base URL for the selected backend.",
    ),
    backend_timeout: float = typer.Option(
        300.0,
        min=1.0,
        help="Per-request Ollama inactivity timeout in seconds.",
    ),
    backend_max_attempts: int = typer.Option(
        3,
        min=1,
        help="Maximum attempts for a failed or malformed culling response.",
    ),
    backend_max_output_tokens: int = typer.Option(
        1024,
        min=1,
        help="Hard Ollama generation ceiling for each structured response.",
    ),
    batch_size: int = typer.Option(
        4,
        min=1,
        help="Maximum images per same-scene cohort sent to the vision backend.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Only process the first N scenes after full-folder grouping.",
    ),
    extract_workers: int = typer.Option(6, min=1),
    score_workers: int = typer.Option(6, min=1),
    min_blur_score: float = typer.Option(110.0, min=0.0),
    min_tenengrad_score: float = typer.Option(45.0, min=0.0),
    learned_iqa: bool = typer.Option(
        True,
        "--learned-iqa/--no-learned-iqa",
        help="Enable learned local quality scores with MUSIQ and NIMA when PyIQA is available.",
    ),
    topiq_ranking: bool = typer.Option(
        True,
        "--topiq-ranking/--no-topiq-ranking",
        help=(
            "Blend TOPIQ NR into folder-relative candidate ranking without "
            "adding it to the hard quality-reject gate."
        ),
    ),
    topiq_rank_weight: float = typer.Option(
        0.25,
        min=0.0,
        max=1.0,
        help="TOPIQ contribution to candidate ranking; calibrated on Xiuling and Longwood.",
    ),
    topiq_shadow: bool = typer.Option(
        True,
        "--topiq-shadow/--no-topiq-shadow",
        help=(
            "Write a TOPIQ disagreement review artifact after culling."
        ),
    ),
    topiq_shadow_workers: int = typer.Option(
        2,
        min=1,
        help="Bounded worker processes used for TOPIQ NR scoring.",
    ),
    topiq_shadow_low_percentile: float = typer.Option(
        0.2,
        min=0.0,
        max=1.0,
        help="Bottom TOPIQ percentile treated as the shadow low-quality band.",
    ),
    topiq_shadow_high_percentile: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Top TOPIQ percentile treated as the shadow high-quality band.",
    ),
    topiq_shadow_max_items: int = typer.Option(
        80,
        min=1,
        help="Maximum photos included in the TOPIQ shadow HTML review.",
    ),
    min_musiq_score: float = typer.Option(50.0, min=0.0),
    min_nima_score: float = typer.Option(4.6, min=0.0),
    brisque: bool = typer.Option(
        True,
        "--brisque/--no-brisque",
        help="Record BRISQUE scores when the optional library is available.",
    ),
    cpbd: bool = typer.Option(
        True,
        "--cpbd/--no-cpbd",
        help="Record CPBD perceptual blur scores when the optional library is available.",
    ),
    max_brisque_score: float = typer.Option(55.0, min=0.0),
    min_cpbd_score: float = typer.Option(0.3, min=0.0),
    use_brisque_for_reject: bool = typer.Option(
        False,
        "--use-brisque-for-reject/--no-use-brisque-for-reject",
        help="Let BRISQUE contribute a weak-quality vote in the local reject gate.",
    ),
    use_cpbd_for_reject: bool = typer.Option(
        False,
        "--use-cpbd-for-reject/--no-use-cpbd-for-reject",
        help="Let CPBD contribute a weak-quality vote in the local reject gate.",
    ),
    local_reject_required_support_votes: int = typer.Option(2, min=1),
    duplicate_hamming_threshold: int = typer.Option(6, min=0),
    max_scene_candidates: int | None = typer.Option(None, min=1),
    cache_previews: bool = typer.Option(
        False,
        help="Persist extracted JPEG previews under runs/<timestamp>/previews.",
    ),
    include_jpegs: bool = typer.Option(
        True,
        "--include-jpegs/--raw-only",
        help="Cull JPEG files too. Paired RAW+JPEG files mirror the RAW decision by default.",
    ),
    mirror_paired_jpegs: bool = typer.Option(
        True,
        "--mirror-paired-jpegs/--score-paired-jpegs",
        help="Mirror cull decisions to same-stem paired JPEGs instead of scoring them separately.",
    ),
    lightroom_auto_edit: bool = typer.Option(
        False,
        "--lightroom-auto-edit/--no-lightroom-auto-edit",
        help=(
            "When writing RAW metadata, apply safe Lightroom sidecar edits. "
            "Lens corrections are written directly; Adaptive Color still "
            "needs Lightroom UI/preset automation."
        ),
    ),
    lightroom_edit_scope: LightroomEditScope = typer.Option(
        LightroomEditScope.ALL,
        "--lightroom-edit-scope",
        help="Which RAW files receive Lightroom sidecar/UI edits: all or kept.",
    ),
    dry_run: bool = typer.Option(True, help="Skip sidecar writes for now."),
) -> None:
    """
    Start a culling run.

    Discovers RAW and JPEG files, scores them, and writes Lightroom-compatible
    culling metadata when dry-run mode is disabled.
    """
    if topiq_shadow_low_percentile >= topiq_shadow_high_percentile:
        raise typer.BadParameter(
            "--topiq-shadow-low-percentile must be below "
            "--topiq-shadow-high-percentile"
        )

    prompt_selection = _resolve_prompt_selection(
        prompt=prompt,
        genre=genre,
        prefer=prefer,
        interactive=interactive,
    )

    config = PipelineConfig(
        path=path,
        prompt=prompt_selection.prompt,
        genre=prompt_selection.genre.value,
        prefer=prompt_selection.prefer,
        prompt_source=prompt_selection.source,
        backend=BackendConfig(
            provider=provider,
            model=model,
            base_url=backend_url,
            timeout_seconds=backend_timeout,
            max_attempts=backend_max_attempts,
            max_output_tokens=backend_max_output_tokens,
        ),
        limit=limit,
        batch_size=batch_size,
        extract_workers=extract_workers,
        score_workers=score_workers,
        min_blur_score=min_blur_score,
        min_tenengrad_score=min_tenengrad_score,
        enable_learned_iqa=learned_iqa,
        enable_topiq_ranking=topiq_ranking,
        topiq_rank_weight=topiq_rank_weight,
        enable_topiq_shadow=topiq_shadow,
        topiq_shadow_workers=topiq_shadow_workers,
        topiq_shadow_low_percentile=topiq_shadow_low_percentile,
        topiq_shadow_high_percentile=topiq_shadow_high_percentile,
        topiq_shadow_max_items=topiq_shadow_max_items,
        min_musiq_score=min_musiq_score,
        min_nima_score=min_nima_score,
        enable_brisque=brisque,
        enable_cpbd=cpbd,
        max_brisque_score=max_brisque_score,
        min_cpbd_score=min_cpbd_score,
        use_brisque_for_reject=use_brisque_for_reject,
        use_cpbd_for_reject=use_cpbd_for_reject,
        local_reject_required_support_votes=local_reject_required_support_votes,
        duplicate_hamming_threshold=duplicate_hamming_threshold,
        max_scene_candidates=max_scene_candidates,
        cache_previews=cache_previews,
        include_jpegs=include_jpegs,
        mirror_paired_jpegs=mirror_paired_jpegs,
        lightroom_auto_edit=lightroom_auto_edit,
        lightroom_edit_scope=lightroom_edit_scope,
        dry_run=dry_run,
    )

    with RichPipelineReporter(console) as reporter:
        items, summary, run_dir = run_pipeline(config, reporter=reporter)
    console.print(f"Discovered {summary.discovered} photo file(s) under {path}")
    console.print(
        f"Assets: RAW={summary.raw_discovered}, JPEG={summary.jpeg_discovered}"
        + (
            f" ({summary.mirrored_jpegs} paired JPEG decision mirror(s))"
            if summary.mirrored_jpegs
            else ""
        )
    )
    if config.limit is not None:
        console.print(f"Limit: first {config.limit} scene(s) after full-folder grouping")
    console.print(
        f"Backend: provider={config.backend.provider} model={config.backend.model}"
    )
    console.print(
        f"Prompt: source={config.prompt_source} genre={config.genre}"
    )
    if config.enable_learned_iqa:
        if learned_iqa_available():
            console.print(
                f"Local IQA: enabled (MUSIQ>={config.min_musiq_score}, NIMA>={config.min_nima_score})"
            )
        else:
            console.print(
                "Local IQA: unavailable, falling back to blur-only local filtering"
                + (f" ({learned_iqa_import_error()})" if learned_iqa_import_error() else "")
            )
    console.print(
        "TOPIQ ranking: "
        + (
            f"on (weight={config.topiq_rank_weight:.0%}; hard-reject gate unchanged)"
            if config.enable_topiq_ranking
            else "off (local-only ordering)"
        )
    )
    console.print(
        f"TOPIQ disagreement review: {'on' if config.enable_topiq_shadow else 'off'}"
    )
    support_errors = support_metric_import_errors()
    if config.enable_brisque or config.enable_cpbd:
        console.print(
            "Support metrics: "
            f"BRISQUE={'on' if config.enable_brisque else 'off'}"
            f" (threshold<={config.max_brisque_score}, vote={'on' if config.use_brisque_for_reject else 'off'})"
            ", "
            f"CPBD={'on' if config.enable_cpbd else 'off'}"
            f" (threshold>={config.min_cpbd_score}, vote={'on' if config.use_cpbd_for_reject else 'off'})"
        )
        if support_errors:
            console.print(
                "Support metric load errors: "
                + ", ".join(f"{name}={error}" for name, error in sorted(support_errors.items()))
            )
    if config.prefer:
        console.print(f"Preference: {config.prefer}")
    console.print(f"Run artifacts: {run_dir}")
    console.print(f"Dry run: {'yes' if config.dry_run else 'no'}")
    shadow_report = run_dir / "topiq-shadow.html"
    if shadow_report.exists():
        console.print(f"TOPIQ ranking review: {shadow_report}")
    console.print(
        "Lightroom RAW auto edit: "
        + ("on" if config.lightroom_auto_edit else "off")
        + f" (scope={config.lightroom_edit_scope.value})"
    )
    if summary.jpeg_discovered:
        console.print(
            "Lightroom JPEG auto edit: use `python main.py lightroom-jpeg-auto "
            f"--path \"{path}\"` after culling."
        )
    console.print(
        "Local filtering complete; images that passed blur screening are ready for vision."
    )

    if items:
        preview = Table(title="Phase 1 Summary")
        preview.add_column("Filename")
        preview.add_column("Scene")
        preview.add_column("Status")
        preview.add_column("Laplacian")
        preview.add_column("Tenengrad")
        preview.add_column("TOPIQ NR")
        preview.add_column("Local rank")
        preview.add_column("Combined rank")
        for item in items[:10]:
            blur_score = (
                f"{item.metrics.blur_score:.2f}"
                if item.metrics and item.metrics.blur_score is not None
                else "-"
            )
            tenengrad_score = (
                f"{item.metrics.tenengrad_score:.2f}"
                if item.metrics and item.metrics.tenengrad_score is not None
                else "-"
            )
            local_rank_score = (
                f"{item.metrics.local_rank_score:.2f}"
                if item.metrics and item.metrics.local_rank_score is not None
                else "-"
            )
            combined_rank_score = (
                f"{item.metrics.combined_rank_score:.3f}"
                if item.metrics and item.metrics.combined_rank_score is not None
                else "-"
            )
            topiq_score = (
                f"{item.metrics.topiq_score:.2f}"
                if item.metrics and item.metrics.topiq_score is not None
                else "-"
            )
            preview.add_row(
                item.asset.filename,
                item.scene_id or "-",
                item.status.value,
                blur_score,
                tenengrad_score,
                topiq_score,
                local_rank_score,
                combined_rank_score,
            )
        console.print(preview)

    stats = Table(title="Counts")
    stats.add_column("Metric")
    stats.add_column("Value")
    stats.add_row("Locally rejected", str(summary.locally_rejected))
    stats.add_row("Rejected total", str(summary.rejected_total))
    stats.add_row("Review", str(summary.reviewed))
    stats.add_row("Pick", str(summary.picked))
    stats.add_row("Sent to vision", str(summary.queued_for_vision))
    stats.add_row("Vision scored", str(summary.scored))
    stats.add_row("Metadata records written", str(summary.sidecars_written))
    stats.add_row(
        "Lightroom sidecar edits written",
        str(summary.lightroom_edits_written),
    )
    stats.add_row("Failed", str(summary.failed))
    stats.add_row("Mirrored paired JPEGs", str(summary.mirrored_jpegs))
    console.print(stats)


@app.command("lightroom-edit")
def lightroom_edit(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Only inspect the first N discovered RAW files.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Show what would be edited without writing sidecars.",
    ),
    edit_scope: LightroomEditScope = typer.Option(
        LightroomEditScope.ALL,
        "--lightroom-edit-scope",
        help="Which RAW files receive Lightroom sidecar edits: all or kept.",
    ),
) -> None:
    """Apply safe Lightroom sidecar edits to RAW files in a folder."""
    assets = discover_raw_assets(path, DEFAULT_EXTENSIONS)
    if limit is not None:
        assets = assets[:limit]

    candidates = 0
    rejected_found = 0
    skipped_rejected = 0
    sidecars_written = 0
    failed: list[tuple[str, str]] = []

    for asset in assets:
        try:
            rejected = sidecar_is_rejected(asset.xmp_path)
        except Exception as exc:
            failed.append((asset.filename, f"sidecar read failed: {exc}"))
            continue
        if rejected:
            rejected_found += 1
        if edit_scope == LightroomEditScope.KEPT and rejected:
            skipped_rejected += 1
            continue

        candidates += 1
        if dry_run:
            continue
        try:
            write_lightroom_edit_sidecar(asset.xmp_path)
        except Exception as exc:
            failed.append((asset.filename, f"sidecar write failed: {exc}"))
            continue
        sidecars_written += 1

    console.print(f"Discovered {len(assets)} RAW file(s) under {path}")
    console.print(f"Edit scope: {edit_scope.value}")
    console.print(f"Edit candidates: {candidates}")
    console.print(f"Rejected RAW files: {rejected_found}")
    console.print(f"Rejected skipped: {skipped_rejected}")
    console.print(f"Sidecars written: {sidecars_written}")
    console.print(f"Dry run: {'yes' if dry_run else 'no'}")
    console.print("Sidecar edit: lens corrections")
    console.print("Adaptive Color: requires Lightroom UI/preset automation")

    if failed:
        errors = Table(title="Lightroom Edit Errors")
        errors.add_column("Filename")
        errors.add_column("Error")
        for filename, error in failed[:10]:
            errors.add_row(filename, error)
        console.print(errors)
        raise typer.Exit(code=1)


@app.command("lightroom-jpeg-auto")
def lightroom_jpeg_auto(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Only inspect the first N discovered JPEG files.",
    ),
    handoff: bool = typer.Option(
        True,
        "--handoff/--no-handoff",
        help="Write a Computer Use checklist and JSON handoff artifact.",
    ),
    edit_scope: LightroomEditScope = typer.Option(
        LightroomEditScope.ALL,
        "--lightroom-edit-scope",
        help="Which JPEG files should receive Lightroom Auto Settings: all or kept.",
    ),
) -> None:
    """Prepare the Computer Use stage for Lightroom Auto Settings on JPEG files."""
    stage = build_jpeg_auto_stage(
        path,
        JPEG_EXTENSIONS,
        limit=limit,
        edit_scope=edit_scope,
    )

    counts = Table(title="Lightroom JPEG Auto Settings Stage")
    counts.add_column("Metric")
    counts.add_column("Value")
    counts.add_row("JPEG files discovered", str(stage.total))
    counts.add_row("Edit scope", stage.edit_scope.value)
    counts.add_row("Edit candidates", str(stage.candidates))
    counts.add_row("Non-rejected JPEG files", str(stage.non_rejected))
    counts.add_row("Rejected JPEG files", str(stage.rejected))
    counts.add_row("Metadata read errors", str(len(stage.errors)))
    console.print(counts)

    if stage.candidates == 0:
        console.print("JPEG Auto Settings stage: no in-scope JPEG files found.")
    else:
        console.print("JPEG Auto Settings stage: ready for Computer Use in Adobe Lightroom.")
        console.print(
            "Filter to JPEG files, select the in-scope JPEG set, and run "
            "Photo > Apply Auto Settings. Do not apply Adaptive Color to JPEGs."
        )
        candidate_preview = ", ".join(stage.candidate_filenames[:5])
        if candidate_preview:
            console.print(f"Next candidates: {candidate_preview}")

    if handoff:
        run_dir = create_run_dir(Path("runs"))
        payload_path, checklist_path = write_jpeg_auto_handoff(stage, run_dir)
        console.print(f"Handoff JSON: {payload_path}")
        console.print(f"Handoff checklist: {checklist_path}")

    if stage.errors:
        errors = Table(title="JPEG Auto Stage Errors")
        errors.add_column("Filename")
        errors.add_column("Error")
        for error in stage.errors:
            errors.add_row(error["filename"], error["error"])
        console.print(errors)
        raise typer.Exit(code=1)


@app.command("lightroom-adaptive-color")
def lightroom_adaptive_color(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Only inspect the first N discovered RAW files.",
    ),
    runs_root: Path = typer.Option(
        Path("runs"),
        file_okay=False,
        dir_okay=True,
        help="Root directory for Computer Use handoff artifacts.",
    ),
    handoff: bool = typer.Option(
        True,
        "--handoff/--no-handoff",
        help="Write a Computer Use checklist and JSON handoff artifact.",
    ),
    assert_complete: bool = typer.Option(
        False,
        "--assert-complete/--no-assert-complete",
        help="Exit with an error when in-scope RAW files still need Adaptive Color.",
    ),
    edit_scope: LightroomEditScope = typer.Option(
        LightroomEditScope.ALL,
        "--lightroom-edit-scope",
        help="Which RAW files should be verified/edited in Lightroom: all or kept.",
    ),
) -> None:
    """Prepare and verify the Computer Use stage for Lightroom Adaptive Color."""
    stage = build_adaptive_color_stage(
        path,
        DEFAULT_EXTENSIONS,
        limit=limit,
        edit_scope=edit_scope,
    )

    counts = Table(title="Lightroom Adaptive Color Stage")
    counts.add_column("Metric")
    counts.add_column("Value")
    counts.add_row("RAW files discovered", str(stage.total))
    counts.add_row("Edit scope", stage.edit_scope.value)
    counts.add_row("Edit candidates", str(stage.candidates))
    counts.add_row("Non-rejected RAW files", str(stage.non_rejected))
    counts.add_row("Rejected RAW files", str(stage.rejected))
    counts.add_row(
        "Already Adaptive Color / AI payload",
        str(stage.already_adaptive_color),
    )
    counts.add_row("Lightroom .acr payload files", str(len(stage.lightroom_acr_filenames)))
    counts.add_row("Embedded DNG AI payload files", str(len(stage.embedded_dng_filenames)))
    counts.add_row(
        "DNG AI refresh candidates",
        str(len(stage.dng_candidate_filenames)),
    )
    counts.add_row("Pending Adaptive Color", str(stage.pending_adaptive_color))
    counts.add_row("Lens corrections enabled", str(stage.lens_corrections_enabled))
    counts.add_row("Sidecar read errors", str(len(stage.errors)))
    console.print(counts)

    if stage.pending_adaptive_color == 0:
        console.print("Adaptive Color stage: complete for discovered in-scope files.")
    else:
        console.print(
            "Adaptive Color stage: ready for Computer Use in Adobe Lightroom."
        )
        console.print(
            "Apply Adaptive Color to one seed photo, copy only the profile/treatment "
            "edit setting, then paste to the in-scope selection."
        )
        pending_preview = ", ".join(stage.pending_filenames[:5])
        if pending_preview:
            console.print(f"Next pending: {pending_preview}")

    if handoff:
        run_dir = create_run_dir(runs_root)
        payload_path, checklist_path = write_adaptive_color_handoff(stage, run_dir)
        console.print(f"Run artifacts: {run_dir}")
        console.print(f"Computer Use checklist: {checklist_path}")
        console.print(f"Computer Use payload: {payload_path}")

    if stage.errors:
        errors = Table(title="Adaptive Color Stage Errors")
        errors.add_column("Filename")
        errors.add_column("Error")
        for error in stage.errors[:10]:
            errors.add_row(error["filename"], error["error"])
        console.print(errors)
        raise typer.Exit(code=1)

    if assert_complete and stage.pending_adaptive_color:
        raise typer.Exit(code=2)


@app.command("suggest-edits")
def suggest_edits(
    path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    prompt: str | None = typer.Option(
        None,
        help="Editing instructions for the model. Defaults to a natural, balanced look.",
    ),
    prefer: str | None = typer.Option(
        None,
        help="Optional short preference appended to the edit prompt.",
    ),
    provider: str = typer.Option("ollama", help="Vision backend provider."),
    model: str = typer.Option(DEFAULT_EDIT_MODEL, help="Vision backend model name."),
    backend_url: str = typer.Option(
        "http://localhost:11434",
        help="Base URL for the selected backend.",
    ),
    backend_timeout: float = typer.Option(
        120.0,
        min=1.0,
        help="Per-request timeout in seconds for edit suggestion model calls.",
    ),
    max_attempts: int = typer.Option(
        1,
        min=1,
        help="Maximum attempts for each edit suggestion model call; defaults to fail-fast.",
    ),
    backend_max_output_tokens: int = typer.Option(
        1024,
        min=1,
        help="Hard Ollama generation ceiling for each structured edit response.",
    ),
    batch_size: int = typer.Option(
        4,
        min=1,
        help="Maximum images per cohort sent to the model in one request.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Only inspect the first N discovered RAW files.",
    ),
    extract_workers: int = typer.Option(6, min=1),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Show suggestions without writing develop settings into sidecars.",
    ),
    include_unculled: bool = typer.Option(
        False,
        "--include-unculled/--skip-unculled",
        help="Also suggest edits for RAW files with no existing XMP sidecar.",
    ),
    with_crop: bool = typer.Option(
        False,
        "--with-crop/--no-with-crop",
        help="Allow the model to suggest composition crop and leveling settings.",
    ),
) -> None:
    """Suggest optional develop edits for culled, non-rejected RAW files.

    The default dry run produces a frozen JSONL recipe for the RapidRAW workflow.
    With --no-dry-run, edits are also written as standard, fully reversible Camera
    Raw settings into XMP. Rejected and unculled RAW files are skipped unless
    --include-unculled is passed.
    """
    edit_prompt = (prompt or DEFAULT_EDIT_PROMPT).strip()
    if prefer and prefer.strip():
        edit_prompt = f"{edit_prompt} Prioritize: {prefer.strip()}."

    assets = discover_raw_assets(path, DEFAULT_EXTENSIONS)
    if limit is not None:
        assets = assets[:limit]

    kept: list[RawAsset] = []
    skipped_rejected = 0
    skipped_unculled = 0
    read_errors: list[tuple[str, str]] = []
    for asset in assets:
        if not asset.xmp_path.exists() and not include_unculled:
            skipped_unculled += 1
            continue
        try:
            rejected = sidecar_is_rejected(asset.xmp_path)
        except Exception as exc:
            read_errors.append((asset.filename, f"sidecar read failed: {exc}"))
            continue
        if rejected:
            skipped_rejected += 1
            continue
        kept.append(asset)

    console.print(f"Discovered {len(assets)} RAW file(s) under {path}")
    console.print(f"Backend: provider={provider} model={model}")
    console.print(f"Edit candidates (culled non-rejected RAW): {len(kept)}")
    console.print(f"Rejected RAW skipped: {skipped_rejected}")
    console.print(f"Unculled RAW skipped: {skipped_unculled}")

    if not kept:
        console.print("No culled non-rejected RAW files to edit.")
        if read_errors:
            _print_error_table("Sidecar Read Errors", read_errors)
            raise typer.Exit(code=1)
        return

    extractor = build_default_extractor()
    previews, extract_errors = _extract_previews_for_assets(kept, extractor, extract_workers)
    read_errors.extend(extract_errors)
    if not previews:
        console.print("No previews could be extracted; nothing to suggest.")
        _print_error_table("Preview Extraction Errors", read_errors)
        raise typer.Exit(code=1)

    try:
        backend = build_backend(
            BackendConfig(
                provider=provider,
                model=model,
                base_url=backend_url,
                timeout_seconds=backend_timeout,
                max_attempts=max_attempts,
                max_output_tokens=backend_max_output_tokens,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run_dir = create_run_dir(Path("runs"))
    suggestions_path = _write_edit_suggestions_header(
        run_dir,
        edit_prompt,
        dry_run,
        with_crop=with_crop,
        provider=provider,
        model=model,
        source_root=path,
    )
    console.print(f"Run artifacts: {run_dir}")

    suggestions: list[tuple[RawAsset, EditSuggestion]] = []
    written = 0
    cohorts = list(_chunked(previews, batch_size))
    for cohort_index, cohort in enumerate(cohorts, start=1):
        cohort_names = ", ".join(preview.asset.filename for preview in cohort)
        console.print(
            f"Suggesting edits cohort {cohort_index}/{len(cohorts)} "
            f"({len(cohort)} photo(s)): {cohort_names}"
        )
        try:
            cohort_pairs = _suggest_edit_pairs_fail_fast(
                backend,
                edit_prompt,
                cohort,
                include_crop=with_crop,
            )
        except (VisionBackendError, ValueError) as exc:
            console.print(
                f"Edit suggestion failed for cohort {cohort_index}/{len(cohorts)}: {exc}"
            )
            console.print("Stopped immediately; no fallback model or per-image retry was used.")
            raise typer.Exit(code=1) from exc

        suggestions.extend(cohort_pairs)
        _append_edit_suggestions(suggestions_path, cohort_pairs)

        cohort_written = 0
        if not dry_run:
            for asset, suggestion in cohort_pairs:
                try:
                    write_develop_sidecar(asset.xmp_path, suggestion)
                except Exception as exc:
                    read_errors.append(
                        (suggestion.filename, f"sidecar write failed: {exc}")
                    )
                    continue
                written += 1
                cohort_written += 1
        console.print(
            f"Completed edit cohort {cohort_index}/{len(cohorts)}: "
            f"suggestions={len(cohort_pairs)} sidecars_written={cohort_written}"
        )

    summary = Table(title="Suggest Edits Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Previews extracted", str(len(previews)))
    summary.add_row("Suggestions returned", str(len(suggestions)))
    summary.add_row(
        "No-op suggestions",
        str(sum(1 for _, suggestion in suggestions if suggestion.is_noop)),
    )
    summary.add_row(
        "Crop suggestions",
        str(sum(1 for _, suggestion in suggestions if suggestion.has_crop)),
    )
    summary.add_row("Model failure policy", "fail-fast")
    summary.add_row("Sidecars written", str(written))
    summary.add_row("Dry run", "yes" if dry_run else "no")
    console.print(summary)

    if suggestions:
        preview_table = Table(title="Suggested Edits (first 10)")
        for column in ("Filename", "Exp", "Contr", "High", "Shad", "Vib", "Crop", "Note"):
            preview_table.add_column(column)
        for _, suggestion in suggestions[:10]:
            preview_table.add_row(
                suggestion.filename,
                f"{suggestion.exposure:+.2f}",
                str(suggestion.contrast),
                str(suggestion.highlights),
                str(suggestion.shadows),
                str(suggestion.vibrance),
                "yes" if suggestion.has_crop else "no",
                (suggestion.summary[:40] + "…")
                if len(suggestion.summary) > 41
                else suggestion.summary,
            )
        console.print(preview_table)

    if read_errors:
        _print_error_table("Suggest Edits Errors", read_errors)
        raise typer.Exit(code=1)


@app.command("rapidraw-stage")
def rapidraw_stage(
    suggestions: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Frozen edit-suggestions JSONL produced by suggest-edits.",
    ),
    output: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
        help="New isolated staging directory; it must not already exist.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Optionally stage only the first N suggestions for a pilot.",
    ),
) -> None:
    """Stage reversible RapidRAW sidecars without modifying original photos."""
    try:
        stage = stage_rapidraw_develop(
            suggestions,
            output,
            limit=limit,
            cull_sh_commit=_current_git_commit(),
        )
    except (FileExistsError, FileNotFoundError, RapidRawError, ValueError) as exc:
        console.print(f"RapidRAW stage failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Staged {stage.photos} photo(s): {stage.root}")
    console.print("Original RAW and XMP files were not modified.")
    console.print(f"Manifest: {stage.manifest_path}")
    console.print(f"Approval template: {stage.approval_template_path}")
    console.print(
        "Next: run rapidraw-preview --stage "
        f"{stage.root} to render the visual approval page."
    )


@app.command("rapidraw-preview")
def rapidraw_preview(
    stage: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="RapidRAW stage created by rapidraw-stage.",
    ),
    rapidraw_binary: Path = typer.Option(
        DEFAULT_RAPIDRAW_BINARY,
        help="RapidRAW headless exporter binary.",
    ),
    quality: int = typer.Option(
        82,
        min=1,
        max=100,
        help="JPEG quality for disposable visual-review renders.",
    ),
) -> None:
    """Render staged edits and create a local before/after approval page."""
    try:
        page = render_rapidraw_review(
            stage,
            rapidraw_binary,
            quality=quality,
        )
    except (FileNotFoundError, RapidRawError, ValueError) as exc:
        console.print(f"RapidRAW preview failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Review page: {page}")
    console.print(
        "Review the before/after renders, choose approvals, and use "
        "Download approvals before final export."
    )


@app.command("rapidraw-export")
def rapidraw_export(
    stage: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="RapidRAW stage created by rapidraw-stage.",
    ),
    approvals: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Approval JSON downloaded from the stage's review.html.",
    ),
    approve_all: bool = typer.Option(
        False,
        "--approve-all/--require-approvals",
        help="Explicitly bypass the approval file and export the full stage.",
    ),
    rapidraw_binary: Path = typer.Option(
        DEFAULT_RAPIDRAW_BINARY,
        help="RapidRAW headless exporter binary.",
    ),
    output_format: str = typer.Option(
        "jpeg",
        "--format",
        help="RapidRAW output format: jpeg, png, webp, avif, tiff, or jxl.",
    ),
    quality: int = typer.Option(92, min=1, max=100),
    keep_metadata: bool = typer.Option(
        True,
        "--keep-metadata/--strip-metadata",
        help="Retain capture metadata in final exports.",
    ),
) -> None:
    """Export approved staged edits, failing fast while preserving resume state."""
    if approvals is not None and approve_all:
        raise typer.BadParameter("use either --approvals or --approve-all, not both")
    try:
        result = export_rapidraw_stage(
            stage,
            rapidraw_binary,
            approvals_path=approvals,
            approve_all=approve_all,
            output_format=output_format.lower(),
            quality=quality,
            keep_metadata=keep_metadata,
        )
    except (FileExistsError, FileNotFoundError, RapidRawError, ValueError) as exc:
        console.print(f"RapidRAW final export failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Approved={result.approved} newly_exported={result.exported} "
        f"resumed={result.resumed}"
    )
    console.print(f"Export manifest: {result.manifest_path}")


@app.command("repair-sidecars")
def repair_sidecars(
    run_dir: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Run directory containing manifest.jsonl. Defaults to the latest run under ./runs.",
    ),
    runs_root: Path = typer.Option(
        Path("runs"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Root directory containing run artifacts when --run-dir is omitted.",
    ),
    lightroom_auto_edit: bool = typer.Option(
        False,
        "--lightroom-auto-edit/--no-lightroom-auto-edit",
        help="Also apply safe Lightroom sidecar edits to repaired sidecars.",
    ),
    lightroom_edit_scope: LightroomEditScope = typer.Option(
        LightroomEditScope.ALL,
        "--lightroom-edit-scope",
        help="Which repaired sidecars receive Lightroom edits: all or kept.",
    ),
) -> None:
    """Rewrite sidecars from an existing manifest without rerunning scoring."""
    try:
        target_run_dir = run_dir or find_latest_run_dir(runs_root)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    records = load_manifest_records(target_run_dir)

    rewritten = 0
    skipped = 0
    for record in records:
        decision = decision_from_manifest_record(record)
        if decision is None:
            skipped += 1
            continue
        raw_path = Path(str(record["raw_path"]))
        asset_kind = AssetKind(str(record.get("asset_kind", AssetKind.RAW.value)))
        xmp_path = Path(str(record.get("xmp_path", raw_path.with_suffix(".xmp"))))
        paired_raw_path = record.get("paired_raw_path")
        asset = RawAsset(
            raw_path=raw_path,
            xmp_path=xmp_path,
            kind=asset_kind,
            paired_raw_path=Path(str(paired_raw_path)) if paired_raw_path else None,
        )
        write_photo_metadata(
            asset,
            decision,
            apply_lightroom_edit=lightroom_auto_edit,
            lightroom_edit_scope=lightroom_edit_scope,
        )
        rewritten += 1

    console.print(f"Run artifacts: {target_run_dir}")
    console.print(f"Sidecars rewritten: {rewritten}")
    console.print(f"Records skipped: {skipped}")
    console.print(
        "Lightroom auto edit: "
        + ("on" if lightroom_auto_edit else "off")
        + f" (scope={lightroom_edit_scope.value})"
    )


def _extract_previews_for_assets(
    assets: list[RawAsset],
    extractor,
    workers: int,
) -> tuple[list[PreviewImage], list[tuple[str, str]]]:
    previews: list[PreviewImage] = []
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extractor.extract_preview_bytes, asset.raw_path): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                image_bytes = future.result()
            except PreviewExtractionError as exc:
                errors.append((asset.filename, f"preview extraction failed: {exc}"))
                continue
            except Exception as exc:
                errors.append((asset.filename, f"unexpected extraction error: {exc}"))
                continue
            previews.append(PreviewImage(asset=asset, image_bytes=image_bytes))
    previews.sort(key=lambda preview: preview.asset.filename)
    return previews, errors


def _chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _suggest_edit_pairs_fail_fast(
    backend,
    prompt: str,
    cohort: list[PreviewImage],
    include_crop: bool = False,
) -> list[tuple[RawAsset, EditSuggestion]]:
    cohort_suggestions = backend.suggest_edits(
        prompt,
        cohort,
        include_crop=include_crop,
    )
    if len(cohort_suggestions) != len(cohort):
        raise VisionBackendError(
            "backend returned an unexpected number of edit suggestions"
        )
    return [
        (preview.asset, suggestion)
        for preview, suggestion in zip(cohort, cohort_suggestions)
    ]


def _write_edit_suggestions_header(
    run_dir: Path,
    prompt: str,
    dry_run: bool,
    with_crop: bool = False,
    provider: str | None = None,
    model: str | None = None,
    source_root: Path | None = None,
) -> Path:
    path = run_dir / "edit-suggestions.jsonl"
    header: dict[str, object] = {
        "prompt": prompt,
        "dry_run": dry_run,
        "with_crop": with_crop,
    }
    if provider is not None:
        header["schema_version"] = 1
        header["kind"] = "cull-sh-edit-suggestions"
        header["provider"] = provider
    if model is not None:
        header["model"] = model
    if source_root is not None:
        header["source_root"] = str(source_root.expanduser().resolve())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(header, sort_keys=True)
            + "\n"
        )
    return path


def _append_edit_suggestions(
    path: Path,
    suggestions: list[tuple[RawAsset, EditSuggestion]],
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for asset, suggestion in suggestions:
            handle.write(
                json.dumps(
                    {
                        "asset_id": suggestion.asset_id or asset.raw_path.as_posix(),
                        "filename": suggestion.filename,
                        "raw_path": str(asset.raw_path),
                        "xmp_path": str(asset.xmp_path),
                        "exposure": suggestion.exposure,
                        "contrast": suggestion.contrast,
                        "highlights": suggestion.highlights,
                        "shadows": suggestion.shadows,
                        "vibrance": suggestion.vibrance,
                        "has_crop": suggestion.has_crop,
                        "crop_left": suggestion.crop_left,
                        "crop_top": suggestion.crop_top,
                        "crop_right": suggestion.crop_right,
                        "crop_bottom": suggestion.crop_bottom,
                        "crop_angle": suggestion.crop_angle,
                        "summary": suggestion.summary,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_edit_suggestions(
    run_dir: Path,
    prompt: str,
    suggestions: list[tuple[RawAsset, EditSuggestion]],
    dry_run: bool,
    with_crop: bool = False,
) -> Path:
    path = _write_edit_suggestions_header(
        run_dir,
        prompt,
        dry_run,
        with_crop=with_crop,
    )
    _append_edit_suggestions(path, suggestions)
    return path


def _print_error_table(title: str, errors: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Filename")
    table.add_column("Error")
    for filename, error in errors[:10]:
        table.add_row(filename, error)
    console.print(table)


def _format_cli_percent(value: object) -> str:
    return "-" if value is None else f"{float(value):.1%}"


def _current_git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _resolve_prompt_selection(
    prompt: str | None,
    genre: GenrePreset,
    prefer: str | None,
    interactive: bool,
):
    if prompt is not None and prompt.strip():
        return resolve_prompt(prompt=prompt, genre=genre, prefer=prefer, source="custom")

    if interactive and sys.stdin.isatty():
        return _run_guided_setup(genre, prefer)

    return resolve_prompt(prompt=None, genre=genre, prefer=prefer, source="preset")


def _run_guided_setup(genre: GenrePreset, prefer: str | None):
    console.print("No prompt provided. Starting guided setup.")
    selected_genre = genre
    if genre == GenrePreset.AUTO:
        raw_value = typer.prompt(
            f"Genre ({supported_genre_labels()})",
            default=GenrePreset.MIXED.value,
        )
        try:
            selected_genre = parse_genre(raw_value)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if selected_genre == GenrePreset.AUTO:
            selected_genre = GenrePreset.MIXED

    selected_prefer = prefer
    if selected_prefer is None:
        selected_prefer = typer.prompt(
            "Anything to prioritize? Press Enter to skip",
            default="",
            show_default=False,
        )

    return resolve_prompt(
        prompt=None,
        genre=selected_genre,
        prefer=selected_prefer,
        source="interactive",
    )


if __name__ == "__main__":
    app()
