from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path
import shutil
import sys

import httpx
import typer
from rich.console import Console
from rich.table import Table

from cull_sh.backends import build_backend
from cull_sh.backends import VisionBackendError
from cull_sh.config import BackendConfig, DEFAULT_EXTENSIONS, JPEG_EXTENSIONS, PipelineConfig
from cull_sh.extractors import PreviewExtractionError
from cull_sh.extractors import build_default_extractor
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
from cull_sh.reporting import RichPipelineReporter
from cull_sh.scanner import discover_raw_assets
from cull_sh.xmp import sidecar_is_rejected
from cull_sh.xmp import write_develop_sidecar
from cull_sh.xmp import write_photo_metadata
from cull_sh.xmp import write_lightroom_edit_sidecar


DEFAULT_EDIT_PROMPT = (
    "Suggest natural, balanced global edits that improve each photo while keeping "
    "a realistic look."
)


app = typer.Typer(
    help="Cull.sh: natural-language photo culling for RAW and JPEG files."
)
console = Console()


@app.command()
def doctor(
    backend_url: str = typer.Option(
        "http://localhost:11434",
        help="Backend URL to probe for Ollama health.",
    ),
) -> None:
    """Report key local dependencies used by the scaffold."""
    table = Table(title="Cull.sh Doctor")
    table.add_column("Dependency")
    table.add_column("Location")

    for tool in ("exiftool", "sips", "ollama"):
        location = shutil.which(tool) or "not found"
        table.add_row(tool, location)

    console.print(table)

    health = Table(title="Backend Health")
    health.add_column("Check")
    health.add_column("Result")
    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        health.add_row("Ollama API", f"reachable ({len(models)} model(s) listed)")
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
        ),
        limit=limit,
        batch_size=batch_size,
        extract_workers=extract_workers,
        score_workers=score_workers,
        min_blur_score=min_blur_score,
        min_tenengrad_score=min_tenengrad_score,
        enable_learned_iqa=learned_iqa,
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
        preview.add_column("Rank")
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
            preview.add_row(
                item.asset.filename,
                item.scene_id or "-",
                item.status.value,
                blur_score,
                tenengrad_score,
                local_rank_score,
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
    model: str = typer.Option("gemma4:12b", help="Vision backend model name."),
    backend_url: str = typer.Option(
        "http://localhost:11434",
        help="Base URL for the selected backend.",
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
) -> None:
    """Suggest gentle Lightroom develop edits for culled, non-rejected RAW files.

    Edits are written as standard, fully reversible Camera Raw settings into the
    XMP sidecar next to each RAW. Rejected and unculled RAW files are skipped
    unless --include-unculled is passed.
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
            BackendConfig(provider=provider, model=model, base_url=backend_url)
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    suggestions: list[tuple[RawAsset, EditSuggestion]] = []
    failed_cohorts = 0
    for cohort in _chunked(previews, batch_size):
        try:
            cohort_suggestions = backend.suggest_edits(edit_prompt, cohort)
        except (VisionBackendError, ValueError) as exc:
            failed_cohorts += 1
            for preview in cohort:
                read_errors.append((preview.asset.filename, f"edit suggestion failed: {exc}"))
            continue
        if len(cohort_suggestions) != len(cohort):
            failed_cohorts += 1
            for preview in cohort:
                read_errors.append(
                    (
                        preview.asset.filename,
                        "edit suggestion failed: backend returned an unexpected number of results",
                    )
                )
            continue
        suggestions.extend(
            (preview.asset, suggestion)
            for preview, suggestion in zip(cohort, cohort_suggestions)
        )

    run_dir = create_run_dir(Path("runs"))
    _write_edit_suggestions(run_dir, edit_prompt, suggestions, dry_run)

    written = 0
    if not dry_run:
        for asset, suggestion in suggestions:
            try:
                write_develop_sidecar(asset.xmp_path, suggestion)
            except Exception as exc:
                read_errors.append((suggestion.filename, f"sidecar write failed: {exc}"))
                continue
            written += 1

    summary = Table(title="Suggest Edits Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Previews extracted", str(len(previews)))
    summary.add_row("Suggestions returned", str(len(suggestions)))
    summary.add_row(
        "No-op suggestions",
        str(sum(1 for _, suggestion in suggestions if suggestion.is_noop)),
    )
    summary.add_row("Failed cohorts", str(failed_cohorts))
    summary.add_row("Sidecars written", str(written))
    summary.add_row("Dry run", "yes" if dry_run else "no")
    console.print(summary)

    if suggestions:
        preview_table = Table(title="Suggested Edits (first 10)")
        for column in ("Filename", "Exp", "Contr", "High", "Shad", "Vib", "Note"):
            preview_table.add_column(column)
        for _, suggestion in suggestions[:10]:
            preview_table.add_row(
                suggestion.filename,
                f"{suggestion.exposure:+.2f}",
                str(suggestion.contrast),
                str(suggestion.highlights),
                str(suggestion.shadows),
                str(suggestion.vibrance),
                (suggestion.summary[:40] + "…")
                if len(suggestion.summary) > 41
                else suggestion.summary,
            )
        console.print(preview_table)

    console.print(f"Run artifacts: {run_dir}")

    if read_errors:
        _print_error_table("Suggest Edits Errors", read_errors)
        raise typer.Exit(code=1)


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


def _write_edit_suggestions(
    run_dir: Path,
    prompt: str,
    suggestions: list[tuple[RawAsset, EditSuggestion]],
    dry_run: bool,
) -> Path:
    path = run_dir / "edit-suggestions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"prompt": prompt, "dry_run": dry_run}, sort_keys=True) + "\n"
        )
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
                        "summary": suggestion.summary,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _print_error_table(title: str, errors: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Filename")
    table.add_column("Error")
    for filename, error in errors[:10]:
        table.add_row(filename, error)
    console.print(table)


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
