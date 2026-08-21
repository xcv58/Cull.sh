from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Callable

from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.benchmark import evaluate_signal
from cull_sh.benchmark import load_human_labels
from cull_sh.benchmark import paired_auc_delta_bootstrap
from cull_sh.config import DEFAULT_EXTENSIONS
from cull_sh.extractors import build_default_extractor
from cull_sh.grouping import assign_scene_groups
from cull_sh.manifests import create_run_dir
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem
from cull_sh.scanner import discover_raw_assets


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class VLMModelSpec:
    label: str
    model: str
    temperature: float = 0.0
    think: bool | str | None = None


@dataclass(frozen=True, slots=True)
class VLMCohort:
    cohort_id: str
    scene_id: str
    assets: tuple[RawAsset, ...]
    human_labels: tuple[str, ...]


def run_vlm_benchmark(
    photo_root: Path,
    prompt: str,
    model_specs: list[VLMModelSpec],
    runs_root: Path,
    *,
    include_subfolders: bool = False,
    batch_size: int = 4,
    canary_cohorts: int | None = None,
    cohort_ids: list[str] | None = None,
    timeout_seconds: float = 600.0,
    max_attempts: int = 3,
    frozen_baseline_run: Path | None = None,
    resume_run: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, object], Path]:
    """Compare Ollama vision models without writing photo metadata."""
    progress = progress or (lambda _message: None)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not model_specs:
        raise ValueError("at least one model is required")

    all_cohorts = build_vlm_cohorts(
        photo_root,
        batch_size=batch_size,
        include_subfolders=include_subfolders,
    )
    if cohort_ids:
        by_id = {cohort.cohort_id: cohort for cohort in all_cohorts}
        missing = [cohort_id for cohort_id in cohort_ids if cohort_id not in by_id]
        if missing:
            raise ValueError("unknown benchmark cohort(s): " + ", ".join(missing))
        cohorts = [by_id[cohort_id] for cohort_id in cohort_ids]
    else:
        cohorts = (
            select_canary_cohorts(all_cohorts, canary_cohorts)
            if canary_cohorts is not None
            else all_cohorts
        )
    if not cohorts:
        raise ValueError(f"no benchmark cohorts found under {photo_root}")

    run_dir = resume_run or create_run_dir(runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": "vlm-culling-benchmark",
        "photo_root": str(photo_root),
        "include_subfolders": include_subfolders,
        "batch_size": batch_size,
        "canary_cohorts": canary_cohorts,
        "cohort_ids": cohort_ids or [],
        "prompt": prompt,
        "models": [asdict(spec) for spec in model_specs],
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "frozen_baseline_run": str(frozen_baseline_run) if frozen_baseline_run else None,
        "metadata_writes": False,
        "preview_policy": "exact embedded JPEG preview extracted by ExifTool",
    }
    _write_or_verify_config(run_dir / "vlm-benchmark-config.json", config)
    _write_cohort_manifest(run_dir / "vlm-cohorts.json", cohorts)

    photo_count = sum(len(cohort.assets) for cohort in cohorts)
    progress(
        f"Locked {len(cohorts)} cohort(s) containing {photo_count} photo(s); "
        "photo metadata writes are disabled."
    )
    previews = _prepare_previews(cohorts, run_dir / "previews", progress)

    evaluation_specs = list(model_specs)
    if frozen_baseline_run is not None:
        baseline_spec = VLMModelSpec(
            label="current-pipeline-frozen",
            model=f"frozen:{frozen_baseline_run}",
        )
        _import_frozen_pipeline_baseline(
            frozen_baseline_run,
            run_dir / f"{_safe_slug(baseline_spec.label)}.jsonl",
            cohorts,
        )
        evaluation_specs.insert(0, baseline_spec)
        progress("Imported the frozen current-pipeline decisions for all selected photos.")

    for spec in model_specs:
        _run_model(
            spec,
            cohorts,
            previews,
            prompt,
            run_dir,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            progress=progress,
        )

    payload = evaluate_vlm_run(run_dir, cohorts, evaluation_specs)
    return payload, run_dir


def build_vlm_cohorts(
    photo_root: Path,
    *,
    batch_size: int,
    include_subfolders: bool,
) -> list[VLMCohort]:
    labels = load_human_labels(
        photo_root,
        include_subfolders=include_subfolders,
    )
    label_by_path = {Path(label.raw_path): label.label for label in labels}
    assets = discover_raw_assets(photo_root, DEFAULT_EXTENSIONS)
    if not include_subfolders:
        assets = [asset for asset in assets if asset.raw_path.parent == photo_root]
    items = [WorkItem(asset=asset) for asset in assets if asset.raw_path in label_by_path]
    assign_scene_groups(items)

    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[item.scene_id or "ungrouped"].append(item)

    cohorts: list[VLMCohort] = []
    for scene_id in sorted(grouped):
        scene_items = sorted(
            grouped[scene_id],
            key=lambda item: (item.scene_index or 0, item.filename),
        )
        for offset in range(0, len(scene_items), batch_size):
            chunk = scene_items[offset : offset + batch_size]
            cohort_number = (offset // batch_size) + 1
            cohorts.append(
                VLMCohort(
                    cohort_id=f"{scene_id}-cohort-{cohort_number:03d}",
                    scene_id=scene_id,
                    assets=tuple(item.asset for item in chunk),
                    human_labels=tuple(label_by_path[item.asset.raw_path] for item in chunk),
                )
            )
    return cohorts


def select_canary_cohorts(
    cohorts: list[VLMCohort],
    count: int,
) -> list[VLMCohort]:
    if count < 1:
        raise ValueError("canary cohort count must be at least 1")
    if count >= len(cohorts):
        return list(cohorts)

    buckets: dict[tuple[bool, bool, bool], list[VLMCohort]] = defaultdict(list)
    for cohort in cohorts:
        labels = set(cohort.human_labels)
        buckets[("pick" in labels, "reject" in labels, "neutral" in labels)].append(cohort)

    priority = [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ]
    selected: list[VLMCohort] = []
    positions = {signature: 0 for signature in priority}
    while len(selected) < count:
        advanced = False
        for signature in priority:
            position = positions[signature]
            candidates = buckets.get(signature, [])
            if position >= len(candidates):
                continue
            selected.append(candidates[position])
            positions[signature] += 1
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            break
    return selected


def evaluate_vlm_run(
    run_dir: Path,
    cohorts: list[VLMCohort],
    model_specs: list[VLMModelSpec],
) -> dict[str, object]:
    human_by_filename: dict[str, str] = {}
    scene_by_filename: dict[str, str] = {}
    for cohort in cohorts:
        for asset, label in zip(cohort.assets, cohort.human_labels, strict=True):
            human_by_filename[asset.filename.casefold()] = label
            scene_by_filename[asset.filename.casefold()] = cohort.scene_id

    model_results: list[dict[str, object]] = []
    score_records: dict[str, dict[str, object]] = {
        filename: {
            "filename": filename,
            "label": human_label,
            "signals": {},
            "burst_group_id": scene_by_filename[filename],
        }
        for filename, human_label in human_by_filename.items()
    }

    for spec in model_specs:
        records = _latest_model_records(run_dir / f"{_safe_slug(spec.label)}.jsonl")
        predictions: dict[str, dict[str, object]] = {}
        elapsed_seconds = 0.0
        attempts = 0
        successful_cohorts = 0
        first_pass_cohorts = 0
        frozen_source = False
        for record in records.values():
            frozen_source = frozen_source or record.get("source") == "frozen-pipeline"
            elapsed_seconds += float(record.get("elapsed_seconds", 0.0))
            attempts += int(record.get("attempts", 0))
            if record.get("status") != "success":
                continue
            successful_cohorts += 1
            if record.get("attempts") == 1:
                first_pass_cohorts += 1
            decisions = record.get("decisions", [])
            if not isinstance(decisions, list):
                continue
            for decision in decisions:
                if isinstance(decision, dict) and isinstance(decision.get("filename"), str):
                    predictions[str(decision["filename"]).casefold()] = decision

        decision_counts: Counter[str] = Counter()
        confusion: Counter[tuple[str, str]] = Counter()
        false_rejects = 0
        covered_human_picks = 0
        predicted_picks = 0
        correct_predicted_picks = 0
        human_picks_predicted_pick = 0
        for filename, prediction in predictions.items():
            human_label = human_by_filename.get(filename)
            if human_label is None:
                continue
            bucket = str(prediction.get("bucket"))
            decision_counts[bucket] += 1
            confusion[(human_label, bucket)] += 1
            if human_label == "pick":
                covered_human_picks += 1
                false_rejects += bucket == "reject"
                human_picks_predicted_pick += bucket == "pick"
            if bucket == "pick":
                predicted_picks += 1
                correct_predicted_picks += human_label == "pick"

            rating = int(prediction.get("rating", 0))
            score = {"reject": 0.0, "review": 1.0, "pick": 2.0}.get(bucket, 1.0)
            score += max(0, min(rating, 5)) / 100.0
            target = score_records[filename]["signals"]
            assert isinstance(target, dict)
            target[spec.label] = score

        signal_result = None
        covered_explicit = [
            record
            for record in score_records.values()
            if record["label"] in {"pick", "reject"}
            and spec.label in record["signals"]
        ]
        if {record["label"] for record in covered_explicit} == {"pick", "reject"}:
            signal_result = asdict(
                evaluate_signal(score_records, spec.label, spec.label)
            )

        model_results.append(
            {
                "label": spec.label,
                "model": spec.model,
                "think": spec.think,
                "temperature": spec.temperature,
                "cohorts_total": len(cohorts),
                "cohorts_successful": successful_cohorts,
                "source": "frozen-pipeline" if frozen_source else "ollama",
                "first_pass_success_rate": (
                    None
                    if frozen_source
                    else (first_pass_cohorts / len(cohorts) if cohorts else 0.0)
                ),
                "attempts": attempts,
                "photos_covered": len(predictions),
                "elapsed_seconds": elapsed_seconds,
                "photos_per_minute": (
                    None
                    if frozen_source
                    else (
                        (len(predictions) * 60.0 / elapsed_seconds)
                        if elapsed_seconds > 0
                        else 0.0
                    )
                ),
                "decision_counts": dict(sorted(decision_counts.items())),
                "false_rejects": false_rejects,
                "false_reject_rate": (
                    false_rejects / covered_human_picks if covered_human_picks else 0.0
                ),
                "pick_precision": (
                    correct_predicted_picks / predicted_picks if predicted_picks else 0.0
                ),
                "pick_recall": (
                    human_picks_predicted_pick / covered_human_picks
                    if covered_human_picks
                    else 0.0
                ),
                "confusion": {
                    f"{human}->{predicted}": count
                    for (human, predicted), count in sorted(confusion.items())
                },
                "signal": signal_result,
            }
        )

    comparisons: list[dict[str, object]] = []
    if len(model_specs) >= 2:
        baseline = model_specs[0]
        for candidate in model_specs[1:]:
            try:
                comparisons.append(
                    paired_auc_delta_bootstrap(
                        score_records,
                        candidate.label,
                        candidate.label,
                        baseline.label,
                        baseline.label,
                    )
                )
            except ValueError:
                continue

    human_counts = Counter(human_by_filename.values())
    payload: dict[str, object] = {
        "ground_truth": {
            "photos": len(human_by_filename),
            "picks": human_counts["pick"],
            "rejects": human_counts["reject"],
            "neutral": human_counts["neutral"],
        },
        "models": model_results,
        "comparisons": comparisons,
    }
    (run_dir / "vlm-benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(run_dir / "vlm-benchmark.md", payload)
    return payload


def _prepare_previews(
    cohorts: list[VLMCohort],
    preview_dir: Path,
    progress: ProgressCallback,
) -> dict[str, Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, RawAsset] = {}
    for cohort in cohorts:
        for asset in cohort.assets:
            assets[asset.filename.casefold()] = asset

    extractor = build_default_extractor()
    output: dict[str, Path] = {}
    for index, asset in enumerate(assets.values(), start=1):
        preview_path = preview_dir / f"{asset.filename}.jpg"
        if not preview_path.exists() or preview_path.stat().st_size == 0:
            preview_path.write_bytes(extractor.extract_preview_bytes(asset.raw_path))
        output[asset.filename.casefold()] = preview_path
        if index % 25 == 0 or index == len(assets):
            progress(f"Prepared embedded previews: {index}/{len(assets)}")
    return output


def _import_frozen_pipeline_baseline(
    baseline_run: Path,
    output_path: Path,
    cohorts: list[VLMCohort],
) -> None:
    manifest_path = baseline_run / "manifest.jsonl"
    by_filename: dict[str, dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if isinstance(record.get("filename"), str):
                by_filename[str(record["filename"]).casefold()] = record

    output: list[str] = []
    for cohort in cohorts:
        decisions: list[dict[str, object]] = []
        for asset in cohort.assets:
            record = by_filename.get(asset.filename.casefold())
            decision = record.get("decision") if record is not None else None
            if not isinstance(decision, dict) or not isinstance(decision.get("bucket"), str):
                raise ValueError(
                    f"frozen baseline omitted a decision for {asset.filename}"
                )
            decisions.append(
                {
                    "filename": asset.filename,
                    "bucket": decision["bucket"],
                    "rating": decision.get("rating", 0),
                    "label": decision.get("label"),
                    "summary": decision.get("summary", ""),
                }
            )
        output.append(
            json.dumps(
                {
                    "cohort_id": cohort.cohort_id,
                    "scene_id": cohort.scene_id,
                    "model_label": "current-pipeline-frozen",
                    "model": str(baseline_run),
                    "source": "frozen-pipeline",
                    "status": "success",
                    "attempts": 0,
                    "elapsed_seconds": 0.0,
                    "filenames": [asset.filename for asset in cohort.assets],
                    "decisions": decisions,
                    "error": None,
                },
                sort_keys=True,
            )
        )
    output_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _run_model(
    spec: VLMModelSpec,
    cohorts: list[VLMCohort],
    preview_paths: dict[str, Path],
    prompt: str,
    run_dir: Path,
    *,
    timeout_seconds: float,
    max_attempts: int,
    progress: ProgressCallback,
) -> None:
    output_path = run_dir / f"{_safe_slug(spec.label)}.jsonl"
    completed = _latest_model_records(output_path)
    backend = OllamaVisionBackend(
        base_url="http://127.0.0.1:11434",
        model=spec.model,
        timeout_seconds=timeout_seconds,
        max_attempts=1,
        temperature=spec.temperature,
        think=spec.think,
    )

    with output_path.open("a", encoding="utf-8") as handle:
        for index, cohort in enumerate(cohorts, start=1):
            existing = completed.get(cohort.cohort_id)
            if existing is not None and existing.get("status") == "success":
                progress(f"{spec.label}: reused {index}/{len(cohorts)} {cohort.cohort_id}")
                continue

            previews = [
                PreviewImage(
                    asset=asset,
                    image_bytes=preview_paths[asset.filename.casefold()].read_bytes(),
                    cache_path=preview_paths[asset.filename.casefold()],
                )
                for asset in cohort.assets
            ]
            total_elapsed = 0.0
            error: str | None = None
            decisions = None
            attempts_used = 0
            for attempt in range(1, max_attempts + 1):
                attempts_used = attempt
                started = time.monotonic()
                try:
                    decisions = backend.score_batch(prompt, previews)
                except (VisionBackendError, ValueError) as exc:
                    error = str(exc)
                else:
                    error = None
                total_elapsed += time.monotonic() - started
                if decisions is not None:
                    break

            record = {
                "cohort_id": cohort.cohort_id,
                "scene_id": cohort.scene_id,
                "model_label": spec.label,
                "model": spec.model,
                "status": "success" if decisions is not None else "failed",
                "attempts": attempts_used,
                "elapsed_seconds": round(total_elapsed, 6),
                "filenames": [asset.filename for asset in cohort.assets],
                "decisions": (
                    [
                        {
                            "filename": decision.filename,
                            "bucket": decision.bucket.value,
                            "rating": decision.rating,
                            "label": decision.label.value if decision.label else None,
                            "summary": decision.summary,
                        }
                        for decision in decisions
                    ]
                    if decisions is not None
                    else []
                ),
                "error": error,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            progress(
                f"{spec.label}: {index}/{len(cohorts)} {cohort.cohort_id} "
                f"{record['status']} in {total_elapsed:.1f}s (attempts={attempts_used})"
            )


def _latest_model_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("cohort_id"), str):
                records[str(record["cohort_id"])] = record
    return records


def _write_or_verify_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("resume run configuration does not match the locked benchmark")
        return
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_cohort_manifest(path: Path, cohorts: list[VLMCohort]) -> None:
    payload = [
        {
            "cohort_id": cohort.cohort_id,
            "scene_id": cohort.scene_id,
            "filenames": [asset.filename for asset in cohort.assets],
            "human_labels": list(cohort.human_labels),
        }
        for cohort in cohorts
    ]
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("resume run cohorts do not match the locked benchmark")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    truth = payload["ground_truth"]
    models = payload["models"]
    comparisons = payload.get("comparisons", [])
    assert isinstance(truth, dict)
    assert isinstance(models, list)
    assert isinstance(comparisons, list)
    lines = [
        "# Vision-model culling benchmark",
        "",
        (
            f"Ground truth: {truth['photos']} photos — {truth['picks']} picked, "
            f"{truth['rejects']} rejected, {truth['neutral']} neutral."
        ),
        "",
        "| Model | Covered | First-pass schema | AUC | Top-1 | Top-3 | False rejects | Pick precision | Pick recall | Photos/min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        assert isinstance(model, dict)
        signal = model.get("signal")
        signal = signal if isinstance(signal, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model["label"]),
                    str(model["photos_covered"]),
                    _percent(model["first_pass_success_rate"]),
                    _number(signal.get("auc")),
                    _percent(signal.get("top1_recall")),
                    _percent(signal.get("top3_recall")),
                    f"{model['false_rejects']} ({_percent(model['false_reject_rate'])})",
                    _percent(model["pick_precision"]),
                    _percent(model["pick_recall"]),
                    _number(model["photos_per_minute"]),
                ]
            )
            + " |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired AUC comparisons",
                "",
                "| Candidate | Baseline | AUC delta | 95% bootstrap CI | Iterations |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            assert isinstance(comparison, dict)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(comparison["left_display_name"]),
                        str(comparison["right_display_name"]),
                        _number(comparison["auc_delta"]),
                        (
                            f"{_number(comparison['delta_ci95_low'])} to "
                            f"{_number(comparison['delta_ci95_high'])}"
                        ),
                        str(comparison["bootstrap_iterations"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "The model score is an ordinal mapping of reject < review < pick. Neutral human labels are excluded from AUC and scene-ranking metrics.",
            "No photo or XMP metadata was written by this benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "model"


def _percent(value: object) -> str:
    return "-" if value is None else f"{float(value):.1%}"


def _number(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"
