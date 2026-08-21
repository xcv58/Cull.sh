from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import random
import re
from typing import Callable

from cull_sh.config import DEFAULT_EXTENSIONS
from cull_sh.manifests import create_run_dir
from cull_sh.models import RawAsset
from cull_sh.vlm_benchmark import _prepare_previews
from cull_sh.vlm_benchmark import _run_model
from cull_sh.vlm_benchmark import VLMCohort
from cull_sh.vlm_benchmark import VLMModelSpec


ProgressCallback = Callable[[str], None]


def run_culling_blind_test(
    frozen_run: Path,
    model_specs: list[VLMModelSpec],
    runs_root: Path,
    *,
    batch_size: int = 4,
    seed: int = 20260821,
    timeout_seconds: float = 600.0,
    max_attempts: int = 2,
    cull_sh_commit: str | None = None,
    resume_run: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, Path]:
    """Run two models on frozen semantic candidates and build a blind review."""
    progress = progress or (lambda _message: None)
    if len(model_specs) != 2:
        raise ValueError("blind culling requires exactly two models")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    frozen_run = frozen_run.expanduser().resolve()
    frozen_config = json.loads(
        (frozen_run / "config.json").read_text(encoding="utf-8")
    )
    prompt = str(frozen_config["prompt"])
    cohorts = build_frozen_semantic_cohorts(frozen_run, batch_size=batch_size)
    run_dir = resume_run or create_run_dir(runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "kind": "blind-culling-model-review",
        "frozen_run": str(frozen_run),
        "frozen_manifest_sha256": _file_sha256(frozen_run / "manifest.jsonl"),
        "prompt": prompt,
        "batch_size": batch_size,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "max_output_tokens": 1024,
        "cull_sh_commit": cull_sh_commit,
        "models": [asdict(spec) for spec in model_specs],
        "photo_metadata_reads": False,
        "photo_metadata_writes": False,
    }
    return _execute_blind_culling(
        cohorts,
        prompt,
        model_specs,
        run_dir,
        config,
        seed=seed,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        progress=progress,
    )


def run_sampled_culling_blind_test(
    photos_root: Path,
    prompt: str,
    model_specs: list[VLMModelSpec],
    runs_root: Path,
    *,
    folder_count: int = 6,
    photos_per_folder: int = 2,
    min_sequence_gap: int = 10,
    seed: int = 20260821,
    excluded_folders: tuple[str, ...] = (),
    timeout_seconds: float = 600.0,
    max_attempts: int = 2,
    cull_sh_commit: str | None = None,
    resume_run: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, Path]:
    """Run a blind comparison on a stratified random multi-folder sample."""
    progress = progress or (lambda _message: None)
    if len(model_specs) != 2:
        raise ValueError("blind culling requires exactly two models")
    photos_root = photos_root.expanduser().resolve()
    cohorts = build_random_folder_cohorts(
        photos_root,
        folder_count=folder_count,
        photos_per_folder=photos_per_folder,
        min_sequence_gap=min_sequence_gap,
        seed=seed,
        excluded_folders=excluded_folders,
    )
    run_dir = resume_run or create_run_dir(runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "kind": "blind-culling-random-folder-review",
        "photos_root": str(photos_root),
        "selection_method": "seeded-folder-stratified-filename-gap",
        "folder_count": folder_count,
        "photos_per_folder": photos_per_folder,
        "min_sequence_gap": min_sequence_gap,
        "excluded_folders": list(excluded_folders),
        "prompt": prompt,
        "batch_size": 1,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "max_output_tokens": 1024,
        "cull_sh_commit": cull_sh_commit,
        "models": [asdict(spec) for spec in model_specs],
        "photo_metadata_reads": False,
        "photo_metadata_writes": False,
    }
    return _execute_blind_culling(
        cohorts,
        prompt,
        model_specs,
        run_dir,
        config,
        seed=seed,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        progress=progress,
    )


def build_random_folder_cohorts(
    photos_root: Path,
    *,
    folder_count: int,
    photos_per_folder: int,
    min_sequence_gap: int,
    seed: int,
    excluded_folders: tuple[str, ...] = (),
) -> list[VLMCohort]:
    if folder_count < 1 or photos_per_folder < 1:
        raise ValueError("folder and photo counts must be at least 1")
    if min_sequence_gap < 1:
        raise ValueError("minimum sequence gap must be at least 1")
    excluded = {name.casefold() for name in excluded_folders}
    candidates: list[tuple[Path, list[Path]]] = []
    for folder in sorted(photos_root.iterdir(), key=lambda path: path.name.casefold()):
        if not folder.is_dir():
            continue
        folded_name = folder.name.casefold()
        if "done" in folded_name or "exported" in folded_name or folded_name in excluded:
            continue
        raws = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.casefold() in DEFAULT_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if len(raws) >= photos_per_folder:
            candidates.append((folder, raws))
    if len(candidates) < folder_count:
        raise ValueError(
            f"need {folder_count} eligible folders but found {len(candidates)} under {photos_root}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_folders = candidates[:folder_count]
    cohorts: list[VLMCohort] = []
    for folder_index, (folder, raws) in enumerate(selected_folders, start=1):
        selected = _sample_with_sequence_gap(
            raws,
            count=photos_per_folder,
            min_sequence_gap=min_sequence_gap,
            rng=rng,
        )
        for photo_index, raw_path in enumerate(selected, start=1):
            asset = RawAsset(raw_path=raw_path, xmp_path=raw_path.with_suffix(".xmp"))
            cohorts.append(
                VLMCohort(
                    cohort_id=(
                        f"folder-{folder_index:02d}-{_slug(folder.name)}-"
                        f"photo-{photo_index:02d}"
                    ),
                    scene_id=folder.name,
                    assets=(asset,),
                    human_labels=("unreviewed",),
                )
            )
    return cohorts


def _execute_blind_culling(
    cohorts: list[VLMCohort],
    prompt: str,
    model_specs: list[VLMModelSpec],
    run_dir: Path,
    config: dict[str, object],
    *,
    seed: int,
    timeout_seconds: float,
    max_attempts: int,
    progress: ProgressCallback,
) -> tuple[Path, Path]:
    _write_or_verify_json(run_dir / "blind-culling-config.json", config)
    _write_cohorts(run_dir / "blind-culling-cohorts.json", cohorts)
    photo_count = sum(len(cohort.assets) for cohort in cohorts)
    progress(
        f"Locked {photo_count} photo(s) in "
        f"{len(cohorts)} cohort(s); XMP is neither read nor written."
    )
    previews = _prepare_previews(cohorts, run_dir / "previews", progress)
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
    page, answer_key = write_blind_culling_review(
        run_dir,
        cohorts,
        model_specs,
        seed=seed,
    )
    return page, answer_key


def _sample_with_sequence_gap(
    paths: list[Path],
    *,
    count: int,
    min_sequence_gap: int,
    rng: random.Random,
) -> list[Path]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    selected: list[Path] = []
    selected_numbers: list[int] = []
    for path in shuffled:
        match = re.search(r"(\d+)$", path.stem)
        sequence_number = int(match.group(1)) if match else None
        if sequence_number is not None and any(
            abs(sequence_number - existing) < min_sequence_gap
            for existing in selected_numbers
        ):
            continue
        selected.append(path)
        if sequence_number is not None:
            selected_numbers.append(sequence_number)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"could not select {count} spaced photos from {paths[0].parent} "
        f"with minimum sequence gap {min_sequence_gap}"
    )


def build_frozen_semantic_cohorts(
    frozen_run: Path,
    *,
    batch_size: int,
) -> list[VLMCohort]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    grouped: dict[str, list[tuple[int, RawAsset]]] = defaultdict(list)
    manifest_path = frozen_run / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            decision = record.get("decision")
            vision_decision = (
                isinstance(decision, dict) and decision.get("source") == "vision"
            )
            vision_failure = str(record.get("error") or "").startswith(
                "vision scoring failed:"
            )
            if not vision_decision and not vision_failure:
                continue
            raw_path = Path(str(record["raw_path"]))
            if not raw_path.is_file():
                raise FileNotFoundError(f"frozen candidate is missing: {raw_path}")
            scene_id = str(record.get("scene_id") or "ungrouped")
            scene_index = int(record.get("scene_index") or 0)
            grouped[scene_id].append(
                (
                    scene_index,
                    RawAsset(raw_path=raw_path, xmp_path=raw_path.with_suffix(".xmp")),
                )
            )
    cohorts: list[VLMCohort] = []
    for scene_id in sorted(grouped):
        assets = [
            item[1]
            for item in sorted(grouped[scene_id], key=lambda item: (item[0], item[1].filename))
        ]
        for offset in range(0, len(assets), batch_size):
            chunk = assets[offset : offset + batch_size]
            cohorts.append(
                VLMCohort(
                    cohort_id=f"{scene_id}-blind-{offset // batch_size + 1:03d}",
                    scene_id=scene_id,
                    assets=tuple(chunk),
                    human_labels=tuple("unreviewed" for _ in chunk),
                )
            )
    if not cohorts:
        raise ValueError("frozen run contains no semantic candidates")
    return cohorts


def write_blind_culling_review(
    run_dir: Path,
    cohorts: list[VLMCohort],
    model_specs: list[VLMModelSpec],
    *,
    seed: int,
) -> tuple[Path, Path]:
    if len(model_specs) != 2:
        raise ValueError("blind review requires exactly two models")
    predictions = {
        spec.label: _load_predictions(run_dir / f"{_slug(spec.label)}.jsonl", cohorts)
        for spec in model_specs
    }
    rng = random.Random(seed)
    cards: list[str] = []
    assignments: dict[str, dict[str, object]] = {}
    for cohort in cohorts:
        for asset in cohort.assets:
            filename = asset.filename
            labels = [model_specs[0].label, model_specs[1].label]
            if rng.choice((False, True)):
                labels.reverse()
            variant_a = predictions[labels[0]].get(filename, _unavailable_decision())
            variant_b = predictions[labels[1]].get(filename, _unavailable_decision())
            scoreable = all(
                str(variant.get("bucket") or "unavailable") != "unavailable"
                for variant in (variant_a, variant_b)
            )
            assignments[filename] = {
                "A": labels[0],
                "B": labels[1],
                "scoreable": scoreable,
            }
            choice_html = (
                f"<div class='choices' data-file='{escape(filename)}' data-scene='{escape(cohort.scene_id)}'>"
                "<button type='button' data-choice='A' aria-pressed='false'>A is better</button>"
                "<button type='button' data-choice='B' aria-pressed='false'>B is better</button>"
                "<button type='button' data-choice='tie' aria-pressed='false'>Tie</button>"
                "<strong class='choice-status' aria-live='polite'></strong>"
                "</div>"
                if scoreable
                else "<div class='not-scoreable'><strong>Excluded from preference scoring: one variant did not return a valid decision.</strong></div>"
            )
            cards.append(
                "<article>"
                f"<h2>{escape(filename)} <span>{escape(cohort.scene_id)}</span></h2>"
                "<div class='comparison'>"
                f"<figure><img src='previews/{escape(filename)}.jpg'><figcaption>Frozen embedded preview</figcaption></figure>"
                + _decision_html("A", variant_a)
                + _decision_html("B", variant_b)
                + "</div>"
                + choice_html
                + "</article>"
            )
    answer_key_path = run_dir / "blind-culling-answer-key.json"
    _write_or_upgrade_answer_key(
        answer_key_path,
        {
            "schema_version": 2,
            "seed": seed,
            "models": [asdict(spec) for spec in model_specs],
            "assignments": assignments,
        },
    )
    page = run_dir / "blind-culling-review.html"
    page.write_text(_review_html("".join(cards)), encoding="utf-8")
    return page, answer_key_path


def score_blind_culling_choices(
    run_dir: Path,
    choices_path: Path,
) -> dict[str, object]:
    answer_key = json.loads(
        (run_dir / "blind-culling-answer-key.json").read_text(encoding="utf-8")
    )
    assignments = answer_key["assignments"]
    model_labels = [str(model["label"]) for model in answer_key["models"]]
    counts = {label: 0 for label in model_labels}
    counts["tie"] = 0
    reviewed = 0
    with choices_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = str(row.get("filename") or "")
            choice = str(row.get("choice") or "").strip()
            assignment = assignments.get(filename)
            if (
                not isinstance(assignment, dict)
                or not assignment.get("scoreable")
                or choice not in {"A", "B", "tie"}
            ):
                continue
            reviewed += 1
            if choice == "tie":
                counts["tie"] += 1
            else:
                counts[str(assignment[choice])] += 1
    payload: dict[str, object] = {
        "reviewed": reviewed,
        "counts": counts,
        "choices": str(choices_path),
    }
    (run_dir / "blind-culling-score.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Blind culling review result", "", f"Reviewed: {reviewed}", ""]
    lines.extend(f"- {label}: {count}" for label, count in counts.items())
    (run_dir / "blind-culling-score.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_predictions(
    path: Path,
    cohorts: list[VLMCohort],
) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                latest[str(record["cohort_id"])] = record
    predictions: dict[str, dict[str, object]] = {}
    for cohort in cohorts:
        record = latest.get(cohort.cohort_id)
        if record is None or record.get("status") != "success":
            for asset in cohort.assets:
                predictions[asset.filename] = _unavailable_decision()
            continue
        for decision in record.get("decisions", []):
            if isinstance(decision, dict) and decision.get("filename"):
                predictions[str(decision["filename"])] = decision
    return predictions


def _unavailable_decision() -> dict[str, object]:
    return {
        "bucket": "unavailable",
        "rating": "-",
        "label": None,
        "summary": "No valid structured decision was returned.",
    }


def _decision_html(name: str, decision: dict[str, object]) -> str:
    bucket = str(decision.get("bucket") or "unavailable")
    rating = str(decision.get("rating") if decision.get("rating") is not None else "-")
    label = str(decision.get("label") or "none")
    summary = str(decision.get("summary") or "No explanation returned.")
    return (
        f"<section class='decision'><h3>Variant {name}</h3>"
        f"<p class='bucket {escape(bucket)}'>{escape(bucket.upper())}</p>"
        f"<p>Rating {escape(rating)} · Label {escape(label)}</p>"
        f"<p>{escape(summary)}</p></section>"
    )


def _review_html(cards: str) -> str:
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Blind Gemma/Qwen culling review</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101010;color:#eee}body{max-width:1700px;margin:auto;padding:28px}
.note{color:#bbb}.toolbar{position:sticky;top:0;z-index:2;background:#101010ee;padding:12px 0;display:flex;gap:12px;align-items:center}
button{padding:9px 13px;border-radius:8px;border:1px solid #666;background:#292929;color:#eee;cursor:pointer}button.selected{border-color:#8ff5df;background:#087f6f;color:#fff;box-shadow:0 0 0 3px #2a9d8f66;font-weight:700}
article{background:#191919;border:1px solid #333;border-radius:14px;padding:16px;margin:18px 0}h2{font-size:1rem}h2 span{color:#999;font-weight:normal}
.comparison{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:14px}figure{margin:0}img{display:block;width:100%;height:420px;object-fit:contain;background:#080808;border-radius:8px}
.decision{border:1px solid #444;border-radius:10px;padding:14px}.bucket{font-weight:700}.pick{color:#52b788}.review{color:#e9c46a}.reject{color:#e76f51}.unavailable{color:#aaa}
.choices{display:flex;gap:10px;align-items:center;margin-top:12px}.not-scoreable{margin-top:12px;color:#e9c46a}@media(max-width:900px){.comparison{grid-template-columns:1fr}.choices{flex-wrap:wrap}img{height:auto}}
</style></head><body><h1>Blind culling-model review</h1>
<p class='note'>For each photograph, choose which hidden model gives the more appropriate pick/review/reject judgment. The answer key is not embedded in this page. No XMP was read or written.</p>
<div class='toolbar'><button id='download'>Download choices CSV</button><strong id='count'></strong></div>""" + cards + """
<script>const key='cull-sh-blind-culling-'+location.pathname;const saved={};
try{Object.assign(saved,JSON.parse(localStorage.getItem(key)||'{}'));}catch(_error){}
function persist(){try{localStorage.setItem(key,JSON.stringify(saved));}catch(_error){}}
function refresh(){let n=0;const groups=document.querySelectorAll('.choices');groups.forEach(group=>{const value=saved[group.dataset.file];group.querySelectorAll('button').forEach(button=>{const selected=button.dataset.choice===value;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));});const out=group.querySelector('.choice-status');out.textContent=value?('Selected: '+value):'';if(value)n++;});document.getElementById('count').textContent=n+' of '+groups.length+' reviewed';persist();}
document.querySelectorAll('.choices button').forEach(button=>button.addEventListener('click',()=>{const group=button.closest('.choices');saved[group.dataset.file]=button.dataset.choice;refresh();}));
document.getElementById('download').onclick=()=>{const rows=[['filename','scene_id','choice']];document.querySelectorAll('.choices').forEach(group=>rows.push([group.dataset.file,group.dataset.scene,saved[group.dataset.file]||'']));const csv=rows.map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(',')).join('\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));link.download='blind-culling-choices.csv';link.click();URL.revokeObjectURL(link.href);};refresh();</script></body></html>"""


def _write_cohorts(path: Path, cohorts: list[VLMCohort]) -> None:
    payload = [
        {
            "cohort_id": cohort.cohort_id,
            "scene_id": cohort.scene_id,
            "raw_paths": [str(asset.raw_path) for asset in cohort.assets],
            "filenames": [asset.filename for asset in cohort.assets],
        }
        for cohort in cohorts
    ]
    _write_or_verify_json(path, payload)


def _write_or_verify_json(path: Path, payload: object) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing locked artifact does not match: {path}")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_or_upgrade_answer_key(path: Path, payload: dict[str, object]) -> None:
    if not path.exists():
        _write_or_verify_json(path, payload)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == payload:
        return
    if existing.get("schema_version") != 1:
        raise ValueError(f"existing locked artifact does not match: {path}")
    existing_assignments = existing.get("assignments")
    updated_assignments = payload.get("assignments")
    if not isinstance(existing_assignments, dict) or not isinstance(updated_assignments, dict):
        raise ValueError(f"existing locked artifact does not match: {path}")
    expected_legacy = {
        filename: {"A": assignment["A"], "B": assignment["B"]}
        for filename, assignment in updated_assignments.items()
    }
    legacy_payload = dict(payload)
    legacy_payload["schema_version"] = 1
    legacy_payload["assignments"] = expected_legacy
    if existing != legacy_payload:
        raise ValueError(f"existing locked artifact does not match: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "model"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
