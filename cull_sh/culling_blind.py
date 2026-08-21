from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import random
from typing import Callable

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
    _write_or_verify_json(run_dir / "blind-culling-config.json", config)
    _write_cohorts(run_dir / "blind-culling-cohorts.json", cohorts)
    photo_count = sum(len(cohort.assets) for cohort in cohorts)
    progress(
        f"Locked {photo_count} frozen semantic candidate(s) in "
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
    assignments: dict[str, dict[str, str]] = {}
    for cohort in cohorts:
        for asset in cohort.assets:
            filename = asset.filename
            labels = [model_specs[0].label, model_specs[1].label]
            if rng.choice((False, True)):
                labels.reverse()
            assignments[filename] = {"A": labels[0], "B": labels[1]}
            variant_a = predictions[labels[0]].get(filename, _unavailable_decision())
            variant_b = predictions[labels[1]].get(filename, _unavailable_decision())
            cards.append(
                "<article>"
                f"<h2>{escape(filename)} <span>{escape(cohort.scene_id)}</span></h2>"
                "<div class='comparison'>"
                f"<figure><img src='previews/{escape(filename)}.jpg'><figcaption>Frozen embedded preview</figcaption></figure>"
                + _decision_html("A", variant_a)
                + _decision_html("B", variant_b)
                + "</div>"
                f"<div class='choices' data-file='{escape(filename)}' data-scene='{escape(cohort.scene_id)}'>"
                f"<button data-choice='A'>A is better</button><button data-choice='B'>B is better</button><button data-choice='tie'>Tie</button><strong id='choice-{escape(filename)}'></strong>"
                "</div></article>"
            )
    answer_key_path = run_dir / "blind-culling-answer-key.json"
    _write_or_verify_json(
        answer_key_path,
        {
            "schema_version": 1,
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
            if filename not in assignments or choice not in {"A", "B", "tie"}:
                continue
            reviewed += 1
            if choice == "tie":
                counts["tie"] += 1
            else:
                counts[str(assignments[filename][choice])] += 1
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
button{padding:9px 13px;border-radius:8px;border:1px solid #555;background:#292929;color:#eee;cursor:pointer}button.selected{border-color:#2a9d8f;background:#174b45}
article{background:#191919;border:1px solid #333;border-radius:14px;padding:16px;margin:18px 0}h2{font-size:1rem}h2 span{color:#999;font-weight:normal}
.comparison{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:14px}figure{margin:0}img{display:block;width:100%;height:420px;object-fit:contain;background:#080808;border-radius:8px}
.decision{border:1px solid #444;border-radius:10px;padding:14px}.bucket{font-weight:700}.pick{color:#52b788}.review{color:#e9c46a}.reject{color:#e76f51}.unavailable{color:#aaa}
.choices{display:flex;gap:10px;align-items:center;margin-top:12px}@media(max-width:900px){.comparison{grid-template-columns:1fr}.choices{flex-wrap:wrap}img{height:auto}}
</style></head><body><h1>Blind culling-model review</h1>
<p class='note'>For each photograph, choose which hidden model gives the more appropriate pick/review/reject judgment. The answer key is not embedded in this page. No XMP was read or written.</p>
<div class='toolbar'><button id='download'>Download choices CSV</button><strong id='count'></strong></div>""" + cards + """
<script>const key='cull-sh-blind-culling-'+location.pathname;const saved=JSON.parse(localStorage.getItem(key)||'{}');
function refresh(){let n=0;document.querySelectorAll('.choices').forEach(group=>{const value=saved[group.dataset.file];group.querySelectorAll('button').forEach(button=>button.classList.toggle('selected',button.dataset.choice===value));const out=document.getElementById('choice-'+CSS.escape(group.dataset.file));out.textContent=value?('Selected: '+value):'';if(value)n++;});document.getElementById('count').textContent=n+' reviewed';localStorage.setItem(key,JSON.stringify(saved));}
document.querySelectorAll('.choices button').forEach(button=>button.onclick=()=>{const group=button.closest('.choices');saved[group.dataset.file]=button.dataset.choice;refresh();});
document.getElementById('download').onclick=()=>{const rows=[['filename','scene_id','choice']];document.querySelectorAll('.choices').forEach(group=>rows.push([group.dataset.file,group.dataset.scene,saved[group.dataset.file]||'']));const csv=rows.map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(',')).join('\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));link.download='blind-culling-choices.csv';link.click();URL.revokeObjectURL(link.href);};refresh();</script></body></html>"""


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


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "model"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
