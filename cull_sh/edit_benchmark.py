from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import csv
import html
import json
import random
import shutil
import statistics
import time
from typing import Callable

from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.benchmark import load_human_labels
from cull_sh.extractors import build_default_extractor
from cull_sh.manifests import create_run_dir
from cull_sh.models import EditSuggestion
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.xmp import write_develop_sidecar


ProgressCallback = Callable[[str], None]
EDIT_FIELDS = ("exposure", "contrast", "highlights", "shadows", "vibrance")


@dataclass(frozen=True, slots=True)
class EditDatasetSpec:
    label: str
    photo_root: Path
    sample_size: int = 24


@dataclass(frozen=True, slots=True)
class EditModelSpec:
    label: str
    model: str
    temperature: float = 0.0
    think: bool | str | None = None


def run_edit_benchmark(
    datasets: list[EditDatasetSpec],
    model_specs: list[EditModelSpec],
    prompt: str,
    runs_root: Path,
    *,
    seed: int = 20260820,
    timeout_seconds: float = 600.0,
    max_attempts: int = 2,
    resume_run: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, object], Path]:
    """Run a deterministic, metadata-safe edit-suggestion comparison."""
    progress = progress or (lambda _message: None)
    if len(model_specs) < 2:
        raise ValueError("edit benchmark requires at least two models")
    if not datasets:
        raise ValueError("edit benchmark requires at least one dataset")

    run_dir = resume_run or create_run_dir(runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": "edit-suggestion-benchmark",
        "prompt": prompt,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "batch_size": 1,
        "crop_enabled": False,
        "metadata_writes": False,
        "datasets": [
            {
                "label": spec.label,
                "photo_root": str(spec.photo_root),
                "sample_size": spec.sample_size,
            }
            for spec in datasets
        ],
        "models": [asdict(spec) for spec in model_specs],
    }
    _write_or_verify_json(run_dir / "edit-benchmark-config.json", config)

    cohort_path = run_dir / "edit-cohort.json"
    if cohort_path.exists():
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    else:
        cohort = _build_cohort(datasets)
        cohort_path.write_text(
            json.dumps(cohort, indent=2, sort_keys=True), encoding="utf-8"
        )
    progress(
        f"Locked a {len(cohort)}-photo cohort; originals and their XMP sidecars are read-only."
    )

    preview_dir = run_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    extractor = build_default_extractor()
    for index, item in enumerate(cohort, start=1):
        preview_path = preview_dir / f"{item['id']}.jpg"
        if not preview_path.exists():
            preview_path.write_bytes(extractor.extract_preview_bytes(Path(item["raw_path"])))
        item["preview_path"] = str(preview_path)
        if index % 12 == 0 or index == len(cohort):
            progress(f"Prepared previews: {index}/{len(cohort)}")

    for spec in model_specs:
        output_path = run_dir / f"{_slug(spec.label)}.jsonl"
        completed = _completed_ids(output_path)
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model=spec.model,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            temperature=spec.temperature,
            think=spec.think,
        )
        pending = [item for item in cohort if item["id"] not in completed]
        progress(
            f"{spec.label}: {len(cohort) - len(pending)}/{len(cohort)} already complete; "
            f"{len(pending)} pending."
        )
        for item_index, item in enumerate(pending, start=1):
            asset = RawAsset(
                raw_path=Path(item["raw_path"]),
                xmp_path=Path(item["raw_path"]).with_suffix(".xmp"),
            )
            preview = PreviewImage(
                asset=asset,
                image_bytes=Path(item["preview_path"]).read_bytes(),
            )
            started = time.monotonic()
            record: dict[str, object] = {
                "id": item["id"],
                "dataset": item["dataset"],
                "filename": item["filename"],
                "model_label": spec.label,
                "model": spec.model,
            }
            try:
                suggestions = backend.suggest_edits(prompt, [preview], include_crop=False)
                if len(suggestions) != 1:
                    raise VisionBackendError(
                        f"backend returned {len(suggestions)} suggestions for one image"
                    )
                record["status"] = "ok"
                record["suggestion"] = asdict(suggestions[0])
            except (VisionBackendError, ValueError) as exc:
                record["status"] = "error"
                record["error"] = str(exc)
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            _append_jsonl(output_path, record)
            if item_index % 4 == 0 or item_index == len(pending):
                progress(
                    f"{spec.label}: {len(cohort) - len(pending) + item_index}/{len(cohort)}"
                )

    records_by_model = {
        spec.label: _read_jsonl(run_dir / f"{_slug(spec.label)}.jsonl")
        for spec in model_specs
    }
    payload = _analyze(cohort, model_specs, records_by_model, config)
    (run_dir / "edit-benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(run_dir / "edit-benchmark.md", payload)
    _write_blind_review(run_dir, cohort, model_specs, records_by_model, seed)
    return payload, run_dir


def stage_lightroom_blind_review(
    run_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Create disposable A/B RAW copies and XMP edits for Lightroom rendering."""
    progress = progress or (lambda _message: None)
    cohort = json.loads((run_dir / "edit-cohort.json").read_text(encoding="utf-8"))
    config = json.loads(
        (run_dir / "edit-benchmark-config.json").read_text(encoding="utf-8")
    )
    answer_key = json.loads(
        (run_dir / "blind-answer-key.json").read_text(encoding="utf-8")
    )
    model_records = {
        str(spec["label"]): {
            str(record["id"]): record
            for record in _read_jsonl(run_dir / f"{_slug(str(spec['label']))}.jsonl")
            if record.get("status") == "ok"
        }
        for spec in config["models"]
    }
    render_input = run_dir / "lightroom-render-input"
    render_input.mkdir(exist_ok=True)
    manifest: list[dict[str, object]] = []
    total = len(cohort) * 2
    completed = 0
    for item in cohort:
        item_id = str(item["id"])
        source = Path(item["raw_path"])
        for side in ("A", "B"):
            model_label = str(answer_key[item_id][side])
            record = model_records[model_label][item_id]
            suggestion = EditSuggestion(**record["suggestion"])
            target_name = f"{item_id}-{side}-{source.name}"
            target_raw = render_input / target_name
            target_xmp = target_raw.with_suffix(".xmp")
            if not target_raw.exists() or target_raw.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target_raw)
            write_develop_sidecar(target_xmp, suggestion)
            manifest.append(
                {
                    "pair_id": item_id,
                    "side": side,
                    "dataset": item["dataset"],
                    "source": str(source),
                    "render_raw": str(target_raw),
                    "render_xmp": str(target_xmp),
                    "expected_jpeg_stem": target_raw.stem,
                }
            )
            completed += 1
            if completed % 12 == 0 or completed == total:
                progress(f"Staged Lightroom variants: {completed}/{total}")
    (run_dir / "lightroom-render-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "lightroom-render-output").mkdir(exist_ok=True)
    return render_input


def write_blind_review_page(run_dir: Path, *, seed: int = 20260820) -> Path:
    """Write a private, local-only A/B review page for completed Lightroom renders."""
    cohort = json.loads((run_dir / "edit-cohort.json").read_text(encoding="utf-8"))
    render_output = run_dir / "lightroom-render-output"
    pairs: list[dict[str, str]] = []
    for item in cohort:
        item_id = str(item["id"])
        candidates = sorted(render_output.glob(f"{item_id}-*-*.jpg"))
        by_side = {
            path.name.split("-", 3)[2]: path
            for path in candidates
        }
        if set(by_side) != {"A", "B"}:
            raise ValueError(f"rendered A/B pair is incomplete for {item_id}")
        pairs.append(
            {
                "pair_id": item_id,
                "dataset": str(item["dataset"]),
                "filename": str(item["filename"]),
                "A": by_side["A"].relative_to(run_dir).as_posix(),
                "B": by_side["B"].relative_to(run_dir).as_posix(),
            }
        )
    random.Random(seed + 1).shuffle(pairs)
    encoded_pairs = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    storage_key = html.escape(f"cull-edit-review-{run_dir.name}", quote=True)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Blind edit review</title>
  <style>
    :root {{ color-scheme: dark; --bg:#111310; --panel:#1a1d18; --ink:#f5f3ea; --muted:#a7aa9f; --line:#343a31; --accent:#d9f99d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; display:flex; gap:18px; align-items:center; padding:14px 22px; background:rgba(17,19,16,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
    h1 {{ margin:0; font-size:18px; letter-spacing:.01em; }}
    .progress {{ flex:1; height:7px; overflow:hidden; border-radius:99px; background:#2a2e28; }}
    .progress span {{ display:block; height:100%; width:0; background:var(--accent); transition:width .2s; }}
    .count {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
    main {{ max-width:1680px; margin:auto; padding:20px 22px 28px; }}
    .meta {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }}
    .pair-title {{ font-size:16px; font-weight:650; }}
    .dataset {{ color:var(--muted); }}
    .images {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    figure {{ margin:0; overflow:hidden; border:1px solid var(--line); border-radius:12px; background:#090a09; }}
    figcaption {{ padding:9px 12px; font-size:16px; font-weight:700; }}
    figure img {{ display:block; width:100%; height:min(68vh,780px); object-fit:contain; background:#080908; cursor:zoom-in; }}
    .controls {{ display:grid; grid-template-columns:1fr auto 1fr; gap:10px; margin-top:16px; align-items:center; }}
    button {{ appearance:none; border:1px solid var(--line); border-radius:10px; padding:12px 16px; background:var(--panel); color:var(--ink); font:inherit; font-weight:650; cursor:pointer; }}
    button:hover {{ border-color:#687260; }}
    button.choice {{ font-size:16px; }}
    button.selected {{ color:#17200c; background:var(--accent); border-color:var(--accent); }}
    .secondary {{ display:flex; justify-content:center; gap:10px; margin-top:12px; }}
    .help {{ margin:14px 0 0; color:var(--muted); text-align:center; }}
    @media (max-width:850px) {{ .images {{ grid-template-columns:1fr; }} figure img {{ height:55vh; }} header {{ flex-wrap:wrap; }} }}
  </style>
</head>
<body>
  <header><h1>Blind edit review</h1><div class="progress"><span id="bar"></span></div><div class="count" id="count"></div></header>
  <main>
    <div class="meta"><div class="pair-title" id="title"></div><div class="dataset" id="dataset"></div></div>
    <div class="images">
      <figure><figcaption>A</figcaption><img id="imgA" alt="Edit variant A"></figure>
      <figure><figcaption>B</figcaption><img id="imgB" alt="Edit variant B"></figure>
    </div>
    <div class="controls">
      <button class="choice" id="pickA">Choose A <small>(A)</small></button>
      <button class="choice" id="tie">Tie <small>(T)</small></button>
      <button class="choice" id="pickB">Choose B <small>(B)</small></button>
    </div>
    <div class="secondary"><button id="prev">Previous</button><button id="next">Next</button><button id="export">Export choices CSV</button></div>
    <p class="help">Click either image for full resolution. Choices stay in this browser until you export them. Model names remain hidden.</p>
  </main>
  <script>
    const pairs = {encoded_pairs};
    const storageKey = "{storage_key}";
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let index = Math.max(0, pairs.findIndex(p => !saved[p.pair_id]));
    if (index < 0) index = 0;
    const el = id => document.getElementById(id);
    function render() {{
      const pair = pairs[index];
      el("title").textContent = `${{pair.pair_id}} · ${{pair.filename}}`;
      el("dataset").textContent = pair.dataset;
      el("imgA").src = pair.A; el("imgB").src = pair.B;
      el("imgA").onclick = () => window.open(pair.A, "_blank");
      el("imgB").onclick = () => window.open(pair.B, "_blank");
      for (const [id,value] of [["pickA","A"],["tie","Tie"],["pickB","B"]]) el(id).classList.toggle("selected", saved[pair.pair_id] === value);
      const done = Object.keys(saved).length;
      el("count").textContent = `${{done}} chosen · ${{index + 1}}/${{pairs.length}}`;
      el("bar").style.width = `${{100 * done / pairs.length}}%`;
      el("prev").disabled = index === 0; el("next").disabled = index === pairs.length - 1;
    }}
    function choose(value) {{
      saved[pairs[index].pair_id] = value;
      localStorage.setItem(storageKey, JSON.stringify(saved));
      if (index < pairs.length - 1) index++;
      render();
    }}
    el("pickA").onclick = () => choose("A"); el("tie").onclick = () => choose("Tie"); el("pickB").onclick = () => choose("B");
    el("prev").onclick = () => {{ index--; render(); }}; el("next").onclick = () => {{ index++; render(); }};
    el("export").onclick = () => {{
      const rows = [["pair_id","dataset","filename","preferred"], ...pairs.map(p => [p.pair_id,p.dataset,p.filename,saved[p.pair_id] || ""])];
      const csv = rows.map(r => r.map(v => `"${{String(v).replaceAll('"','""')}}"`).join(",")).join("\\n");
      const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], {{type:"text/csv"}})); link.download = "blind-review-choices.csv"; link.click(); URL.revokeObjectURL(link.href);
    }};
    document.addEventListener("keydown", event => {{
      if (event.key.toLowerCase() === "a") choose("A");
      else if (event.key.toLowerCase() === "b") choose("B");
      else if (event.key.toLowerCase() === "t") choose("Tie");
      else if (event.key === "ArrowLeft" && index > 0) {{ index--; render(); }}
      else if (event.key === "ArrowRight" && index < pairs.length - 1) {{ index++; render(); }}
    }});
    render();
  </script>
</body>
</html>
"""
    output = run_dir / "blind-review.html"
    output.write_text(page, encoding="utf-8")
    return output


def select_evenly_spaced(items: list[object], sample_size: int) -> list[object]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if len(items) < sample_size:
        raise ValueError(f"requested {sample_size} items from only {len(items)} candidates")
    return [items[int((index + 0.5) * len(items) / sample_size)] for index in range(sample_size)]


def _build_cohort(datasets: list[EditDatasetSpec]) -> list[dict[str, object]]:
    cohort: list[dict[str, object]] = []
    sequence = 1
    for dataset in datasets:
        picks = sorted(
            (
                label
                for label in load_human_labels(dataset.photo_root, include_subfolders=False)
                if label.label == "pick"
            ),
            key=lambda label: label.filename.casefold(),
        )
        selected = select_evenly_spaced(picks, dataset.sample_size)
        for label in selected:
            cohort.append(
                {
                    "id": f"edit-{sequence:03d}",
                    "dataset": dataset.label,
                    "filename": label.filename,
                    "raw_path": label.raw_path,
                    "selection": "human-pick filename-order quantile",
                }
            )
            sequence += 1
    return cohort


def _analyze(
    cohort: list[dict[str, object]],
    model_specs: list[EditModelSpec],
    records_by_model: dict[str, list[dict[str, object]]],
    config: dict[str, object],
) -> dict[str, object]:
    model_results: list[dict[str, object]] = []
    ok_maps: dict[str, dict[str, dict[str, object]]] = {}
    for spec in model_specs:
        records = records_by_model[spec.label]
        ok = {
            str(record["id"]): record
            for record in records
            if record.get("status") == "ok"
        }
        ok_maps[spec.label] = ok
        durations = [float(record["duration_seconds"]) for record in records]
        suggestions = [record["suggestion"] for record in ok.values()]
        vectors = [tuple(suggestion[field] for field in EDIT_FIELDS) for suggestion in suggestions]
        sliders = {
            field: _summarize_numbers([float(suggestion[field]) for suggestion in suggestions])
            for field in EDIT_FIELDS
        }
        model_results.append(
            {
                "label": spec.label,
                "model": spec.model,
                "covered": len(ok),
                "errors": len(records) - len(ok),
                "validity_rate": len(ok) / len(cohort) if cohort else 0.0,
                "mean_seconds": statistics.fmean(durations) if durations else None,
                "median_seconds": statistics.median(durations) if durations else None,
                "unique_vectors": len(set(vectors)),
                "duplicate_vector_rate": (
                    1.0 - len(set(vectors)) / len(vectors) if vectors else None
                ),
                "noops": sum(all(float(s[field]) == 0.0 for field in EDIT_FIELDS) for s in suggestions),
                "sliders": sliders,
            }
        )

    comparisons: list[dict[str, object]] = []
    for left_index, left in enumerate(model_specs):
        for right in model_specs[left_index + 1 :]:
            shared = sorted(set(ok_maps[left.label]) & set(ok_maps[right.label]))
            field_deltas = {
                field: statistics.fmean(
                    abs(
                        float(ok_maps[left.label][item_id]["suggestion"][field])
                        - float(ok_maps[right.label][item_id]["suggestion"][field])
                    )
                    for item_id in shared
                )
                if shared
                else None
                for field in EDIT_FIELDS
            }
            comparisons.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "shared_photos": len(shared),
                    "mean_absolute_slider_delta": field_deltas,
                }
            )
    return {
        "config": config,
        "cohort_photos": len(cohort),
        "models": model_results,
        "comparisons": comparisons,
        "interpretation": (
            "Validity, latency, and output diversity are diagnostic only. "
            "The winning edit model must be chosen from the blind rendered A/B review."
        ),
    }


def _summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "stddev": None, "min": None, "max": None, "unique": 0}
    return {
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "unique": len(set(values)),
    }


def _write_blind_review(
    run_dir: Path,
    cohort: list[dict[str, object]],
    model_specs: list[EditModelSpec],
    records_by_model: dict[str, list[dict[str, object]]],
    seed: int,
) -> None:
    if len(model_specs) != 2:
        return
    result_maps = {
        label: {
            str(record["id"]): record
            for record in records
            if record.get("status") == "ok"
        }
        for label, records in records_by_model.items()
    }
    rng = random.Random(seed)
    answer_key: dict[str, dict[str, str]] = {}
    rows: list[dict[str, object]] = []
    for item in cohort:
        item_id = str(item["id"])
        if any(item_id not in result_maps[spec.label] for spec in model_specs):
            continue
        labels = [spec.label for spec in model_specs]
        rng.shuffle(labels)
        answer_key[item_id] = {"A": labels[0], "B": labels[1]}
        row: dict[str, object] = {
            "pair_id": item_id,
            "dataset": item["dataset"],
            "filename": item["filename"],
            "preview_path": item["preview_path"],
        }
        for side, label in (("A", labels[0]), ("B", labels[1])):
            suggestion = result_maps[label][item_id]["suggestion"]
            for field in EDIT_FIELDS:
                row[f"{side}_{field}"] = suggestion[field]
            row[f"{side}_summary"] = suggestion["summary"]
        row.update({"preferred": "", "confidence_1_to_5": "", "notes": ""})
        rows.append(row)

    fieldnames = list(rows[0]) if rows else []
    with (run_dir / "blind-review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "blind-answer-key.json").write_text(
        json.dumps(answer_key, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Edit-suggestion model benchmark",
        "",
        f"Cohort: {payload['cohort_photos']} human-picked photos; batch size 1; no crop; no metadata writes.",
        "",
        "| Model | Valid | Mean sec/photo | Median | Unique vectors | Duplicate rate | No-ops |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in payload["models"]:
        duplicate_rate = model["duplicate_vector_rate"]
        lines.append(
            f"| {model['label']} | {model['covered']}/{payload['cohort_photos']} "
            f"({model['validity_rate']:.1%}) | {_fmt(model['mean_seconds'])} | "
            f"{_fmt(model['median_seconds'])} | {model['unique_vectors']} | "
            f"{_pct(duplicate_rate)} | {model['noops']} |"
        )
    lines.extend(
        [
            "",
            "These diagnostics do not establish edit quality. Use `blind-review.csv` after both variants are rendered through the same engine; keep `blind-answer-key.json` hidden until choices are locked.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _pct(value: object) -> str:
    return "-" if value is None else f"{float(value):.1%}"


def _write_or_verify_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"resume configuration differs from locked benchmark: {path}")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _completed_ids(path: Path) -> set[str]:
    return {str(record["id"]) for record in _read_jsonl(path)}


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-").lower()
