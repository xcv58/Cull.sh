from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from bisect import bisect_left
from bisect import bisect_right
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import random
import sqlite3
from statistics import fmean
from typing import Callable
import xml.etree.ElementTree as ET

from cull_sh.config import DEFAULT_EXTENSIONS
from cull_sh.extractors import build_default_extractor
from cull_sh.quality import analyze_local_quality
from cull_sh.quality import build_local_decision_trace
from cull_sh.quality import score_topiq_quality
from cull_sh.scanner import discover_raw_assets
from cull_sh.xmp import XMP_DM_NS
from cull_sh.xmp import XMP_NS


ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class HumanLabel:
    filename: str
    raw_path: str
    label: str
    rating: int | None
    pick: int | None
    good: bool | None
    color: str | None


@dataclass(slots=True)
class SignalResult:
    name: str
    display_name: str
    available: int
    picks: int
    rejects: int
    pick_mean: float
    reject_mean: float
    auc: float
    pairwise_accuracy: float | None
    pairwise_pairs: int
    top1_recall: float | None
    top3_recall: float | None
    comparable_bursts: int
    zero_false_reject_coverage: float
    zero_false_reject_rejects: int
    coverage_points: list[dict[str, float | int]]


FACET_SIGNALS = {
    "topiq_score": "TOPIQ NR",
    "aesthetic_iaa": "TOPIQ IAA",
    "face_quality_iqa": "TOPIQ NR-Face",
    "liqe_score": "LIQE",
    "aggregate": "Facet aggregate",
    "tech_sharpness": "Facet technical sharpness",
    "subject_sharpness": "BiRefNet subject sharpness",
}

CURRENT_SIGNALS = {
    "topiq_shadow_score": "TOPIQ NR (Cull.sh shadow)",
    "musiq_score": "MUSIQ",
    "nima_score": "NIMA",
    "blur_score": "Laplacian variance",
    "tenengrad_score": "Tenengrad",
    "cpbd_score": "CPBD",
    "brisque_score": "BRISQUE (inverted)",
    "local_rank_score": "Cull.sh local rank",
}


def run_internal_benchmark(
    photo_root: Path,
    facet_db: Path | None,
    run_dir: Path,
    *,
    include_subfolders: bool = False,
    shadow_run: Path | None = None,
    compute_current: bool = True,
    current_workers: int = 2,
    current_cache: Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    progress = progress or (lambda _message: None)
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = load_human_labels(photo_root, include_subfolders=include_subfolders)
    if not labels:
        raise ValueError(f"no RAW/XMP pairs with culling labels found under {photo_root}")

    progress(
        "Loaded human ground truth: "
        f"{sum(label.label == 'pick' for label in labels)} picks, "
        f"{sum(label.label == 'reject' for label in labels)} rejects, "
        f"{sum(label.label == 'neutral' for label in labels)} neutral."
    )

    records: dict[str, dict[str, object]] = {
        label.filename.casefold(): {
            **asdict(label),
            "signals": {},
            "burst_group_id": None,
            "date_taken": None,
        }
        for label in labels
    }

    if facet_db is not None:
        facet_rows = load_facet_signals(facet_db)
        matched = merge_source_records(records, facet_rows)
        progress(f"Matched {matched}/{len(records)} photos to cached Facet results.")

    if shadow_run is not None:
        shadow_rows = load_cull_run_signals(shadow_run)
        matched = merge_source_records(records, shadow_rows)
        progress(
            f"Matched {matched}/{len(records)} photos to frozen Cull.sh shadow-run results."
        )

    if compute_current:
        cache_path = current_cache or (
            run_dir.parent / "benchmark-cache" / f"{_safe_slug(photo_root.name)}-current.jsonl"
        )
        current_rows = compute_current_signals(
            photo_root,
            cache_path,
            workers=current_workers,
            include_subfolders=include_subfolders,
            progress=progress,
        )
        matched = merge_source_records(records, current_rows)
        progress(f"Matched {matched}/{len(records)} photos to current Cull.sh signals.")

    signal_specs = _available_signal_specs(records)
    results = [
        evaluate_signal(records, name, display_name, direction=direction)
        for name, display_name, direction in signal_specs
    ]
    results.sort(key=lambda result: result.auc, reverse=True)
    hard_gates = []
    if any(
        isinstance(record["signals"], dict)
        and record["signals"].get("local_gate_reject") is not None
        for record in records.values()
    ):
        hard_gates.append(
            evaluate_reject_gate(
                records,
                "local_gate_reject",
                "Current Cull.sh local quality gate",
            )
        )
    if any(
        isinstance(record["signals"], dict)
        and record["signals"].get("pipeline_reject") is not None
        for record in records.values()
    ):
        hard_gates.append(
            evaluate_reject_gate(
                records,
                "pipeline_reject",
                "Frozen Cull.sh final reject decision",
            )
        )
    comparisons = []
    available_signal_names = {name for name, _, _ in signal_specs}
    topiq_comparison_name = (
        "topiq_shadow_score"
        if "topiq_shadow_score" in available_signal_names
        else "topiq_score"
    )
    if {topiq_comparison_name, "musiq_score"}.issubset(available_signal_names):
        comparisons.append(
            paired_auc_delta_bootstrap(
                records,
                topiq_comparison_name,
                dict((name, display) for name, display, _ in signal_specs)[
                    topiq_comparison_name
                ],
                "musiq_score",
                "MUSIQ",
            )
        )

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "photo_root": str(photo_root),
        "facet_db": str(facet_db) if facet_db is not None else None,
        "shadow_run": str(shadow_run) if shadow_run is not None else None,
        "ground_truth": {
            "definition": (
                "pick = positive xmpDM:Pick or positive xmp:Rating; "
                "reject = negative pick or rating; neutral excluded from binary metrics"
            ),
            "total": len(labels),
            "picks": sum(label.label == "pick" for label in labels),
            "rejects": sum(label.label == "reject" for label in labels),
            "neutral": sum(label.label == "neutral" for label in labels),
        },
        "signals": [asdict(result) for result in results],
        "hard_gates": hard_gates,
        "comparisons": comparisons,
    }
    (run_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_photo_scores_csv(run_dir / "photo-scores.csv", records, signal_specs)
    write_benchmark_report(run_dir / "benchmark.md", payload)
    return payload


def load_human_labels(
    photo_root: Path,
    *,
    include_subfolders: bool = True,
) -> list[HumanLabel]:
    labels: list[HumanLabel] = []
    assets = discover_raw_assets(photo_root, DEFAULT_EXTENSIONS)
    if not include_subfolders:
        assets = [asset for asset in assets if asset.raw_path.parent == photo_root]
    for asset in assets:
        if not asset.xmp_path.exists():
            continue
        attributes: dict[str, str] = {}
        root = ET.parse(asset.xmp_path).getroot()
        for element in root.iter():
            attributes.update(element.attrib)

        rating = _parse_xmp_integer(attributes.get(f"{{{XMP_NS}}}Rating"))
        pick = _parse_xmp_integer(attributes.get(f"{{{XMP_DM_NS}}}Pick"))
        good = _parse_xmp_boolean(attributes.get(f"{{{XMP_DM_NS}}}good"))
        color = attributes.get(f"{{{XMP_NS}}}Label")
        if good is not None:
            # Lightroom Desktop Local mode stores its visible flag state in
            # xmpDM:good. Legacy Rating/Pick values can remain in the same
            # sidecar after a human changes the flag, so good is authoritative
            # whenever Lightroom has written it.
            label = "pick" if good else "reject"
        else:
            rejected = rating == -1 or pick == -1
            picked = (rating is not None and rating > 0) or (pick is not None and pick > 0)
            if rejected and picked:
                raise ValueError(f"contradictory pick/reject metadata in {asset.xmp_path}")
            label = "reject" if rejected else "pick" if picked else "neutral"
        labels.append(
            HumanLabel(
                filename=asset.filename,
                raw_path=str(asset.raw_path),
                label=label,
                rating=rating,
                pick=pick,
                good=good,
                color=color,
            )
        )
    return labels


def load_facet_signals(database_path: Path) -> list[dict[str, object]]:
    uri = f"file:{database_path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT filename, path, date_taken, burst_group_id,
                   topiq_score, aesthetic_iaa, face_quality_iqa, liqe_score,
                   aggregate, tech_sharpness, subject_sharpness
            FROM photos
            """
        ).fetchall()
    finally:
        connection.close()

    output: list[dict[str, object]] = []
    for row in rows:
        signals = {
            name: float(row[name])
            for name in FACET_SIGNALS
            if row[name] is not None and math.isfinite(float(row[name]))
        }
        output.append(
            {
                "filename": str(row["filename"]),
                "raw_path": str(row["path"]),
                "date_taken": row["date_taken"],
                "burst_group_id": row["burst_group_id"],
                "signals": signals,
            }
        )
    return output


def load_cull_run_signals(run_dir: Path) -> list[dict[str, object]]:
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cull.sh shadow run has no manifest: {manifest_path}")
    output: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
                continue
            signals: dict[str, object] = {}
            for name in CURRENT_SIGNALS:
                source_name = "topiq_score" if name == "topiq_shadow_score" else name
                value = record.get(source_name)
                if value is not None:
                    signals[name] = value
            local_trace = record.get("local_trace")
            if isinstance(local_trace, dict) and local_trace.get("quality_reject") is not None:
                signals["local_gate_reject"] = bool(local_trace["quality_reject"])
            decision = record.get("decision")
            if isinstance(decision, dict) and decision.get("bucket") is not None:
                signals["pipeline_reject"] = decision.get("bucket") == "reject"
            scene_id = record.get("scene_id")
            output.append(
                {
                    "filename": record["filename"],
                    "raw_path": record.get("raw_path"),
                    "date_taken": None,
                    "burst_group_id": f"cull:{scene_id}" if scene_id else None,
                    "signals": signals,
                }
            )
    return output


def merge_source_records(
    records: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
) -> int:
    matched = 0
    for row in rows:
        key = str(row["filename"]).casefold()
        target = records.get(key)
        if target is None:
            continue
        matched += 1
        target_signals = target["signals"]
        assert isinstance(target_signals, dict)
        source_signals = row.get("signals", {})
        if isinstance(source_signals, dict):
            target_signals.update(source_signals)
        for name in ("burst_group_id", "date_taken"):
            if row.get(name) is not None:
                target[name] = row[name]
    return matched


def compute_current_signals(
    photo_root: Path,
    cache_path: Path,
    *,
    workers: int = 2,
    include_subfolders: bool = True,
    progress: ProgressCallback | None = None,
) -> list[dict[str, object]]:
    progress = progress or (lambda _message: None)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    assets = discover_raw_assets(photo_root, DEFAULT_EXTENSIONS)
    if not include_subfolders:
        assets = [asset for asset in assets if asset.raw_path.parent == photo_root]
    cached = _load_current_cache(cache_path)
    completed: dict[str, dict[str, object]] = {}
    for asset in assets:
        record = cached.get(asset.filename.casefold())
        if record is not None and _cache_matches_asset(record, asset.raw_path):
            completed[asset.filename.casefold()] = record

    pending_assets = [
        asset for asset in assets if asset.filename.casefold() not in completed
    ]
    progress(
        f"Current Cull.sh scoring cache: {len(completed)} hit(s), "
        f"{len(pending_assets)} photo(s) pending."
    )
    if not pending_assets:
        return list(completed.values())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = build_default_extractor()
    pending_futures: dict[object, tuple[str, str, int, int]] = {}
    submitted = 0
    finished = len(completed)
    maximum_pending = max(workers * 2, 2)

    with cache_path.open("a", encoding="utf-8") as cache_handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for asset in pending_assets:
                preview_bytes = extractor.extract_preview_bytes(asset.raw_path)
                stat = asset.raw_path.stat()
                future = executor.submit(_score_current_preview, preview_bytes)
                pending_futures[future] = (
                    asset.filename,
                    str(asset.raw_path),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
                submitted += 1
                if len(pending_futures) >= maximum_pending:
                    finished += _collect_current_futures(
                        pending_futures,
                        completed,
                        cache_handle,
                        wait_for_all=False,
                    )
                    if finished % 10 == 0 or finished == len(assets):
                        progress(f"Scored current Cull.sh signals: {finished}/{len(assets)}")

            while pending_futures:
                finished += _collect_current_futures(
                    pending_futures,
                    completed,
                    cache_handle,
                    wait_for_all=False,
                )
                if finished % 10 == 0 or finished == len(assets):
                    progress(f"Scored current Cull.sh signals: {finished}/{len(assets)}")

    if submitted != len(pending_assets):
        raise RuntimeError("not all pending benchmark photos were submitted")
    return list(completed.values())


def _collect_current_futures(
    pending_futures: dict[object, tuple[str, str, int, int]],
    completed: dict[str, dict[str, object]],
    cache_handle,
    *,
    wait_for_all: bool,
) -> int:
    done, _ = wait(
        pending_futures,
        return_when=FIRST_COMPLETED if not wait_for_all else None,
    )
    for future in done:
        filename, raw_path, size, mtime_ns = pending_futures.pop(future)
        signals = future.result()
        record = {
            "filename": filename,
            "raw_path": raw_path,
            "raw_size": size,
            "raw_mtime_ns": mtime_ns,
            "signals": signals,
        }
        completed[filename.casefold()] = record
        cache_handle.write(json.dumps(record, sort_keys=True) + "\n")
        cache_handle.flush()
    return len(done)


def _score_current_preview(preview_bytes: bytes) -> dict[str, float | bool | None]:
    metrics = analyze_local_quality(
        preview_bytes,
        include_portrait_metrics=False,
        enable_learned_iqa=True,
        enable_brisque=True,
        enable_cpbd=True,
    )
    trace = build_local_decision_trace(
        metrics,
        min_blur_score=110.0,
        min_tenengrad_score=45.0,
        min_musiq_score=50.0,
        min_nima_score=4.6,
        max_brisque_score=55.0,
        min_cpbd_score=0.3,
        use_brisque_for_reject=False,
        use_cpbd_for_reject=False,
        local_reject_required_support_votes=2,
    )
    output: dict[str, float | bool | None] = {
        name: getattr(metrics, name)
        for name in CURRENT_SIGNALS
        if name != "topiq_shadow_score"
    }
    try:
        output["topiq_shadow_score"] = score_topiq_quality(preview_bytes)
    except Exception:
        output["topiq_shadow_score"] = None
    output["local_gate_reject"] = bool(trace["quality_reject"])
    return output


def evaluate_signal(
    records: dict[str, dict[str, object]],
    signal_name: str,
    display_name: str,
    *,
    direction: int = 1,
) -> SignalResult:
    labeled: list[tuple[str, float, str, object]] = []
    for record in records.values():
        label = str(record["label"])
        if label not in {"pick", "reject"}:
            continue
        signals = record["signals"]
        assert isinstance(signals, dict)
        raw_score = signals.get(signal_name)
        if raw_score is None:
            continue
        score = float(raw_score) * direction
        if not math.isfinite(score):
            continue
        labeled.append(
            (
                str(record["filename"]),
                score,
                label,
                record.get("burst_group_id"),
            )
        )

    picks = [score for _, score, label, _ in labeled if label == "pick"]
    rejects = [score for _, score, label, _ in labeled if label == "reject"]
    if not picks or not rejects:
        raise ValueError(f"signal {signal_name} lacks both pick and reject scores")

    pairwise_accuracy, pair_count, top1, top3, burst_count = _burst_metrics(labeled)
    coverage_points = _coverage_curve(labeled)
    zero_rejects = sum(score < min(picks) for score in rejects)
    return SignalResult(
        name=signal_name,
        display_name=display_name,
        available=len(labeled),
        picks=len(picks),
        rejects=len(rejects),
        pick_mean=fmean(picks) * direction,
        reject_mean=fmean(rejects) * direction,
        auc=_auc(picks, rejects),
        pairwise_accuracy=pairwise_accuracy,
        pairwise_pairs=pair_count,
        top1_recall=top1,
        top3_recall=top3,
        comparable_bursts=burst_count,
        zero_false_reject_coverage=zero_rejects / len(labeled),
        zero_false_reject_rejects=zero_rejects,
        coverage_points=coverage_points,
    )


def evaluate_reject_gate(
    records: dict[str, dict[str, object]],
    signal_name: str,
    display_name: str,
) -> dict[str, float | int | str]:
    picks = 0
    rejects = 0
    predicted_rejects = 0
    false_rejects = 0
    true_rejects = 0
    for record in records.values():
        label = str(record["label"])
        if label not in {"pick", "reject"}:
            continue
        signals = record["signals"]
        assert isinstance(signals, dict)
        prediction = signals.get(signal_name)
        if prediction is None:
            continue
        picks += int(label == "pick")
        rejects += int(label == "reject")
        if bool(prediction):
            predicted_rejects += 1
            false_rejects += int(label == "pick")
            true_rejects += int(label == "reject")

    total = picks + rejects
    return {
        "name": signal_name,
        "display_name": display_name,
        "available": total,
        "picks": picks,
        "rejects": rejects,
        "auto_rejected": predicted_rejects,
        "true_rejects": true_rejects,
        "false_rejects": false_rejects,
        "coverage": predicted_rejects / total if total else 0.0,
        "reject_precision": true_rejects / predicted_rejects if predicted_rejects else 0.0,
        "false_reject_rate": false_rejects / picks if picks else 0.0,
        "reject_recall": true_rejects / rejects if rejects else 0.0,
    }


def paired_auc_delta_bootstrap(
    records: dict[str, dict[str, object]],
    left_name: str,
    left_display_name: str,
    right_name: str,
    right_display_name: str,
    *,
    iterations: int = 2000,
    seed: int = 20260812,
) -> dict[str, float | int | str]:
    picks: list[tuple[float, float]] = []
    rejects: list[tuple[float, float]] = []
    for record in records.values():
        label = str(record["label"])
        if label not in {"pick", "reject"}:
            continue
        signals = record["signals"]
        assert isinstance(signals, dict)
        left = signals.get(left_name)
        right = signals.get(right_name)
        if left is None or right is None:
            continue
        pair = (float(left), float(right))
        (picks if label == "pick" else rejects).append(pair)
    if not picks or not rejects:
        raise ValueError("paired AUC comparison requires picks and rejects")

    observed_left = _fast_auc([pair[0] for pair in picks], [pair[0] for pair in rejects])
    observed_right = _fast_auc([pair[1] for pair in picks], [pair[1] for pair in rejects])
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled_picks = [rng.choice(picks) for _ in picks]
        sampled_rejects = [rng.choice(rejects) for _ in rejects]
        left_auc = _fast_auc(
            [pair[0] for pair in sampled_picks],
            [pair[0] for pair in sampled_rejects],
        )
        right_auc = _fast_auc(
            [pair[1] for pair in sampled_picks],
            [pair[1] for pair in sampled_rejects],
        )
        deltas.append(left_auc - right_auc)
    deltas.sort()
    low_index = max(0, math.floor(iterations * 0.025) - 1)
    high_index = min(iterations - 1, math.ceil(iterations * 0.975) - 1)
    return {
        "left_name": left_name,
        "left_display_name": left_display_name,
        "right_name": right_name,
        "right_display_name": right_display_name,
        "left_auc": observed_left,
        "right_auc": observed_right,
        "auc_delta": observed_left - observed_right,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "delta_ci95_low": deltas[low_index],
        "delta_ci95_high": deltas[high_index],
    }


def _auc(picks: list[float], rejects: list[float]) -> float:
    correct = 0.0
    for pick_score in picks:
        for reject_score in rejects:
            if pick_score > reject_score:
                correct += 1.0
            elif pick_score == reject_score:
                correct += 0.5
    return correct / (len(picks) * len(rejects))


def _fast_auc(picks: list[float], rejects: list[float]) -> float:
    ordered_rejects = sorted(rejects)
    correct = 0.0
    for pick_score in picks:
        lower = bisect_left(ordered_rejects, pick_score)
        upper = bisect_right(ordered_rejects, pick_score)
        correct += lower + ((upper - lower) * 0.5)
    return correct / (len(picks) * len(rejects))


def _burst_metrics(
    labeled: list[tuple[str, float, str, object]],
) -> tuple[float | None, int, float | None, float | None, int]:
    groups: dict[object, list[tuple[str, float, str]]] = {}
    for filename, score, label, group_id in labeled:
        if group_id is None:
            continue
        groups.setdefault(group_id, []).append((filename, score, label))

    correct_pairs = 0.0
    pair_count = 0
    top1_hits = 0
    top3_hits = 0
    comparable_groups = 0
    for rows in groups.values():
        picks = [row for row in rows if row[2] == "pick"]
        rejects = [row for row in rows if row[2] == "reject"]
        if not picks or not rejects:
            continue
        comparable_groups += 1
        for _, pick_score, _ in picks:
            for _, reject_score, _ in rejects:
                pair_count += 1
                if pick_score > reject_score:
                    correct_pairs += 1.0
                elif pick_score == reject_score:
                    correct_pairs += 0.5

        ranked = sorted(rows, key=lambda row: (-row[1], row[0].casefold()))
        top1_hits += int(ranked[0][2] == "pick")
        top3_hits += int(any(row[2] == "pick" for row in ranked[:3]))

    if not comparable_groups:
        return None, pair_count, None, None, 0
    pairwise = correct_pairs / pair_count if pair_count else None
    return (
        pairwise,
        pair_count,
        top1_hits / comparable_groups,
        top3_hits / comparable_groups,
        comparable_groups,
    )


def _coverage_curve(
    labeled: list[tuple[str, float, str, object]],
) -> list[dict[str, float | int]]:
    ranked = sorted(labeled, key=lambda row: (row[1], row[0].casefold()))
    total_picks = sum(label == "pick" for _, _, label, _ in ranked)
    output: list[dict[str, float | int]] = []
    for requested_percent in (5, 10, 20, 30, 40, 50):
        count = max(1, math.floor(len(ranked) * requested_percent / 100))
        rejected_rows = ranked[:count]
        false_rejects = sum(label == "pick" for _, _, label, _ in rejected_rows)
        true_rejects = count - false_rejects
        output.append(
            {
                "requested_coverage_percent": requested_percent,
                "actual_coverage": count / len(ranked),
                "auto_rejected": count,
                "true_rejects": true_rejects,
                "false_rejects": false_rejects,
                "false_reject_rate": false_rejects / total_picks,
                "reject_precision": true_rejects / count,
            }
        )
    return output


def write_photo_scores_csv(
    path: Path,
    records: dict[str, dict[str, object]],
    signal_specs: list[tuple[str, str, int]],
) -> None:
    signal_names = [name for name, _, _ in signal_specs]
    fieldnames = [
        "filename",
        "label",
        "rating",
        "pick",
        "color",
        "burst_group_id",
        "date_taken",
        *signal_names,
        "local_gate_reject",
        "pipeline_reject",
        "raw_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records.values(), key=lambda row: str(row["filename"]).casefold()):
            signals = record["signals"]
            assert isinstance(signals, dict)
            row = {name: record.get(name) for name in fieldnames}
            for signal_name in signal_names:
                row[signal_name] = signals.get(signal_name)
            row["local_gate_reject"] = signals.get("local_gate_reject")
            row["pipeline_reject"] = signals.get("pipeline_reject")
            writer.writerow(row)


def write_benchmark_report(path: Path, payload: dict[str, object]) -> None:
    truth = payload["ground_truth"]
    signals = payload["signals"]
    hard_gates = payload.get("hard_gates", [])
    comparisons = payload.get("comparisons", [])
    assert isinstance(truth, dict)
    assert isinstance(signals, list)
    assert isinstance(hard_gates, list)
    assert isinstance(comparisons, list)
    photo_root = Path(str(payload["photo_root"]))
    lines = [
        f"# {photo_root.name} internal culling benchmark",
        "",
        f"Generated: {payload['created_at']}",
        "",
        "## Ground truth",
        "",
        (
            f"Human-reviewed photos: {truth['total']} total; {truth['picks']} explicit picks, "
            f"{truth['rejects']} explicit rejects, and {truth['neutral']} neutral/unflagged. "
            "Neutral photos are preserved in the CSV but excluded from binary metrics."
        ),
        "",
        "## Model comparison",
        "",
        "| Signal | AUC | Burst pairwise | Top-1 | Top-3 | Zero-FR coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for signal in signals:
        assert isinstance(signal, dict)
        lines.append(
            "| "
            + str(signal["display_name"])
            + f" | {float(signal['auc']):.3f}"
            + f" | {_format_optional_percent(signal['pairwise_accuracy'])}"
            + f" | {_format_optional_percent(signal['top1_recall'])}"
            + f" | {_format_optional_percent(signal['top3_recall'])}"
            + f" | {float(signal['zero_false_reject_coverage']):.1%} "
            + f"({signal['zero_false_reject_rejects']}) |"
        )

    lines.extend(
        [
            "",
            "AUC measures whether a human pick scores above a human reject across all explicit labels. "
            "Burst metrics use available Facet burst or frozen Cull.sh scene groups containing at "
            "least one explicit pick and reject. "
            "Zero-FR coverage is the share of explicitly labeled photos that could be rejected below the "
            "lowest-scoring human pick on this dataset; it is descriptive, not a production threshold.",
            "",
            "## Auto-reject safety",
            "",
            "False-reject rate is the fraction of all human picks discarded when the bottom-scoring "
            "portion of explicitly labeled photos is auto-rejected.",
            "",
        ]
    )
    if comparisons:
        lines.extend(["## Direct model comparison", ""])
        for comparison in comparisons:
            assert isinstance(comparison, dict)
            lines.append(
                f"{comparison['left_display_name']} minus {comparison['right_display_name']} "
                f"AUC: {float(comparison['auc_delta']):+.3f}; paired stratified bootstrap "
                f"95% interval {float(comparison['delta_ci95_low']):+.3f} to "
                f"{float(comparison['delta_ci95_high']):+.3f} "
                f"({comparison['bootstrap_iterations']} resamples)."
            )
        lines.extend(
            [
                "",
                "An interval crossing zero means this one folder does not establish a reliable "
                "global AUC difference, even when the observed score is higher.",
                "",
            ]
        )
    if hard_gates:
        lines.extend(
            [
                "## Current hard-gate policy",
                "",
                "| Gate | Coverage | Reject precision | False-reject rate | False rejects | Reject recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for gate in hard_gates:
            assert isinstance(gate, dict)
            lines.append(
                f"| {gate['display_name']}"
                f" | {float(gate['coverage']):.1%}"
                f" | {float(gate['reject_precision']):.1%}"
                f" | {float(gate['false_reject_rate']):.1%}"
                f" | {gate['false_rejects']}"
                f" | {float(gate['reject_recall']):.1%} |"
            )
        lines.append("")
    for signal in signals:
        assert isinstance(signal, dict)
        lines.extend(
            [
                f"### {signal['display_name']}",
                "",
                "| Coverage | Reject precision | False-reject rate | False rejects |",
                "|---:|---:|---:|---:|",
            ]
        )
        points = signal["coverage_points"]
        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, dict)
            lines.append(
                f"| {float(point['actual_coverage']):.1%}"
                f" | {float(point['reject_precision']):.1%}"
                f" | {float(point['false_reject_rate']):.1%}"
                f" | {point['false_rejects']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _available_signal_specs(
    records: dict[str, dict[str, object]],
) -> list[tuple[str, str, int]]:
    specs: list[tuple[str, str, int]] = []
    for name, display_name in {**FACET_SIGNALS, **CURRENT_SIGNALS}.items():
        direction = -1 if name == "brisque_score" else 1
        labels_with_values = {
            str(record["label"])
            for record in records.values()
            if isinstance(record["signals"], dict)
            and record["signals"].get(name) is not None
            and record["label"] in {"pick", "reject"}
        }
        if labels_with_values == {"pick", "reject"}:
            specs.append((name, display_name, direction))
    return specs


def _load_current_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        return {}
    output: dict[str, dict[str, object]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("filename"), str):
                output[str(record["filename"]).casefold()] = record
    return output


def _cache_matches_asset(record: dict[str, object], raw_path: Path) -> bool:
    try:
        stat = raw_path.stat()
    except OSError:
        return False
    signals = record.get("signals")
    return (
        record.get("raw_path") == str(raw_path)
        and record.get("raw_size") == stat.st_size
        and record.get("raw_mtime_ns") == stat.st_mtime_ns
        and isinstance(signals, dict)
        and all(name in signals for name in CURRENT_SIGNALS)
    )


def _parse_xmp_integer(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_xmp_boolean(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _safe_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "photos"


def _format_optional_percent(value: object) -> str:
    return "-" if value is None else f"{float(value):.1%}"
