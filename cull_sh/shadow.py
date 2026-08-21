from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re

from cull_sh.models import DecisionBucket
from cull_sh.models import WorkItem


@dataclass(slots=True)
class TopiqShadowSummary:
    scored: int
    low_kept: int
    high_rejected: int
    scene_winner_disagreements: int
    review_items: int
    json_path: Path
    html_path: Path


def write_topiq_shadow_report(
    items: list[WorkItem],
    run_dir: Path,
    *,
    low_percentile: float = 0.2,
    high_percentile: float = 0.8,
    max_items: int = 80,
    rank_weight: float = 0.0,
) -> TopiqShadowSummary | None:
    if not 0.0 <= low_percentile < high_percentile <= 1.0:
        raise ValueError("TOPIQ shadow percentiles must satisfy 0 <= low < high <= 1")
    if max_items < 1:
        raise ValueError("TOPIQ shadow max_items must be at least 1")
    if not 0.0 <= rank_weight <= 1.0:
        raise ValueError("TOPIQ rank weight must be between 0 and 1")

    scoreable = [
        item
        for item in items
        if item.metrics is not None
        and item.metrics.topiq_score is not None
        and item.decision is not None
        and not item.asset.mirrors_paired_raw
    ]
    if not scoreable:
        return None

    ordered = sorted(
        scoreable,
        key=lambda item: (float(item.metrics.topiq_score), item.filename.casefold()),
    )
    percentiles = {
        str(item.asset.raw_path): (index / (len(ordered) - 1) if len(ordered) > 1 else 0.5)
        for index, item in enumerate(ordered)
    }
    low_kept = [
        item
        for item in ordered
        if percentiles[str(item.asset.raw_path)] <= low_percentile
        and item.decision is not None
        and item.decision.bucket != DecisionBucket.REJECT
    ]
    high_rejected = [
        item
        for item in reversed(ordered)
        if percentiles[str(item.asset.raw_path)] >= high_percentile
        and item.decision is not None
        and item.decision.bucket == DecisionBucket.REJECT
    ]
    scene_disagreements = _find_scene_winner_disagreements(scoreable)

    reasons: dict[str, list[str]] = defaultdict(list)
    review_order: list[WorkItem] = []

    def add_review_item(item: WorkItem, reason: str) -> None:
        key = str(item.asset.raw_path)
        if reason not in reasons[key]:
            reasons[key].append(reason)
        if all(existing.asset.raw_path != item.asset.raw_path for existing in review_order):
            review_order.append(item)

    review_categories: list[list[tuple[WorkItem, str]]] = [
        [
            (item, "TOPIQ bottom band, current pipeline kept")
            for item in low_kept
        ],
        [
            (item, "TOPIQ top band, current pipeline rejected")
            for item in high_rejected
        ],
    ]
    scene_review_items: list[tuple[WorkItem, str]] = []
    for disagreement in scene_disagreements:
        scene_review_items.extend(
            [
                (disagreement["topiq_winner"], "TOPIQ scene winner"),
                (
                    disagreement["local_winner"],
                    "Current local-rank scene winner",
                ),
            ]
        )
    review_categories.append(scene_review_items)

    for index in range(max((len(category) for category in review_categories), default=0)):
        for category in review_categories:
            if index < len(category):
                add_review_item(*category[index])

    review_items = review_order[:max_items]
    thumbnail_dir = run_dir / "topiq-shadow-thumbnails"
    thumbnail_paths = _write_thumbnails(review_items, thumbnail_dir)
    review_payload = [
        _item_payload(
            item,
            percentile=percentiles[str(item.asset.raw_path)],
            reasons=reasons[str(item.asset.raw_path)],
            thumbnail_path=thumbnail_paths.get(str(item.asset.raw_path)),
            run_dir=run_dir,
        )
        for item in review_items
    ]
    payload = {
        "schema_version": 1,
        "mode": "ranking_review" if rank_weight else "shadow_only",
        "decision_influence": "ranking_only" if rank_weight else "none",
        "rank_weight": rank_weight,
        "note": (
            f"TOPIQ contributes {rank_weight:.0%} to folder-relative candidate ranking "
            "but does not participate in the hard quality-reject gate."
            if rank_weight
            else (
                "TOPIQ did not change flags, ratings, labels, duplicate suppression, "
                "scene ranking, or vision decisions."
            )
        ),
        "thresholds": {
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "max_review_items": max_items,
        },
        "counts": {
            "scored": len(scoreable),
            "low_kept": len(low_kept),
            "high_rejected": len(high_rejected),
            "scene_winner_disagreements": len(scene_disagreements),
            "review_items_available": len(review_order),
            "review_items_written": len(review_items),
        },
        "review_items": review_payload,
        "scene_winner_disagreements": [
            {
                "scene_id": disagreement["scene_id"],
                "topiq_winner": disagreement["topiq_winner"].filename,
                "local_winner": disagreement["local_winner"].filename,
                "topiq_winner_score": disagreement["topiq_winner"].metrics.topiq_score,
                "local_winner_topiq_score": disagreement["local_winner"].metrics.topiq_score,
                "topiq_winner_local_rank": disagreement[
                    "topiq_winner"
                ].metrics.local_rank_score,
                "local_winner_local_rank": disagreement[
                    "local_winner"
                ].metrics.local_rank_score,
            }
            for disagreement in scene_disagreements
        ],
    }
    json_path = run_dir / "topiq-shadow.json"
    html_path = run_dir / "topiq-shadow.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return TopiqShadowSummary(
        scored=len(scoreable),
        low_kept=len(low_kept),
        high_rejected=len(high_rejected),
        scene_winner_disagreements=len(scene_disagreements),
        review_items=len(review_items),
        json_path=json_path,
        html_path=html_path,
    )


def _find_scene_winner_disagreements(
    items: list[WorkItem],
) -> list[dict[str, object]]:
    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[item.scene_id or "ungrouped"].append(item)

    output: list[dict[str, object]] = []
    for scene_id in sorted(grouped):
        scene_items = grouped[scene_id]
        if len(scene_items) < 2:
            continue
        topiq_winner = max(
            scene_items,
            key=lambda item: (
                item.metrics.topiq_score or 0.0,
                item.filename.casefold(),
            ),
        )
        local_winner = max(
            scene_items,
            key=lambda item: (
                item.metrics.local_rank_score or 0.0,
                item.filename.casefold(),
            ),
        )
        if topiq_winner.asset.raw_path == local_winner.asset.raw_path:
            continue
        output.append(
            {
                "scene_id": scene_id,
                "topiq_winner": topiq_winner,
                "local_winner": local_winner,
            }
        )
    return output


def _item_payload(
    item: WorkItem,
    *,
    percentile: float,
    reasons: list[str],
    thumbnail_path: Path | None,
    run_dir: Path,
) -> dict[str, object]:
    assert item.metrics is not None
    assert item.decision is not None
    return {
        "filename": item.filename,
        "raw_path": str(item.asset.raw_path),
        "scene_id": item.scene_id,
        "decision_bucket": item.decision.bucket.value,
        "decision_source": item.decision.source.value,
        "decision_summary": item.decision.summary,
        "topiq_score": item.metrics.topiq_score,
        "topiq_percentile": percentile,
        "musiq_score": item.metrics.musiq_score,
        "nima_score": item.metrics.nima_score,
        "local_rank_score": item.metrics.local_rank_score,
        "reasons": reasons,
        "thumbnail": (
            thumbnail_path.relative_to(run_dir).as_posix()
            if thumbnail_path is not None
            else None
        ),
    }


def _write_thumbnails(
    items: list[WorkItem],
    thumbnail_dir: Path,
) -> dict[str, Path]:
    import cv2
    import numpy as np

    output: dict[str, Path] = {}
    for index, item in enumerate(items, start=1):
        if item.preview is None:
            continue
        array = np.frombuffer(item.preview.image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        long_edge = max(height, width)
        if long_edge > 520:
            scale = 520 / long_edge
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            continue
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", item.asset.raw_path.stem)
        path = thumbnail_dir / f"{index:03d}-{safe_stem}.jpg"
        path.write_bytes(encoded.tobytes())
        output[str(item.asset.raw_path)] = path
    return output


def _render_html(payload: dict[str, object]) -> str:
    counts = payload["counts"]
    thresholds = payload["thresholds"]
    items = payload["review_items"]
    assert isinstance(counts, dict)
    assert isinstance(thresholds, dict)
    assert isinstance(items, list)
    cards: list[str] = []
    for raw_item in items:
        assert isinstance(raw_item, dict)
        thumbnail = raw_item.get("thumbnail")
        image = (
            f'<img src="{escape(str(thumbnail), quote=True)}" alt="{escape(str(raw_item["filename"]), quote=True)}">'
            if thumbnail
            else '<div class="missing">No thumbnail</div>'
        )
        reason_values = raw_item.get("reasons", [])
        assert isinstance(reason_values, list)
        reasons = "".join(f"<li>{escape(str(reason))}</li>" for reason in reason_values)
        cards.append(
            "<article class=\"card\">"
            f"{image}"
            "<div class=\"body\">"
            f"<h2>{escape(str(raw_item['filename']))}</h2>"
            f"<p class=\"decision\">Current: {escape(str(raw_item['decision_bucket']))} "
            f"({escape(str(raw_item['decision_source']))})</p>"
            f"<p>TOPIQ <strong>{float(raw_item['topiq_score']):.3f}</strong> · "
            f"percentile {float(raw_item['topiq_percentile']):.1%} · "
            f"scene {escape(str(raw_item.get('scene_id') or 'ungrouped'))}</p>"
            f"<ul>{reasons}</ul>"
            f"<p class=\"summary\">{escape(str(raw_item.get('decision_summary') or ''))}</p>"
            "</div></article>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOPIQ ranking review</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; background:#111; color:#eee; }}
body {{ margin:0; padding:28px; max-width:1500px; margin-inline:auto; }}
h1 {{ margin:0 0 8px; }} .notice {{ padding:14px 16px; border:1px solid #66551d; background:#29230f; border-radius:10px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 24px; }}
.stat {{ background:#1d1d1d; padding:10px 14px; border-radius:9px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); gap:16px; }}
.card {{ background:#1b1b1b; border:1px solid #333; border-radius:12px; overflow:hidden; }}
.card img,.missing {{ display:block; width:100%; aspect-ratio:3/2; object-fit:contain; background:#080808; }}
.missing {{ display:grid; place-items:center; color:#888; }} .body {{ padding:14px; }}
h2 {{ font-size:1rem; margin:0 0 8px; }} p {{ color:#bbb; margin:7px 0; }}
.decision {{ color:#fff; }} ul {{ color:#f0c96c; padding-left:20px; }} .summary {{ font-size:.88rem; }}
</style>
</head>
<body>
<h1>TOPIQ ranking review</h1>
<p class="notice"><strong>Ranking influence only.</strong> {escape(str(payload['note']))}</p>
<div class="stats">
<span class="stat">{counts['scored']} scored</span>
<span class="stat">{counts['low_kept']} low-band kept</span>
<span class="stat">{counts['high_rejected']} high-band rejected</span>
<span class="stat">{counts['scene_winner_disagreements']} scene disagreements</span>
<span class="stat">bands: ≤{float(thresholds['low_percentile']):.0%} / ≥{float(thresholds['high_percentile']):.0%}</span>
</div>
<main class="grid">{''.join(cards)}</main>
</body>
</html>
"""
