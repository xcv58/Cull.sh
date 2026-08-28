"""Final unattended album selection, independent of assisted triage buckets.

Reuses frozen technical scores only with matching RAW hashes. Every decodable
frame receives contextual selection; duplicates are suppressed AFTER selection.
No source sidecars or frozen records are modified.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from cull_sh.backends import build_backend
from cull_sh.config import BackendConfig
from cull_sh.extractors import ExifToolPreviewExtractor
from cull_sh.manifests import load_manifest_records
from cull_sh.models import DecisionBucket, PreviewImage, RawAsset

POLICY = "unattended-album-v1"
PROMPT = "Assemble a complete travel album with meaningful coverage and restrained repetition. Technical scores are advisory, not the definition of a memorable photo."


def _digest(path):
    with path.open("rb") as handle:
        from hashlib import file_digest

        return file_digest(handle, "sha256").hexdigest()


def _write(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_raw_hashes(path):
    hashes = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        filename = Path(name.lstrip("*")).name
        if not re.fullmatch("[0-9a-fA-F]{64}", digest) or filename in hashes:
            raise ValueError("invalid or ambiguous RAW hash ledger")
        hashes[filename] = digest.lower()
    return hashes


def _sequence(filename):
    match = re.fullmatch(r"(.*?)(\d+)", Path(filename).stem)
    return (match[1], int(match[2])) if match else None


def resolve_album_duplicates(records, previews):
    """Conservative nearby lookalikes only, with a surviving final representative."""
    rows = deepcopy(records)
    selected = [r for r in rows if r["decision"]["bucket"] == "pick"]
    selected.sort(
        key=lambda r: (-float(r.get("combined_rank_score") or 0), r["filename"])
    )
    tiny = {}

    def rgb(name):
        if name not in tiny:
            with Image.open(BytesIO(previews[name])) as image:
                tiny[name] = np.asarray(
                    image.convert("RGB").resize((64, 64)), dtype=float
                )
        return tiny[name]

    retained = []
    for candidate in selected:
        seq = _sequence(candidate["filename"])
        ch = candidate.get("perceptual_hash")
        anchor = None
        if seq and ch:
            for other in retained:
                os = _sequence(other["filename"])
                oh = other.get("perceptual_hash")
                if not os or seq[0] != os[0] or abs(seq[1] - os[1]) > 3 or not oh:
                    continue
                if (int(ch, 16) ^ int(oh, 16)).bit_count() > 2:
                    continue
                if (
                    float(
                        np.abs(
                            rgb(candidate["filename"]) - rgb(other["filename"])
                        ).mean()
                    )
                    <= 8
                ):
                    anchor = other
                    break
        if anchor:
            candidate["album_trace"]["represented_by"] = anchor["filename"]
            candidate["decision"] = {
                **candidate["decision"],
                "bucket": "reject",
                "keep": False,
                "rating": 0,
                "label": None,
                "summary": f"Not separately included: near-identical adjacent frame represented by selected {anchor['filename']}",
            }
        else:
            retained.append(candidate)
    picked = {r["filename"] for r in retained}
    for row in rows:
        representative = row["album_trace"].get("represented_by")
        if representative and representative not in picked:
            raise AssertionError("suppressed duplicate has no selected representative")
    return rows


def run_album_selection(
    frozen_run,
    source,
    output,
    raw_hashes,
    backend,
    *,
    batch_size=6,
    progress=print,
    extractor=None,
    prepare_only=False,
):
    frozen_run, source, output = [
        p.expanduser().resolve() for p in (frozen_run, source, output)
    ]
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    for protected in [frozen_run, source]:
        if (
            output == protected
            or protected in output.parents
            or output in protected.parents
        ):
            raise ValueError("album stage must be outside source and frozen run")
    rows = load_manifest_records(frozen_run)
    names = [str(r["filename"]) for r in rows]
    if (
        not names
        or len({Path(n).stem for n in names}) != len(names)
        or any(Path(n).name != n or n.startswith(".") for n in names)
    ):
        raise ValueError("manifest must contain unique visible camera filenames")
    if set(names) - raw_hashes.keys():
        raise ValueError("RAW checksum ledger does not cover every frozen record")
    if any(r.get("error") for r in rows):
        raise ValueError(
            "frozen input has errors; resolve them before final album selection"
        )
    expected = {
        "policy": POLICY,
        "frozen_manifest_sha256": _digest(frozen_run / "manifest.jsonl"),
        "raw_hashes": {n: raw_hashes[n] for n in sorted(names)},
        "source": str(source),
        "batch_size": batch_size,
        "prompt": PROMPT,
        "model": getattr(backend, "model", "test"),
        "think": getattr(backend, "think", True),
        "max_output_tokens": getattr(backend, "max_output_tokens", 4096),
    }
    checkpoint = output / "album-state.json"
    if output.exists() and any(output.iterdir()) and not checkpoint.is_file():
        raise ValueError("refusing nonempty untracked album output")
    output.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.is_file()
        else {"config": expected, "decisions": {}, "status": "running"}
    )
    if state["config"] != expected:
        raise ValueError("album resume configuration changed; use a new stage")
    state["status"] = "running"
    _write(checkpoint, state)
    extractor = extractor or ExifToolPreviewExtractor()
    cache = output / "previews"
    cache.mkdir(exist_ok=True)
    previews = {}
    try:
        for i, row in enumerate(rows, 1):
            name = row["filename"]
            raw = source / name
            if _digest(raw) != raw_hashes[name]:
                raise ValueError(
                    f"RAW hash mismatch; refusing stale score reuse: {name}"
                )
            path = cache / (Path(name).stem + ".jpg")
            if path.is_file() and name in state.get("preview_hashes", {}):
                data = path.read_bytes()
                if sha256(data).hexdigest() != state.get("preview_hashes", {}).get(
                    name
                ):
                    raise ValueError(f"untracked or changed preview: {name}")
            else:
                data = extractor.extract_preview_bytes(raw)
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                # A crash before checkpointing can leave an untracked preview.
                # Regenerate it from the hash-verified RAW; never trust it.
                temporary = path.with_suffix(".jpg.tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                state.setdefault("preview_hashes", {})[name] = sha256(data).hexdigest()
                _write(checkpoint, state)
            previews[name] = data
            if i % 40 == 0 or i == len(rows):
                progress(f"Verified/cached {i}/{len(rows)} RAW previews")
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("scene_id") or row["filename"]].append(row)
        cohorts = []
        for group in groups.values():
            group.sort(key=lambda r: r["filename"])
            cohorts.extend(
                group[i : i + batch_size] for i in range(0, len(group), batch_size)
            )
        if prepare_only:
            summary = {
                "photos": len(rows),
                "cohorts": len(cohorts),
                "source_writes": 0,
                "decisions_completed": len(state["decisions"]),
            }
            state.update(status="prepared", preparation=summary)
            _write(checkpoint, state)
            return summary
        for index, cohort in enumerate(cohorts, 1):
            cohort_names = [r["filename"] for r in cohort]
            done = [n in state["decisions"] for n in cohort_names]
            if all(done):
                continue
            if any(done):
                raise ValueError("partial cohort checkpoint is invalid")
            progress(
                f"Album decision {index}/{len(cohorts)}: " + ", ".join(cohort_names)
            )
            images = [
                PreviewImage(
                    RawAsset(source / n, (source / n).with_suffix(".xmp")), previews[n]
                )
                for n in cohort_names
            ]
            decisions = backend.select_album_batch(PROMPT, images)
            if len(decisions) != len(images):
                raise ValueError("album decision count mismatch")
            parsed = {}
            for name, decision in zip(cohort_names, decisions):
                if (
                    decision.filename != name
                    or decision.bucket == DecisionBucket.REVIEW
                    or not decision.summary.strip()
                ):
                    raise ValueError(
                        "album decision must identify a final outcome and reason for each frame"
                    )
                parsed[name] = {
                    **asdict(decision),
                    "keep": decision.bucket == DecisionBucket.PICK,
                }
            state["decisions"].update(parsed)
            _write(checkpoint, state)
        resolved = []
        for original in rows:
            row = deepcopy(original)
            name = row["filename"]
            row.update(
                raw_path=str(source / name),
                xmp_path=str((source / name).with_suffix(".xmp")),
                decision=state["decisions"][name],
                sidecar_written=False,
                lightroom_edit_written=False,
                status="scored",
            )
            row["album_trace"] = {
                "policy": POLICY,
                "frozen_decision": original.get("decision"),
                "contextual_decision": state["decisions"][name],
                "original_local_trace": original.get("local_trace"),
                "represented_by": None,
            }
            # Historical local rejects are not current decisions; keep them only in provenance.
            row["local_trace"] = {
                "final_local_action": "contextual_album_selection",
                "scores_reused": True,
            }
            row["vision_trace"] = state["decisions"][name]
            resolved.append(row)
        resolved = resolve_album_duplicates(resolved, previews)
        summary = {
            "photos": len(rows),
            "selected": sum(r["decision"]["bucket"] == "pick" for r in resolved),
            "duplicates_represented": sum(
                bool(r["album_trace"]["represented_by"]) for r in resolved
            ),
            "frozen_buckets_of_selected": dict(
                Counter(
                    r["album_trace"]["frozen_decision"]["bucket"]
                    for r in resolved
                    if r["decision"]["bucket"] == "pick"
                )
            ),
            "source_writes": 0,
            "policy": POLICY,
            "kind": "development-reselection",
        }
        final = output / "manifest.jsonl"
        temp = output / "manifest.jsonl.tmp"
        temp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in resolved))
        temp.replace(final)
        _write(output / "album-summary.json", summary)
        state.update(status="complete", summary=summary, manifest_sha256=_digest(final))
        _write(checkpoint, state)
        return summary
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        _write(checkpoint, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["frozen-run", "source", "output", "raw-checksums"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify RAW hashes and cache previews without calling the model.",
    )
    args = parser.parse_args()
    backend = build_backend(BackendConfig(max_output_tokens=4096, timeout_seconds=600))
    result = run_album_selection(
        args.frozen_run,
        args.source,
        args.output,
        load_raw_hashes(args.raw_checksums),
        backend,
        batch_size=args.batch_size,
        progress=lambda message: print(message, flush=True),
        prepare_only=args.prepare_only,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
