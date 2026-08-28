"""Run an explicitly selected, isolated development pilot, with resumable records."""

import argparse
import json
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path

from cull_sh.album_selection import load_raw_hashes, run_album_selection
from cull_sh.backends import build_backend
from cull_sh.config import DEFAULT_PRODUCTION_MODEL, BackendConfig
from cull_sh.edit_feedback import export_feedback_stage, run_feedback_pilot
from cull_sh.models import RawAsset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cull-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument(
        "--album-output",
        type=Path,
        help="After the pilot passes, run full unattended reselection here.",
    )
    parser.add_argument("--raw-checksums", type=Path)
    parser.add_argument("--context-tokens", type=int, default=None)
    parser.add_argument(
        "--runtime-change-reason",
        "--context-change-reason",
        dest="context_change_reason",
        help="Explicitly record a context/image-transport recovery while retaining completed work.",
    )
    args = parser.parse_args()
    if args.album_output and not args.raw_checksums:
        parser.error("--album-output requires --raw-checksums")
    source = args.source.resolve()
    root = args.output.resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("source and output trees must be separate")
    if len(set(args.stems)) != len(args.stems) or any(
        Path(s).name != s or "." in s for s in args.stems
    ):
        raise ValueError(
            "stems must be unique plain camera filenames without extension"
        )
    assets = [
        RawAsset(source / f"{s}.ARW", source / f"{s}.xmp") for s in sorted(args.stems)
    ]
    if any(not a.raw_path.is_file() for a in assets):
        raise ValueError("a selected RAW is missing")
    if args.raw_checksums:
        expected = load_raw_hashes(args.raw_checksums)
        for asset in assets:
            with asset.raw_path.open("rb") as raw:
                if file_digest(raw, "sha256").hexdigest() != expected.get(
                    asset.filename
                ):
                    raise ValueError(
                        f"RAW differs from frozen checksum ledger: {asset.filename}"
                    )
    if root.exists() and any(root.iterdir()) and not (root / "status.json").is_file():
        raise ValueError("refusing nonempty untracked pilot output")
    root.mkdir(parents=True, exist_ok=True)
    backend_config = BackendConfig(
        max_output_tokens=4096,
        timeout_seconds=600,
        think=True,
        max_attempts=1,
        context_tokens=args.context_tokens,
    )
    backend = build_backend(backend_config)
    prompt = (
        "Produce a natural, readable travel photograph while preserving lighting and mood. "
        "Daylight should not look unintentionally dim or yellow-green. Preserve white cloud detail, "
        "open useful subject detail without flattening sunset silhouettes, and correct genuinely tilted "
        "horizons when evidence supports it. Do not introduce milky edge effects. "
        "The shown baseline may already be camera-midtone calibrated: output absolute recipe values."
    )
    state = {
        "kind": "development-pilot-not-independent-benchmark",
        "model": DEFAULT_PRODUCTION_MODEL,
        "stems": sorted(args.stems),
        "source": str(source),
        "album_output": str(args.album_output.resolve()) if args.album_output else None,
        "context_tokens": args.context_tokens,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    status_path = root / "status.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text())
        state["started_at"] = previous.get("started_at", state["started_at"])
        state["attempt_history"] = previous.get("attempt_history", []) + [
            {
                key: previous.get(key)
                for key in ("status", "error", "updated_at", "context_tokens")
            }
        ]
        state["resumed_at"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    try:
        run_feedback_pilot(
            assets,
            root / "stage",
            args.binary,
            backend,
            prompt=prompt,
            model=DEFAULT_PRODUCTION_MODEL,
            cull_run=args.cull_run,
            human_baseline=root / "unused-human-baseline",
            selection_scope="development-pilot",
            baseline_mode="camera-midtones-v1",
            context_change_reason=args.context_change_reason,
            quality=95,
            progress=lambda msg: print(msg, flush=True),
        )
        result = export_feedback_stage(
            root / "stage", args.binary, root / "delivery", quality=95
        )
        state.update(
            status="pilot-complete", exported=result.exported, resumed=result.resumed
        )
        if args.album_output:
            state["status"] = "selecting-album"
            status_path.write_text(json.dumps(state, indent=2) + "\n")
            state["album"] = run_album_selection(
                args.cull_run,
                source,
                args.album_output,
                load_raw_hashes(args.raw_checksums),
                backend,
                progress=lambda msg: print(msg, flush=True),
            )
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
