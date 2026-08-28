"""Explicit development correction of failed edits in a separate feedback stage.

Never invoked automatically by the production pipeline. Passed pixels and checks
are copied unchanged; failed records get one new correction and a fresh final
check under the original delivery instructions.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cull_sh.backends import build_backend
from cull_sh.config import BackendConfig
from cull_sh.edit_feedback import (
    VALIDATION_POLICY, _asset_from_record, _file_sha256, _recipe_sha256,
    _records, _write_json, export_feedback_stage, run_feedback_pilot,
)
from cull_sh.rapidraw import rapidraw_render_environment

CORRECTION_POLICY = "explicit-failed-edit-correction-v1"
CORRECTION_GUIDANCE = (
    "This is one explicitly authorized development correction of a previously failed final render. "
    "The edited image is that failed candidate, not a new initial edit. "
    "Diagnose its visible problem against the baseline and repair it without adding a new style. "
    "Slider values are absolute: zero undoes an existing calibrated value. "
    "If the daylight became too dark, restore the appropriate baseline brightness and ease "
    "unnecessary contrast/dehaze instead of reducing brightness further. Keep unrelated controls "
    "unchanged. The baseline is an allowed outcome if it is better; do not invent changes. "
    "Do not accept unchanged failed pixels merely to get a pass. A separate final check will "
    "judge the new rendered pixels under the original instructions, without this diagnosis."
)


def _verify_parent(parent, binary):
    if parent.get("schema_version") != 4 or parent.get("validation_policy") != VALIDATION_POLICY:
        raise ValueError("correction requires a current delivery-validated parent stage")
    if parent.get("renderer_sha256") != _file_sha256(binary):
        raise ValueError("parent renderer binary changed")
    if parent.get("renderer_environment") != rapidraw_render_environment(binary):
        raise ValueError("parent renderer environment changed")
    records = _records(parent)
    if not records or len({r["filename"] for r in records}) != len(records):
        raise ValueError("parent selection is empty or duplicated")
    failed = []
    for record in records:
        name = record["filename"]
        if Path(name).name != name or name.startswith("."):
            raise ValueError("invalid parent filename")
        check = record.get("delivery_validation", {})
        if check.get("policy") != VALIDATION_POLICY or check.get("status") not in {"passed", "failed"}:
            raise ValueError(f"parent validation incomplete: {name}")
        for field, digest in (
            ("source", record["source_sha256"]),
            ("staged_raw", record["source_sha256"]),
            ("baseline_render", record["baseline_sha256"]),
            ("first_render", record["first_sha256"]),
            ("final_render", check["render_sha256"]),
        ):
            if _file_sha256(Path(record[field])) != digest:
                raise ValueError(f"parent {field} changed: {name}")
        if _recipe_sha256(record["final_suggestion"]) != check["recipe_sha256"]:
            raise ValueError(f"parent final recipe changed: {name}")
        if check["status"] == "failed":
            failed.append(name)
    if not failed:
        raise ValueError("parent has no failed edits to correct")
    return failed


def prepare_correction_stage(parent_root, root, binary, *, reason):
    parent_root, root, binary = (p.expanduser().resolve() for p in (parent_root, root, binary))
    if root == parent_root or root in parent_root.parents or parent_root in root.parents:
        raise ValueError("parent and correction stage must be separate trees")
    if not reason.strip():
        raise ValueError("an explicit correction reason is required")
    parent_path = parent_root / "feedback-manifest.json"
    parent = json.loads(parent_path.read_text())
    failed = _verify_parent(parent, binary)
    for record in parent["records"]:
        source_root = Path(record["source"]).resolve().parent
        if root == source_root or root in source_root.parents or source_root in root.parents:
            raise ValueError("correction stage must be separate from source photos")
    provenance = {
        "policy": CORRECTION_POLICY,
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": _file_sha256(parent_path),
        "reason": reason,
        "guidance": CORRECTION_GUIDANCE,
        "failed_filenames": failed,
        "maximum_additional_refinements": 1,
        "independent_benchmark": False,
    }
    path = root / "feedback-manifest.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("correction_experiment") != provenance:
            raise ValueError("correction parent, reason or policy changed; use a new stage")
        return saved
    if root.exists() and any(root.iterdir()):
        raise ValueError("refusing nonempty untracked correction stage")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("input", "baseline", "first-pass", "final"):
        (root / name).mkdir()
    manifest = copy.deepcopy(parent)
    manifest["correction_experiment"] = provenance
    for record in manifest["records"]:
        previous = copy.deepcopy(record)
        name, stem = record["filename"], Path(record["filename"]).stem
        record["inherited_from_manifest_sha256"] = provenance["parent_manifest_sha256"]
        copies = {
            "staged_raw": (previous["staged_raw"], root / "input" / name),
            "baseline_render": (previous["baseline_render"], root / "baseline" / f"{stem}.jpg"),
            "first_render": (previous["first_render"], root / "first-pass" / f"{stem}.jpg"),
        }
        if name in failed:
            record["prior_experiment_record"] = previous
            record["initial_suggestion"] = copy.deepcopy(previous["final_suggestion"])
            record["first_sha256"] = previous["delivery_validation"]["render_sha256"]
            record["first_metrics"] = previous["final_metrics"]
            copies["first_render"] = (previous["final_render"], copies["first_render"][1])
            for key in ("review", "final_suggestion", "final_render", "final_metrics", "delivery_validation"):
                record.pop(key, None)
        else:
            copies["final_render"] = (previous["final_render"], root / "final" / f"{stem}.jpg")
        for field, (source, destination) in copies.items():
            shutil.copy2(source, destination)
            if _file_sha256(destination) != _file_sha256(Path(source)):
                raise ValueError(f"correction copy differs: {name} {field}")
            record[field] = str(destination)
    _write_json(path, manifest)
    return manifest


class _CorrectionBackend:
    def __init__(self, backend, manifest):
        self.backend = backend
        self.manifest = manifest

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def suggest_edits(self, *args, **kwargs):
        raise RuntimeError("correction stage must reuse all initial suggestions")

    def review_edits(self, prompt, pairs):
        if not all(p.delivery_check for p in pairs):
            if any(p.delivery_check for p in pairs):
                raise ValueError("mixed correction and delivery requests")
            prior = {r["filename"]: r.get("prior_experiment_record") for r in self.manifest["records"]}
            diagnoses = []
            for pair in pairs:
                previous = prior.get(pair.asset.filename)
                if not previous:
                    raise ValueError("cannot correct a previously passed photo")
                diagnoses.append({"filename": pair.asset.filename,
                                  "previous_failure": previous["delivery_validation"]["summary"]})
            prompt += "\n\n" + CORRECTION_GUIDANCE + "\n" + json.dumps(diagnoses)
        # Final validation receives the exact original prompt, no prior verdict or diagnosis.
        return self.backend.review_edits(prompt, pairs)


def run_correction(parent_root, root, binary, backend, *, reason, progress=print, command_runner=None):
    manifest = prepare_correction_stage(parent_root, root, binary, reason=reason)
    return run_feedback_pilot(
        [_asset_from_record(r) for r in manifest["records"]], root, binary,
        _CorrectionBackend(backend, manifest),
        prompt=manifest["prompt"], model=manifest["model"],
        cull_run=Path(manifest["cull_run"]), human_baseline=Path(manifest["human_baseline"]),
        include_crop=manifest["include_crop"], selection_scope=manifest["selection_scope"],
        suggestion_batch_size=manifest["suggestion_batch_size"],
        review_batch_size=manifest["review_batch_size"], quality=manifest["quality"],
        baseline_mode=manifest["baseline_mode"], progress=progress, command_runner=command_runner,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--context-tokens", type=int, default=65536)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    stage = root / "stage"
    # Prepare/verify before creating status; never mark an invalid old directory as tracked.
    if root.exists() and any(root.iterdir()) and not (stage / "feedback-manifest.json").is_file():
        raise ValueError("refusing untracked correction output")
    prepare_correction_stage(args.parent_stage, stage, args.binary, reason=args.reason)
    backend = build_backend(BackendConfig(max_output_tokens=4096, timeout_seconds=600,
                                         think=True, max_attempts=1, context_tokens=args.context_tokens))
    status_path = root / "status.json"
    state = {"status": "running", "kind": CORRECTION_POLICY, "album_output": None,
             "started_at": datetime.now(timezone.utc).isoformat()}
    if status_path.is_file():
        old = json.loads(status_path.read_text())
        state["attempt_history"] = old.get("attempt_history", []) + [
            {k: v for k, v in old.items() if k != "attempt_history"}]
    _write_json(status_path, state)
    try:
        run_correction(args.parent_stage, stage, args.binary, backend,
                       reason=args.reason, progress=lambda msg: print(msg, flush=True))
        result = export_feedback_stage(stage, args.binary, root / "delivery", quality=95)
        state.update(status="complete", exported=result.exported, resumed=result.resumed)
    except BaseException as exc:
        state.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(status_path, state)


if __name__ == "__main__":
    main()
