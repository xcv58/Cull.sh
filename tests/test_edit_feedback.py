from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from cull_sh.backends.base import VisionBackend
from cull_sh.edit_feedback import export_feedback_stage
from cull_sh.edit_feedback import run_feedback_pilot
from cull_sh.edit_feedback import select_feedback_picks
from cull_sh.edit_feedback import select_machine_picks
from cull_sh.edit_feedback import _validate_render_dimensions
from cull_sh.models import EditReview
from cull_sh.models import EditReviewPair
from cull_sh.models import EditReviewVerdict
from cull_sh.models import EditSuggestion
from cull_sh.models import FinalDecision
from cull_sh.models import PreviewImage


class _FakeBackend(VisionBackend):
    def score_batch(
        self, prompt: str, previews: list[PreviewImage]
    ) -> list[FinalDecision]:
        raise NotImplementedError

    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> list[EditSuggestion]:
        return [
            EditSuggestion(
                filename=preview.asset.filename,
                asset_id=str(preview.asset.raw_path),
                exposure=0.2,
                brightness=0.1,
                contrast=5,
                highlights=-10,
                shadows=12,
                whites=6,
                blacks=-4,
                temperature=5,
                tint=-2,
                vibrance=3,
                saturation=2,
                clarity=7,
                dehaze=3,
                structure=2,
                sharpness=5,
                luma_noise_reduction=10,
                color_noise_reduction=6,
                vignette_amount=-4,
                has_crop=True,
                crop_left=0.05,
                crop_top=0.05,
                crop_right=0.95,
                crop_bottom=0.95,
                crop_angle=-1.0,
                additional_edits=["Consider a subject mask."],
                summary="Open the darker midtones.",
            )
            for preview in previews
        ]

    def review_edits(
        self, prompt: str, pairs: list[EditReviewPair]
    ) -> list[EditReview]:
        return [
            EditReview(
                filename=pair.asset.filename,
                verdict=EditReviewVerdict.REFINE,
                final_suggestion=EditSuggestion(
                    filename=pair.asset.filename,
                    asset_id=str(pair.asset.raw_path),
                    exposure=0.1,
                    brightness=0.05,
                    contrast=3,
                    highlights=-8,
                    shadows=8,
                    whites=4,
                    blacks=-3,
                    temperature=3,
                    tint=-1,
                    vibrance=2,
                    saturation=1,
                    clarity=5,
                    dehaze=2,
                    structure=1,
                    sharpness=4,
                    luma_noise_reduction=8,
                    color_noise_reduction=5,
                    vignette_amount=-3,
                    has_crop=True,
                    crop_left=0.05,
                    crop_top=0.05,
                    crop_right=0.95,
                    crop_bottom=0.95,
                    crop_angle=-0.5,
                    additional_edits=["Consider a subject mask."],
                    summary="Use a gentler lift.",
                ),
                summary="The first pass is slightly too bright.",
            )
            for pair in pairs
        ]


class _FailIfCalledBackend(_FakeBackend):
    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> list[EditSuggestion]:
        raise AssertionError("completed suggestions should be resumed")

    def review_edits(
        self, prompt: str, pairs: list[EditReviewPair]
    ) -> list[EditReview]:
        raise AssertionError("completed reviews should be resumed")


class EditFeedbackPilotTests(unittest.TestCase):
    def test_rejects_implausibly_small_crop_render(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = root / "baseline.jpg"
            rendered = root / "rendered.jpg"
            Image.new("RGB", (1000, 500), "gray").save(baseline)
            Image.new("RGB", (90, 80), "gray").save(rendered)
            suggestion = EditSuggestion(
                filename="frame.ARW",
                has_crop=True,
                crop_left=0.0,
                crop_top=0.1,
                crop_right=0.9,
                crop_bottom=1.0,
            )

            with self.assertRaisesRegex(RuntimeError, "unexpectedly small"):
                _validate_render_dimensions(baseline, rendered, suggestion)

    def test_selects_machine_picks_not_present_in_human_baseline(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            for stem in ("human", "machine", "review"):
                (source / f"{stem}.ARW").write_bytes(stem.encode())
            _write_xmp(backup / "human.xmp", good="True")
            _write_xmp(backup / "machine.xmp")
            _write_xmp(backup / "review.xmp")
            _write_manifest(
                run,
                source,
                [("human", "pick"), ("machine", "pick"), ("review", "review")],
            )

            selected = select_machine_picks(run, backup, source)

            self.assertEqual([asset.filename for asset in selected], ["machine.ARW"])

    def test_selects_union_of_protected_and_machine_picks(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            for stem in ("protected", "machine", "review"):
                (source / f"{stem}.ARW").write_bytes(stem.encode())
            _write_xmp(backup / "protected.xmp", good="True")
            _write_xmp(backup / "machine.xmp")
            _write_xmp(backup / "review.xmp")
            _write_manifest(
                run,
                source,
                [
                    ("protected", "reject"),
                    ("machine", "pick"),
                    ("review", "review"),
                ],
            )

            selected = select_feedback_picks(
                run,
                backup,
                source,
                include_existing_picks=True,
            )

            self.assertEqual(
                [asset.filename for asset in selected],
                ["machine.ARW", "protected.ARW"],
            )

    def test_fails_when_protected_pick_is_missing_from_frozen_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            (source / "protected.ARW").write_bytes(b"raw")
            _write_xmp(backup / "protected.xmp", good="True")
            (run / "manifest.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing from the frozen"):
                select_feedback_picks(
                    run,
                    backup,
                    source,
                    include_existing_picks=True,
                )

    def test_frozen_machine_scope_ignores_preexisting_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            for stem in ("picked-again", "protected-only"):
                (source / f"{stem}.ARW").write_bytes(stem.encode())
                _write_xmp(backup / f"{stem}.xmp", good="True")
            _write_manifest(
                run,
                source,
                [("picked-again", "pick"), ("protected-only", "reject")],
            )

            selected = select_feedback_picks(
                run,
                backup,
                source,
                include_existing_picks=False,
                ignore_baseline_picks=True,
            )

            self.assertEqual(
                [asset.filename for asset in selected],
                ["picked-again.ARW"],
            )

    def test_runs_resumable_render_feedback_loop_without_touching_original(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            stage = root / "stage"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            raw = source / "machine.ARW"
            raw.write_bytes(b"original-raw")
            _write_xmp(backup / "machine.xmp")
            _write_manifest(run, source, [("machine", "pick")])
            binary = root / "RapidRAW"
            binary.write_bytes(b"fake")
            assets = select_machine_picks(run, backup, source)

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                output = Path(command[command.index("--output") + 1])
                Image.new("RGB", (32, 24), "gray").save(output)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result = run_feedback_pilot(
                assets,
                stage,
                binary,
                _FakeBackend(),
                prompt="Natural edits",
                model="qwen",
                cull_run=run,
                human_baseline=backup,
                command_runner=runner,
            )

            self.assertEqual(raw.read_bytes(), b"original-raw")
            self.assertEqual(result.photos, 1)
            self.assertEqual(result.refined, 1)
            self.assertTrue(result.review_page.is_file())
            self.assertTrue((stage / "baseline/machine.jpg").is_file())
            self.assertTrue((stage / "first-pass/machine.jpg").is_file())
            self.assertTrue((stage / "final/machine.jpg").is_file())
            payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["originals_modified"])
            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(payload["suggestion_batch_size"], 1)
            self.assertEqual(payload["review_batch_size"], 1)
            self.assertEqual(payload["maximum_refinements"], 1)
            self.assertEqual(payload["records"][0]["review"]["verdict"], "refine")
            sidecar = json.loads(
                (stage / "input/machine.ARW.rrdata").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["adjustments"]["temperature"], 3)
            self.assertEqual(sidecar["adjustments"]["clarity"], 5)
            self.assertEqual(sidecar["adjustments"]["rotation"], -0.5)
            review_html = result.review_page.read_text(encoding="utf-8")
            self.assertIn("additional: Consider a subject mask.", review_html)

            resumed = run_feedback_pilot(
                assets,
                stage,
                binary,
                _FailIfCalledBackend(),
                prompt="Natural edits",
                model="qwen",
                cull_run=run,
                human_baseline=backup,
                command_runner=runner,
            )
            self.assertEqual(resumed.refined, 1)

    def test_exports_validated_feedback_as_resumable_delivery_jpeg(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            stage = root / "stage"
            delivery = root / "delivery"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            raw = source / "protected.ARW"
            raw.write_bytes(b"original-raw")
            _write_xmp(backup / "protected.xmp", good="True")
            _write_manifest(run, source, [("protected", "reject")])
            binary = root / "RapidRAW"
            binary.write_bytes(b"#!/bin/sh\n")
            binary.chmod(0o755)
            assets = select_feedback_picks(
                run,
                backup,
                source,
                include_existing_picks=True,
            )

            def pilot_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                output = Path(command[command.index("--output") + 1])
                Image.new("RGB", (32, 24), "gray").save(output)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            pilot = run_feedback_pilot(
                assets,
                stage,
                binary,
                _FakeBackend(),
                prompt="Natural edits",
                model="qwen",
                cull_run=run,
                human_baseline=backup,
                selection_scope="protected-and-machine-picks",
                command_runner=pilot_runner,
            )
            commands: list[list[str]] = []

            def export_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                Image.new("RGB", (32, 24), "green").save(output)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result = export_feedback_stage(
                pilot.root,
                binary,
                delivery,
                quality=95,
                keep_metadata=True,
                command_runner=export_runner,
            )

            self.assertEqual(result.photos, 1)
            self.assertEqual(result.exported, 1)
            self.assertEqual(result.resumed, 0)
            self.assertEqual([path.name for path in delivery.iterdir()], ["protected.jpg"])
            self.assertIn("--keep-metadata", commands[0])
            self.assertEqual(commands[0][commands[0].index("--quality") + 1], "95")
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "unattended")
            self.assertEqual(manifest["records"][0]["selection"], "protected-existing-pick")
            self.assertEqual(manifest["records"][0]["status"], "exported")
            self.assertTrue(manifest["records"][0]["sha256"])
            self.assertEqual(manifest["photos"], 1)
            self.assertEqual(manifest["completed_photos"], 1)

            resumed = export_feedback_stage(
                pilot.root,
                binary,
                delivery,
                quality=95,
                keep_metadata=True,
                command_runner=export_runner,
            )
            self.assertEqual(resumed.exported, 0)
            self.assertEqual(resumed.resumed, 1)
            self.assertEqual(len(commands), 1)

            with self.assertRaisesRegex(ValueError, "outside the source"):
                export_feedback_stage(
                    pilot.root,
                    binary,
                    source / "OUTPUT",
                    command_runner=export_runner,
                )

    def test_feedback_export_rejects_incomplete_validation(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage = root / "stage"
            stage.mkdir()
            (stage / "feedback-manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "cull-sh-rendered-edit-feedback-pilot",
                        "records": [{"id": "pilot-0001", "filename": "frame.ARW"}],
                    }
                ),
                encoding="utf-8",
            )
            binary = root / "RapidRAW"
            binary.write_bytes(b"#!/bin/sh\n")
            binary.chmod(0o755)

            with self.assertRaisesRegex(ValueError, "is incomplete"):
                export_feedback_stage(stage, binary, root / "delivery")


def _write_manifest(
    run: Path, source: Path, decisions: list[tuple[str, str]]
) -> None:
    with (run / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for stem, bucket in decisions:
            handle.write(
                json.dumps(
                    {
                        "filename": f"{stem}.ARW",
                        "asset_kind": "raw",
                        "raw_path": str(source / f"{stem}.ARW"),
                        "xmp_path": str(source / f"{stem}.xmp"),
                        "decision": {
                            "rating": 4 if bucket == "pick" else 0,
                            "label": "Green" if bucket == "pick" else None,
                            "bucket": bucket,
                            "keep": bucket == "pick",
                            "source": "vision",
                            "summary": "fixture",
                        },
                    }
                )
                + "\n"
            )


def _write_xmp(path: Path, good: str | None = None) -> None:
    good_attribute = f'xmpDM:good="{good}"' if good is not None else ""
    path.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF><rdf:Description rdf:about="" {good_attribute}/></rdf:RDF>
</x:xmpmeta>
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
