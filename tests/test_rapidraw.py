from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from cull_sh.rapidraw import export_rapidraw_stage
from cull_sh.rapidraw import rapidraw_adjustments_from_suggestion
from cull_sh.rapidraw import RapidRawError
from cull_sh.rapidraw import render_rapidraw_review
from cull_sh.rapidraw import stage_rapidraw_develop


class RapidRawWorkflowTests(unittest.TestCase):
    def test_maps_qwen_edit_fields_and_normalized_crop(self) -> None:
        adjustments = rapidraw_adjustments_from_suggestion(
            {
                "exposure": 0.3,
                "contrast": 10,
                "highlights": -20,
                "shadows": 25,
                "vibrance": 5,
                "has_crop": True,
                "crop_left": 0.1,
                "crop_top": 0.2,
                "crop_right": 0.9,
                "crop_bottom": 0.8,
                "crop_angle": 1.5,
            }
        )

        self.assertEqual(adjustments["exposure"], 0.3)
        self.assertEqual(adjustments["highlights"], -20)
        self.assertEqual(
            adjustments["crop"],
            {"unit": "%", "x": 10.0, "y": 20.0, "width": 80.0, "height": 60.0},
        )
        self.assertEqual(adjustments["rotation"], 1.5)

    def test_rejects_out_of_range_slider_and_extreme_crop(self) -> None:
        with self.assertRaisesRegex(ValueError, "contrast"):
            rapidraw_adjustments_from_suggestion({"contrast": 101})
        with self.assertRaisesRegex(ValueError, "retain at least 10%"):
            rapidraw_adjustments_from_suggestion(
                {
                    "has_crop": True,
                    "crop_left": 0.45,
                    "crop_top": 0.45,
                    "crop_right": 0.55,
                    "crop_bottom": 0.55,
                }
            )

    def test_stage_copies_raw_writes_sidecar_and_requires_new_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            source = base / "source.ARW"
            source.write_bytes(b"raw-data")
            suggestions = _write_suggestions(base, source)

            stage = stage_rapidraw_develop(
                suggestions,
                base / "stage",
                cull_sh_commit="abc123",
            )

            self.assertEqual(stage.photos, 1)
            self.assertEqual((stage.input_dir / source.name).read_bytes(), b"raw-data")
            sidecar = json.loads(
                (stage.input_dir / "source.ARW.rrdata").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["adjustments"]["exposure"], 0.2)
            manifest = json.loads(stage.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["originals_modified"])
            self.assertTrue(manifest["approval_required"])
            self.assertEqual(manifest["cull_sh_commit"], "abc123")
            template = json.loads(
                stage.approval_template_path.read_text(encoding="utf-8")
            )
            self.assertEqual(template["manifest_sha256"], stage.manifest_sha256)
            self.assertEqual(template["decisions"], {"edit-0001": False})

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                stage_rapidraw_develop(suggestions, base / "stage")

    def test_review_renders_before_after_and_downloadable_approval_page(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            source = base / "source.ARW"
            source.write_bytes(b"raw-data")
            stage = stage_rapidraw_develop(
                _write_suggestions(base, source), base / "stage"
            )
            binary = _fake_binary(base)
            commands: list[list[str]] = []

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                Image.new("RGB", (20, 10), "green").save(output)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            page = render_rapidraw_review(
                stage.root,
                binary,
                command_runner=runner,
                preview_extractor=lambda _path: _jpeg_bytes("gray"),
            )

            self.assertEqual(len(commands), 1)
            html = page.read_text(encoding="utf-8")
            self.assertIn("Before: embedded RAW preview", html)
            self.assertIn("After: RapidRAW preview render", html)
            self.assertIn("Download approvals", html)
            self.assertIn(stage.manifest_sha256, html)

    def test_export_requires_matching_approval_and_exports_only_approved(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            first = base / "first.ARW"
            second = base / "second.ARW"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            suggestions = base / "suggestions.jsonl"
            suggestions.write_text(
                json.dumps({"kind": "edit-suggestions", "model": "qwen"})
                + "\n"
                + json.dumps(_suggestion(first))
                + "\n"
                + json.dumps(_suggestion(second))
                + "\n",
                encoding="utf-8",
            )
            stage = stage_rapidraw_develop(suggestions, base / "stage")
            binary = _fake_binary(base)

            with self.assertRaisesRegex(RapidRawError, "requires --approvals"):
                export_rapidraw_stage(stage.root, binary)

            approvals = base / "approvals.json"
            approvals.write_text(
                json.dumps(
                    {
                        "manifest_sha256": stage.manifest_sha256,
                        "decisions": {"edit-0001": True, "edit-0002": False},
                    }
                ),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"rendered")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result = export_rapidraw_stage(
                stage.root,
                binary,
                approvals_path=approvals,
                command_runner=runner,
            )

            self.assertEqual(result.approved, 1)
            self.assertEqual(result.exported, 1)
            self.assertEqual(len(commands), 1)
            self.assertIn("--keep-metadata", commands[0])
            export_manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(export_manifest["approved_ids"], ["edit-0001"])
            self.assertEqual(export_manifest["records"][0]["status"], "exported")

            resumed = export_rapidraw_stage(
                stage.root,
                binary,
                approvals_path=approvals,
                command_runner=runner,
            )
            self.assertEqual(resumed.resumed, 1)
            self.assertEqual(len(commands), 1)

    def test_export_fails_fast_and_records_failure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            source = base / "source.ARW"
            source.write_bytes(b"raw-data")
            stage = stage_rapidraw_develop(
                _write_suggestions(base, source), base / "stage"
            )
            binary = _fake_binary(base)

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 7, "", "decoder failed")

            with self.assertRaisesRegex(RapidRawError, "decoder failed"):
                export_rapidraw_stage(
                    stage.root,
                    binary,
                    approve_all=True,
                    command_runner=runner,
                )
            payload = json.loads(
                (stage.root / "rapidraw-export.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["records"][0]["status"], "failed")


def _write_suggestions(base: Path, source: Path) -> Path:
    path = base / "suggestions.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "edit-suggestions",
                "model": "orcarouter/Qwen3.8-27B-Uncensored",
                "prompt": "Natural edits",
            }
        )
        + "\n"
        + json.dumps(_suggestion(source))
        + "\n",
        encoding="utf-8",
    )
    return path


def _suggestion(source: Path) -> dict[str, object]:
    return {
        "asset_id": str(source),
        "filename": source.name,
        "raw_path": str(source),
        "exposure": 0.2,
        "contrast": 10,
        "highlights": -15,
        "shadows": 20,
        "vibrance": 10,
        "has_crop": False,
        "summary": "Natural tonal balance.",
    }


def _jpeg_bytes(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 10), color).save(output, format="JPEG")
    return output.getvalue()


def _fake_binary(base: Path) -> Path:
    binary = base / "RapidRAW"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


if __name__ == "__main__":
    unittest.main()
