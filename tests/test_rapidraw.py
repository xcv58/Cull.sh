from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from cull_sh.models import EditSuggestion
from cull_sh.rapidraw import export_rapidraw_stage
from cull_sh.rapidraw import rapidraw_adjustments_from_suggestion
from cull_sh.rapidraw import RapidRawError
from cull_sh.rapidraw import render_rapidraw_review
from cull_sh.rapidraw import stage_rapidraw_develop


class RapidRawWorkflowTests(unittest.TestCase):
    def test_noop_tracks_every_executable_control_but_not_future_intents(self) -> None:
        self.assertTrue(
            EditSuggestion(
                filename="frame.ARW",
                additional_edits=["Consider a subject mask."],
            ).is_noop
        )
        self.assertFalse(
            EditSuggestion(filename="frame.ARW", temperature=1).is_noop
        )
        self.assertFalse(
            EditSuggestion(filename="frame.ARW", crop_angle=0.5).is_noop
        )

    def test_maps_qwen_edit_fields_and_normalized_crop(self) -> None:
        adjustments = rapidraw_adjustments_from_suggestion(
            {
                "exposure": 0.3,
                "brightness": 0.1,
                "contrast": 10,
                "highlights": -20,
                "shadows": 25,
                "whites": 12,
                "blacks": -8,
                "temperature": 7,
                "tint": -3,
                "vibrance": 5,
                "saturation": 2,
                "clarity": 9,
                "dehaze": 4,
                "structure": 3,
                "sharpness": 11,
                "luma_noise_reduction": 14,
                "color_noise_reduction": 8,
                "vignette_amount": -6,
                "has_crop": True,
                "crop_left": 0.1,
                "crop_top": 0.2,
                "crop_right": 0.9,
                "crop_bottom": 0.8,
                "crop_angle": 1.5,
            },
            image_size=(1000, 500),
        )

        self.assertEqual(adjustments["exposure"], 0.3)
        self.assertEqual(adjustments["brightness"], 0.1)
        self.assertEqual(adjustments["highlights"], -20)
        self.assertEqual(adjustments["whites"], 12)
        self.assertEqual(adjustments["blacks"], -8)
        self.assertEqual(adjustments["temperature"], 7)
        self.assertEqual(adjustments["tint"], -3)
        self.assertEqual(adjustments["clarity"], 9)
        self.assertEqual(adjustments["lumaNoiseReduction"], 14)
        self.assertEqual(adjustments["colorNoiseReduction"], 8)
        self.assertEqual(adjustments["vignetteAmount"], -6)
        self.assertEqual(
            adjustments["crop"],
            {"x": 100.0, "y": 100.0, "width": 800.0, "height": 300.0},
        )
        self.assertEqual(adjustments["rotation"], 1.5)

    def test_rejects_rotation_without_a_crop(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotation requires a crop"):
            rapidraw_adjustments_from_suggestion(
                {
                    "has_crop": False,
                    "crop_angle": -2.25,
                }
            )
        with self.assertRaisesRegex(ValueError, "non-default crop bounds"):
            rapidraw_adjustments_from_suggestion(
                {
                    "has_crop": True,
                    "crop_angle": -2.25,
                    "crop_left": 0.0,
                    "crop_top": 0.0,
                    "crop_right": 1.0,
                    "crop_bottom": 1.0,
                },
                image_size=(1000, 500),
            )

    def test_rotation_insets_an_optimistic_crop_to_avoid_fill_edges(self) -> None:
        adjustments = rapidraw_adjustments_from_suggestion(
            {
                "has_crop": True,
                "crop_angle": 5.0,
                "crop_left": 0.01,
                "crop_top": 0.01,
                "crop_right": 0.99,
                "crop_bottom": 0.99,
            },
            image_size=(1000, 500),
        )

        crop = adjustments["crop"]
        self.assertGreater(crop["x"], 70.0)
        self.assertGreater(crop["y"], 35.0)
        self.assertLess(crop["x"] + crop["width"], 930.0)
        self.assertLess(crop["y"] + crop["height"], 465.0)

    def test_rejects_out_of_range_slider_and_extreme_crop(self) -> None:
        with self.assertRaisesRegex(ValueError, "contrast"):
            rapidraw_adjustments_from_suggestion({"contrast": 101})
        with self.assertRaisesRegex(ValueError, "luma_noise_reduction"):
            rapidraw_adjustments_from_suggestion({"luma_noise_reduction": -1})
        with self.assertRaisesRegex(ValueError, "retain at least 10%"):
            rapidraw_adjustments_from_suggestion(
                {
                    "has_crop": True,
                    "crop_left": 0.45,
                    "crop_top": 0.45,
                    "crop_right": 0.55,
                    "crop_bottom": 0.55,
                },
                image_size=(1000, 500),
            )

    def test_crop_mapping_requires_image_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "source image dimensions"):
            rapidraw_adjustments_from_suggestion(
                {
                    "has_crop": True,
                    "crop_left": 0.1,
                    "crop_top": 0.1,
                    "crop_right": 0.9,
                    "crop_bottom": 0.9,
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
            self.assertEqual(sidecar["adjustments"]["temperature"], 6)
            self.assertEqual(sidecar["adjustments"]["clarity"], 8)
            self.assertEqual(sidecar["adjustments"]["lumaNoiseReduction"], 12)
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
            self.assertIn("temperature 6", html)
            self.assertIn("Consider a subtle subject mask.", html)
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
        "brightness": 0.1,
        "contrast": 10,
        "highlights": -15,
        "shadows": 20,
        "whites": 5,
        "blacks": -4,
        "temperature": 6,
        "tint": -2,
        "vibrance": 10,
        "saturation": 2,
        "clarity": 8,
        "dehaze": 3,
        "structure": 2,
        "sharpness": 7,
        "luma_noise_reduction": 12,
        "color_noise_reduction": 6,
        "vignette_amount": -4,
        "has_crop": False,
        "crop_angle": 0.0,
        "additional_edits": ["Consider a subtle subject mask."],
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
