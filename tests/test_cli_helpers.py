from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from cull_sh.backends.base import VisionBackendError
from cull_sh.cli import _append_edit_suggestions
from cull_sh.cli import _suggest_edit_pairs_fail_fast
from cull_sh.cli import _write_edit_suggestions_header
from cull_sh.models import EditSuggestion
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset


class CliHelperTests(unittest.TestCase):
    def test_edit_suggestions_are_written_incrementally(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            asset = RawAsset(
                raw_path=run_dir / "frame.ARW",
                xmp_path=run_dir / "frame.xmp",
            )
            suggestion = EditSuggestion(
                filename="frame.ARW",
                asset_id=str(asset.raw_path),
                exposure=0.1,
                contrast=5,
                highlights=-10,
                shadows=4,
                vibrance=3,
                has_crop=True,
                crop_left=0.1,
                crop_top=0.05,
                crop_right=0.9,
                crop_bottom=0.95,
                crop_angle=-1.5,
                summary="Slightly dark foreground.",
            )

            path = _write_edit_suggestions_header(
                run_dir,
                "prompt",
                dry_run=False,
                with_crop=True,
            )
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)

            _append_edit_suggestions(path, [(asset, suggestion)])

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                rows[0],
                {"prompt": "prompt", "dry_run": False, "with_crop": True},
            )
            self.assertEqual(rows[1]["filename"], "frame.ARW")
            self.assertEqual(rows[1]["raw_path"], str(asset.raw_path))
            self.assertTrue(rows[1]["has_crop"])
            self.assertEqual(rows[1]["crop_left"], 0.1)
            self.assertEqual(rows[1]["crop_top"], 0.05)
            self.assertEqual(rows[1]["crop_right"], 0.9)
            self.assertEqual(rows[1]["crop_bottom"], 0.95)
            self.assertEqual(rows[1]["crop_angle"], -1.5)
            self.assertEqual(rows[1]["summary"], "Slightly dark foreground.")

    def test_edit_suggestion_header_records_inference_policy(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = _write_edit_suggestions_header(
                Path(tmp_dir),
                "prompt",
                dry_run=True,
                provider="ollama",
                model="qwen:test",
                backend_think=True,
                backend_max_attempts=1,
            )

            header = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(header["backend_think"])
        self.assertEqual(header["backend_max_attempts"], 1)

    def test_edit_suggestion_failure_does_not_retry_or_fallback(self) -> None:
        previews = [
            _build_preview("/tmp/a.ARW"),
            _build_preview("/tmp/b.ARW"),
        ]
        backend = MagicMock()
        backend.suggest_edits.side_effect = VisionBackendError("bad cohort")

        with self.assertRaisesRegex(VisionBackendError, "bad cohort"):
            _suggest_edit_pairs_fail_fast(
                backend,
                "prompt",
                previews,
                include_crop=True,
            )

        self.assertEqual(backend.suggest_edits.call_count, 1)
        self.assertEqual(
            [call.kwargs for call in backend.suggest_edits.call_args_list],
            [{"include_crop": True}],
        )

    def test_edit_suggestion_count_mismatch_fails_fast(self) -> None:
        previews = [
            _build_preview("/tmp/a.ARW"),
            _build_preview("/tmp/b.ARW"),
        ]
        backend = MagicMock()
        backend.suggest_edits.return_value = [
            EditSuggestion(filename="a.ARW", summary="A needs contrast.")
        ]

        with self.assertRaisesRegex(VisionBackendError, "unexpected number"):
            _suggest_edit_pairs_fail_fast(backend, "prompt", previews)

        self.assertEqual(backend.suggest_edits.call_count, 1)


def _build_preview(raw_path: str) -> PreviewImage:
    path = Path(raw_path)
    return PreviewImage(
        asset=RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")),
        image_bytes=b"jpeg-bytes",
    )


if __name__ == "__main__":
    unittest.main()
