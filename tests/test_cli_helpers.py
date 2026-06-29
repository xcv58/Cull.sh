from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from cull_sh.backends.base import VisionBackendError
from cull_sh.cli import _append_edit_suggestions
from cull_sh.cli import _suggest_edit_pairs_with_fallback
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
                summary="Slightly dark foreground.",
            )

            path = _write_edit_suggestions_header(run_dir, "prompt", dry_run=False)
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)

            _append_edit_suggestions(path, [(asset, suggestion)])

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0], {"prompt": "prompt", "dry_run": False})
            self.assertEqual(rows[1]["filename"], "frame.ARW")
            self.assertEqual(rows[1]["raw_path"], str(asset.raw_path))
            self.assertEqual(rows[1]["summary"], "Slightly dark foreground.")

    def test_edit_suggestion_fallback_retries_each_image(self) -> None:
        previews = [
            _build_preview("/tmp/a.ARW"),
            _build_preview("/tmp/b.ARW"),
        ]
        backend = MagicMock()
        backend.suggest_edits.side_effect = [
            VisionBackendError("bad cohort"),
            [EditSuggestion(filename="a.ARW", summary="A needs contrast.")],
            [EditSuggestion(filename="b.ARW", summary="B needs shadows.")],
        ]

        pairs, failed, fallback, errors = _suggest_edit_pairs_with_fallback(
            backend,
            "prompt",
            previews,
        )

        self.assertEqual([asset.filename for asset, _ in pairs], ["a.ARW", "b.ARW"])
        self.assertEqual(failed, 1)
        self.assertEqual(fallback, 2)
        self.assertEqual(errors, [])
        self.assertEqual(backend.suggest_edits.call_count, 3)


def _build_preview(raw_path: str) -> PreviewImage:
    path = Path(raw_path)
    return PreviewImage(
        asset=RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")),
        image_bytes=b"jpeg-bytes",
    )


if __name__ == "__main__":
    unittest.main()
