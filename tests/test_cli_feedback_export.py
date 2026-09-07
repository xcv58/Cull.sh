from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from click import unstyle
from typer.testing import CliRunner

from cull_sh.cli import app


class FeedbackExportCliTests(unittest.TestCase):
    def test_requires_explicit_unattended_acknowledgement(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage = root / "stage"
            stage.mkdir()

            result = CliRunner().invoke(
                app,
                [
                    "rapidraw-feedback-export",
                    "--stage",
                    str(stage),
                    "--output",
                    str(root / "delivery"),
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("pass --unattended", unstyle(result.output))


if __name__ == "__main__":
    unittest.main()
