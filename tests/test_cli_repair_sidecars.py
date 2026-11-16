from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from cull_sh.cli import app
from cull_sh.xmp import sidecar_is_picked, sidecar_is_rejected


class RepairSidecarsCliTests(unittest.TestCase):
    def test_dry_run_preserves_existing_picks_and_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            picked_xmp = root / "picked.xmp"
            neutral_xmp = root / "neutral.xmp"
            self._write_xmp(picked_xmp, 'xmpDM:good="True"')
            self._write_xmp(neutral_xmp, "")
            original_pick = picked_xmp.read_bytes()
            original_neutral = neutral_xmp.read_bytes()
            self._write_manifest(run_dir, picked_xmp, neutral_xmp)

            result = CliRunner().invoke(
                app,
                [
                    "repair-sidecars",
                    "--run-dir",
                    str(run_dir),
                    "--preserve-existing-picks",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Mode: dry run", result.output)
            self.assertIn("Sidecars that would be rewritten: 1", result.output)
            self.assertIn("Existing picks preserved: 1", result.output)
            self.assertEqual(picked_xmp.read_bytes(), original_pick)
            self.assertEqual(neutral_xmp.read_bytes(), original_neutral)

    def test_write_mode_preserves_pick_and_replays_everything_else(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            picked_xmp = root / "picked.xmp"
            neutral_xmp = root / "neutral.xmp"
            self._write_xmp(picked_xmp, 'xmpDM:good="True"')
            self._write_xmp(neutral_xmp, "")
            original_pick = picked_xmp.read_bytes()
            self._write_manifest(run_dir, picked_xmp, neutral_xmp)

            result = CliRunner().invoke(
                app,
                [
                    "repair-sidecars",
                    "--run-dir",
                    str(run_dir),
                    "--preserve-existing-picks",
                    "--no-dry-run",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Mode: write", result.output)
            self.assertIn("Sidecars rewritten: 1", result.output)
            self.assertIn("Existing picks preserved: 1", result.output)
            self.assertEqual(picked_xmp.read_bytes(), original_pick)
            self.assertTrue(sidecar_is_picked(picked_xmp))
            self.assertTrue(sidecar_is_rejected(neutral_xmp))

    def test_embedded_dng_replay_does_not_create_an_xmp_sidecar(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dng = root / "edited.DNG"
            dng.touch()
            record = self._record(dng.with_suffix(".xmp"), "pick", 4)
            record.update(filename=dng.name, raw_path=str(dng))
            (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")

            args = ["repair-sidecars", "--run-dir", str(root), "--embed-dng-metadata", "--no-dry-run"]
            with patch("cull_sh.cli.write_embedded_cull_metadata") as write:
                result = CliRunner().invoke(app, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(write.call_args.args[0], dng)
            self.assertFalse(dng.with_suffix(".xmp").exists())

            dng.with_suffix(".xmp").touch()
            with patch("cull_sh.cli.write_embedded_cull_metadata") as write:
                result = CliRunner().invoke(app, args)
            self.assertNotEqual(result.exit_code, 0, result.output)
            write.assert_not_called()

    @staticmethod
    def _write_xmp(path: Path, attributes: str) -> None:
        path.write_text(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF><rdf:Description rdf:about="" {attributes}/></rdf:RDF>
</x:xmpmeta>
''',
            encoding="utf-8",
        )

    @staticmethod
    def _write_manifest(run_dir: Path, picked_xmp: Path, neutral_xmp: Path) -> None:
        records = [
            RepairSidecarsCliTests._record(picked_xmp, "pick", 4),
            RepairSidecarsCliTests._record(neutral_xmp, "reject", -1),
        ]
        with (run_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")

    @staticmethod
    def _record(xmp_path: Path, bucket: str, rating: int) -> dict[str, object]:
        return {
            "filename": f"{xmp_path.stem}.ARW",
            "asset_kind": "raw",
            "raw_path": str(xmp_path.with_suffix(".ARW")),
            "xmp_path": str(xmp_path),
            "decision": {
                "rating": rating,
                "label": "Green" if bucket == "pick" else "Red",
                "bucket": bucket,
                "keep": bucket == "pick",
                "source": "local",
                "summary": "fixture",
            },
        }


if __name__ == "__main__":
    unittest.main()
