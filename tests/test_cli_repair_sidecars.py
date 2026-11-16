from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from cull_sh.cli import app
from cull_sh.xmp import EmbeddedDngState
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

    def test_mixed_dng_replay_keeps_metadata_in_its_active_location(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edited = root / "edited.DNG"
            external = root / "external.DNG"
            pristine = root / "pristine.DNG"
            records = []
            for dng in (edited, external, pristine):
                dng.touch()
                record = self._record(dng.with_suffix(".xmp"), "pick", 4)
                record.update(filename=dng.name, raw_path=str(dng))
                records.append(record)
            (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            self._write_xmp(external.with_suffix(".xmp"), 'xmp:CreatorTool="Lightroom"')

            args = ["repair-sidecars", "--run-dir", str(root), "--no-dry-run"]
            states = {edited: EmbeddedDngState(has_develop_settings=True), pristine: EmbeddedDngState()}
            with (
                patch("cull_sh.xmp.read_embedded_dng_state", side_effect=states.__getitem__),
                patch("cull_sh.cli.write_embedded_cull_metadata") as write,
            ):
                result = CliRunner().invoke(app, args)
                self.assertNotEqual(result.exit_code, 0, result.output)
                write.assert_not_called()
                self.assertFalse(pristine.with_suffix(".xmp").exists())

                result = CliRunner().invoke(app, [*args, "--embed-dng-metadata"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Sidecars rewritten: 2", result.output)
            self.assertIn("Embedded DNG metadata records: 1", result.output)
            write.assert_called_once()
            self.assertEqual(write.call_args.args[0], edited)
            self.assertFalse(edited.with_suffix(".xmp").exists())
            self.assertTrue(sidecar_is_picked(pristine.with_suffix(".xmp")))
            self.assertTrue(sidecar_is_picked(external.with_suffix(".xmp")))
            self.assertIn('xmp:CreatorTool="Lightroom"', external.with_suffix(".xmp").read_text())

    def test_collision_with_raw_omitted_from_manifest_stops_replay_and_lens_writes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "same.ARW").touch()
            (root / "same.DNG").touch()
            self._write_manifest(root, root / "unrelated.xmp", root / "same.xmp")

            runner = CliRunner()
            for args in (
                ["repair-sidecars", "--run-dir", str(root), "--embed-dng-metadata"],
                ["lightroom-edit", "--path", str(root), "--limit", "1"],
                ["cull", "--path", str(root), "--prompt", "test"],
            ):
                with self.subTest(command=args[0]):
                    result = runner.invoke(app, [*args, "--no-dry-run"], env={"COLUMNS": "140"})
                    self.assertNotEqual(result.exit_code, 0, result.output)
                    self.assertIn("share the XMP sidecar", result.output)
                    self.assertEqual(list(root.glob("*.xmp")), [])

    def test_lens_pass_preserves_dng_with_embedded_culling_or_lens(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            picked = root / "picked.DNG"
            corrected = root / "corrected.DNG"
            for dng in (picked, corrected):
                dng.touch()
            states = {
                picked: EmbeddedDngState(has_culling_metadata=True),
                corrected: EmbeddedDngState(has_lens_corrections=True),
            }

            args = ["lightroom-edit", "--path", str(root)]
            runner = CliRunner()
            with (
                patch("cull_sh.xmp.read_embedded_dng_state", side_effect=states.__getitem__),
                patch("cull_sh.cli.write_embedded_lens_corrections") as write,
            ):
                result = runner.invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Embedded DNG lens edits planned: 1", result.output)
                write.assert_not_called()

                result = runner.invoke(app, [*args, "--no-dry-run"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Embedded DNG lens edits written: 1", result.output)
            self.assertIn("DNGs with lens corrections already embedded: 1", result.output)
            write.assert_called_once_with(picked)
            self.assertEqual(list(root.glob("*.xmp")), [])

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
