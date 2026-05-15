from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from cull_sh.config import DEFAULT_EXTENSIONS
from cull_sh.lightroom_ui import build_adaptive_color_stage
from cull_sh.lightroom_ui import write_adaptive_color_handoff
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LightroomEditScope
from cull_sh.xmp import write_xmp_sidecar


class LightroomUiTests(unittest.TestCase):
    def test_build_adaptive_color_stage_counts_candidates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for filename in ("a.ARW", "b.ARW", "c.ARW", "d.ARW", "e.ARW"):
                (root / filename).write_bytes(b"raw")
            write_xmp_sidecar(
                root / "b.xmp",
                FinalDecision(
                    filename="b.ARW",
                    rating=-1,
                    label=ColorLabel.RED,
                    bucket=DecisionBucket.REJECT,
                    source=DecisionSource.LOCAL,
                ),
            )
            _write_adaptive_color_sidecar(root / "c.xmp", ai_payload=True)
            _write_adaptive_color_sidecar(root / "d.xmp", ai_payload=False)
            (root / "e.acr").write_bytes(b"lightroom-ai-payload")

            stage = build_adaptive_color_stage(root, DEFAULT_EXTENSIONS)

            self.assertEqual(stage.total, 5)
            self.assertEqual(stage.rejected, 1)
            self.assertEqual(stage.non_rejected, 4)
            self.assertEqual(stage.candidates, 5)
            self.assertEqual(stage.already_adaptive_color, 2)
            self.assertEqual(stage.pending_adaptive_color, 3)
            self.assertEqual(stage.lens_corrections_enabled, 2)
            self.assertEqual(stage.rejected_filenames, ["b.ARW"])
            self.assertEqual(stage.adaptive_color_filenames, ["c.ARW", "e.ARW"])
            self.assertEqual(stage.lightroom_acr_filenames, ["e.ARW"])
            self.assertEqual(stage.pending_filenames, ["a.ARW", "b.ARW", "d.ARW"])

    def test_build_adaptive_color_stage_can_scope_to_kept_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for filename in ("a.ARW", "b.ARW"):
                (root / filename).write_bytes(b"raw")
            write_xmp_sidecar(
                root / "b.xmp",
                FinalDecision(
                    filename="b.ARW",
                    rating=-1,
                    label=ColorLabel.RED,
                    bucket=DecisionBucket.REJECT,
                    source=DecisionSource.LOCAL,
                ),
            )

            stage = build_adaptive_color_stage(
                root,
                DEFAULT_EXTENSIONS,
                edit_scope=LightroomEditScope.KEPT,
            )

            self.assertEqual(stage.total, 2)
            self.assertEqual(stage.rejected, 1)
            self.assertEqual(stage.non_rejected, 1)
            self.assertEqual(stage.candidates, 1)
            self.assertEqual(stage.pending_filenames, ["a.ARW"])

    def test_build_adaptive_color_stage_counts_embedded_dng_payloads(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "drone.DNG").write_bytes(b"dng")

            exiftool_result = Mock(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "AILookActive": True,
                            "LookName": "Adaptive Color",
                            "LensProfileEnable": 1,
                        }
                    ]
                ),
            )
            with (
                patch("cull_sh.lightroom_ui.shutil.which", return_value="/usr/bin/exiftool"),
                patch("cull_sh.lightroom_ui.subprocess.run", return_value=exiftool_result),
            ):
                stage = build_adaptive_color_stage(root, DEFAULT_EXTENSIONS)

            self.assertEqual(stage.total, 1)
            self.assertEqual(stage.candidates, 1)
            self.assertEqual(stage.already_adaptive_color, 1)
            self.assertEqual(stage.pending_adaptive_color, 0)
            self.assertEqual(stage.lens_corrections_enabled, 1)
            self.assertEqual(stage.lightroom_acr_filenames, [])
            self.assertEqual(stage.embedded_dng_filenames, ["drone.DNG"])

    def test_write_adaptive_color_handoff(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "frame.ARW").write_bytes(b"raw")
            stage = build_adaptive_color_stage(root, DEFAULT_EXTENSIONS)
            run_dir = root / "run"

            payload_path, checklist_path = write_adaptive_color_handoff(stage, run_dir)

            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            checklist = checklist_path.read_text(encoding="utf-8")
            self.assertEqual(payload["edit_scope"], "all")
            self.assertEqual(payload["edit_candidates"], 1)
            self.assertEqual(payload["non_rejected_candidates"], 1)
            self.assertEqual(payload["pending_adaptive_color"], 1)
            self.assertEqual(
                payload["automation_mode"],
                "seed_profile_selective_copy_paste",
            )
            self.assertEqual(payload["seed_filename"], "frame.ARW")
            self.assertEqual(payload["verification_filenames"], ["frame.ARW"])
            self.assertIn("Copy Edit Settings", payload["computer_use_instruction"])
            self.assertIn("only the profile/treatment", payload["computer_use_instruction"])
            self.assertIn("AI settings update", payload["computer_use_instruction"])
            self.assertIn("Profile dropdown", checklist)


def _write_adaptive_color_sidecar(path: Path, ai_payload: bool) -> None:
    ai_look = '<crs:AILook crs:Active="true" crs:AILookData="abc"/>' if ai_payload else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="" crs:LensProfileEnable="1">
      {ai_look}
      <crs:Look>
        <rdf:Description crs:Name="Adaptive Color"/>
      </crs:Look>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
