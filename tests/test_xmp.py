from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import xml.etree.ElementTree as ET

from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LightroomEditScope
from cull_sh.xmp import CRS_NS
from cull_sh.xmp import NAMESPACES
from cull_sh.xmp import jpeg_is_rejected
from cull_sh.xmp import sidecar_is_rejected
from cull_sh.xmp import write_jpeg_metadata
from cull_sh.xmp import write_lightroom_edit_sidecar
from cull_sh.xmp import write_xmp_sidecar


class XmpSidecarTests(unittest.TestCase):
    def test_write_new_sidecar(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            decision = FinalDecision(
                filename="frame.ARW",
                rating=-1,
                label=ColorLabel.RED,
                bucket=DecisionBucket.REJECT,
                source=DecisionSource.LOCAL,
            )

            write_xmp_sidecar(target, decision)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(
                description.get("{http://ns.adobe.com/xap/1.0/}Rating"),
                "-1",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xap/1.0/}Label"),
                "Red",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}Pick"),
                "-1",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}good"),
                "False",
            )

    def test_merge_existing_sidecar_preserves_unrelated_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            target.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <rdf:RDF>
    <rdf:Description rdf:about="" dc:format="image/x-sony-arw" xmp:Rating="2" xmp:Label="Blue"/>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )
            decision = FinalDecision(
                filename="frame.ARW",
                rating=5,
                label=None,
                bucket=DecisionBucket.PICK,
                source=DecisionSource.VISION,
            )

            write_xmp_sidecar(target, decision)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(
                description.get("{http://ns.adobe.com/xap/1.0/}Rating"),
                "5",
            )
            self.assertIsNone(description.get("{http://ns.adobe.com/xap/1.0/}Label"))
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}Pick"),
                "1",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}good"),
                "True",
            )
            self.assertEqual(
                description.get("{http://purl.org/dc/elements/1.1/}format"),
                "image/x-sony-arw",
            )

    def test_vision_reject_is_forced_to_yellow(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            decision = FinalDecision(
                filename="frame.ARW",
                rating=-1,
                label=None,
                bucket=DecisionBucket.REJECT,
                source=DecisionSource.VISION,
            )

            write_xmp_sidecar(target, decision)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(
                description.get("{http://ns.adobe.com/xap/1.0/}Rating"),
                "-1",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xap/1.0/}Label"),
                "Yellow",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}Pick"),
                "-1",
            )
            self.assertEqual(
                description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}good"),
                "False",
            )

    def test_review_clears_pick_and_reject_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            target.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF>
    <rdf:Description rdf:about="" xmp:Rating="5" xmp:Label="Green" xmpDM:Pick="1" xmpDM:good="True"/>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )
            decision = FinalDecision(
                filename="frame.ARW",
                rating=0,
                label=None,
                bucket=DecisionBucket.REVIEW,
                source=DecisionSource.VISION,
            )

            write_xmp_sidecar(target, decision)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertIsNone(description.get("{http://ns.adobe.com/xap/1.0/}Rating"))
            self.assertIsNone(description.get("{http://ns.adobe.com/xap/1.0/}Label"))
            self.assertIsNone(description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}Pick"))
            self.assertIsNone(description.get("{http://ns.adobe.com/xmp/1.0/DynamicMedia/}good"))

    def test_write_lightroom_edit_sidecar_applies_lens_corrections(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"

            write_lightroom_edit_sidecar(target)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(description.get(f"{{{CRS_NS}}}LensProfileEnable"), "1")
            self.assertEqual(
                description.get(f"{{{CRS_NS}}}LensProfileSetup"),
                "LensDefaults",
            )
            self.assertEqual(
                description.get(f"{{{CRS_NS}}}LensProfileDistortionScale"),
                "100",
            )
            self.assertIsNone(description.get(f"{{{CRS_NS}}}CameraProfile"))
            self.assertIsNone(description.find("crs:Look", NAMESPACES))

    def test_write_xmp_sidecar_applies_lens_corrections_to_kept_decision(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            decision = FinalDecision(
                filename="frame.ARW",
                rating=4,
                label=ColorLabel.GREEN,
                bucket=DecisionBucket.PICK,
                source=DecisionSource.VISION,
            )

            write_xmp_sidecar(target, decision, apply_lightroom_edit=True)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(description.get(f"{{{CRS_NS}}}LensProfileEnable"), "1")
            self.assertIsNone(description.find("crs:Look", NAMESPACES))

    def test_write_xmp_sidecar_applies_lens_corrections_to_reject_by_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            decision = FinalDecision(
                filename="frame.ARW",
                rating=-1,
                label=ColorLabel.RED,
                bucket=DecisionBucket.REJECT,
                source=DecisionSource.LOCAL,
            )

            write_xmp_sidecar(target, decision, apply_lightroom_edit=True)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertEqual(description.get(f"{{{CRS_NS}}}LensProfileEnable"), "1")
            self.assertIsNone(description.find("crs:Look", NAMESPACES))

    def test_write_xmp_sidecar_can_skip_lightroom_edit_for_reject(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            decision = FinalDecision(
                filename="frame.ARW",
                rating=-1,
                label=ColorLabel.RED,
                bucket=DecisionBucket.REJECT,
                source=DecisionSource.LOCAL,
            )

            write_xmp_sidecar(
                target,
                decision,
                apply_lightroom_edit=True,
                lightroom_edit_scope=LightroomEditScope.KEPT,
            )

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertIsNone(description.get(f"{{{CRS_NS}}}LensProfileEnable"))
            self.assertIsNone(description.find("crs:Look", NAMESPACES))

    def test_lightroom_edit_removes_static_adaptive_look_without_ai_payload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            target.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <crs:Look>
        <rdf:Description crs:Name="Adaptive Color"/>
      </crs:Look>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            write_lightroom_edit_sidecar(target)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertIsNone(description.find("crs:Look", NAMESPACES))
            self.assertIsNone(description.get(f"{{{CRS_NS}}}CameraProfile"))

    def test_lightroom_edit_preserves_adaptive_look_with_ai_payload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "frame.xmp"
            target.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <crs:AILook crs:Active="true" crs:AILookData="abc"/>
      <crs:Look>
        <rdf:Description crs:Name="Adaptive Color"/>
      </crs:Look>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            write_lightroom_edit_sidecar(target)

            tree = ET.parse(target)
            description = tree.getroot().find("rdf:RDF/rdf:Description", NAMESPACES)
            self.assertIsNotNone(description)
            assert description is not None
            self.assertIsNotNone(description.find("crs:AILook", NAMESPACES))
            self.assertIsNotNone(description.find("crs:Look", NAMESPACES))

    def test_sidecar_is_rejected_reads_rating_and_pick(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rejected_by_rating = Path(tmp_dir) / "rating.xmp"
            rejected_by_pick = Path(tmp_dir) / "pick.xmp"
            kept = Path(tmp_dir) / "kept.xmp"
            rejected_by_rating.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="" xmp:Rating="-1"/>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )
            rejected_by_pick.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF>
    <rdf:Description rdf:about="" xmpDM:Pick="-1"/>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )
            kept.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="" xmp:Rating="4"/>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            self.assertTrue(sidecar_is_rejected(rejected_by_rating))
            self.assertTrue(sidecar_is_rejected(rejected_by_pick))
            self.assertFalse(sidecar_is_rejected(kept))
            self.assertFalse(sidecar_is_rejected(Path(tmp_dir) / "missing.xmp"))

    def test_write_jpeg_metadata_uses_embedded_xmp_tags(self) -> None:
        decision = FinalDecision(
            filename="frame.JPG",
            rating=-1,
            label=ColorLabel.RED,
            bucket=DecisionBucket.REJECT,
            source=DecisionSource.LOCAL,
        )
        result = Mock(returncode=0, stdout="", stderr="")

        with (
            patch("cull_sh.xmp.shutil.which", return_value="/usr/bin/exiftool"),
            patch("cull_sh.xmp.subprocess.run", return_value=result) as run,
        ):
            write_jpeg_metadata(Path("/tmp/frame.JPG"), decision)

        command = run.call_args.args[0]
        self.assertIn("-overwrite_original", command)
        self.assertIn("-XMP-xmp:Rating=-1", command)
        self.assertIn("-XMP-xmpDM:Pick=-1", command)
        self.assertIn("-XMP-xmp:Label=Red", command)

    def test_write_jpeg_metadata_clears_review_flags(self) -> None:
        decision = FinalDecision(
            filename="frame.JPG",
            rating=0,
            label=None,
            bucket=DecisionBucket.REVIEW,
            source=DecisionSource.VISION,
        )
        result = Mock(returncode=0, stdout="", stderr="")

        with (
            patch("cull_sh.xmp.shutil.which", return_value="/usr/bin/exiftool"),
            patch("cull_sh.xmp.subprocess.run", return_value=result) as run,
        ):
            write_jpeg_metadata(Path("/tmp/frame.JPG"), decision)

        command = run.call_args.args[0]
        self.assertIn("-XMP-xmp:Rating=", command)
        self.assertIn("-XMP-xmpDM:Pick=", command)
        self.assertIn("-XMP-xmp:Label=", command)

    def test_jpeg_is_rejected_reads_embedded_xmp(self) -> None:
        result = Mock(
            returncode=0,
            stdout='[{"SourceFile":"/tmp/frame.JPG","Rating":-1}]',
            stderr="",
        )

        with (
            patch("cull_sh.xmp.shutil.which", return_value="/usr/bin/exiftool"),
            patch("cull_sh.xmp.subprocess.run", return_value=result),
        ):
            self.assertTrue(jpeg_is_rejected(Path("/tmp/frame.JPG")))


if __name__ == "__main__":
    unittest.main()
