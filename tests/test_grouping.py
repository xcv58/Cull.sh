from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.grouping import assign_scene_groups
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem


class SceneGroupingTests(unittest.TestCase):
    def test_assign_scene_groups_splits_on_sequence_gap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = [
                root / "DSC0001.ARW",
                root / "DSC0002.ARW",
                root / "DSC0008.ARW",
            ]
            for path in paths:
                path.write_bytes(b"raw")

            items = [
                WorkItem(asset=RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")))
                for path in paths
            ]

            scene_count = assign_scene_groups(items, max_gap_seconds=999.0, max_sequence_gap=2)

            self.assertEqual(scene_count, 2)
            self.assertEqual(items[0].scene_id, "scene-0001")
            self.assertEqual(items[1].scene_id, "scene-0001")
            self.assertEqual(items[2].scene_id, "scene-0002")
            self.assertEqual(items[0].scene_index, 1)
            self.assertEqual(items[1].scene_index, 2)
            self.assertEqual(items[2].scene_index, 1)


if __name__ == "__main__":
    unittest.main()
