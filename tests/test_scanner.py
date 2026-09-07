from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.scanner import discover_jpeg_assets
from cull_sh.scanner import discover_photo_assets
from cull_sh.scanner import discover_raw_assets


class ScannerTests(unittest.TestCase):
    def test_discover_raw_assets_sorts_supported_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ("b.ARW", "a.ARW", "c.ARW", "ignore.txt"):
                (root / name).write_bytes(b"")

            assets = discover_raw_assets(root, (".arw",))

            self.assertEqual([asset.filename for asset in assets], ["a.ARW", "b.ARW", "c.ARW"])

    def test_discover_raw_assets_ignores_appledouble_companions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "frame.ARW").write_bytes(b"raw")
            (root / "._frame.ARW").write_bytes(b"appledouble")

            assets = discover_raw_assets(root, (".arw",))

            self.assertEqual([asset.filename for asset in assets], ["frame.ARW"])

    def test_discover_photo_assets_includes_jpegs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ("a.ARW", "b.JPG", "c.jpeg", "ignore.txt"):
                (root / name).write_bytes(b"")

            assets = discover_photo_assets(
                root,
                raw_extensions=(".arw",),
                include_jpegs=True,
                jpeg_extensions=(".jpg", ".jpeg"),
            )

            self.assertEqual(
                [asset.filename for asset in assets],
                ["a.ARW", "b.JPG", "c.jpeg"],
            )
            self.assertEqual([asset.is_jpeg for asset in assets], [False, True, True])

    def test_discover_jpeg_assets_ignores_appledouble_companions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "frame.JPG").write_bytes(b"jpeg")
            (root / "._frame.JPG").write_bytes(b"appledouble")

            assets = discover_jpeg_assets(root, (".jpg",))

            self.assertEqual([asset.filename for asset in assets], ["frame.JPG"])

    def test_discover_jpeg_assets_marks_same_stem_raw_pairs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "frame.ARW").write_bytes(b"")
            (root / "frame.JPG").write_bytes(b"")
            (root / "orphan.JPG").write_bytes(b"")
            raw_assets = discover_raw_assets(root, (".arw",))

            jpeg_assets = discover_jpeg_assets(
                root,
                (".jpg",),
                raw_assets=raw_assets,
                mirror_paired_jpegs=True,
            )

            by_name = {asset.filename: asset for asset in jpeg_assets}
            self.assertEqual(by_name["frame.JPG"].paired_raw_path, root / "frame.ARW")
            self.assertTrue(by_name["frame.JPG"].mirrors_paired_raw)
            self.assertIsNone(by_name["orphan.JPG"].paired_raw_path)


if __name__ == "__main__":
    unittest.main()
