from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.scanner import discover_raw_assets


class ScannerTests(unittest.TestCase):
    def test_discover_raw_assets_sorts_supported_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ("b.ARW", "a.ARW", "c.ARW", "ignore.txt"):
                (root / name).write_bytes(b"")

            assets = discover_raw_assets(root, (".arw",))

            self.assertEqual([asset.filename for asset in assets], ["a.ARW", "b.ARW", "c.ARW"])


if __name__ == "__main__":
    unittest.main()
