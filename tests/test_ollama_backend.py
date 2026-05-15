from __future__ import annotations

import unittest

from cull_sh.backends.ollama import _normalize_label
from cull_sh.models import ColorLabel


class OllamaBackendTests(unittest.TestCase):
    def test_normalize_label_accepts_null_like_values(self) -> None:
        self.assertIsNone(_normalize_label(None))
        self.assertIsNone(_normalize_label(""))
        self.assertIsNone(_normalize_label("null"))
        self.assertIsNone(_normalize_label("None"))

    def test_normalize_label_maps_supported_colors(self) -> None:
        self.assertEqual(_normalize_label("green"), ColorLabel.GREEN)
        self.assertEqual(_normalize_label("Purple"), ColorLabel.PURPLE)


if __name__ == "__main__":
    unittest.main()
