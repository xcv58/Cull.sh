from __future__ import annotations

import unittest

from cull_sh.edit_benchmark import select_evenly_spaced


class EditBenchmarkTests(unittest.TestCase):
    def test_select_evenly_spaced_uses_quantile_centers(self) -> None:
        self.assertEqual(select_evenly_spaced(list(range(10)), 4), [1, 3, 6, 8])

    def test_select_evenly_spaced_rejects_oversampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested 3 items from only 2"):
            select_evenly_spaced([1, 2], 3)


if __name__ == "__main__":
    unittest.main()
