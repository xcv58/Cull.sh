from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.vlm_benchmark import build_vlm_cohorts
from cull_sh.vlm_benchmark import select_canary_cohorts
from cull_sh.vlm_benchmark import _write_markdown_report


class VLMBenchmarkTests(unittest.TestCase):
    def test_build_cohorts_uses_lightroom_desktop_labels_and_top_level_scope(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nested = root / "DONE"
            nested.mkdir()
            self._write_photo(root, "DSC0001", "True")
            self._write_photo(root, "DSC0002", "False")
            self._write_photo(nested, "DSC0003", "True")

            cohorts = build_vlm_cohorts(
                root,
                batch_size=4,
                include_subfolders=False,
            )

            self.assertEqual(len(cohorts), 1)
            self.assertEqual(
                [asset.filename for asset in cohorts[0].assets],
                ["DSC0001.ARW", "DSC0002.ARW"],
            )
            self.assertEqual(cohorts[0].human_labels, ("pick", "reject"))

    def test_canary_selection_spans_label_mixtures(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels = ["True", "False", None, "True", "False", None]
            for index, value in enumerate(labels, start=1):
                self._write_photo(root, f"DSC{index * 10:04d}", value)
            cohorts = build_vlm_cohorts(
                root,
                batch_size=1,
                include_subfolders=False,
            )

            selected = select_canary_cohorts(cohorts, 3)

            self.assertEqual(len(selected), 3)
            self.assertEqual(
                {cohort.human_labels[0] for cohort in selected},
                {"pick", "reject", "neutral"},
            )

    def test_markdown_report_includes_top3_and_paired_confidence_interval(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            report = Path(tmp_dir) / "report.md"
            _write_markdown_report(
                report,
                {
                    "ground_truth": {"photos": 2, "picks": 1, "rejects": 1, "neutral": 0},
                    "models": [
                        {
                            "label": "candidate",
                            "photos_covered": 2,
                            "first_pass_success_rate": 1.0,
                            "false_rejects": 0,
                            "false_reject_rate": 0.0,
                            "pick_precision": 1.0,
                            "pick_recall": 1.0,
                            "photos_per_minute": 2.0,
                            "signal": {"auc": 0.75, "top1_recall": 0.5, "top3_recall": 1.0},
                        }
                    ],
                    "comparisons": [
                        {
                            "left_display_name": "candidate",
                            "right_display_name": "baseline",
                            "auc_delta": -0.25,
                            "delta_ci95_low": -0.4,
                            "delta_ci95_high": -0.1,
                            "bootstrap_iterations": 2000,
                        }
                    ],
                },
            )

            contents = report.read_text(encoding="utf-8")
            self.assertIn("| Top-1 | Top-3 |", contents)
            self.assertIn("| candidate | baseline | -0.250 | -0.400 to -0.100 | 2000 |", contents)

    @staticmethod
    def _write_photo(root: Path, stem: str, good: str | None) -> None:
        (root / f"{stem}.ARW").write_bytes(b"raw")
        attribute = f'xmpDM:good="{good}"' if good is not None else ""
        (root / f"{stem}.xmp").write_text(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/" {attribute}/>
  </rdf:RDF>
</x:xmpmeta>
''',
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
