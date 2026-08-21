from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.benchmark import _auc
from cull_sh.benchmark import evaluate_signal
from cull_sh.benchmark import evaluate_reject_gate
from cull_sh.benchmark import load_human_labels
from cull_sh.benchmark import load_cull_run_signals
from cull_sh.benchmark import merge_source_records
from cull_sh.benchmark import paired_auc_delta_bootstrap


class BenchmarkTests(unittest.TestCase):
    def test_load_human_labels_reads_pick_reject_and_neutral(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_photo(root, "pick", 'xmp:Rating="5" xmpDM:Pick="1"')
            self._write_photo(root, "reject", 'xmp:Rating="-1" xmpDM:Pick="-1"')
            self._write_photo(root, "neutral", "")

            labels = {label.filename: label for label in load_human_labels(root)}

            self.assertEqual(labels["pick.ARW"].label, "pick")
            self.assertEqual(labels["reject.ARW"].label, "reject")
            self.assertEqual(labels["neutral.ARW"].label, "neutral")

    def test_load_human_labels_prefers_lightroom_desktop_good_flag(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_photo(
                root,
                "desktop-pick",
                'xmp:Rating="-1" xmpDM:Pick="-1" xmpDM:good="True"',
            )
            self._write_photo(
                root,
                "desktop-reject",
                'xmp:Rating="5" xmpDM:Pick="1" xmpDM:good="False"',
            )

            labels = {label.filename: label for label in load_human_labels(root)}

            self.assertEqual(labels["desktop-pick.ARW"].label, "pick")
            self.assertTrue(labels["desktop-pick.ARW"].good)
            self.assertEqual(labels["desktop-reject.ARW"].label, "reject")
            self.assertFalse(labels["desktop-reject.ARW"].good)

    def test_load_human_labels_can_match_lightroom_current_folder_scope(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nested = root / "DONE"
            nested.mkdir()
            self._write_photo(root, "top-level", 'xmpDM:good="True"')
            self._write_photo(nested, "nested", 'xmpDM:good="False"')

            labels = load_human_labels(root, include_subfolders=False)

            self.assertEqual([label.filename for label in labels], ["top-level.ARW"])

    def test_auc_gives_half_credit_to_ties(self) -> None:
        self.assertEqual(_auc([2.0, 1.0], [1.0]), 0.75)

    def test_evaluate_signal_reports_burst_ranking_and_safe_coverage(self) -> None:
        records = {
            "pick.arw": self._record("pick.ARW", "pick", 3.0, 1),
            "reject-a.arw": self._record("reject-a.ARW", "reject", 1.0, 1),
            "reject-b.arw": self._record("reject-b.ARW", "reject", 2.0, 1),
        }

        result = evaluate_signal(records, "score", "Score")

        self.assertEqual(result.auc, 1.0)
        self.assertEqual(result.pairwise_accuracy, 1.0)
        self.assertEqual(result.top1_recall, 1.0)
        self.assertEqual(result.top3_recall, 1.0)
        self.assertEqual(result.zero_false_reject_rejects, 2)
        self.assertAlmostEqual(result.zero_false_reject_coverage, 2 / 3)

    def test_merge_source_records_matches_filenames_case_insensitively(self) -> None:
        records = {
            "frame.arw": {
                "filename": "FRAME.ARW",
                "label": "pick",
                "signals": {},
                "burst_group_id": None,
                "date_taken": None,
            }
        }

        matched = merge_source_records(
            records,
            [
                {
                    "filename": "frame.arw",
                    "signals": {"topiq_score": 7.0},
                    "burst_group_id": 4,
                }
            ],
        )

        self.assertEqual(matched, 1)
        self.assertEqual(records["frame.arw"]["signals"]["topiq_score"], 7.0)
        self.assertEqual(records["frame.arw"]["burst_group_id"], 4)

    def test_paired_bootstrap_reports_observed_auc_delta(self) -> None:
        records = {
            "pick.arw": {
                "filename": "pick.ARW",
                "label": "pick",
                "signals": {"left": 3.0, "right": 2.0},
            },
            "reject.arw": {
                "filename": "reject.ARW",
                "label": "reject",
                "signals": {"left": 1.0, "right": 2.0},
            },
        }

        result = paired_auc_delta_bootstrap(
            records,
            "left",
            "Left",
            "right",
            "Right",
            iterations=20,
        )

        self.assertEqual(result["left_auc"], 1.0)
        self.assertEqual(result["right_auc"], 0.5)
        self.assertEqual(result["auc_delta"], 0.5)

    def test_evaluate_reject_gate_reports_false_reject_safety(self) -> None:
        records = {
            "pick.arw": {
                "filename": "pick.ARW",
                "label": "pick",
                "signals": {"gate": False},
            },
            "reject.arw": {
                "filename": "reject.ARW",
                "label": "reject",
                "signals": {"gate": True},
            },
        }

        result = evaluate_reject_gate(records, "gate", "Gate")

        self.assertEqual(result["reject_precision"], 1.0)
        self.assertEqual(result["false_reject_rate"], 0.0)
        self.assertEqual(result["reject_recall"], 1.0)

    def test_load_cull_run_signals_reuses_frozen_shadow_and_decision(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            record = {
                "filename": "frame.ARW",
                "raw_path": "/photos/frame.ARW",
                "scene_id": "scene-0001",
                "topiq_score": 6.5,
                "musiq_score": 70.0,
                "local_rank_score": 900.0,
                "local_trace": {"quality_reject": False},
                "decision": {"bucket": "reject"},
            }
            (run_dir / "manifest.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            rows = load_cull_run_signals(run_dir)

            self.assertEqual(rows[0]["burst_group_id"], "cull:scene-0001")
            self.assertEqual(rows[0]["signals"]["topiq_shadow_score"], 6.5)
            self.assertFalse(rows[0]["signals"]["local_gate_reject"])
            self.assertTrue(rows[0]["signals"]["pipeline_reject"])

    @staticmethod
    def _record(filename: str, label: str, score: float, burst: int) -> dict[str, object]:
        return {
            "filename": filename,
            "label": label,
            "signals": {"score": score},
            "burst_group_id": burst,
        }

    @staticmethod
    def _write_photo(root: Path, stem: str, attributes: str) -> None:
        (root / f"{stem}.ARW").write_bytes(b"raw")
        xmp = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/" {attributes}/>
  </rdf:RDF>
</x:xmpmeta>
'''
        (root / f"{stem}.xmp").write_text(xmp, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
