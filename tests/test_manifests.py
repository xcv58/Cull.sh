from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.config import PipelineConfig
from cull_sh.manifests import decision_from_manifest_record
from cull_sh.manifests import find_latest_run_dir
from cull_sh.manifests import write_run_config
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource


class ManifestTests(unittest.TestCase):
    def test_write_run_config_includes_limit(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            config = PipelineConfig(
                path=Path("/tmp/photos"),
                prompt="test prompt",
                genre="flowers",
                limit=12,
            )

            config_path = write_run_config(run_dir, config)
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["limit"], 12)
            self.assertEqual(payload["limit_kind"], "scene")
            self.assertEqual(payload["genre"], "flowers")
            self.assertEqual(payload["min_tenengrad_score"], 45.0)
            self.assertEqual(payload["duplicate_hamming_threshold"], 6)
            self.assertIsNone(payload["max_scene_candidates"])
            self.assertFalse(payload["lightroom_auto_edit"])
            self.assertEqual(payload["lightroom_edit_scope"], "all")
            self.assertTrue(payload["include_jpegs"])
            self.assertTrue(payload["mirror_paired_jpegs"])
            self.assertTrue(payload["enable_topiq_shadow"])
            self.assertTrue(payload["enable_topiq_ranking"])
            self.assertEqual(payload["topiq_rank_weight"], 0.25)
            self.assertEqual(payload["topiq_shadow_workers"], 2)
            self.assertEqual(payload["topiq_shadow_low_percentile"], 0.2)
            self.assertEqual(payload["topiq_shadow_high_percentile"], 0.8)
            self.assertEqual(payload["topiq_shadow_max_items"], 80)
            self.assertEqual(payload["backend_timeout_seconds"], 300.0)
            self.assertEqual(payload["backend_max_attempts"], 3)
            self.assertEqual(payload["backend_max_output_tokens"], 1024)

    def test_find_latest_run_dir_returns_last_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "20260411-100000-000000").mkdir()
            (root / "20260411-110000-000000").mkdir()

            latest = find_latest_run_dir(root)

            self.assertEqual(latest.name, "20260411-110000-000000")

    def test_decision_from_manifest_record_parses_reject(self) -> None:
        record = {
            "filename": "frame.ARW",
            "decision": {
                "rating": -1,
                "label": "Red",
                "bucket": "reject",
                "keep": False,
                "source": "vision",
                "summary": "Rejected",
            },
        }

        decision = decision_from_manifest_record(record)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.rating, -1)
        self.assertEqual(decision.label, ColorLabel.RED)
        self.assertEqual(decision.bucket, DecisionBucket.REJECT)
        self.assertEqual(decision.source, DecisionSource.VISION)

    def test_decision_from_manifest_record_falls_back_to_legacy_keep(self) -> None:
        record = {
            "filename": "frame.ARW",
            "decision": {
                "rating": 4,
                "label": "Green",
                "keep": True,
                "source": "vision",
                "summary": "Picked",
            },
        }

        decision = decision_from_manifest_record(record)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.bucket, DecisionBucket.PICK)


if __name__ == "__main__":
    unittest.main()
